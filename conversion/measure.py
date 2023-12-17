from exllamav2.model import \
(
    ExLlamaV2Embedding,
    ExLlamaV2Attention,
    ExLlamaV2MLP,
    ExLlamaV2MoEMLP,
    ExLlamaV2Linear,
    ExLlamaV2RMSNorm
)

from safetensors import safe_open
from safetensors.torch import save_file
from conversion.qparams import QParams, qparams_headoptions, qparams_attn, qparams_mlp, get_qparams_reduced
from conversion.adaptivegptq import AdaptiveGPTQ
import torch
from torch import nn
import os, time, math, json
import torch.nn.functional as F
import gc

def list_live_tensors():

    tensors = {}
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in tensors.keys():
                    tensors[d] += 1
                else:
                    tensors[d] = 1
        except:
            pass

    print("-----------")
    for k, v in tensors.items():
        print(f"{v} : {k}")


# Get initial token embeddings

def embeddings(job, save_fn, model, measure = False):

    module = model.modules[0]
    assert isinstance(module, ExLlamaV2Embedding)

    with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
        input_ids = f.get_tensor("input_ids")

    module.load()
    input_ids[input_ids >= module.native_vocab_size] = 0
    hidden_state = module.forward(input_ids)
    module.unload()

    embeddings_dict = { f"row.{i:05}": hidden_state[i:i+1, :, :] for i in range(hidden_state.shape[0]) }
    save_file(embeddings_dict, os.path.join(job["out_dir"], "hidden_states.safetensors"))


# Test quantization options

def test_quant(source: ExLlamaV2Linear,
               lq: AdaptiveGPTQ,
               qparams: list):

    variants = []
    variants_bits = []

    original = nn.Linear(source.in_features, source.out_features, False, device = "meta", dtype = torch.float16)
    original.weight = nn.Parameter(source.linear.weight.clone())

    for qp in qparams:

        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
        lq.quantize()
        quantized = lq.apply_temp()
        quantized.to("cpu")

        variants.append(quantized)
        total_bits = qp.total_bits(quantized.weight.T.shape)
        variants_bits.append(total_bits)

        bpw = total_bits / quantized.weight.numel()
        desc = qp.desc

        print(f" -- {source.key:50} {desc:50} {bpw:2.2f} bpw")

    return variants, variants_bits


def test_error(module, hidden_states, target_states, cache, attn_mask):

    rfn_sum = 0
    rfn_count = 0
    for x, xref in zip(hidden_states, target_states):
        x = x.cuda()
        xref = xref.cuda()
        xtest = module.forward(x, cache, attn_mask)
        xtest = xtest[0].float()
        xref = xref[0].float()
        rfn_sum += torch.linalg.norm(xtest - xref, 'fro') / torch.linalg.norm(xref, 'fro')
        rfn_count += 1

    return max(1e-6, 1 - (rfn_sum / rfn_count)).item()


def measure_attn(module, hidden_states, target_states, quantizers, cache, attn_mask):

    qjobs, qmaps = get_qparams_reduced(qparams_attn)
    results = []

    quantizers["q_proj"].prepare()
    quantizers["k_proj"].reuse_h(quantizers["q_proj"])
    quantizers["v_proj"].reuse_h(quantizers["q_proj"])
    quantizers["o_proj"].prepare()

    options_q, bits_q = test_quant(module.q_proj, quantizers["q_proj"], qjobs[0])
    options_k, bits_k = test_quant(module.k_proj, quantizers["k_proj"], qjobs[1])
    options_v, bits_v = test_quant(module.v_proj, quantizers["v_proj"], qjobs[2])
    options_o, bits_o = test_quant(module.o_proj, quantizers["o_proj"], qjobs[3])

    total_numel = module.q_proj.numel()
    total_numel += module.k_proj.numel()
    total_numel += module.v_proj.numel()
    total_numel += module.o_proj.numel()

    (q_, k_, v_, o_) = (-1, -1, -1, -1)
    for (q, k, v, o) in qmaps:

        if q != q_: module.q_proj.linear.weight = nn.Parameter(options_q[q].weight.cuda())
        if k != k_: module.k_proj.linear.weight = nn.Parameter(options_k[k].weight.cuda())
        if v != v_: module.v_proj.linear.weight = nn.Parameter(options_v[v].weight.cuda())
        if o != o_: module.o_proj.linear.weight = nn.Parameter(options_o[o].weight.cuda())
        (q_, k_, v_, o_) = (q, k, v, o)

        total_bits = bits_q[q]
        total_bits += bits_k[k]
        total_bits += bits_v[v]
        total_bits += bits_o[o]
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_mask)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "q_proj": qjobs[0][q].get_dict(),
              "k_proj": qjobs[1][k].get_dict(),
              "v_proj": qjobs[2][v].get_dict(),
              "o_proj": qjobs[3][o].get_dict() }
        results.append(r)

    return results


def measure_mlp(module, hidden_states, target_states, quantizers, cache, attn_mask):

    qjobs, qmaps = get_qparams_reduced(qparams_mlp)
    results = []

    quantizers["gate_proj"].prepare()
    quantizers["up_proj"].reuse_h(quantizers["gate_proj"])
    quantizers["down_proj"].prepare()

    options_g, bits_g = test_quant(module.gate_proj, quantizers[f"gate_proj"], qjobs[0])
    options_u, bits_u = test_quant(module.up_proj, quantizers[f"up_proj"], qjobs[1])
    options_d, bits_d = test_quant(module.down_proj, quantizers[f"down_proj"], qjobs[2])

    total_numel = module.gate_proj.numel()
    total_numel += module.up_proj.numel()
    total_numel += module.down_proj.numel()

    (g_, u_, d_) = (-1, -1, -1)
    for (g, u, d) in qmaps:

        if g != g_: module.gate_proj.linear.weight = nn.Parameter(options_g[g].weight.cuda())
        if u != u_: module.up_proj.linear.weight = nn.Parameter(options_u[u].weight.cuda())
        if d != d_: module.down_proj.linear.weight = nn.Parameter(options_d[d].weight.cuda())
        (g_, u_, d_) = (g, u, d)

        total_bits = bits_g[g]
        total_bits += bits_u[u]
        total_bits += bits_d[d]
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_mask)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "gate_proj": qjobs[0][g].get_dict(),
              "up_proj": qjobs[1][u].get_dict(),
              "down_proj": qjobs[2][d].get_dict() }
        results.append(r)

    return results


def measure_moe_mlp(module, hidden_states, target_states, quantizers, cache, attn_mask):

    qjobs, qmaps = get_qparams_reduced(qparams_mlp)
    num_experts = module.model.config.num_experts
    results = []

    quantizers["w1.0"].prepare()
    for i in range(num_experts):
        if i > 0: quantizers[f"w1.{i}"].reuse_h(quantizers["w1.0"])
        quantizers[f"w2.{i}"].prepare()
        quantizers[f"w3.{i}"].reuse_h(quantizers["w1.0"])

    options_g, bits_g = [], []
    options_u, bits_u = [], []
    options_d, bits_d = [], []
    for i in range(num_experts):
        options_g_, bits_g_ = test_quant(module.w1[i], quantizers[f"w1.{i}"], qjobs[0])
        del quantizers[f"w1.{i}"]
        options_u_, bits_u_ = test_quant(module.w3[i], quantizers[f"w3.{i}"], qjobs[1])
        del quantizers[f"w3.{i}"]
        options_d_, bits_d_ = test_quant(module.w2[i], quantizers[f"w2.{i}"], qjobs[2])
        del quantizers[f"w2.{i}"]
        options_g.append(options_g_)
        options_u.append(options_u_)
        options_d.append(options_d_)
        bits_g.append(bits_g_)
        bits_u.append(bits_u_)
        bits_d.append(bits_d_)

    quantizers.clear()
    gc.collect()
    torch.cuda.empty_cache()

    total_numel = sum(module.w1[i].numel() for i in range(num_experts))
    total_numel += sum(module.w3[i].numel() for i in range(num_experts))
    total_numel += sum(module.w2[i].numel() for i in range(num_experts))

    (g_, u_, d_) = (-1, -1, -1)
    for (g, u, d) in qmaps:

        for i in range(num_experts):
            if g != g_: module.w1[i].linear.weight = nn.Parameter(options_g[i][g].weight.cuda())
            if u != u_: module.w3[i].linear.weight = nn.Parameter(options_u[i][u].weight.cuda())
            if d != d_: module.w2[i].linear.weight = nn.Parameter(options_d[i][d].weight.cuda())
        (g_, u_, d_) = (g, u, d)

        total_bits = sum(bits_g[i][g] for i in range(num_experts))
        total_bits += sum(bits_u[i][u] for i in range(num_experts))
        total_bits += sum(bits_d[i][d] for i in range(num_experts))
        total_bpw = total_bits / total_numel

        accuracy = test_error(module, hidden_states, target_states, cache, attn_mask)
        print(f" -- {total_bpw:1.4f} bpw  accuracy: {accuracy:1.8f}")

        torch.cuda.empty_cache()

        r = { "accuracy": accuracy,
              "total_bits": total_bits,
              "w1": qjobs[0][g].get_dict(),
              "w3": qjobs[1][u].get_dict(),
              "w2": qjobs[2][d].get_dict() }
        results.append(r)

    return results


@torch.inference_mode()
def measure_quant(job, save_fn, model):

    snapshot_interval = 10
    temp_filename = os.path.join(job["out_dir"], "hidden_states_temp.safetensors")
    states_filename = os.path.join(job["out_dir"], "hidden_states.safetensors")
    measurement = job.get("measurement", {})

    # Quantize

    if not "last_module_idx" in job:
        job["last_module_idx"] = 0

    hidden_states = []
    with safe_open(states_filename, framework = "pt", device = "cpu") as f:
        for k in sorted(f.keys()):
            hidden_states.append(f.get_tensor(k))

    index = job["last_module_idx"]
    while True:

        index += 1
        if index >= len(model.modules): break

        # Prepare module

        module = model.modules[index]
        module.load()

        print(f" -- Layer: {module.key} ({module.name})")

        # Create quantizers

        quantizers = {}

        if isinstance(module, ExLlamaV2Attention):
            mode = "self_attn"
            quantizers["q_proj"] = AdaptiveGPTQ(module.q_proj.linear)
            quantizers["k_proj"] = AdaptiveGPTQ(module.k_proj.linear)
            quantizers["v_proj"] = AdaptiveGPTQ(module.v_proj.linear)
            quantizers["o_proj"] = AdaptiveGPTQ(module.o_proj.linear)

        elif isinstance(module, ExLlamaV2MLP):
            mode = "mlp"
            quantizers["gate_proj"] = AdaptiveGPTQ(module.gate_proj.linear)
            quantizers["up_proj"] = AdaptiveGPTQ(module.up_proj.linear)
            quantizers["down_proj"] = AdaptiveGPTQ(module.down_proj.linear)

        elif isinstance(module, ExLlamaV2MoEMLP):
            mode = "block_sparse_moe"
            for i in range(model.config.num_experts):
                quantizers[f"w1.{i}"] = AdaptiveGPTQ(module.w1[i].linear)
                quantizers[f"w3.{i}"] = AdaptiveGPTQ(module.w3[i].linear)
                quantizers[f"w2.{i}"] = AdaptiveGPTQ(module.w2[i].linear)

        elif isinstance(module, ExLlamaV2Linear):
            mode = "linear"
            # Don't measure head layer

        elif isinstance(module, ExLlamaV2RMSNorm):
            mode = "norm"

        # Reference forward pass

        cache = None
        attn_mask = model.build_attn_mask(1, hidden_states[0].shape[1], 0, None, "cuda:0") if mode == "self_attn" else None

        target_states = []
        if mode == "block_sparse_moe":
            uncalibrated_experts = [0 for _ in range(model.config.num_experts)]

        for i in range(len(hidden_states)):

            x = hidden_states[i].to("cuda:0")
            outputs = module.forward(x, cache, attn_mask, intermediates = True)

            # Hessians

            if mode == "self_attn":
                quantizers["q_proj"].add_batch(outputs["post_norm"])  # Reuse H for K and V
                quantizers["o_proj"].add_batch(outputs["attn_output"])

            if mode == "mlp":
                quantizers["gate_proj"].add_batch(outputs["post_norm"])  # Reuse H for up_proj
                quantizers["down_proj"].add_batch(outputs["pre_down"])

            if mode == "block_sparse_moe":
                for j in range(model.config.num_experts):
                    if f"pre_down.{j}" in outputs:
                        quantizers[f"w1.{j}"].add_batch(outputs["post_norm"])
                        quantizers[f"w2.{j}"].add_batch(outputs[f"pre_down.{j}"])
                        if outputs[f"pre_down.{j}"].shape[0] < outputs["post_norm"].shape[0] / 10:
                            uncalibrated_experts[j] += 1
                    else:
                        uncalibrated_experts[j] += 1

            target_states.append(outputs["hidden_states"].to("cpu"))

        # For MoE layers, warn if any layer received less than 10% of a calibration batch

        if mode == "block_sparse_moe":
            for j in range(model.config.num_experts):
                ue = uncalibrated_experts[j]
                if ue > 0:
                    print(f" !! Warning: w2.{j} has less than 10% calibration for {ue}/{len(hidden_states)} rows")

        # Measurement

        m = None

        if mode == "self_attn":
            m = measure_attn(module, hidden_states, target_states, quantizers, cache, attn_mask)

        if mode == "mlp":
            m = measure_mlp(module, hidden_states, target_states, quantizers, cache, attn_mask)

        if mode == "block_sparse_moe":
            m = measure_moe_mlp(module, hidden_states, target_states, quantizers, cache, attn_mask)

        measurement[module.key + "." + mode] = m

        # Unload module

        module.unload()
        torch.cuda.empty_cache()

        # Advance

        hidden_states = target_states

        # Checkpoint

        if index % snapshot_interval == 0 or index == len(model.modules) - 1:

            save_dict = {f"row.{idx:05}": h for idx, h in enumerate(hidden_states)}
            save_file(save_dict, temp_filename)
            save_dict = None

            job["invalid"] = True
            save_fn()

            os.remove(states_filename)
            os.rename(temp_filename, states_filename)

            job["measurement"] = measurement.copy()
            job["last_module_idx"] = index

            del job["invalid"]
            save_fn()

    # Export measurement

    exp_measurement = { "measurement": job["measurement"],
                        "last_module_idx": job["last_module_idx"] }

    measurement_files = [os.path.join(job["out_dir"], "measurement.json")]
    if job["output_measurement"] is not None:
        measurement_files += [job["output_measurement"]]
        print(f" -- Writing {job['output_measurement']}")

    for filename in measurement_files:
        with open(filename, "w", encoding = "utf8") as f:
            f.write(json.dumps(exp_measurement, indent = 4))


