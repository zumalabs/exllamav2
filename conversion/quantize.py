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


# Quantize

def quant_linear(job: dict,
                 source: ExLlamaV2Linear,
                 lq: AdaptiveGPTQ,
                 qparams: dict,
                 drop = False):

    qp = QParams.from_dict(qparams)
    print(f" -- Linear: {source.key} -> {qp.get_desc()}, {qp.bpw(source.linear.weight.T.shape):.2f} bpw")

    # Quantize

    lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
    lq.quantize(keep_qweight = True, apply = True, drop = drop)

    # Pack and save quantized layer

    packed_dict = lq.pack(source.key, qp)
    tensorfile = os.path.join(job["out_dir"], "out_tensor/" + source.key + ".safetensors")
    save_file(packed_dict, tensorfile)

    # Drop buffers from quantizer to free VRAM

    if drop: lq.drop_buffers()

    # Reconstruct from packed layer

    recons_linear = ExLlamaV2Linear(source.model, source.key, source.in_features, source.out_features, False)
    recons_linear.device_idx = source.device_idx
    recons_dict = {}
    for k in ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups"]:
        recons_dict[k] = packed_dict[source.key + "." + k]
    recons_dict["q_perm"] = torch.argsort(recons_dict["q_invperm"]).to(torch.int)
    recons_linear.load(recons_dict)

    # Sanity test to ensure reconstructed matrix matches unpacked matrix

    quant_w = source.linear.weight.T
    recons_w = recons_linear.get_weight_tensor_dq()

    ident = torch.eye(recons_linear.in_features, dtype = torch.half).cuda()
    recons_w2 = recons_linear.forward(ident, force_cuda = True)

    recons_w2.sub_(quant_w)
    recons_w2.abs_()
    diff2 = torch.max(recons_w2)

    quant_w.sub_(recons_w)
    quant_w.abs_()
    diff1 = torch.max(quant_w)
    quant_w = None

    if diff1 > 0.01 or diff2 > 0.01:
        print(" ## Quantization error (2)")
        os._exit(0)

    # Free reconstructed linear layer

    recons_linear.unload()

    # Apply reconstructed matrix to source layer

    source.linear.weight.data = recons_w.T


def quant_attn(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat):

    quantizers["q_proj"].prepare()
    quantizers["k_proj"].reuse_h(quantizers["q_proj"])
    quantizers["v_proj"].reuse_h(quantizers["q_proj"])
    quantizers["o_proj"].prepare()

    quant_linear(job, module.q_proj, quantizers["q_proj"], strat["q_proj"])
    quant_linear(job, module.k_proj, quantizers["k_proj"], strat["k_proj"])
    quant_linear(job, module.v_proj, quantizers["v_proj"], strat["v_proj"])
    quant_linear(job, module.o_proj, quantizers["o_proj"], strat["o_proj"])


def quant_mlp(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat):

    quantizers["gate_proj"].prepare()
    quantizers["up_proj"].reuse_h(quantizers["gate_proj"])
    quantizers["down_proj"].prepare()

    quant_linear(job, module.gate_proj, quantizers["gate_proj"], strat["gate_proj"])
    del quantizers[f"gate_proj"]
    quant_linear(job, module.up_proj, quantizers["up_proj"], strat["up_proj"])
    del quantizers[f"up_proj"]
    quant_linear(job, module.down_proj, quantizers["down_proj"], strat["down_proj"])
    del quantizers[f"down_proj"]


def quant_moe_mlp(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat):

    num_experts = module.model.config.num_experts

    quantizers["w1.0"].prepare()
    for i in range(num_experts):
        if i > 0: quantizers[f"w1.{i}"].reuse_h(quantizers["w1.0"])
        quantizers[f"w2.{i}"].prepare()
        quantizers[f"w3.{i}"].reuse_h(quantizers["w1.0"])

    for i in range(num_experts):
        quant_linear(job, module.w1[i], quantizers[f"w1.{i}"], strat["w1"])
        del quantizers[f"w1.{i}"]
        quant_linear(job, module.w3[i], quantizers[f"w3.{i}"], strat["w3"])
        del quantizers[f"w3.{i}"]
        quant_linear(job, module.w2[i], quantizers[f"w2.{i}"], strat["w2"])
        del quantizers[f"w2.{i}"]


def quant_lm_head(job, module, hidden_states, quantizers, cache, attn_mask):

    quantizers["lm_head"].prepare()

    qp = qparams_headoptions[job["head_bits"]]
    quant_linear(job, module, quantizers["lm_head"], qp.get_dict())


# def testc(module, states, target_states, norm, layers):
#
#     rows = len(states)
#     cols = states[0].shape[1]
#     dim = module.model.config.hidden_size
#
#     a_batch = torch.empty((rows * cols, dim), dtype = torch.float, device = "cuda:0")
#     b_batch = torch.empty((rows * cols, dim), dtype = torch.float, device = "cuda:0")
#
#     r = 0
#     for state, target_state in zip(states, target_states):
#         a = norm.forward(state.to("cuda:0"))
#         b = norm.forward(target_state.to("cuda:0"))
#         a_batch[r:r+cols] = a.view(-1, dim)
#         b_batch[r:r+cols] = b.view(-1, dim)
#         r += cols
#
#     # diff = F.mse_loss(b_batch, a_batch)
#     m_a = torch.mean(a_batch.abs(), dim = 0)
#     m_b = torch.mean(b_batch.abs(), dim = 0)
#     m_ab = m_b / m_a
#     # a_batch *= m_ab
#     # diff = F.mse_loss(b_batch, a_batch)
#     norm.weight.data *= m_ab
#
#     # s = torch.linalg.lstsq(a_batch, b_batch)
#     # s = s.solution
#     #
#     # for linear in layers:
#     #     m = torch.matmul(s, linear.linear.weight.data.T.float())
#     #     linear.linear.weight.data = nn.Parameter(m.T.half())
#
#     xx = 0


@torch.inference_mode()
def quant(job, save_fn, model):

    snapshot_interval = 10
    temp_filename = os.path.join(job["out_dir"], "hidden_states_temp.safetensors")
    states_filename = os.path.join(job["out_dir"], "hidden_states.safetensors")
    strategy = job["strategy"]

    # Quantize

    if not "q_last_module_idx" in job:
        job["q_last_module_idx"] = 0

    hidden_states = []
    # hidden_i_states = []
    with safe_open(states_filename, framework = "pt", device = "cpu") as f:
        for k in sorted(f.keys()):
            if k.startswith("row"):
                hidden_states.append(f.get_tensor(k))
            # elif k.startswith("i_row"):
            #     hidden_i_states.append(f.get_tensor(k))

    index = job["q_last_module_idx"]
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
            # if index > 1: testc(module, hidden_states, hidden_i_states, module.input_layernorm, [module.q_proj, module.k_proj, module.v_proj])
            quantizers["q_proj"] = AdaptiveGPTQ(module.q_proj.linear)
            quantizers["k_proj"] = AdaptiveGPTQ(module.k_proj.linear)
            quantizers["v_proj"] = AdaptiveGPTQ(module.v_proj.linear)
            quantizers["o_proj"] = AdaptiveGPTQ(module.o_proj.linear)

        elif isinstance(module, ExLlamaV2MLP):
            mode = "mlp"
            # testc(module, hidden_states, hidden_i_states, module.post_attention_layernorm, [module.gate_proj, module.up_proj])
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
            assert module.key == "lm_head"
            quantizers["lm_head"] = AdaptiveGPTQ(module.linear)

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

            if mode == "linear":
                quantizers["lm_head"].add_batch(x)

            if mode != "linear":
                target_states.append(outputs["hidden_states"].to("cpu"))

        # For MoE layers, warn if any layer received less than 10% of a calibration batch

        if mode == "block_sparse_moe":
            for j in range(model.config.num_experts):
                ue = uncalibrated_experts[j]
                if ue > len(hidden_states) * 0.10:
                    print(f" !! Warning: w2.{j} has less than 10% calibration for {ue}/{len(hidden_states)} rows")

        # Conversion

        if mode == "self_attn":
            strat = strategy[module.key + "." + mode]
            quant_attn(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat)

        if mode == "mlp":
            strat = strategy[module.key + "." + mode]
            quant_mlp(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat)

        if mode == "block_sparse_moe":
            strat = strategy[module.key + "." + mode]
            quant_moe_mlp(job, module, hidden_states, target_states, quantizers, cache, attn_mask, strat)

        if mode == "linear":
            quant_lm_head(job, module, hidden_states, quantizers, cache, attn_mask)

        quantizers.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # Post-quantization forward pass

        if mode == "linear":
            with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
                cal_ids = f.get_tensor("input_ids")

        rfn_sum = 0
        rfn_count = 0
        logprob_sum = 0.0
        logprob_count = 0

        q_states = []
        for i in range(len(hidden_states)):

            if mode != "linear":

                x = hidden_states[i].to("cuda:0")
                output = module.forward(x, cache, attn_mask)
                q_states.append(output.to("cpu"))

                output = output[0].float()
                output_ref = target_states[i].to("cuda:0")
                output_ref = output_ref[0].float()

                rfn_sum += (torch.linalg.norm(output - output_ref, 'fro') / torch.linalg.norm(output_ref, 'fro')).item()
                rfn_count += 1

            elif i < job["measurement_rows"]:

                x = hidden_states[i].to("cuda:0")
                output = module.forward(x, cache, attn_mask)
                if module.padding > 0: output = output[:, :, :-module.padding]

                logits = output[:, :-1, :]
                logits = logits.float() + 1e-10
                target_ids = cal_ids[i:i+1, 1:].to("cuda:0")

                log_probs = F.log_softmax(logits, dim = -1)
                token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                logprob_sum += token_log_probs.sum().item()
                logprob_count += target_ids.numel()

        if mode != "linear":

            err = rfn_sum / rfn_count
            print(f" -- Module quantized, rfn_error: {err:1.6f}")

        else:

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)

            print(f" -- Module quantized, calibration perplexity (quant): {perplexity:.4f}")

        # Unload module

        module.unload()
        torch.cuda.empty_cache()

        # Advance

        if mode != "linear":
            # hidden_i_states = hidden_states
            # hidden_states = target_states
            # hidden_states = [(x + y) / 2 for x, y in zip(target_states, q_states)]
            hidden_states = q_states

        # Checkpoint

        if index % snapshot_interval == 0 or index == len(model.modules) - 1:

            if mode != "linear":
                save_dict = {f"row.{idx:05}": h for idx, h in enumerate(hidden_states)}
                # save_dict |= {f"i_row.{idx:05}": h for idx, h in enumerate(hidden_i_states)}
                save_file(save_dict, temp_filename)
                save_dict = None

            job["invalid"] = True
            save_fn()

            if mode != "linear":
                os.remove(states_filename)
                os.rename(temp_filename, states_filename)

            job["q_last_module_idx"] = index

            del job["invalid"]
            save_fn()

