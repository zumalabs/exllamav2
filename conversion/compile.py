from exllamav2.model import \
(
    ExLlamaV2Embedding,
    ExLlamaV2Attention,
    ExLlamaV2MLP,
    ExLlamaV2MoEMLP,
    ExLlamaV2Linear,
    ExLlamaV2RMSNorm
)

import os, glob, shutil
from safetensors import safe_open
from safetensors.torch import save_file

def _tsize(t):

    return t.nelement() * t.element_size()

def _dsize(d):

    size = 0
    for _, v in d.items(): size += _tsize(v)
    return size


def get_f_module(job, module):

    mod_dict = {}
    module.load()
    mod_dict[module.key + ".weight"] = module.get_weight()
    return mod_dict


def get_q_module(job, module):

    mod_dict = {}
    filename = os.path.join(job["out_dir"], "out_tensor/" + module.key + ".safetensors")
    with safe_open(filename, framework = "pt", device = "cpu") as f:
        for k in f.keys():
            mod_dict[k] = f.get_tensor(k)
    return mod_dict


def compile_model(job, save_fn, model):

    out_dict = {}
    current_size = 0
    file_index = 1
    index = 0
    shard_bytes = job["shard_size"] * 1024 ** 2

    while index < len(model.modules):

        module = model.modules[index]

        if isinstance(module, ExLlamaV2Embedding):

            d = get_f_module(job, module); out_dict |= d; current_size += _dsize(d)

        if isinstance(module, ExLlamaV2Attention):

            d = get_f_module(job, module.input_layernorm); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.q_proj); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.k_proj); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.v_proj); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.o_proj); out_dict |= d; current_size += _dsize(d)

        if isinstance(module, ExLlamaV2MLP):

            d = get_f_module(job, module.post_attention_layernorm); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.gate_proj); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.up_proj); out_dict |= d; current_size += _dsize(d)
            d = get_q_module(job, module.down_proj); out_dict |= d; current_size += _dsize(d)

        if isinstance(module, ExLlamaV2MoEMLP):

            d = get_f_module(job, module.post_attention_layernorm); out_dict |= d; current_size += _dsize(d)
            d = get_f_module(job, module.gate); out_dict |= d; current_size += _dsize(d)
            for i in range(model.config.num_experts):
                d = get_q_module(job, module.w1[i]); out_dict |= d; current_size += _dsize(d)
                d = get_q_module(job, module.w3[i]); out_dict |= d; current_size += _dsize(d)
                d = get_q_module(job, module.w2[i]); out_dict |= d; current_size += _dsize(d)

        if isinstance(module, ExLlamaV2RMSNorm):

            d = get_f_module(job, module); out_dict |= d; current_size += _dsize(d)

        if isinstance(module, ExLlamaV2Linear):

            assert module.key == "lm_head"
            d = get_q_module(job, module); out_dict |= d; current_size += _dsize(d)

        index += 1

        # Save shard

        if current_size > shard_bytes or index == len(model.modules):

            save_dict = {}
            dont_save_dict = {}
            this_shard_size = 0
            for k, v in out_dict.items():
                tsize = _tsize(v)
                if this_shard_size + tsize <= shard_bytes:
                    this_shard_size += tsize
                    current_size -= tsize
                    save_dict[k] = v
                else:
                    dont_save_dict[k] = v

            if len(save_dict) == 0:

                print(f" ## Error: Unable to fit output tensor in single shard.")
                os._exit(0)

            while True:

                print(f" -- Writing shard {file_index}...")

                out_dir = job["out_dir"]
                if job["compile_full"] is not None: out_dir = job["compile_full"]
                if not os.path.exists(out_dir):
                    print(f" -- Creating directory {out_dir}")
                    os.makedirs(out_dir)

                out_filename = os.path.join(out_dir, f"output_temp_{file_index}.safetensors")
                save_file(save_dict, out_filename)
                file_index += 1

                out_dict = dont_save_dict

                if index == len(model.modules) and len(out_dict) > 0:
                    save_dict = dont_save_dict
                    dont_save_dict = {}
                    continue

                break

    num_files = file_index - 1
    if num_files == 1:

        final_filename = os.path.join(out_dir, "output.safetensors")
        os.rename(out_filename, final_filename)

        filesize = os.path.getsize(final_filename) // (1024 ** 2)
        print(f" --   {final_filename} ({filesize:,} MB)")

    else:

        print(f" -- Saved model weights:")

        for i in range(num_files):
            temp_filename = os.path.join(out_dir, f"output_temp_{i + 1}.safetensors")
            final_filename = os.path.join(out_dir, f"output-{i + 1:05}-of-{num_files:05}.safetensors")
            os.rename(temp_filename, final_filename)

            filesize = os.path.getsize(final_filename) // (1024 ** 2)
            print(f" --   {final_filename} ({filesize:,} MB)")

    # Copy all non-tensor files from the model's directory if compiling a full model

    if job["compile_full"] is not None:

        print(f" -- Copying non-tensor files to output directory {out_dir}")

        input_dir = model.config.model_dir
        all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        tensor_files = glob.glob(os.path.join(input_dir, "*.safetensors"))
        tensor_files_set = set(tensor_files)
        non_tensor_files = [f for f in all_files if os.path.join(input_dir, f) not in tensor_files_set]

        for f in non_tensor_files:
            print(f" --   {f}")
            source_file_path = os.path.join(input_dir, f)
            target_file_path = os.path.join(out_dir, f)
            shutil.copy(source_file_path, target_file_path)




