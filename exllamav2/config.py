import torch
from safetensors import safe_open
import os, glob, json

class ExLlamaV2Config:

    debug_mode = False
    model_dir: str = None                       # Directory containing model files

    max_seq_len: int = 2048                     # Maximum sequence length. Sequences longer than this will throw an exception
    max_batch_size: int = 1                     # Maximum size of batches to process
    max_input_len: int = 2048                   # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
    max_attention_size: int = 2048 ** 2         # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this

    scale_pos_emb: float = 1.0                  # Factor by which to scale positional embeddings, e.g. for 4096-token sequence use a scaling factor of 2.0, requires finetuned model or LoRA
    scale_alpha_value: float = 1.0              # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb but works without finetuned model

    no_flash_attn: bool = False                 # Implementation will automatically use flash-attn-2 when available

    # Loaded/set by .prepare():

    architecture: str

    model_config: str
    tensor_file_map: dict
    tensor_files: list

    tokenizer_path: str

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    hidden_size: int
    initializer_range: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    rotary_embedding_base: float = 10000.0      # Constant for all Llama models, nodified by .prepare() if scale_alpha_value != 1.0
    head_dim: int = 128                         # Constant for all Llama models, except 3b
    num_experts: int = 1
    num_experts_per_token: int = 1

    qkv_embed: bool = False


    def __init__(self):
        pass


    # Set low-mem options

    def set_low_mem(self):

        self.qkv_embed = True
        self.max_input_len = 1024
        self.max_attention_size = 1024 ** 2


    # Populate config with required files from model_dir

    def prepare(self, no_tensors = False):

        assert self.model_dir is not None, "No model_dir specified in ExLlamaV2Config"
        assert os.path.exists(self.model_dir), "Can't find " + self.model_dir

        # Load config.json

        self.model_config = os.path.join(self.model_dir, "config.json")
        assert os.path.exists(self.model_config), "Can't find " + self.model_config

        with open(self.model_config, encoding = "utf8") as f:
            read_config = json.load(f)

            layer_keys = []
            expect_keys = []
            layer_keys_llama_norms = [["input_layernorm"],
                                      ["post_attention_layernorm"]]
            layer_keys_yi_norms = [["ln1", "input_layernorm"],
                                   ["ln2", "post_attention_layernorm"]]
            layer_keys_llama_attn = [["self_attn.q_proj"],
                                     ["self_attn.k_proj"],
                                     ["self_attn.v_proj"],
                                     ["self_attn.o_proj"]]
            layer_keys_llama_mlp = [["mlp.down_proj"],
                                    ["mlp.gate_proj"],
                                    ["mlp.up_proj"]]
            expect_keys_llama = [["lm_head"],
                                 ["model.norm"],
                                 ["model.embed_tokens"]]

            if "LlamaForCausalLM" in read_config["architectures"]:
                self.architecture = "Llama"
                layer_keys += \
                    layer_keys_llama_norms + \
                    layer_keys_llama_attn + \
                    layer_keys_llama_mlp
                expect_keys += \
                    expect_keys_llama

            elif "MistralForCausalLM" in read_config["architectures"]:
                self.architecture = "Llama"
                layer_keys += \
                    layer_keys_llama_norms + \
                    layer_keys_llama_attn + \
                    layer_keys_llama_mlp
                expect_keys += \
                    expect_keys_llama

            elif "YiForCausalLM" in read_config["architectures"]:
                self.architecture = "Yi"
                layer_keys += \
                    layer_keys_yi_norms + \
                    layer_keys_llama_attn + \
                    layer_keys_llama_mlp
                expect_keys += \
                    expect_keys_llama

            elif "MixtralForCausalLM" in read_config["architectures"]:
                self.architecture = "Mixtral"
                self.num_experts = read_config["num_local_experts"]
                self.num_experts_per_token = read_config["num_experts_per_tok"]
                layer_keys += \
                    layer_keys_llama_norms + \
                    layer_keys_llama_attn + \
                    [[f"block_sparse_moe.experts.{e}.w{w}" for e in range(8) for w in range(3)]] + \
                    [["block_sparse_moe.gate"]]
                expect_keys += \
                    expect_keys_llama

            else:
                print(f" !! Warning, unknown architecture: {repr(read_config['architectures'])}")
                print(f" !! Loading as LlamaForCausalLM")
                self.architecture = "Llama"
                layer_keys += \
                    layer_keys_llama_norms + \
                    layer_keys_llama_attn + \
                    layer_keys_llama_mlp
                expect_keys += \
                    expect_keys_llama

            self.bos_token_id = read_config["bos_token_id"] if "bos_token_id" in read_config else 1
            self.eos_token_id = read_config["eos_token_id"] if "eos_token_id" in read_config else 2
            self.pad_token_id = read_config["pad_token_id"] if "pad_token_id" in read_config else 0

            self.hidden_size = read_config["hidden_size"]
            self.initializer_range = read_config["initializer_range"]
            self.intermediate_size = read_config["intermediate_size"]
            self.num_attention_heads = read_config["num_attention_heads"]
            self.num_hidden_layers = read_config["num_hidden_layers"]
            self.rms_norm_eps = read_config["rms_norm_eps"]
            self.vocab_size = read_config["vocab_size"]

            self.rotary_embedding_base = read_config["rope_theta"] if "rope_theta" in read_config else 10000.0

            if "num_key_value_heads" in read_config:
                self.num_key_value_heads = read_config["num_key_value_heads"]
                self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
            else:
                self.num_key_value_heads = self.num_attention_heads
                self.num_key_value_groups = 1

            if "max_sequence_length" in read_config: self.max_seq_len = read_config["max_sequence_length"]
            elif "max_position_embeddings" in read_config: self.max_seq_len = read_config["max_position_embeddings"]

        # Model dimensions

        self.head_dim = self.hidden_size // self.num_attention_heads

        # Create map of model tensors

        if no_tensors: return

        self.tensor_file_map = {}

        st_pattern = os.path.join(self.model_dir, "*.safetensors")
        self.tensor_files = glob.glob(st_pattern)

        if len(self.tensor_files) == 0:
            raise ValueError(f" ## No .safetensors files found in {self.model_dir}")

        for st_file in self.tensor_files:

            with safe_open(st_file, framework = "pt", device = "cpu") as f:
                for key in f.keys():
                    self.tensor_file_map[key] = st_file

        # Make sure we found all the layers we need

        for layer_idx in range(self.num_hidden_layers):
            for ks in layer_keys:
                prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                expect_keys.append(prefixes)

        for prefixes in expect_keys:
            for prefix in prefixes:
                if any(key.startswith(prefix) for key in self.tensor_file_map):
                    break
            else:
                raise ValueError(f" ## Could not find {prefix}.* in model")

