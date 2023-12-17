
import sys
min_version = (3, 8)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

# Set CUDA context to lazy loading since we won't need 95% of the modules in Torch

import os
os.environ['CUDA_MODULE_LOADING']='LAZY'

import torch
import math
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.embedding import ExLlamaV2Embedding
# from exllamav2.util import list_live_tensors, print_vram_usage, set_snapshot, diff_snapshot, print_vram_usage_peak
from exllamav2.compat import safe_move_tensor
import gc

def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2DeviceTensors:

    model = None
    device_idx: int
    ready: bool

    scratch_bytes: int
    scratch_idx: int

    sin: torch.tensor
    cos: torch.tensor

    scratch: torch.tensor = None


    def __init__(self, model, device_idx, scratch_bytes):

        self.model = model
        self.device_idx = device_idx
        self.ready = False
        self.scratch_bytes = scratch_bytes
        self.scratch_idx = 0


    def prepare(self, scratch):

        self.prepare_sincos()

        if scratch:
            self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = _torch_device(self.device_idx))

        self.ready = True


    def begin_scratch_alloc(self):

        self.scratch_idx = 0


    def get_scratch_slice(self, size_bytes):

        if self.scratch is None: self.prepare(True)

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, self.scratch_idx, size_half)
        self.scratch_idx += size_half
        return scratch_slice


    def prepare_sincos(self):

        base = self.model.config.rotary_embedding_base
        alpha = self.model.config.scale_alpha_value
        scale = self.model.config.scale_pos_emb
        head_dim = self.model.config.head_dim
        device = _torch_device(self.device_idx)

        if alpha != 1.0: base *= alpha ** (self.model.config.head_dim / (self.model.config.head_dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device = device).float() / head_dim))
        t = torch.arange(self.model.config.max_seq_len, device = device, dtype = torch.float32)

        if scale != 1.0: t /= scale

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.sin = emb.sin()[None, None, :, :].half()
        self.cos = emb.cos()[None, None, :, :].half()


class ExLlamaV2:

    config: ExLlamaV2Config
    modules: list = []
    modules_dict: dict = {}
    device_tensors: list = []
    cache_map: dict
    last_kv_layer_idx: int
    head_layer_idx: int
    loaded: bool


    def __init__(self, config: ExLlamaV2Config, lazy_load = False):

        self.config = config
        self.modules = []
        self.modules_dict = {}
        self.device_tensors = []
        self.cache_map = {}
        self.loaded = False

        # Build model

        self.modules.append(ExLlamaV2Embedding(self, "model.embed_tokens"))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]

        for layer_idx in range(self.config.num_hidden_layers):

            self.modules.append(ExLlamaV2Attention(self, f"model.layers.{layer_idx}", layer_idx))
            for m in self.modules[-1].submodules: self.modules_dict[m.key] = m
            if self.config.architecture == "Mixtral":
                self.modules.append(ExLlamaV2MoEMLP(self, f"model.layers.{layer_idx}", layer_idx))
            else:
                self.modules.append(ExLlamaV2MLP(self, f"model.layers.{layer_idx}", layer_idx))
            for m in self.modules[-1].submodules: self.modules_dict[m.key] = m


        self.modules.append(ExLlamaV2RMSNorm(self, "model.norm"))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]

        self.head_layer_idx = len(self.modules)
        self.modules.append(ExLlamaV2Linear(self, "lm_head", self.config.hidden_size, self.config.vocab_size, False))
        self.modules_dict[self.modules[-1].key] = self.modules[-1]

        # Find last layer that affects k/v cache

        layer_idx = len(self.modules)
        while True:
            layer_idx -= 1
            if isinstance(self.modules[layer_idx], ExLlamaV2Attention):
                break

        self.last_kv_layer_idx = layer_idx


    def set_device_map(self, allocation, embed_cpu = True):

        self.cache_map = {}

        # Constant shared between layers

        sincos_size = self.config.head_dim * self.config.max_seq_len * 2
        constant_size = sincos_size * 2

        # Max size of hidden state
        # TODO: Option to reserve space for cache while loading model

        state_size = self.config.hidden_size * self.config.max_input_len * self.config.max_batch_size * 2
        mask_size = self.config.max_input_len ** 2 * 2

        # Bytes remaining per device

        allocation_bytes =  [a * 1024**3 - (constant_size + state_size + mask_size) for a in allocation]

        # Scratch space required per device

        reserve_bytes = [0 for a in allocation]
        reserve_bytes_attn = [0 for a in allocation]
        fixed_bytes = [0 for a in allocation]

        current_idx = 0
        for idx, module in enumerate(self.modules):

            # Special case for token embeddings on CPU

            if idx == 0 and embed_cpu:

                module.set_device_idx(-1)
                continue

            # Special case for attention

            attn_bytes_current = 0
            if isinstance(module, ExLlamaV2Attention): attn_bytes_current = module.temp_attn_size()

            # Advance current_idx until module fits in allocation

            footprint = module.weight_footprint()   # Footprint, in bytes
            scratch = module.scratch_space()        # Scratch space required by module

            while True:
                assert current_idx < len(allocation_bytes), "Insufficient space in device allocation"
                dev_scratch = max(scratch, reserve_bytes[current_idx])
                dev_scratch_attn = max(attn_bytes_current, reserve_bytes_attn[current_idx])
                if footprint + dev_scratch + dev_scratch_attn <= allocation_bytes[current_idx]: break
                current_idx += 1

            # Size for fixed tensors

            scratch_fixed = module.scratch_space_fixed()
            fixed_bytes[current_idx] = max(scratch_fixed, fixed_bytes[current_idx])

            # Subtract module size from allocation

            reserve_bytes[current_idx] = dev_scratch
            reserve_bytes_attn[current_idx] = dev_scratch_attn
            allocation_bytes[current_idx] -= footprint

            module.set_device_idx(current_idx)

        # Prepare to prepare device tensors

        self.device_tensors = []
        for idx, scratch_bytes in enumerate(fixed_bytes):
            self.device_tensors.append(ExLlamaV2DeviceTensors(self, idx, scratch_bytes))

        # Create map for cache

        self.set_cache_map()

        # Return unused space, in GB

        return [(ab - rb - rba) / 1024**3 for (ab, rb, rba) in zip(allocation_bytes, reserve_bytes, reserve_bytes_attn)]


    def load(self, gpu_split = None, lazy = False, stats = False, callback = None, callback_gen = None):
        f = self.load_gen(gpu_split, lazy, stats, callback, callback_gen)
        for item in f: return item

    def load_gen(self, gpu_split = None, lazy = False, stats = False, callback = None, callback_gen = None):

        assert not self.config.qkv_embed or not lazy, "Lazy initialization is unsupported when config.qkv_embed = True"

        with torch.inference_mode():

            stats_ = self.set_device_map(gpu_split or [99999])

            # Load module weights

            if not lazy:

                for idx, module in enumerate(self.modules):

                    if callback is not None: callback(idx, len(self.modules))
                    if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                    module.load()

                if callback is not None: callback(len(self.modules), len(self.modules))
                if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            # Cache map

            self.set_cache_map()

            self.loaded = True
            if stats: yield gpu_split, stats_
            else: yield gpu_split


    def load_autosplit(self, cache, reserve_vram = None, last_id_only = False, callback = None, callback_gen = None):
        f = self.load_autosplit_gen(cache, reserve_vram, last_id_only, callback, callback_gen)
        for item in f: x = item

    def load_autosplit_gen(self, cache, reserve_vram = None, last_id_only = False, callback = None, callback_gen = None):

        assert not self.config.qkv_embed, "Auto GPU split is unsupported when config.qkv_embed = True"

        minimum_reserve_vram = 192 * 1024**2
        last_touched_device = -1
        current_device = 0
        num_devices = torch.torch.cuda.device_count()
        loras = None  # TODO:

        with torch.inference_mode():

            self.device_tensors = []

            # Reserved space

            if reserve_vram is None:
                reserve_vram = [192 * 1024**2] + [0] * (num_devices - 1)

            reserved_vram_tensors = []
            minimum_reserve_tensor = None

            # Largest hidden state to ever forward through model

            hidden_state = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
            batch_size, seq_len = hidden_state.shape
            past_len = 0
            attn_mask = None

            # Size of fixed scratch space

            scratch_fixed = 0
            for module in self.modules:
                scratch_fixed = max(scratch_fixed, module.scratch_space_fixed())

            # Load modules and create cache tensors sequentially

            self.cache_map = {}
            for idx, module in enumerate(self.modules):

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                # Embedding layer on CPU

                if idx == 0:

                    module.set_device_idx(-1)
                    module.load()
                    hidden_state = module.forward(hidden_state)
                    continue

                while True:

                    # If we've reached a new device, allocate fixed tensors and attention mask

                    if current_device > last_touched_device:

                        self.device_tensors.append(ExLlamaV2DeviceTensors(self, current_device, scratch_fixed))
                        if attn_mask is not None:
                            reserved_vram_tensors.append(attn_mask)
                            attn_mask = safe_move_tensor(attn_mask, _torch_device(current_device))
                        else:
                            attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, None, _torch_device(current_device))

                        b = reserve_vram[current_device]
                        reserved_vram_tensors.append(torch.empty((b,), dtype = torch.int8, device = _torch_device(current_device)))
                        minimum_reserve_tensor = torch.empty((minimum_reserve_vram,), dtype = torch.int8, device = _torch_device(current_device))

                        last_touched_device = current_device

                    # Attempt to load module and forward state

                    module.set_device_idx(current_device)

                    hidden_state_backup = safe_move_tensor(hidden_state, "cpu").clone()

                    try:
                        if isinstance(module, ExLlamaV2Attention):
                            self.cache_map[module.layer_idx] = module.device()
                            cache.update_cache_tensors()

                        module.load()

                        if idx == self.head_layer_idx:
                            if last_id_only:
                                hidden_state = hidden_state.narrow(-2, -1, 1)

                        hidden_state = safe_move_tensor(hidden_state, _torch_device(current_device))
                        hidden_state = module.forward(hidden_state, cache = cache, attn_mask = attn_mask, past_len = past_len, loras = loras)
                        fail = False

                    except Exception as e:

                        test = 0
                        if ("CUDA out of memory" in str(e)) or ("HIP out of memory" in str(e)):
                            fail = True  # Exception object will hold references to tensors so we can't free them here
                        else:
                            raise

                    # If we failed, roll back and advance to next device

                    if fail:

                        module.unload()
                        hidden_state = None

                        if minimum_reserve_tensor is not None: del minimum_reserve_tensor
                        minimum_reserve_tensor = None

                        gc.collect()
                        torch.cuda.empty_cache()
                        hidden_state = hidden_state_backup.clone()

                        current_device += 1
                        if current_device >= num_devices:
                            raise RuntimeError("Insufficient VRAM for model and cache")

                        continue

                    break

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            hidden_state = None
            attn_mask = None
            reserved_vram_tensors = None

            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = True

        if 'yield' in locals():
            yield


    def unload(self):

        for module in self.modules:
            module.unload()

        self.modules = []
        self.modules_dict = {}
        self.device_tensors = []


    def set_cache_map(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): self.cache_map[module.layer_idx] = module.device()


    def get_cache_devices(self):

        return list(set(self.cache_map.values()))


    def create_device_tensors(self, scratch_bytes):

        for idx, bytes in enumerate(scratch_bytes):

            tensors = ExLlamaV2DeviceTensors(self, idx, bytes)
            self.device_tensors.append(tensors)


    def get_device_tensors(self, device_idx, scratch = True):

        tensors = self.device_tensors[device_idx]
        if not tensors.ready: tensors.prepare(scratch)
        return tensors


    def get_modules(self):

        return [module for module in self.modules]


    def update_loras(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): module.update_loras()
            if isinstance(module, ExLlamaV2MLP): module.update_loras()


    def is_quant(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention):
                if module.is_quant(): return True

        return False


    def build_attn_mask(self, batch_size, seq_len, past_len, input_mask, device):

        if input_mask is None and seq_len == 1: return None

        if isinstance(past_len, tuple):

            attn_masks = []

            for i in range(len(past_len[1])):

                attn_mask = torch.zeros((1, 1, seq_len, past_len[1][i] + seq_len), dtype = torch.float16, device = device)
                attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.))
                attn_mask[:, :, : seq_len - 1, past_len[1][i] + 1: past_len[1][i] + seq_len] = attn_mask_triu

                if input_mask is not None:
                    min_mask_width = min(input_mask[i].shape[-1], seq_len + past_len[1][i])
                    input_mask_part = safe_move_tensor(input_mask[i][:, :min_mask_width], attn_mask.device)
                    input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
                    attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)

                attn_masks.append(attn_mask)

            return attn_masks

        else:

            attn_mask = torch.zeros((batch_size, 1, seq_len, past_len + seq_len), dtype = torch.float16, device = device)
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

            if input_mask is not None:
                min_mask_width = min(input_mask.shape[-1], seq_len + past_len)
                input_mask_part = safe_move_tensor(input_mask[:, :min_mask_width], attn_mask.device)
                input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
                attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)

            return attn_mask

    @torch.inference_mode()
    def forward(self,
                input_ids,
                cache = None,
                input_mask = None,
                preprocess_only = False,
                last_id_only = False,
                loras = None,
                return_last_state = False,
                position_offsets = None):

        q_len = input_ids.shape[-1]
        remaining_q_len = q_len
        bsz = input_ids.shape[0]

        # Attn and MLP layers have preallocated buffers for temp states, sized by the model config. Effective max input
        # length depends on the current batch size

        effective_max_input_len = self.config.max_input_len * self.config.max_batch_size // bsz

        # Without a cache we can't process the sequence in chunks, so forward the whole thing and assume the input length
        # is less than config.max_input_len

        if cache is None or not isinstance(cache, ExLlamaV2CacheBase):

            assert q_len <= effective_max_input_len, "Maximum input length exceeded in model.forward"

            result, last_state = self._forward(input_ids = input_ids,
                                               cache = cache,
                                               input_mask = input_mask,
                                               preprocess_only = preprocess_only,
                                               last_id_only = last_id_only,
                                               loras = loras,
                                               return_last_state = return_last_state,
                                               position_offsets = position_offsets)

            if last_state is None:
                return result
            else:
                return result, last_state

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # Split sequence

        result = None
        last_state = None

        chunk_begin = 0
        while chunk_begin < q_len:

            # Limit chunk_size to max_input_len

            chunk_size = min(remaining_q_len, effective_max_input_len)

            # Limit chunk_size to keep size of attention operation <= max_attention_size

            past_len = cache.current_seq_len
            attn_size = (past_len + remaining_q_len) * remaining_q_len
            max_a = self.config.max_attention_size
            if attn_size > max_a:
                cs = (math.sqrt(past_len ** 2 + 4 * max_a) - past_len) / 2
                chunk_size = min(chunk_size, math.floor(cs))

            # Process chunk

            chunk_end = min(chunk_begin + chunk_size, q_len)

            # print(f"Forward chunk length: {chunk_end - chunk_begin}")

            _last_id_only = last_id_only
            _preprocess_only = preprocess_only or (chunk_end < q_len and last_id_only)

            r, ls = self._forward(input_ids = input_ids[:, chunk_begin : chunk_end],
                                  cache = cache,
                                  input_mask = input_mask,
                                  preprocess_only = _preprocess_only,
                                  last_id_only = _last_id_only,
                                  loras = loras,
                                  return_last_state = return_last_state and remaining_q_len <= chunk_size,
                                  position_offsets = position_offsets)

            if not _preprocess_only:
                result = r if result is None else torch.cat((result, r), dim = 1)
                r = None

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size
            last_state = ls

        if last_state is None:
            return result
        else:
            return result, last_state


    @torch.inference_mode()
    def _forward(self,
                 input_ids,
                 cache = None,
                 input_mask = None,
                 preprocess_only = False,
                 last_id_only = False,
                 loras = None,
                 return_last_state = False,
                 position_offsets = None):

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if cache is not None:
            if isinstance(cache, ExLlamaV2CacheBase):
                past_len = cache.current_seq_len
            else:
                pl = [c.current_seq_len for c in cache]
                past_len = torch.tensor(pl, dtype = torch.int)
                past_len = (past_len, past_len)

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids
        prev_device = None
        attn_mask = None
        last_state = None

        for idx, module in enumerate(self.modules):

            device = _torch_device(module.device_idx)

            # Build attention mask

            if device != prev_device and device != "cpu":

                prev_device = device
                attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, input_mask, device)
                if isinstance(past_len, tuple): past_len = (safe_move_tensor(past_len[0], device), past_len[1])
                if position_offsets is not None: position_offsets = safe_move_tensor(position_offsets, device)

            # Onward

            if idx == self.head_layer_idx:
                if last_id_only and return_last_state:
                    x = x.narrow(-2, -1, 1)
                    last_state = x
                elif last_id_only:
                    x = x.narrow(-2, -1, 1)
                elif return_last_state:
                    last_state = x.narrow(-2, -1, 1)

            x = safe_move_tensor(x, device)
            x = module.forward(x, cache = cache, attn_mask = attn_mask, past_len = past_len, loras = loras, position_offsets = position_offsets)

            if preprocess_only and idx == self.last_kv_layer_idx:
                x = None
                break

        # Advance cache

        if cache is not None:
            if isinstance(cache, list):
                for c in cache: c.current_seq_len += seq_len
            else:
                cache.current_seq_len += seq_len

        # Set padding logits to -inf

        if x is not None:
            head_padding = self.modules[-1].padding
            if head_padding > 0:
                x[:, :, -head_padding:] = -65504.

        return x, last_state