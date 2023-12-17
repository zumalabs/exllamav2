import torch
import torch.nn.functional as F
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2 import ext

class ExLlamaV2MoEMLP(ExLlamaV2Module):

    layer_idx: int
    post_attention_layernorm: ExLlamaV2RMSNorm
    w1: list
    w2: list
    w3: list
    gate: ExLlamaV2Linear
    num_experts: int
    num_experts_per_token: int

    name: str = "MoE MLP"
    submodules: list

    q_handle: int or None = None

    temp_lora_size: int = 0

    def __init__(self, model, key, layer_idx):
        super().__init__(model, key)

        self.layer_idx = layer_idx

        hidden_size = self.model.config.hidden_size
        intermediate_size = self.model.config.intermediate_size
        self.num_experts = self.model.config.num_experts
        self.num_experts_per_token = self.model.config.num_experts_per_token

        self.post_attention_layernorm = ExLlamaV2RMSNorm(model, key + ".post_attention_layernorm")
        self.w1 = [ExLlamaV2Linear(model, key + f".block_sparse_moe.experts.{e}.w1", hidden_size, intermediate_size, False) for e in range(self.num_experts)]
        self.w2 = [ExLlamaV2Linear(model, key + f".block_sparse_moe.experts.{e}.w2", intermediate_size, hidden_size, False) for e in range(self.num_experts)]
        self.w3 = [ExLlamaV2Linear(model, key + f".block_sparse_moe.experts.{e}.w3", hidden_size, intermediate_size, False) for e in range(self.num_experts)]
        self.gate = ExLlamaV2Linear(model, key + ".block_sparse_moe.gate", hidden_size, self.num_experts, False, pad32 = False)

        self.submodules = [self.post_attention_layernorm,
                           self.gate] + \
                          self.w1 + \
                          self.w2 + \
                          self.w3

    def numel(self):

        return sum(l.numel() for l in self.w1 + self.w2 + self.w3)


    def load(self):

        self.post_attention_layernorm.load()
        self.gate.load()
        for e in range(self.num_experts):
            self.w1[e].load()
            self.w2[e].load()
            self.w3[e].load()

        if self.w1[0].is_quant():
            device_tensors = self.model.get_device_tensors(self.device_idx)
            device_tensors.begin_scratch_alloc()
            self.q_handle = ext_c.make_q_moe_mlp(self.post_attention_layernorm.weight,
                                                 self.post_attention_layernorm.variance_epsilon,
                                                 self.gate.linear.weight,
                                                 self.num_experts,
                                                 self.num_experts_per_token,
                                                 [w.q_handle for w in self.w1],
                                                 [w.q_handle for w in self.w2],
                                                 [w.q_handle for w in self.w3],
                                                 device_tensors.get_scratch_slice(self.temp_state_size()),
                                                 device_tensors.get_scratch_slice(self.temp_gathered_state_size()),
                                                 device_tensors.get_scratch_slice(self.temp_a_size()),
                                                 device_tensors.get_scratch_slice(self.temp_b_size()),
                                                 device_tensors.get_scratch_slice(self.temp_logit_size()),
                                                 device_tensors.get_scratch_slice(self.temp_dq_size()),
                                                 self.model.config.max_input_len * self.model.config.max_batch_size)


    def unload(self):
        # if self.q_handle is not None:
        #     ext_c.free_q_mlp(self.q_handle)
        #     self.q_handle = None

        self.post_attention_layernorm.unload()
        self.gate.unload()
        for e in range(self.num_experts):
            self.w1[e].unload()
            self.w2[e].unload()
            self.w3[e].unload()


    def weight_footprint(self):

        return self.post_attention_layernorm.weight_footprint() + \
               self.gate.weight_footprint() + \
               sum(self.w1[e].weight_footprint() for e in range(self.num_experts)) + \
               sum(self.w2[e].weight_footprint() for e in range(self.num_experts)) + \
               sum(self.w3[e].weight_footprint() for e in range(self.num_experts))


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        assert self.model.config.intermediate_size >= self.model.config.hidden_size
        return self.temp_state_size() + \
               self.temp_gathered_state_size() + \
               self.temp_a_size() + \
               self.temp_b_size() + \
               self.temp_logit_size() + \
               self.temp_dq_size()


    def temp_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_gathered_state_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.hidden_size * 2 + 128


    def temp_a_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_b_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.intermediate_size * 2 + 128


    def temp_dq_size(self):

        return max(self.w1[0].temp_dq_size(),
                   self.w2[0].temp_dq_size(),
                   self.w3[0].temp_dq_size())


    def temp_logit_size(self):

        return self.model.config.max_input_len * self.model.config.max_batch_size * self.model.config.num_experts * 2 + 128


    def set_device_idx(self, idx):
        super().set_device_idx(idx)

        self.post_attention_layernorm.set_device_idx(idx)
        self.gate.set_device_idx(idx)
        for e in range(self.num_experts):
            self.w1[e].set_device_idx(idx)
            self.w2[e].set_device_idx(idx)
            self.w3[e].set_device_idx(idx)


    def forward(self, hidden_states, cache = None, attn_mask = None, past_len = None, intermediates = False, loras = None, position_offsets = None):

        assert loras is None or len(loras) == 0, "LoRA support not yet implemented for MoE MLP layers"
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # if True:
        if self.q_handle is None or intermediates or batch_size * sequence_length > 4 or self.num_experts not in [4, 8]:
            return self.forward_torch(hidden_states, cache, attn_mask, intermediates, loras = loras)

        # if loras is None or self.temp_lora_size == 0:
        #     pass_loras = []
        #     pass_lora_temp = ext.none_tensor
        # else:
        #     pass_loras = [id(x) for x in loras]
        #     pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        # ref = self.forward_torch(hidden_states, cache, attn_mask, intermediates, loras = loras)
        ext_c.q_moe_mlp_forward_(self.q_handle, hidden_states.view(-1, hidden_states.shape[-1]))

        return hidden_states


    def forward_torch(self, hidden_states, cache = None, attn_mask = None, intermediates = False, loras = None, position_offsets = None):

        residual = hidden_states

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        if intermediates: result = { "post_norm": hidden_states }

        router_logits = self.gate.forward(hidden_states, loras = loras)  #[:, :self.num_experts]

        routing_weights = F.softmax(router_logits, dim = -1, dtype = torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_token, dim = -1)
        routing_weights /= routing_weights.sum(dim = -1, keepdim = True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

            gate = self.w1[expert_idx].forward(current_state, loras = loras)
            up = self.w3[expert_idx].forward(current_state, loras = loras)

            current_hidden_states = F.silu(gate) * up
            if intermediates: result[f"pre_down.{expert_idx}"] = current_hidden_states

            current_hidden_states = self.w2[expert_idx].forward(current_hidden_states, loras = loras)
            current_hidden_states *= routing_weights[top_x_list, idx_list, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        final_hidden_states += residual

        if intermediates:
            result["hidden_states"] = final_hidden_states
            return result
        else:
            return final_hidden_states


    def update_loras(self):
        pass


    def is_quant(self):
        return self.q_handle is not None

