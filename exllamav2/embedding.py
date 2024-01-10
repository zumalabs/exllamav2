import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module

class ExLlamaV2Embedding(ExLlamaV2Module):

    embedding: nn.Embedding or None
    embedding_q: nn.Embedding or None
    embedding_k: nn.Embedding or None
    embedding_v: nn.Embedding or None

    name: str = "Embedding"
    native_vocab_size: int = None

    def __init__(self, model, key):
        super().__init__(model, key)


    def load(self):

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        pad_token_id = self.model.config.pad_token_id

        w = self.load_weight()
        assert isinstance(w, nn.Parameter)
        self.native_vocab_size = w.shape[0]

        # pad_id = self.model.config.pad_token_id

        # Padding token should embed a zero vector, but sometimes it doesn't (?)

        # if not torch.is_grad_enabled():
        #     w[pad_id] *= 0

        self.embedding = nn.Embedding(vocab_size, hidden_size, pad_token_id, device ="meta")
        self.embedding.weight = w


    def unload(self):

        del self.embedding
        self.embedding = None


    def get_weight(self):

        return self.embedding.weight.data


    def weight_footprint(self):

        vocab_size = self.model.config.vocab_size
        hidden_size = self.model.config.hidden_size
        kv_size = self.model.config.num_key_value_heads * self.model.config.head_dim

        return vocab_size * hidden_size * 2


    def scratch_space_fixed(self):

        return 0


    def scratch_space(self):

        return 0


    def forward(self, hidden_states, cache = None, attn_params = None, past_len = None, intermediates = False, loras = None):

        hidden_states = self.embedding.forward(hidden_states)

        if intermediates:
            return {"hidden_states": hidden_states}
        else:
            return hidden_states

