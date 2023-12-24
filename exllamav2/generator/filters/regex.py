from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Tokenizer
)

from exllamav2.generator.filters.base import ExLlamaV2Filter
import functools
import re


@functools.lru_cache(maxsize=1000000)
def filter_tokens(branch_str, char_trie, pattern):
    pass_tokens = set()
    for c in char_trie.children:
        w = char_trie.children[c]
        if re.fullmatch(pattern, branch_str + c):
            if len(w.leaf) > 0:
                pass_tokens.update(w.leaf)
            pass_tokens.update(filter_tokens(branch_str + c, w, pattern))
    return pass_tokens


class ExLlamaV2RegexFilter(ExLlamaV2Filter):

    pattern: str
    prefix: str

    def __init__(self, model, tokenizer, pattern):
        super().__init__(model, tokenizer)
        self.pattern = pattern


    def begin(self, prefix_str = ""):
        self.sequence_str = ""
        self.prefix_str = prefix_str


    def feed(self, token):
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        piece = id_to_piece[token]
        self.sequence_str += piece


    def filter_tokens(self, branch_str, char_trie, pass_tokens):
        for c in char_trie.children:
            w = char_trie.children[c]
            if re.fullmatch(self.pattern, branch_str + c):
                if len(w.leaf) > 0:
                    pass_tokens.update(w.leaf)
                self.filter_tokens(branch_str + c, w, pass_tokens)


    def next(self):
        char_trie = self.tokenizer.get_char_trie()
        pass_tokens = set()
        end_tokens = set()
        if self.prefix_str and not self.sequence_str:
            for c in self.prefix_str:
                char_trie = char_trie.children[c]
        prefix_token = self.tokenizer.get_piece_to_id_dict()[self.prefix_str]
        pass_tokens.add(prefix_token)
        gen_string = self.sequence_str.replace(self.prefix_str, "")
        self.filter_tokens(gen_string, char_trie, pass_tokens)
        # pass_tokens.update(filter_tokens(gen_string, char_trie, self.pattern))
        return pass_tokens, end_tokens
