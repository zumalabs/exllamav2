from exllamav2 import *
from exllamav2.generator import *
import sys, torch

config = ExLlamaV2Config()
config.model_dir = "/mnt/str/models/mixtral-8x7b-instruct-exl2/3.0bpw/"
config.prepare()

model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)

print("Loading model...")
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
generator.set_stop_conditions([tokenizer.eos_token_id])
gen_settings = ExLlamaV2Sampler.Settings()

while True:

    print()
    instruction = input("User: ")
    print()
    print("Assistant:", end = "")

    instruction_ids = tokenizer.encode(f"[INST] {instruction} [/INST]", add_bos = True)
    context_ids = instruction_ids if generator.sequence_ids is None \
        else torch.cat([generator.sequence_ids, instruction_ids], dim = -1)

    generator.begin_stream(context_ids, gen_settings)

    while True:
        chunk, eos, _ = generator.stream()
        if eos: break
        print(chunk, end = "")
        sys.stdout.flush()

    print()
