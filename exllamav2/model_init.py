
import argparse, sys, os, glob

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)

def add_args(parser):

    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
    parser.add_argument("-gs", "--gpu_split", type = str, help = "\"auto\", or VRAM allocation per GPU in GB")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length")
    parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")
    parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")
    parser.add_argument("-lm", "--low_mem", action = "store_true", help = "Enable VRAM optimizations, potentially trading off speed")


def print_options(args):

    print(f" -- Model: {args.model_dir}")

    print_opts = []
    if args.gpu_split: print_opts += [f"gpu_split: {args.gpu_split}"]
    if args.length: print_opts += [f"length: {args.length}"]
    print_opts += [f"rope_scale {args.rope_scale}"]
    print_opts += [f"rope_alpha {args.rope_alpha}"]
    if args.no_flash_attn: print_opts += ["no_flash_attn"]
    if args.low_mem: print_opts += ["low_mem"]
    print(f" -- Options: {print_opts}")


def check_args(args):

    if not args.model_dir:
        print(" ## Error: No model directory specified")
        sys.exit()

    if not os.path.exists(args.model_dir):
        print(f" ## Error: Can't find model directory: {args.model_dir}")
        sys.exit()

    required_files = ["config.json",
                      ["tokenizer.model", "tokenizer.json"],
                      "*.safetensors"]

    for filename in required_files:
        if isinstance(filename, str):
            filename = [filename]
        all_matches = []
        for file in filename:
            path = os.path.join(args.model_dir, file)
            matches = glob.glob(path)
            all_matches += matches
        if len(all_matches) == 0:
            print(f" ## Error: Cannot find {filename} in {args.model_dir}")
            sys.exit()


def init(args, quiet = False, allow_auto_split = False, skip_load = False):

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    # Set config options

    if args.length: config.max_seq_len = args.length
    config.scale_pos_emb = args.rope_scale
    config.scale_alpha_value = args.rope_alpha
    config.no_flash_attn = args.no_flash_attn

    # Set low-mem options

    if args.low_mem: config.set_low_mem()

    # Load model
    # If --gpu_split auto, return unloaded model. Model must be loaded with model.load_autosplit() supplying cache
    # created in lazy mode

    model = ExLlamaV2(config)

    split = None
    if args.gpu_split and args.gpu_split != "auto":
        split = [float(alloc) for alloc in args.gpu_split.split(",")]

    if args.gpu_split != "auto" and not skip_load:
        if not quiet: print(" -- Loading model...")
        model.load(split)
    else:
        assert allow_auto_split, "Auto split not allowed."

    # Load tokenizer

    if not quiet: print(" -- Loading tokenizer...")

    tokenizer = ExLlamaV2Tokenizer(config)

    return model, tokenizer