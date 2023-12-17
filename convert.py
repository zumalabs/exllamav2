from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
import argparse, os, shutil
import sys
import json
from conversion.tokenize import tokenize
from conversion.quantize import embeddings, measure_quant, quant
from conversion.optimize import optimize, optimize_new
from conversion.compile import compile_model
from conversion.qparams import qparams_headoptions

# import tracemalloc
# tracemalloc.start()

parser = argparse.ArgumentParser(description = "Convert model to ExLlamaV2")
parser.add_argument("-i", "--in_dir", type = str, help = "Input directory", default = "")
parser.add_argument("-o", "--out_dir", type = str, help = "Output (working) directory")
parser.add_argument("-nr", "--no_resume", action = "store_true", help = "Do not resume an interrupted job (deletes all files in the output directory)")
parser.add_argument("-cf", "--compile_full", type = str, help = "Output folder for compiled model with all config/tokenizer files")
parser.add_argument("-om", "--output_measurement", type = str, help = "Only perform measurement pass, then save measurement to the specified file")
parser.add_argument("-c", "--cal_dataset", type = str, help = "Calibration dataset (.parquet file)", default = "")
parser.add_argument("-r", "--dataset_rows", type = int, default = 100, help = "Number of rows to apply from dataset")
parser.add_argument("-mr", "--measurement_rows", type = int, default = 16, help = "Number of rows to apply from dataset when measuring")
parser.add_argument("-gr", "--gpu_rows", type = int, default = 0, help = "Threshold for paging hidden state to CPU")
parser.add_argument("-l", "--length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-ml", "--measurement_length", type = int, default = 2048, help = "Max no. tokens per sample when measuring")
parser.add_argument("-b", "--bits", type = float, default = 4.125, help = "Target bits per weight")
parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Target bits per weight (head layer)")
parser.add_argument("-m", "--measurement", type = str, help = "Reuse previous measurement")
parser.add_argument("-ss", "--shard_size", type = float, help = "Max shard size in MB (default: 8192)", default = 8192)
parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")
parser.add_argument("-kld", "--kld_estimate", action = "store_true", help = "Do not use measurement, instead optimize according to built-in KL divergence estimates")
parser.add_argument("-gq", "--gigaquant", action = "store_true", help = "Gigaquant mode (don't use this)")

args = parser.parse_args()

# Check some args

if not args.in_dir:
    print(" ## Please specify input model directory (-i, --in_dir)")
    sys.exit()

if not args.out_dir:
    print(" ## Please specify output/working directory (-o, --out_dir)")
    sys.exit()

# if not args.cal_dataset:
#     print(" ## Please specify dataset Parquet file (-c, --cal_dataset)")
#     sys.exit()

if args.length > 2048 or args.measurement_length > 2048:
    print(" !! Warning: calibration rows > 2048 tokens may result in excessive VRAM use")

if not args.head_bits in qparams_headoptions:
    print(f" ## Error: {args.head_bits} is not a supported option for head layer bitrate")
    sys.exit()

if args.bits < 2 or args.bits > 8:
    print(f" !! Warning: target bitrate {args.bits} will likely not be attainable")

if args.output_measurement is not None and args.compile_full is not None:
    print(" ## Conflicting options: --output_measurement and --compile_full")
    sys.exit()

if args.output_measurement is not None and args.kld_estimate:
    print(" ## Conflicting options: --output_measurement and --kld_estimate")
    sys.exit()

if args.measurement is not None and args.kld_estimate:
    print(" ## Conflicting options: --measurement and --kld_estimate")
    sys.exit()

# Arguments

in_dir = None if args.in_dir == "" else os.path.abspath(args.in_dir)
out_dir = os.path.abspath(args.out_dir)
cal_dataset = None if args.cal_dataset == "" else os.path.abspath(args.cal_dataset)
dataset_rows = args.dataset_rows
measurement_rows = args.measurement_rows
gpu_rows = args.gpu_rows
length = args.length
measurement_length = args.measurement_length
bits = args.bits
head_bits = args.head_bits
reuse_measurement = args.measurement
shard_size = args.shard_size if args.shard_size > 0 else 1024 ** 3  # 1 PB = unlimited
no_resume = args.no_resume
output_measurement = args.output_measurement
if output_measurement is not None:
    if os.path.isdir(output_measurement):
        output_measurement = os.path.join(output_measurement, "measurement.json")
rope_scale = args.rope_scale
rope_alpha = args.rope_alpha
kld_estimate = args.kld_estimate
gigaquant = args.gigaquant

compile_full = args.compile_full

if not os.path.exists(out_dir):
    print(f" ## Error: Directory not found: {out_dir}")
    sys.exit()

# Create config

config = ExLlamaV2Config()
config.model_dir = in_dir
config.qkv_embed = False
config.prepare()

# Tokenizer

tokenizer = ExLlamaV2Tokenizer(config)

# Job file

job_file = os.path.join(out_dir, "job.json")

# Create new job

def save_job():
    global job_file, job
    with open(job_file, "w", encoding = "utf8") as f:
        f.write(json.dumps(job, indent = 4))

if no_resume or not os.path.exists(job_file):

    print(f" -- Beginning new job")

    if len(os.listdir(out_dir)) != 0:
        print(f" !! Warning: Output directory is not empty: {out_dir}")
        if no_resume:
            print(f" !! Cleaning output directory: {out_dir}")
            for filename in os.listdir(out_dir):
                file_path = os.path.join(out_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    if in_dir is None:
        print(f" ## Error: No input directory specified")
        sys.exit()

    # if cal_dataset is None:
    #     print(f" ## Error: No calibration dataset specified")
    #     sys.exit()

    job = { "in_dir": in_dir,
            "out_dir": out_dir,
            "cal_dataset": cal_dataset,
            "dataset_rows": dataset_rows,
            "measurement_rows": measurement_rows,
            "gpu_rows": gpu_rows,
            "length": length,
            "measurement_length": measurement_length,
            "bits": bits,
            "head_bits": head_bits,
            "progress": "begin",
            "shard_size": shard_size,
            "output_measurement": output_measurement,
            "compile_full": compile_full,
            "rope_scale": rope_scale,
            "rope_alpha": rope_alpha,
            "kld_estimate": kld_estimate,
            "gigaquant": gigaquant
            }

    if reuse_measurement is not None:

        with open(reuse_measurement, "r", encoding = "utf8") as f:

            imp_measurement = json.load(f)
            job["measurement"] = imp_measurement["measurement"]
            job["last_module_idx"] = imp_measurement["last_module_idx"]
            job["base_perplexity"] = imp_measurement["base_perplexity"]
            job["reuse_measurement"] = reuse_measurement

    save_job()

# Resume existing job

else:

    print(f" -- Resuming job")
    print(f" !! Note: Overriding options with settings from existing job")

    with open(job_file, "r", encoding = "utf8") as f:
        job = json.load(f)

    if "invalid" in job:
        print(" ** Error: Corrupted job")
        sys.exit()

    if "shard_size" not in job: job["shard_size"] = shard_size
    if "output_measurement" not in job: job["output_measurement"] = output_measurement
    if "compile_full" not in job: job["compile_full"] = compile_full

    job["out_dir"] = out_dir

# Feedback

print(f" -- Input: {job['in_dir']}")
print(f" -- Output: {out_dir}")
if job['cal_dataset'] is not None:
    print(f" -- Calibration dataset: {job['cal_dataset']}, {job['dataset_rows']} / {job['measurement_rows']} ({job['gpu_rows']}) rows, {job['length']} tokens per sample")
else:
    print(f" -- Using default calibration dataset")
if job["output_measurement"] is None:
    print(f" -- Target bits per weight: {job['bits']} (decoder), {job['head_bits']} (head)")
    print(f" -- Max shard size: {job['shard_size']} MB")
else:
    print(f" -- Measurement will be saved to {job['output_measurement']}")
    print(f" !! Conversion script will end after measurement pass")

if job["rope_scale"] is not None:
    print(f" -- RoPE scale: {job['rope_scale']:.2f}")

if job["rope_alpha"] is not None:
    print(f" -- RoPE alpha: {job['rope_alpha']:.2f}")

if job["gigaquant"]:
    print(f" !! Running in gigaquant mode")

# Make sure subfolders exist

if job["compile_full"] is not None:
    print(f" -- Full model will be compiled to: {job['compile_full']}")
    if os.path.exists(job["compile_full"]):
        if not os.path.isdir(job["compile_full"]):
            print(f" ## Error: Output path {job['compile_full']} exists but is not a directory")
            sys.exit()
        if len(os.listdir(job["compile_full"])) > 0:
            print(f" !! Warning: Output path {job['compile_full']} exists but is not empty")

out_tensor_dir = os.path.join(job["out_dir"], "out_tensor")
if not os.path.exists(out_tensor_dir):
    os.makedirs(out_tensor_dir)

# Allocate space for hidden state

max_l = max(job["measurement_length"], job["length"])
config.max_input_len = max_l
config.max_attention_size = max_l ** 2

# Set scaling for input model

if job["rope_scale"] is not None:
    config.scale_pos_emb = job["rope_scale"]

if job["rope_alpha"] is not None:
    config.scale_alpha_value = job["rope_alpha"]

# Create model without loading weights

model = ExLlamaV2(config)
model.load(lazy = True)

# Do the things

while True:

    progress = job["progress"]

    if progress == "begin":

        if "reuse_measurement" in job:

            print(f" -- Reusing measurement: {job['reuse_measurement']}")
            job["progress"] = "optimize"
            save_job()

        elif job.get("kld_estimate", False):

            print(f" -- Using built-in KL divergence estimates")
            job["progress"] = "optimize_new"
            save_job()

        else:

            print(f" -- Tokenizing samples (measurement)...")
            tokenize(job, save_job, tokenizer, measure = True)
            job["progress"] = "initial_embeddings"
            save_job()

    if progress == "initial_embeddings":

        print(f" -- Token embeddings (measurement)...")
        embeddings(job, save_job, model)
        job["progress"] = "measure_quant"
        save_job()

    if progress == "measure_quant":

        print(f" -- Measuring quantization impact...")
        measure_quant(job, save_job, model)
        if job["output_measurement"] is None:
            job["progress"] = "optimize"
        else:
            job["progress"] = "finished"
        save_job()

    if progress == "optimize":

        print(f" -- Optimizing...")
        optimize(job, save_job)
        job["progress"] = "tokens_cal"
        save_job()

    if progress == "optimize_new":

        print(f" -- Optimizing...")
        optimize_new(job, save_job, model)
        job["progress"] = "tokens_cal"
        save_job()

    if progress == "tokens_cal":

        print(f" -- Tokenizing samples...")
        tokenize(job, save_job, tokenizer)
        job["progress"] = "embeddings"
        save_job()

    if progress == "embeddings":
        print(f" -- Token embeddings again...")
        embeddings(job, save_job, model)
        job["progress"] = "quant"
        save_job()

    if progress == "quant":

        print(f" -- Quantizing...")
        quant(job, save_job, model)
        job["progress"] = "compile" if not job["gigaquant"] else "finished"
        save_job()

    if progress == "compile":

        print(f" -- Compiling output file...")
        compile_model(job, save_job, model)
        job["progress"] = "finished"
        save_job()

    if progress == "finished": break

print(f" -- Finished")