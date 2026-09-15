"""Quantize a local floating-point model using the tested W8A8 bridge contract.

Third-party imports are delayed until arguments are validated and CPU mode has
hidden CUDA devices. This command runs in its own process so those settings do
not affect another application's GPU workload.
"""

import argparse
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path

DEFAULT_IGNORES = ("lm_head", "re:.*\\.gate$", "re:.*\\.router$", "re:.*\\.shared_expert_gate$")
GIB = 1024**3


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_gib(value):
    number = float(value)
    if not math.isfinite(number) or number * GIB < 1:
        raise argparse.ArgumentTypeError("must be a finite positive GiB budget of at least one byte")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True, help="Fresh compressed-tensors output directory")
    parser.add_argument("--offload-dir", type=Path, required=True, help="Fresh directory for disk offload; prefer NVMe")
    parser.add_argument("--method", choices=("gptq", "rtn"), default="gptq")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--cpu-memory-gib",
        type=positive_gib,
        default=64,
        help="CPU weight-placement budget, not a limit on total process RAM",
    )
    parser.add_argument(
        "--calibration-data", type=Path, help="Local JSONL records with text or messages; required for GPTQ"
    )
    parser.add_argument("--samples", type=positive_int, default=512)
    parser.add_argument("--sequence-length", type=positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ignore", action="append", default=[], help="Additional LLM-Compressor layer name or re: pattern"
    )
    parser.add_argument(
        "--sequential-target",
        action="append",
        default=None,
        help="Optional decoder-layer class/name for sequential GPTQ tracing",
    )
    args = parser.parse_args(argv)
    try:
        validate_paths(args)
        args.records = read_calibration(args)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return args


def validate_paths(args):
    paths = [args.model_path.resolve(), args.save_path.resolve(), args.offload_dir.resolve()]
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("model, output and offload directories must be separate and non-overlapping")
    for path in paths[1:]:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"directory must be empty: {path}")
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("quantization_config")
        or (config.get("text_config") or {}).get("quantization_config")
        or (args.model_path / "quant_model_description.json").exists()
    ):
        raise ValueError("start from floating-point source weights, not an already quantized checkpoint")
    if not any(args.model_path.glob("*.safetensors")):
        raise ValueError("model-path must contain a local safetensors checkpoint")
    if args.method == "gptq" and args.calibration_data is None:
        raise ValueError("GPTQ requires --calibration-data with representative local samples")
    if args.method == "rtn" and args.calibration_data is not None:
        raise ValueError("RTN is data-free; use --method gptq to calibrate with samples")


def read_calibration(args):
    if args.calibration_data is None:
        return []
    records = []
    with args.calibration_data.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"calibration line {line_number}: expected an object")
            has_text = isinstance(record.get("text"), str) and bool(record["text"].strip())
            messages = record.get("messages")
            has_messages = (
                isinstance(messages, list)
                and bool(messages)
                and all(
                    isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                    for m in messages
                )
            )
            if not (has_text or has_messages):
                raise ValueError(f"calibration line {line_number}: expected nonempty text or a messages list")
            records.append(record)
            if len(records) == args.samples:
                break
    if len(records) < args.samples:
        raise ValueError(f"requested {args.samples} samples but found {len(records)}; lower --samples or add data")
    return records


def quantize(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Optional dependencies and accelerator discovery occur only after CPU isolation.
    import torch
    from compressed_tensors.offload import get_offloaded_device
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.utils import load_context
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a CUDA-enabled PyTorch environment and a visible GPU")
    if args.device == "cpu" and torch.accelerator.is_available():
        raise RuntimeError("CPU mode requires accelerator isolation; another accelerator is still visible")
    # This is a PTQ-only process. In particular, disk-cache dtype conversions
    # must not build autograd graphs during the save/restore offload round trip.
    torch.set_grad_enabled(False)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define a pad token or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for record in args.records:
        text = record.get("text")
        is_chat = not isinstance(text, str) or not text.strip()
        if is_chat:
            text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(
            text, max_length=args.sequence_length, truncation=True, padding=False, add_special_tokens=not is_chat
        )
        if len(tokens["input_ids"]) < 2:
            raise ValueError("each calibration sample must contain at least two tokens")
        rows.append({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    with load_context():
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=getattr(torch, args.dtype),
            device_map="auto_offload",
            max_memory={"cpu": int(args.cpu_memory_gib * GIB)},
            offload_folder=str(args.offload_dir),
        )
    disk_modules = [name for name, module in model.named_modules() if get_offloaded_device(module) == "disk"]
    ignores = list(DEFAULT_IGNORES) + args.ignore
    recipe = (GPTQModifier if args.method == "gptq" else QuantizationModifier)(
        scheme="W8A8",
        targets=["Linear"],
        ignore=ignores,
    )
    calibration = {}
    if rows:
        calibration = {
            "dataset": Dataset.from_list(rows),
            "num_calibration_samples": len(rows),
            "max_seq_length": args.sequence_length,
            "pipeline": "sequential",
            "sequential_targets": args.sequential_target,
            "sequential_offload_device": "cpu",
            "shuffle_calibration_samples": False,
            "moe_calibrate_all_experts": True,
        }
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=str(args.save_path),
        save_compressed=True,
        clear_sparse_session=True,
        **calibration,
    )
    manifest = {
        "method": args.method,
        "scheme": "W8A8",
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "model_type": model.config.model_type,
        "cpu_weight_budget_bytes": int(args.cpu_memory_gib * GIB),
        "disk_offloaded_modules": disk_modules,
        "calibration_samples": len(rows),
        "sequence_length": args.sequence_length,
        "calibration_tokens_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "ignore": ignores,
        "versions": {name: version(name) for name in ("torch", "transformers", "llmcompressor", "compressed-tensors")},
        "validation": "calibration and compressed checkpoint save only; Ascend export and inference not validated",
    }
    (args.save_path / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved compressed W8A8 checkpoint to {args.save_path}")


if __name__ == "__main__":
    quantize(parse_args())
