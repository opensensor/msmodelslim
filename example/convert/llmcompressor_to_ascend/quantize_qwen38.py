"""Calibrate original Qwen3.8-Flash-Next with bounded CPU/disk weight placement.

Uses the same arguments as quantize_w8a8.py. The architecture adapter is local,
text-only and uncached; vision/MTP weights are preserved without calibration.
"""

import hashlib
import json
import os
import time
from importlib.metadata import version

from quantize_w8a8 import GIB, parse_args


def quantize(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from qwen38.export import save_checkpoint
    from qwen38.model import load_model
    from transformers import AutoTokenizer, set_seed

    if args.ignore or args.sequential_target:
        raise ValueError("Qwen38 uses a fixed expert-only policy and decoder-layer sequential targets")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_grad_enabled(False)
    set_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    # tokenizer_config.json selects the existing Qwen2 tokenizer independently
    # of the unavailable installed Qwen4Exp model implementation.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for record in args.records:
        text = record.get("text")
        chat = not isinstance(text, str) or not text.strip()
        if chat:
            text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, truncation=True, max_length=args.sequence_length, add_special_tokens=not chat)
        if len(tokens["input_ids"]) < 2:
            raise ValueError("each sample must contain at least two tokens")
        rows.append({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        args.model_path,
        args.offload_dir,
        dtype=getattr(torch, args.dtype),
        device=args.device,
        cpu_budget_bytes=int(args.cpu_memory_gib * GIB),
    )
    recipe = (GPTQModifier if args.method == "gptq" else QuantizationModifier)(
        scheme="W8A8", targets=[r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"]
    )
    calibration = {}
    if rows:
        calibration = {
            "dataset": Dataset.from_list(rows),
            "num_calibration_samples": len(rows),
            "max_seq_length": args.sequence_length,
            "pipeline": "sequential",
            "sequential_targets": ["Qwen4ExpTextDecoderLayer"],
            "sequential_offload_device": "cpu",
            "shuffle_calibration_samples": False,
            "moe_calibrate_all_experts": True,
        }
    oneshot(model=model, tokenizer=tokenizer, recipe=recipe, output_dir=None, clear_sparse_session=True, **calibration)
    result = save_checkpoint(model, args.save_path, float_dtype=getattr(torch, args.dtype))
    result.update(
        method=args.method,
        dtype=args.dtype,
        device=args.device,
        target=args.target,
        elapsed_seconds=time.monotonic() - started,
        cpu_weight_bytes=model.cpu_weight_bytes,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
        calibration_samples=len(rows),
        calibration_tokens_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        versions={name: version(name) for name in ("torch", "transformers", "llmcompressor", "compressed-tensors")},
        validation="text calibration and compressed-tensors export; Ascend inference and quality not validated",
        ple="source shards preserved; memory-mapped row lookup during calibration",
        vision="preserved, not calibrated",
        mtp="preserved in floating point, not calibrated or enabled",
    )
    (args.save_path / "calibration_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    quantize(parse_args())
