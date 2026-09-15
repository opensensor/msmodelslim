"""Calibrate original Qwen3.8-Flash-Next with bounded CPU/disk weight placement.

Uses the same arguments as quantize_w8a8.py. The architecture adapter is local,
text-only and uncached; vision/MTP weights are preserved without calibration.
"""

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from quantize_w8a8 import GIB, positive_int, read_calibration
from quantize_w8a8 import parse_args as parse_common_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-data", type=Path, help="Held-out JSONL for paired pre/post quantization perplexity")
    parser.add_argument("--quality-report-dir", type=Path, help="Fresh directory outside model/output/offload paths")
    parser.add_argument("--quality-samples", type=positive_int, default=32)
    parser.add_argument("--quality-sequence-length", type=positive_int, default=1024)
    parser.add_argument("--quality-logit-chunk-size", type=positive_int, default=128)
    args = parse_common_args(argv, parser=parser)
    try:
        if bool(args.quality_data) != bool(args.quality_report_dir):
            raise ValueError("--quality-data and --quality-report-dir must be provided together")
        args.quality_records = []
        if args.quality_data:
            path = args.quality_report_dir.resolve()
            for other in (args.model_path, args.save_path, args.offload_dir):
                other = other.resolve()
                if path == other or path in other.parents or other in path.parents:
                    raise ValueError("quality report directory must not overlap model/output/offload paths")
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise ValueError("quality report directory must be empty")
            args.quality_records = read_calibration(
                SimpleNamespace(calibration_data=args.quality_data, samples=args.quality_samples)
            )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return args


def tokenize_records(tokenizer, records, sequence_length):
    rows = []
    for record in records:
        text = record.get("text")
        chat = not isinstance(text, str) or not text.strip()
        if chat:
            text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(text, truncation=True, max_length=sequence_length, add_special_tokens=not chat)
        if len(tokens["input_ids"]) < 2:
            raise ValueError("each sample must contain at least two tokens")
        rows.append({"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]})
    return rows


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
    from qwen38.quality import compare_reports, evaluate, require_held_out, write_report
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
    rows = tokenize_records(tokenizer, args.records, args.sequence_length)
    quality_rows = tokenize_records(tokenizer, args.quality_records, args.quality_sequence_length)
    if quality_rows:
        require_held_out(rows, quality_rows)
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        args.model_path,
        args.offload_dir,
        dtype=getattr(torch, args.dtype),
        device=args.device,
        cpu_budget_bytes=int(args.cpu_memory_gib * GIB),
    )
    if quality_rows:
        args.quality_report_dir.mkdir(parents=True, exist_ok=True)
        before = evaluate(model, quality_rows, device=args.device, chunk_size=args.quality_logit_chunk_size)
        write_report(args.quality_report_dir / "before.json", before)
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
    quality = None
    if quality_rows:
        after = evaluate(
            model, quality_rows, device=args.device, chunk_size=args.quality_logit_chunk_size, quantized=True
        )
        write_report(args.quality_report_dir / "after.json", after)
        quality = compare_reports(before, after)
        write_report(args.quality_report_dir / "comparison.json", quality)
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
        quality=quality,
        validation="text calibration and compressed-tensors export; Ascend inference not validated",
        ple="source shards preserved; memory-mapped row lookup during calibration",
        vision="preserved, not calibrated",
        mtp="preserved in floating point, not calibrated or enabled",
    )
    (args.save_path / "calibration_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    quantize(parse_args())
