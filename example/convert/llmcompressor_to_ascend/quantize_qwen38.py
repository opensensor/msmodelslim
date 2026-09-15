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
    parser.add_argument("--checkpoint-dir", type=Path, help="Durable GPTQ stage checkpoints; separate from scratch")
    parser.add_argument("--resume", action="store_true", help="Resume the last committed stage in checkpoint-dir")
    parser.add_argument("--ple-cache-dir", type=Path, help="Persistent exact PLE row cache for these input datasets")
    parser.add_argument("--reuse-baseline", type=Path, help="Explicitly reuse a matching baseline from this source")
    args = parse_common_args(argv, parser=parser)
    try:
        if args.resume and not args.checkpoint_dir:
            raise ValueError("--resume requires --checkpoint-dir")
        if args.checkpoint_dir and args.method != "gptq":
            raise ValueError("stage checkpoints currently require --method gptq")
        if args.reuse_baseline and not args.quality_data:
            raise ValueError("--reuse-baseline requires --quality-data")
        paths = [
            p.resolve()
            for p in (
                args.model_path,
                args.save_path,
                args.offload_dir,
                args.quality_report_dir,
                args.checkpoint_dir,
                args.ple_cache_dir,
            )
            if p
        ]
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                if path == other or path in other.parents or other in path.parents:
                    raise ValueError("model/output/offload/quality/checkpoint/PLE paths must not overlap")
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
    try:
        return _quantize(args)
    finally:
        if getattr(args, "stage_store", None) is not None:
            args.stage_store.close()


def _quantize(args):
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from qwen38.checkpoint import Checkpoint
    from qwen38.export import save_checkpoint
    from qwen38.model import build_config, load_model
    from qwen38.ple_cache import prepare_ple_cache
    from qwen38.quality import compare_reports, evaluate, require_held_out, validate_baseline, write_report
    from qwen38.resume import StageStore, atomic_json, fingerprint
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
    store = None
    if args.checkpoint_dir:
        store = args.stage_store = StageStore(args.checkpoint_dir, fingerprint(args, rows, quality_rows), args.resume)
    ple_cache = None
    if args.ple_cache_dir:
        checkpoint = Checkpoint(args.model_path)
        checkpoint.require_complete()
        ple_cache = prepare_ple_cache(
            checkpoint,
            build_config(checkpoint.text_config),
            [(rows, args.sequence_length), (quality_rows, args.quality_sequence_length)],
            args.ple_cache_dir,
            getattr(torch, args.dtype),
        )
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        args.model_path,
        args.offload_dir,
        dtype=getattr(torch, args.dtype),
        device=args.device,
        cpu_budget_bytes=int(args.cpu_memory_gib * GIB),
        ple_cache=ple_cache,
    )
    if quality_rows:
        args.quality_report_dir.mkdir(parents=True, exist_ok=True)
        baseline = store.path / "before.json" if store else None
        reuse = baseline if baseline and baseline.exists() else args.reuse_baseline
        if reuse:
            before = json.loads(reuse.read_text())
            validate_baseline(before, model, quality_rows, args.device, args.quality_logit_chunk_size)
            print(f"Reusing validated floating baseline: {reuse}", flush=True)
        else:
            before = evaluate(model, quality_rows, device=args.device, chunk_size=args.quality_logit_chunk_size)
        if store and not baseline.exists():
            atomic_json(baseline, before)
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
            "pipeline": "qwen38_resumable" if store else "sequential",
            "sequential_targets": ["Qwen4ExpTextDecoderLayer"],
            "sequential_offload_device": "cpu",
            "shuffle_calibration_samples": False,
            "moe_calibrate_all_experts": True,
        }
    if store:
        model.stage_store = store
    oneshot(model=model, tokenizer=tokenizer, recipe=recipe, output_dir=None, clear_sparse_session=True, **calibration)
    quality = None
    if quality_rows:
        saved_after = store.path / "after.json" if store else None
        if saved_after and saved_after.exists():
            after = json.loads(saved_after.read_text())
        else:
            after = evaluate(
                model, quality_rows, device=args.device, chunk_size=args.quality_logit_chunk_size, quantized=True
            )
            if saved_after:
                atomic_json(saved_after, after)
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
        ple="source shards preserved; exact prepared row cache" if ple_cache is not None else "memory-mapped rows",
        checkpoint_dir=str(args.checkpoint_dir) if store else None,
        vision="preserved, not calibrated",
        mtp="preserved in floating point, not calibrated or enabled",
    )
    (args.save_path / "calibration_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    quantize(parse_args())
