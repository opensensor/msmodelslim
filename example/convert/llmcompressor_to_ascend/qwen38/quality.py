"""Paired text perplexity on the offloaded calibration model, before export.

Scores every next token, including user and assistant text. This measures eager
floating-point/CT QDQ reference execution, not deployed Ascend kernels.
"""

import hashlib
import json
import math
import time

import torch
from compressed_tensors.quantization import QuantizationStatus

from .checkpoint import EXPERT_WEIGHT


def require_held_out(calibration, evaluation):
    """Reject duplicate inputs, including a shared prefix truncated to either limit."""
    seen = []
    for row in evaluation:
        ids = row["input_ids"]
        for other in [*calibration, *seen]:
            previous = other["input_ids"]
            length = min(len(ids), len(previous))
            if ids[:length] == previous[:length]:
                raise ValueError("held-out inputs duplicate calibration/evaluation tokens or their truncated prefix")
        seen.append(row)


def check_quantization(model, quantized):
    count = 0
    expected = model.config.num_hidden_layers * model.config.num_experts * 3
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None:
            continue
        if not quantized:
            raise ValueError("floating baseline already has quantization attached")
        if not EXPERT_WEIGHT.fullmatch(name + ".weight"):
            raise ValueError(f"unexpected quantized module: {name}")
        if (
            not getattr(module, "quantization_enabled", True)
            or getattr(module, "quantization_status", None) != QuantizationStatus.FROZEN
        ):
            raise ValueError(f"quantization must be enabled and frozen: {name}")
        weights, activation = scheme.weights, scheme.input_activations
        if (
            weights is None
            or weights.num_bits != 8
            or weights.type != "int"
            or not weights.symmetric
            or weights.dynamic
            or activation is None
            or activation.num_bits != 8
            or activation.type != "int"
            or not activation.symmetric
            or activation.dynamic is not True
            or scheme.output_activations is not None
        ):
            raise ValueError(f"expected symmetric W8A8 with dynamic INT8 inputs: {name}")
        count += 1
    if quantized and count != expected:
        raise ValueError(f"expected {expected} quantized expert projections, found {count}")
    return count


@torch.no_grad()
def evaluate(model, rows, *, device="cpu", chunk_size=128, quantized=False):
    if not rows or chunk_size <= 0:
        raise ValueError("evaluation requires samples and a positive logit chunk size")
    count = check_quantization(model, quantized)
    started = time.monotonic()
    training = model.training
    model.eval()
    samples = []
    try:
        for index, row in enumerate(rows):
            ids = row["input_ids"]
            if len(ids) < 2 or row["attention_mask"] != [1] * len(ids):
                raise ValueError("evaluation requires unpadded sequences with at least two tokens")
            inputs = torch.tensor([ids], device=device)
            hidden = model.model(inputs, attention_mask=torch.ones_like(inputs))
            nll = torch.zeros((), dtype=torch.float64, device=hidden.device)
            for start in range(0, len(ids) - 1, chunk_size):
                stop = min(start + chunk_size, len(ids) - 1)
                # Never materialize sequence_length x vocabulary logits. Only the
                # small final hidden states and one output-head chunk remain live.
                logits = model.lm_head(hidden[:, start:stop])
                loss = torch.nn.functional.cross_entropy(
                    logits[0].float(), inputs[0, start + 1 : stop + 1].to(logits.device), reduction="sum"
                )
                nll += loss.to(nll.device, dtype=nll.dtype)
                del logits, loss
            value = nll.item()  # One sync per completed sample, outside model hot paths.
            if not math.isfinite(value):
                raise ValueError(f"non-finite evaluation loss at sample {index}")
            samples.append({"nll": value, "predicted_tokens": len(ids) - 1})
            del hidden, inputs, nll
            print(f"Quality {'quantized' if quantized else 'baseline'}: {index + 1}/{len(rows)}", flush=True)
    finally:
        model.train(training)
    tokens = sum(sample["predicted_tokens"] for sample in samples)
    mean = sum(sample["nll"] for sample in samples) / tokens
    return {
        "model_path": str(model.config._name_or_path),
        "samples": samples,
        "predicted_tokens": tokens,
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        "token_ids_sha256": hashlib.sha256(json.dumps([row["input_ids"] for row in rows]).encode()).hexdigest(),
        "quantized_projections": count,
        "device": str(device),
        "dtype": str(model.dtype),
        "logit_chunk_size": chunk_size,
        "elapsed_seconds": time.monotonic() - started,
        "metric": "all-token next-token perplexity, including user and assistant text",
        "execution": "compressed-tensors QDQ reference" if quantized else "floating-point reference",
        "ascend_inference_validated": False,
    }


def compare_reports(before, after):
    for key in ("model_path", "token_ids_sha256", "predicted_tokens", "metric", "device", "dtype", "logit_chunk_size"):
        if before[key] != after[key]:
            raise ValueError(f"quality reports must match: {key}")
    if before["quantized_projections"] or not after["quantized_projections"]:
        raise ValueError("comparison requires a floating baseline and quantized result")
    return {
        "before": before,
        "after": after,
        "mean_nll_delta": after["mean_nll"] - before["mean_nll"],
        "relative_perplexity_change_percent": 100 * (after["perplexity"] / before["perplexity"] - 1),
        "scope": "paired eager text reference; not GGUF, generation quality or Ascend serving validation",
    }


def validate_baseline(report, model, rows, device, chunk_size):
    """Validate an explicitly supplied legacy baseline, before attaching GPTQ.

    Legacy reports do not hash the source weights: callers must retain the
    original source. New automatic reuse is additionally bound to StageStore's
    source/code/token fingerprint.
    """
    expected = {
        "model_path": str(model.config._name_or_path),
        "token_ids_sha256": hashlib.sha256(json.dumps([row["input_ids"] for row in rows]).encode()).hexdigest(),
        "predicted_tokens": sum(len(row["input_ids"]) - 1 for row in rows),
        "device": str(device),
        "dtype": str(model.dtype),
        "logit_chunk_size": chunk_size,
        "quantized_projections": 0,
        "execution": "floating-point reference",
        "metric": "all-token next-token perplexity, including user and assistant text",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"baseline mismatch: {key}")
    if len(report["samples"]) != len(rows):
        raise ValueError("baseline sample count mismatch")
    for sample, row in zip(report["samples"], rows, strict=True):
        if sample["predicted_tokens"] != len(row["input_ids"]) - 1 or not math.isfinite(sample["nll"]):
            raise ValueError("invalid baseline sample")
    mean = sum(sample["nll"] for sample in report["samples"]) / expected["predicted_tokens"]
    if not math.isclose(mean, report["mean_nll"]) or not math.isclose(math.exp(mean), report["perplexity"]):
        raise ValueError("baseline aggregate mismatch")


def write_report(path, report):
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
