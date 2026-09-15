"""Measure first-layer Qwen38 MoE GPTQ with real inputs and CPU/disk offload.

Only first-layer attention, gated residuals, MoE weights and token embeddings
are read. This component probe can run before the whole checkpoint downloads.
It does not export a model or measure model perplexity/Ascend inference.
"""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import torch
from accelerate import init_empty_weights
from compressed_tensors.offload import offload_module
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from pydantic import PrivateAttr
from quantize_qwen38 import tokenize_records
from quantize_w8a8 import positive_int, read_calibration
from qwen38.checkpoint import Checkpoint
from qwen38.model import LinearExperts, build_config
from qwen38.quality import require_held_out
from qwen38.reference import Qwen4ExpTextGatedDeltaNet, Qwen4ExpTextGatedResidual, Qwen4ExpTextSparseMoeBlock
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

LAYER_PREFIX = "model.language_model.layers.0"


def load_part(source, module, prefix, *, device, offload_dir=None):
    """Load one projection at a time; disk placement deliberately stresses offload."""
    for name, child in module.named_modules():
        for key, parameter in list(child.named_parameters(recurse=False)):
            full_name = ".".join(part for part in (prefix, name, key) if part)
            value = source.read(full_name, dtype=torch.float16)
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid source parameter: {full_name}")
            child._parameters[key] = nn.Parameter(value, requires_grad=False)
        if offload_dir is not None and (child._parameters or child._buffers):
            offload_module(child, device, "disk", offload_dir=str(offload_dir))
    if offload_dir is None:
        module.to(device)
    return module.eval()


@torch.no_grad()
def capture_inputs(source, config, rows, device):
    if config.layer_types[0] != "linear_attention" or 1 in config.ple_layer_ids:
        raise ValueError("probe requires first-layer GDN without PLE")
    with init_empty_weights():
        attention_hc = Qwen4ExpTextGatedResidual(config)
        attention = Qwen4ExpTextGatedDeltaNet(config, 0)
        mlp_hc = Qwen4ExpTextGatedResidual(config)
    attention_hc = load_part(source, attention_hc, LAYER_PREFIX + ".attn_hyper_connection", device=device)
    attention = load_part(source, attention, LAYER_PREFIX + ".linear_attn", device=device)
    mlp_hc = load_part(source, mlp_hc, LAYER_PREFIX + ".mlp_hyper_connection", device=device)
    captures = []
    for row in rows:
        ids = torch.tensor([row["input_ids"]])
        name = "model.language_model.embed_tokens.weight"
        with source.open(name) as handle:
            hidden = handle.get_tensor(name)[ids].to(device=device, dtype=torch.float16)
        hidden, residual, injection = attention_hc(hidden.repeat(1, 1, config.hc_count))
        hidden = attention(hidden, cache_params=None, attention_mask=torch.ones_like(ids, device=device))
        hidden = residual + (hidden.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        hidden, _, _ = mlp_hc(hidden)
        if not torch.isfinite(hidden).all():
            raise ValueError("nonfinite captured first-layer inputs")
        captures.append(hidden[0].cpu())
    return captures


class CapturedMoe(PreTrainedModel):
    _no_split_modules: ClassVar[list[str]] = ["Qwen4ExpTextSparseMoeBlock"]

    def __init__(self, features, moe):
        super().__init__(
            PretrainedConfig(vocab_size=features.shape[0], hidden_size=features.shape[1], tie_word_embeddings=False)
        )
        self.capture = nn.Embedding.from_pretrained(features, freeze=True)
        self.mlp = moe

    def get_input_embeddings(self):
        return self.capture

    def forward(self, input_ids, attention_mask=None):
        return self.mlp(self.capture(input_ids))


class MeasuredGPTQ(GPTQModifier):
    """Observe the installed compressor's live matrices without changing its math."""

    _observations: list = PrivateAttr(default_factory=list)

    def compress_modules(self):
        if self._hessians:
            self._observations.append(
                {
                    "matrices": len(self._hessians),
                    "hessian_bytes": sum(value.numel() * value.element_size() for value in self._hessians.values()),
                    "minimum_samples": min(int(value) for value in self._num_samples.values()),
                }
            )
            print("GPTQ matrices: " + json.dumps(self._observations[-1]), flush=True)
        return super().compress_modules()


def probe(args):
    if args.device == "cpu" and torch.accelerator.is_available():
        raise ValueError("CPU probe requires accelerator isolation before Python starts; hide CUDA devices")
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    source = Checkpoint(args.model_path)
    config = build_config(source.text_config)
    tokenizer = AutoTokenizer.from_pretrained(source.path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = tokenize_records(
        tokenizer,
        read_calibration(SimpleNamespace(calibration_data=args.calibration_data, samples=args.samples)),
        args.sequence_length,
    )
    test = tokenize_records(
        tokenizer,
        read_calibration(SimpleNamespace(calibration_data=args.quality_data, samples=args.quality_samples)),
        args.sequence_length,
    )
    require_held_out(train, test)
    captures = capture_inputs(source, config, train + test, args.device)
    features = torch.cat(captures).to(args.device)
    offsets, position = [], 0
    for capture in captures:
        offsets.append(list(range(position, position + len(capture))))
        position += len(capture)
    with init_empty_weights():
        moe = Qwen4ExpTextSparseMoeBlock(config)
        moe.experts = LinearExperts(config)
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    moe = load_part(source, moe, LAYER_PREFIX + ".mlp", device=args.device, offload_dir=args.offload_dir)
    model = CapturedMoe(features, moe).eval()
    expected = [model(torch.tensor([row], device=args.device)).cpu() for row in offsets[len(train) :]]
    recipe = MeasuredGPTQ(scheme="W8A8", targets=[r"re:.*experts\.\d+\.(gate_proj|up_proj|down_proj)"])
    calibration_started = time.monotonic()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=Dataset.from_dict(
            {"input_ids": offsets[: len(train)], "attention_mask": [[1] * len(r) for r in offsets[: len(train)]]}
        ),
        num_calibration_samples=len(train),
        max_seq_length=args.sequence_length,
        pipeline="sequential",
        sequential_targets=["Qwen4ExpTextSparseMoeBlock"],
        sequential_offload_device="cpu",
        shuffle_calibration_samples=False,
        moe_calibrate_all_experts=True,
        output_dir=None,
        clear_sparse_session=True,
    )
    calibration_seconds = time.monotonic() - calibration_started
    quantized = [module for module in moe.modules() if getattr(module, "quantization_scheme", None) is not None]
    if len(quantized) != config.num_experts * 3 or any(not m.quantization_enabled for m in quantized):
        raise ValueError("not every expert projection has active quantization")
    error_sum = reference_sum = elements = 0
    for row, reference in zip(offsets[len(train) :], expected, strict=True):
        actual = model(torch.tensor([row], device=args.device)).float().cpu()
        if not torch.isfinite(actual).all():
            raise ValueError("nonfinite quantized MoE output")
        error_sum += (actual - reference.float()).square().sum().item()
        reference_sum += reference.float().square().sum().item()
        elements += actual.numel()
    report = {
        "model_path": str(source.path),
        "layer": 0,
        "experts": config.num_experts,
        "quantized_projections": len(quantized),
        "device": args.device,
        "dtype": "float16",
        "weight_storage": "disk",
        "calibration_samples": len(train),
        "calibration_tokens": sum(len(row["input_ids"]) for row in train),
        "held_out_samples": len(test),
        "held_out_tokens": sum(len(row["input_ids"]) for row in test),
        "token_rows_sha256": hashlib.sha256(json.dumps(train + test, sort_keys=True).encode()).hexdigest(),
        "hessian_observations": recipe._observations,
        "held_out_moe_output_rmse": (error_sum / elements) ** 0.5,
        "held_out_moe_reference_rms": (reference_sum / elements) ** 0.5,
        "calibration_seconds": calibration_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
        "peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "first-layer MoE with real captured inputs; not full-model quality, export or Ascend validation",
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--quality-data", type=Path, required=True)
    parser.add_argument("--samples", type=positive_int, default=8)
    parser.add_argument("--quality-samples", type=positive_int, default=2)
    parser.add_argument("--sequence-length", type=positive_int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    paths = [args.model_path.resolve(), args.offload_dir.resolve(), args.report.resolve()]
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                parser.error("model, offload and report paths must not overlap")
    if args.offload_dir.exists() and (not args.offload_dir.is_dir() or any(args.offload_dir.iterdir())):
        parser.error("offload directory must be fresh")
    if args.report.exists() or not args.report.parent.is_dir():
        parser.error("report must be new with an existing parent directory")
    return args


if __name__ == "__main__":
    probe(parse_args())
