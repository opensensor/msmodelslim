"""Write calibrated experts plus every untouched source tensor in bounded shards."""

import json
import shutil
from pathlib import Path

import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.compressors.naive_quantized import IntQuantizationCompressor
from compressed_tensors.quantization import QuantizationStatus
from safetensors.torch import save_file

from .checkpoint import EXPERT_WEIGHT


def save_checkpoint(model, destination, shard_bytes=1024**3, float_dtype=torch.float16):
    destination = Path(destination)
    if shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("export requires an empty destination")
    if destination.resolve() == model.checkpoint.path or model.checkpoint.path in destination.resolve().parents:
        raise ValueError("export must be outside the source checkpoint")
    quantized = {}
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is not None:
            if not EXPERT_WEIGHT.fullmatch(name + ".weight"):
                raise ValueError(f"only routed expert projections may be quantized: {name}")
            quantized[name] = module
    config = model.checkpoint.text_config
    expected = config["num_hidden_layers"] * config["num_experts"] * 3
    if len(quantized) != expected:
        raise ValueError(f"expected {expected} calibrated expert projections, found {len(quantized)}")
    compressor = ModelCompressor.from_pretrained_model(model, quantization_format="int-quantized")
    compressor.quantization_config.quantization_status = QuantizationStatus.COMPRESSED
    destination.mkdir(parents=True, exist_ok=True)
    pending, weight_map = {}, {}
    pending_bytes = total_bytes = shard_index = 0

    def flush():
        nonlocal pending, pending_bytes, shard_index
        if not pending:
            return
        shard_index += 1
        filename = f"model-{shard_index:05d}.safetensors"
        save_file(pending, str(destination / filename), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(pending, filename))
        pending, pending_bytes = {}, 0

    def append(name, value):
        nonlocal pending_bytes, total_bytes
        value = value.detach().cpu().contiguous()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite export tensor: {name}")
        size = value.numel() * value.element_size()
        if pending_bytes + size > shard_bytes:
            flush()
        if name in pending or name in weight_map:
            raise ValueError(f"duplicate export tensor: {name}")
        pending[name] = value
        pending_bytes += size
        total_bytes += size
        if pending_bytes >= shard_bytes:
            flush()

    with torch.no_grad():
        for name, module in quantized.items():
            state = {key: value.detach().cpu() for key, value in module.state_dict().items()}
            compressed = IntQuantizationCompressor.compress(state, module.quantization_scheme)
            for key, value in compressed.items():
                append(f"{name}.{key}", value)
        for name in model.checkpoint.weight_map:
            if name.startswith("model.language_model.") and name.endswith(
                (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")
            ):
                continue
            value = model.checkpoint.read(name)
            if value.is_floating_point():
                # Preserve explicitly FP32 source tensors; convert BF16 for 310P.
                value = value.to(torch.float32 if value.dtype == torch.float32 else float_dtype)
            append(name, value)
        flush()
    for source in model.checkpoint.path.iterdir():
        if (
            source.is_file()
            and (source.suffix in (".json", ".jinja", ".txt") or source.name in ("LICENSE", "README.md"))
            and source.name not in ("model.safetensors.index.json", "download_provenance.json")
        ):
            shutil.copyfile(source, destination / source.name)
    compressor.update_config(str(destination))
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return {"quantized_projections": len(quantized), "tensor_bytes": total_bytes, "shards": shard_index}
