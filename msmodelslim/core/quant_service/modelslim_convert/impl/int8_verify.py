"""Verify an Ascend W8A8_DYNAMIC export against its compressed-tensors source."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from msmodelslim.core.quant_service.modelslim_convert.config_mapper import spec_to_convert_config
from msmodelslim.core.quant_service.modelslim_convert.impl.int8_source import validate_int8_checkpoint
from msmodelslim.infra.io.checkpoint_reader import CheckpointReader

DTYPE_BYTES = {"I8": 1, "U8": 1, "BOOL": 1, "F16": 2, "BF16": 2, "I16": 2, "F32": 4, "I32": 4, "F64": 8, "I64": 8}


def inventory(directory):
    result = {}
    for file in sorted(directory.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for key in keys:
                if key in result:
                    raise ValueError(f"Duplicate tensor in checkpoint shards: {key}")
                tensor = handle.get_slice(key)
                result[key] = (file, tuple(tensor.get_shape()), tensor.get_dtype())
    if not result:
        raise ValueError(f"No safetensors tensors under {directory}")
    indices = list(directory.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError(f"Multiple checkpoint indices under {directory}")
    if indices:
        weight_map = json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"]
        if set(weight_map) != set(result) or any(weight_map[key] != result[key][0].name for key in result):
            raise ValueError(f"Checkpoint index does not match tensor shards: {indices[0]}")
    return result


def pieces(handle, key, shape, chunk_rows):
    if not shape:
        yield handle.get_tensor(key)
    else:
        tensor = handle.get_slice(key)
        for start in range(0, shape[0], chunk_rows):
            yield tensor[start : start + chunk_rows]


def verify(source, destination, *, check_values=False, chunk_rows=1024):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    source, destination = Path(source), Path(destination)
    before, after = inventory(source), inventory(destination)
    config = spec_to_convert_config(
        {"linears": [{"match": ["*"], "target": "W8A8_DYNAMIC", "route": "auto"}]}, str(source), str(destination)
    )
    reader = CheckpointReader(source)
    validate_int8_checkpoint(reader, reader.read_catalog(), config)
    quantized = {
        key.removesuffix(".weight")
        for key, (_, _, dtype) in before.items()
        if key.endswith(".weight") and dtype == "I8"
    }
    zero_points = {key for key in before if key.endswith(".weight_zero_point")}
    expected_keys = (set(before) - zero_points) | {p + ".weight_offset" for p in quantized}
    if set(after) != expected_keys:
        raise ValueError(
            f"Tensor inventory mismatch: missing={sorted(expected_keys - set(after))[:8]}, "
            f"unexpected={sorted(set(after) - expected_keys)[:8]}"
        )
    description = json.loads((destination / "quant_model_description.json").read_text(encoding="utf-8"))
    if description.get("model_quant_type") != "W8A8_DYNAMIC":
        raise ValueError("Expected model_quant_type W8A8_DYNAMIC")
    source_config = reader.read_model_config()
    source_config.pop("quantization_config", None)
    if isinstance(source_config.get("text_config"), dict):
        source_config["text_config"].pop("quantization_config", None)
    output_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    if source_config != output_config:
        raise ValueError("Model configuration differs beyond removal of quantization_config")

    for key, (file, shape, dtype) in after.items():
        prefix, _, suffix = key.rpartition(".")
        native_scale = prefix in quantized and suffix in ("weight_scale", "weight_offset")
        expected_type = (
            "W8A8_DYNAMIC" if prefix in quantized and suffix in ("weight", "weight_scale", "weight_offset") else "FLOAT"
        )
        if native_scale:
            expected_shape, expected_dtype = (before[prefix + ".weight"][1][0], 1), "F32"
        else:
            _, expected_shape, expected_dtype = before[key]
            if prefix in quantized and suffix == "bias":
                expected_dtype = "F32"
        if (shape, dtype) != (expected_shape, expected_dtype):
            raise ValueError(f"Tensor layout mismatch: {key}: {(shape, dtype)} != {(expected_shape, expected_dtype)}")
        if description.get(key) != expected_type:
            raise ValueError(f"Quantization description mismatch: {key}")
        if not check_values:
            continue
        with ExitStack() as stack:
            output = stack.enter_context(safe_open(file, framework="pt", device="cpu"))
            output_parts = pieces(output, key, shape, chunk_rows)
            if suffix == "weight_offset" and prefix in quantized:
                if any(torch.count_nonzero(part) for part in output_parts):
                    raise ValueError(f"Nonzero weight offset: {key}")
                continue
            original = stack.enter_context(safe_open(before[key][0], framework="pt", device="cpu"))
            input_parts = pieces(original, key, before[key][1], chunk_rows)
            for input_part, output_part in zip(input_parts, output_parts, strict=True):
                expected = input_part.to(output_part.dtype).reshape(output_part.shape)
                if not torch.equal(expected, output_part):
                    raise ValueError(f"Tensor values differ: {key}")
                if native_scale and (not torch.isfinite(output_part).all() or not (output_part > 0).all()):
                    raise ValueError(f"Invalid weight scale: {key}")
    if check_values:
        for key in zero_points:
            file, shape, _ = before[key]
            with safe_open(file, framework="pt", device="cpu") as handle:
                if any(torch.count_nonzero(part) for part in pieces(handle, key, shape, chunk_rows)):
                    raise ValueError(f"Nonzero source zero point: {key}")
    return {
        "validation": "values" if check_values else "headers",
        "quantized_linears": len(quantized),
        "source_tensors": len(before),
        "export_tensors": len(after),
        "source_tensor_bytes": sum(math.prod(shape) * DTYPE_BYTES[dtype] for _, shape, dtype in before.values()),
        "export_tensor_bytes": sum(math.prod(shape) * DTYPE_BYTES[dtype] for _, shape, dtype in after.values()),
        "ascend_inference_validated": False,
    }
