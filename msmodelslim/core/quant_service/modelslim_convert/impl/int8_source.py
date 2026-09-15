"""Validate a compressed-tensors INT8 checkpoint before any Ascend output is written.

This bridge preserves calibrated weights. It does not infer activation semantics
from an INT8 dtype, requantize weights, or adapt model architectures.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from msmodelslim.core.convert.types import IRKind


def validate_int8_source(reader, catalog, config) -> None:
    validate_int8_destination(config)
    validate_int8_checkpoint(reader, catalog, config)


def validate_int8_destination(config) -> None:
    """Check before the CLI's existing cleanup, allowing its generated recipe."""
    source, destination = Path(config.model_path).resolve(), Path(config.save_path).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("INT8 import requires separate, non-overlapping source and destination directories")
    if destination.exists() and (
        not destination.is_dir()
        or any(item.name != "convert_best_practice.yaml" or not item.is_file() for item in destination.iterdir())
    ):
        raise ValueError("INT8 import requires an empty destination directory")


def validate_int8_checkpoint(reader, catalog, config) -> None:
    """Validate the source contract independently of output-directory policy."""
    model_config = reader.read_model_config()
    quant = model_config.get("quantization_config") or {}
    if not quant:
        quant = (model_config.get("text_config") or {}).get("quantization_config") or {}
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("W8A8_DYNAMIC import requires a compressed-tensors quantization_config")
    if quant.get("format") != "int-quantized" or quant.get("quantization_status") != "compressed":
        raise ValueError("Expected a compressed int-quantized checkpoint, not fake-quantized or packed weights")
    if quant.get("kv_cache_scheme") or quant.get("transform_config") or model_config.get("transform_config"):
        raise ValueError("KV-cache quantization and runtime transforms are not supported by the INT8 bridge")
    if model_config.get("compression_config") or quant.get("sparsity_config"):
        raise ValueError("Additional compression_config or sparsity_config is not supported by the INT8 bridge")
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Missing compressed-tensors config_groups")
    for name, group in groups.items():
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if not _matches(weight, strategy="channel", dynamic=False):
            raise ValueError(f"{name}: require symmetric static INT8 per-channel weights")
        if not _matches(activation, strategy="token", dynamic=True):
            raise ValueError(f"{name}: require symmetric dynamic INT8 per-token input activations")
        if group.get("output_activations") or group.get("format") not in (None, "int-quantized"):
            raise ValueError(f"{name}: output activation quantization or format override is unsupported")

    # Only headers are read here, even when the original checkpoint exceeds RAM.
    reader.enrich_catalog(catalog)
    quantized_paths = set()
    for key, entry in catalog.items():
        if key.endswith(".weight") and entry.dtype == "I8":
            path = key.removesuffix(".weight")
            rule = next((r for r in config.convert_rules if fnmatch.fnmatch(path, r.match)), None)
            if rule is None or rule.action != "transform" or rule.target_ir != IRKind.W8A8_DYNAMIC:
                raise ValueError(f"Quantized layer {path} is not selected for W8A8_DYNAMIC conversion")
            module_rule = next((r for r in config.module_rules if fnmatch.fnmatch(path, r.match)), None)
            if module_rule is None or module_rule.source_ir not in (None, IRKind.INT8_PER_CHANNEL):
                raise ValueError(f"{path}: missing or conflicting INT8 module rule")
            for suffix in ("weight", "weight_scale", "weight_zero_point", "bias"):
                if catalog.get(path + "." + suffix) is not None:
                    binding = module_rule.tensor_map.get(suffix, "").replace("{module}", path)
                    if binding != path + "." + suffix:
                        raise ValueError(f"{path}: missing or remapped {suffix} binding")
            if len(entry.shape) != 2:
                raise ValueError(f"{key}: expected a 2D linear weight; fused expert tensors must be exported unfused")
            scale = catalog.get(path + ".weight_scale")
            if scale is None or scale.shape not in ((entry.shape[0],), (entry.shape[0], 1)):
                raise ValueError(f"{path}: missing or invalid per-output-channel weight_scale")
            if scale.dtype not in ("F16", "BF16", "F32"):
                raise ValueError(f"{path}: expected floating-point per-output-channel scales (FP16/BF16/FP32)")
            bias = catalog.get(path + ".bias")
            if bias is not None and (bias.shape != (entry.shape[0],) or bias.dtype not in ("F16", "BF16", "F32")):
                raise ValueError(f"{path}: invalid linear bias")
            quantized_paths.add(path)
        elif key.endswith(".weight") and entry.dtype not in ("BF16", "F16", "F32", "F64"):
            raise ValueError(f"Unsupported source weight dtype {entry.dtype}: {key}")
        elif entry.dtype == "I8" and not key.endswith(".weight_zero_point"):
            raise ValueError(f"Unsupported INT8 tensor layout: {key}; export unfused 2D linear weights")
    if not quantized_paths:
        raise ValueError("No compressed INT8 linear weights found")
    for key, entry in catalog.items():
        if key.endswith((".weight_scale", ".weight_zero_point")):
            if key.rsplit(".", 1)[0] not in quantized_paths:
                raise ValueError(f"Orphan quantization parameter: {key}")
        elif key.endswith(
            (
                ".weight_packed",
                ".weight_shape",
                ".input_scale",
                ".input_zero_point",
                ".output_scale",
                ".output_zero_point",
                ".weight_g_idx",
                ".weight_scale_inv",
                ".input_global_scale",
                ".weight_global_scale",
            )
        ):
            raise ValueError(f"Unsupported quantization tensor: {key}")


def _matches(args: dict, *, strategy: str, dynamic: bool) -> bool:
    return (
        args.get("num_bits") == 8
        and args.get("type") == "int"
        and args.get("symmetric") is True
        and args.get("strategy") == strategy
        and args.get("dynamic", False) is dynamic
        and args.get("group_size") is None
        and args.get("block_structure") is None
        # GPTQ static/weight ordering is undone before serialization; no runtime permutation.
        and args.get("actorder") in ((None, "static", "weight") if not dynamic else (None,))
    )
