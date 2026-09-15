"""Account for Qwen3.8-Flash-Next storage from a complete safetensors header inventory.

This plans hypothetical layouts. It does not convert weights or certify runtime fit.
"""

import argparse
import json
import math
import re
from pathlib import Path

GIB = 1024**3
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
EXPERT = re.compile(r"(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$")


def storage(tensor):
    shape = tensor["shape"]
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("tensor dimensions must be positive integers")
    if tensor["dtype"] not in DTYPE_BYTES:
        raise ValueError("expected original floating-point checkpoint and I64 auxiliary tensors")
    elements = math.prod(shape)
    size = elements * DTYPE_BYTES[tensor["dtype"]]
    start, end = tensor["data_offsets"]
    if start < 0 or end - start != size:
        raise ValueError("tensor payload disagrees with shape/dtype")
    return elements, size


def plan(inventory, config, devices=4, memory_gb=48, reserve_gib=6, host_memory_gb=256):
    if type(devices) is not int or devices <= 0:
        raise ValueError("devices must be a positive integer")
    if not math.isfinite(memory_gb) or not math.isfinite(reserve_gib) or memory_gb <= 0 or reserve_gib < 0:
        raise ValueError("invalid memory budget")
    per_device_budget = int(memory_gb * 10**9 - reserve_gib * GIB)
    if per_device_budget <= 0:
        raise ValueError("reserve consumes the nominal device memory")
    if not math.isfinite(host_memory_gb) or host_memory_gb <= 0:
        raise ValueError("host memory must be finite and positive")
    if config.get("model_type") != "qwen4_exp" or config.get("quantization_config"):
        raise ValueError("expected original qwen4_exp configuration")
    text = config["text_config"]
    if text.get("quantization_config"):
        raise ValueError("expected floating-point text weights")
    tensors = inventory["tensors"]
    if not tensors:
        raise ValueError("empty tensor inventory")
    groups = {}
    expert_names = set()
    ngram_names = set()
    expert_int8 = {False: 0, True: 0}
    float_bytes = {False: 0, True: 0}
    ple_bytes = {16: 0, 8: 0, 4: 0}
    for name, tensor in tensors.items():
        elements, size = storage(tensor)
        mtp = name.startswith("mtp.")
        match = EXPERT.fullmatch(name)
        if match:
            expert_names.add(name)
            shape = tensor["shape"]
            expected = [text["num_experts"], 2 * text["moe_intermediate_size"], text["hidden_size"]]
            if match[1] == "down_proj":
                expected = [text["num_experts"], text["hidden_size"], text["moe_intermediate_size"]]
            if shape != expected or tensor["dtype"] not in ("BF16", "F16"):
                raise ValueError(f"unexpected fused expert layout: {name}")
            # Splitting gate/up preserves row count. Native dynamic W8A8 stores
            # one FP32 scale and one FP32 zero offset per output channel.
            expert_int8[mtp] += elements + shape[0] * shape[1] * 8
            group = "mtp_experts" if mtp else "main_experts"
        elif ".ngram_embedding." in name:
            ngram_names.add(name)
            shape = tensor["shape"]
            if len(shape) != 2 or tensor["dtype"] not in ("BF16", "F16"):
                raise ValueError("expected two-dimensional floating-point n-gram tables")
            rows, width = shape
            ple_bytes[16] += size
            # Hypothetical symmetric row-wise INT8/INT4 lookup storage with
            # one FP32 scale per row and no serialized zero point.
            for bits in (8, 4):
                ple_bytes[bits] += rows * ((width * bits + 7) // 8 + 4)
            group = "ngram_tables"
        else:
            if ".experts." in name:
                raise ValueError(f"unrecognized expert tensor: {name}")
            float_bytes[mtp] += size
            group = "mtp_other" if mtp else "vision" if name.startswith("model.visual.") else "main_other"
        bucket = groups.setdefault(group, {"tensors": 0, "elements": 0, "bytes": 0})
        bucket["tensors"] += 1
        bucket["elements"] += elements
        bucket["bytes"] += size
    expected_experts = {
        f"{prefix}.layers.{layer}.mlp.experts.{projection}"
        for prefix, count in (
            ("model.language_model", text["num_hidden_layers"]),
            ("mtp", text["mtp_num_hidden_layers"]),
        )
        for layer in range(count)
        for projection in ("gate_up_proj", "down_proj")
    }
    expected_ngrams = {
        f"model.language_model.layers.{layer - 1}.ple.ple_embedding.ngram_embedding.shard_{part}.weight"
        for layer in text["ple_layer_ids"]
        for part in range(text["split_ngram_parts"])
    }
    if expert_names != expected_experts or ngram_names != expected_ngrams:
        raise ValueError("incomplete or unexpected expert/ngram inventory")
    total = sum(g["bytes"] for g in groups.values())
    if total != inventory["source_tensor_bytes"]:
        raise ValueError("inventory total does not match declared source bytes")
    scenarios = []
    for include_mtp in (False, True):
        core = expert_int8[False] + float_bytes[False]
        if include_mtp:
            core += expert_int8[True] + float_bytes[True]
        for bits, placement in ((16, "device"), (16, "host"), (8, "device"), (8, "host"), (4, "device")):
            npu = core + (ple_bytes[bits] if placement == "device" else 0)
            host_table = ple_bytes[bits] if placement == "host" else 0
            scenarios.append(
                {
                    "expert_weight_bits": 8,
                    "expert_activation_bits": 8,
                    "include_mtp": include_mtp,
                    "ple_storage_bits": bits,
                    "ple_placement": placement,
                    "checkpoint_tensor_bytes": core + ple_bytes[bits],
                    "npu_tensor_bytes_before_runtime_layout": npu,
                    "host_table_bytes_one_distributed_copy": host_table,
                    "host_bytes_remaining_before_other_allocations": int(host_memory_gb * 10**9) - host_table,
                    "ideal_average_npu_tensor_bytes": (npu + devices - 1) // devices,
                    "aggregate_exceeds_planning_budget": npu > devices * per_device_budget,
                }
            )
    return {
        "repo_id": inventory["repo_id"],
        "revision": inventory["revision"],
        "source_tensor_bytes": total,
        "source_groups": groups,
        "devices": devices,
        "nominal_memory_gb_per_device": memory_gb,
        "nominal_host_memory_gb": host_memory_gb,
        "assumed_runtime_reserve_gib_per_device": reserve_gib,
        "planning_weight_budget_bytes_per_device": per_device_budget,
        "attention_heads": text["num_attention_heads"],
        "kv_heads": text["num_key_value_heads"],
        "kv_replication_warning": devices > text["num_key_value_heads"],
        "scenarios": scenarios,
        "assumptions": [
            "Only routed expert projections use W8A8_DYNAMIC; other tensors retain their source byte width.",
            "BF16 floating-point runtime tensors would need FP16 handling on 310P; equal storage size does not prove numerical parity.",
            "INT8/INT4 PLE entries are proposed row-wise formats, not supported native Ascend checkpoints.",
            "Host PLE size assumes one table distributed across ranks, not a complete private copy per rank.",
            "Ideal averages exclude rank imbalance, replicated parameters, NZ padding, KV/recurrent state and runtime workspaces.",
            "A scenario below the aggregate planning budget is not a verified per-device fit.",
        ],
        "conversion_implemented": False,
        "ascend_inference_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--memory-gb-per-device", type=float, default=48)
    parser.add_argument("--reserve-gib-per-device", type=float, default=6)
    parser.add_argument("--host-memory-gb", type=float, default=256)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    try:
        result = plan(
            json.loads(args.inventory.read_text()),
            json.loads(args.config.read_text()),
            args.devices,
            args.memory_gb_per_device,
            args.reserve_gib_per_device,
            args.host_memory_gb,
        )
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved hypothetical storage plan: {args.report}")


if __name__ == "__main__":
    main()
