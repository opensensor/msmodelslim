"""Check checkpoint accounting without downloading model weights or using a GPU."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "example/convert/llmcompressor_to_ascend"


def load_planner():
    spec = importlib.util.spec_from_file_location("qwen38_planner_test", EXAMPLE / "plan_qwen38.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_inventory():
    tensors = {}

    def add(name, shape):
        size = 2
        for width in shape:
            size *= width
        tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [0, size]}

    for prefix in ("model.language_model", "mtp"):
        add(f"{prefix}.layers.0.mlp.experts.gate_up_proj", [2, 4, 4])
        add(f"{prefix}.layers.0.mlp.experts.down_proj", [2, 4, 2])
        add(f"{prefix}.norm.weight", [4])
    add("model.visual.weight", [4, 4])
    for shard in range(2):
        add(f"model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_{shard}.weight", [3, 5])
    inventory = {"repo_id": "test", "revision": "a" * 40, "source_tensor_bytes": 300, "tensors": tensors}
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "num_hidden_layers": 1,
            "mtp_num_hidden_layers": 1,
            "num_experts": 2,
            "hidden_size": 4,
            "moe_intermediate_size": 2,
            "ple_layer_ids": [1],
            "split_ngram_parts": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
    }
    return inventory, config


def test_expert_metadata_ple_scales_and_mtp_accounting():
    inventory, config = fixture_inventory()
    result = load_planner().plan(inventory, config)
    assert result["source_tensor_bytes"] == 300
    assert result["source_groups"]["main_experts"]["bytes"] == 96
    scenarios = {(s["include_mtp"], s["ple_storage_bits"], s["ple_placement"]): s for s in result["scenarios"]}
    # 48 INT8 expert values + 16 output channels * 8 metadata bytes + 40 FLOAT bytes.
    host = scenarios[False, 16, "host"]
    assert host["npu_tensor_bytes_before_runtime_layout"] == 216
    assert host["host_table_bytes_one_distributed_copy"] == 60
    assert host["host_bytes_remaining_before_other_allocations"] == 256 * 10**9 - 60
    assert host["checkpoint_tensor_bytes"] == 276
    assert host["ideal_average_npu_tensor_bytes"] == 54
    assert scenarios[True, 16, "host"]["npu_tensor_bytes_before_runtime_layout"] == 400
    # Six 5-wide rows: 5 INT8 bytes + 4 scale bytes, or 3 packed INT4 bytes + 4 scale bytes.
    assert scenarios[False, 8, "host"]["host_table_bytes_one_distributed_copy"] == 54
    assert scenarios[False, 4, "device"]["npu_tensor_bytes_before_runtime_layout"] == 258
    assert result["kv_replication_warning"] is True
    assert result["conversion_implemented"] is False
    assert result["ascend_inference_validated"] is False


def test_host_capacity_is_explicit_and_does_not_certify_fit():
    planner = load_planner()
    inventory, config = fixture_inventory()
    result = planner.plan(inventory, config, host_memory_gb=0.00000001)
    host = next(s for s in result["scenarios"] if s["ple_placement"] == "host")
    assert host["host_bytes_remaining_before_other_allocations"] == -50
    assert result["ascend_inference_validated"] is False
    with pytest.raises(ValueError, match="host memory"):
        planner.plan(inventory, config, host_memory_gb=float("nan"))


@pytest.mark.parametrize("case", ["expert_missing", "ngram_missing", "expert_shape", "quantized", "total", "span"])
def test_rejects_incomplete_or_incompatible_sources(case):
    inventory, config = fixture_inventory()
    expert = "model.language_model.layers.0.mlp.experts.down_proj"
    if case == "expert_missing":
        del inventory["tensors"][expert]
    elif case == "ngram_missing":
        del inventory["tensors"][next(n for n in inventory["tensors"] if "ngram_embedding" in n)]
    elif case == "expert_shape":
        inventory["tensors"][expert]["shape"] = [2, 2, 4]
    elif case == "quantized":
        config["quantization_config"] = {"quant_method": "fp8"}
    elif case == "total":
        inventory["source_tensor_bytes"] -= 1
    else:
        inventory["tensors"][expert]["data_offsets"][1] -= 1
    with pytest.raises(ValueError):
        load_planner().plan(inventory, config)


@pytest.mark.parametrize("devices,memory,reserve", [(0, 48, 6), (4, float("nan"), 6), (4, 48, -1), (4, 1, 2)])
def test_rejects_invalid_device_budgets(devices, memory, reserve):
    with pytest.raises(ValueError):
        load_planner().plan(*fixture_inventory(), devices, memory, reserve)


def test_planner_cli_preserves_existing_report(tmp_path):
    inventory, config = fixture_inventory()
    source = tmp_path / "inventory.json"
    model = tmp_path / "config.json"
    report = tmp_path / "report.json"
    source.write_text(json.dumps(inventory))
    model.write_text(json.dumps(config))
    command = [
        sys.executable,
        str(EXAMPLE / "plan_qwen38.py"),
        "--inventory",
        str(source),
        "--config",
        str(model),
        "--report",
        str(report),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    original = report.read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert report.read_bytes() == original


def test_header_validation_and_range_refusal():
    # Run in a subprocess so the example's sibling import does not modify pytest's import path.
    code = '''
from unittest.mock import patch
from fetch_qwen38_headers import read_range, validate_header
tensors = {"x": {"dtype":"BF16", "shape":[2], "data_offsets":[0,4]}}
validate_header(tensors, 64, 76)
for size in (75, 77):
    try: validate_header(tensors, 64, size)
    except ValueError: pass
    else: raise AssertionError("bad file size accepted")
with patch("urllib.request.urlopen") as opener:
    response=opener.return_value.__enter__.return_value
    response.status=200
    try: read_range("https://example.test/weights", 0, 7, 76)
    except ValueError: pass
    else: raise AssertionError("full-body response accepted")
    response.read.assert_not_called()
    response.status=206
    response.headers={"Content-Range":"bytes 0-7/76"}
    response.read.return_value=b"12345678"
    assert read_range("https://example.test/weights", 0, 7, 76)==b"12345678"
    response.read.return_value=b"123"
    try: read_range("https://example.test/weights", 0, 7, 76)
    except ValueError: pass
    else: raise AssertionError("truncated range accepted")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=EXAMPLE, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
