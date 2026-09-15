"""Independently detect incomplete or corrupted native exports on CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from msmodelslim.core.quant_service.modelslim_convert.impl.int8_verify import verify
from test.integration.test_int8_to_ascend import convert, make_source


@pytest.mark.parametrize("sharded", [False, True])
def test_verify_counts_bytes_and_exact_values(tmp_path, sharded):
    source, output = tmp_path / "source", tmp_path / "output"
    _, original = make_source(source, sharded=sharded)
    convert(source, output)
    report = verify(source, output, check_values=True, chunk_rows=1)
    assert report["validation"] == "values"
    assert report["quantized_linears"] == 2
    assert report["source_tensor_bytes"] == sum(t.numel() * t.element_size() for t in original.values())
    assert report["ascend_inference_validated"] is False
    assert verify(source, output)["validation"] == "headers"
    automatic = json.loads((output / "conversion_report.json").read_text())
    assert automatic["validation"] == "headers"
    assert automatic["quantized_linears"] == 2
    if not sharded:
        script = Path(__file__).resolve().parents[2] / "example/convert/llmcompressor_to_ascend/verify_w8a8.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--source",
                str(source),
                "--export",
                str(output),
                "--check-values",
                "--chunk-rows",
                "1",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["validation"] == "values"


@pytest.mark.parametrize(
    "change,message",
    [
        ("missing", "Tensor inventory mismatch"),
        ("shape", "Tensor layout mismatch"),
        ("tag", "description mismatch"),
        ("config", "Model configuration differs"),
        ("value", "Tensor values differ"),
        ("offset", "Nonzero weight offset"),
        ("duplicate", "Duplicate tensor"),
        ("index", "index does not match"),
    ],
)
def test_verify_rejects_bad_exports(tmp_path, change, message):
    source, output = tmp_path / "source", tmp_path / "output"
    make_source(source)
    convert(source, output)
    file = next(output.glob("*.safetensors"))
    tensors = load_file(file)
    key = "model.layers.0.self_attn.q_proj.weight"
    if change == "missing":
        del tensors["model.layers.0.mlp.gate.e_score_correction_bias"]
        index = next(output.glob("*.safetensors.index.json"))
        data = json.loads(index.read_text())
        del data["weight_map"]["model.layers.0.mlp.gate.e_score_correction_bias"]
        index.write_text(json.dumps(data))
    elif change == "shape":
        tensors[key] = tensors[key].T.contiguous()
    elif change == "value":
        tensors[key][0, 0] -= 1
    elif change == "offset":
        tensors["model.layers.0.self_attn.q_proj.weight_offset"][0, 0] = 1
    elif change == "tag":
        path = output / "quant_model_description.json"
        description = json.loads(path.read_text())
        description[key] = "FLOAT"
        path.write_text(json.dumps(description))
    elif change == "config":
        path = output / "config.json"
        config = json.loads(path.read_text())
        config["vocab_size"] = 9999
        path.write_text(json.dumps(config))
    elif change == "duplicate":
        save_file({key: tensors[key]}, output / "stale.safetensors")
    elif change == "index":
        next(output.glob("*.safetensors.index.json")).write_text(
            json.dumps({"weight_map": {key: "missing.safetensors"}})
        )
    save_file(tensors, file)
    if change in ("value", "offset"):
        # Metadata-only validation must not claim that values have been checked.
        assert verify(source, output)["validation"] == "headers"
    with pytest.raises(ValueError, match=message):
        verify(source, output, check_values=True, chunk_rows=1)


def test_verification_rejects_invalid_chunk_budget(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        verify(tmp_path, tmp_path, chunk_rows=0)
