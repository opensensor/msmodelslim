"""Exercise the public calibration command with local models and sequential GPTQ."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from msmodelslim.core.quant_service.modelslim_convert.impl.int8_verify import verify

pytest.importorskip("llmcompressor")
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3NextConfig,
    Qwen3NextForCausalLM,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example/convert/llmcompressor_to_ascend"
CPU_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def create_source(path, architecture="dense", num_layers=1):
    torch.manual_seed(42)
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attention_bias": True,
        "tie_word_embeddings": False,
    }
    if architecture == "moe":
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(**kwargs, moe_intermediate_size=48, num_experts=4, num_experts_per_tok=2)
        )
    elif architecture == "glm":
        model = Glm4MoeForCausalLM(
            Glm4MoeConfig(
                **{**kwargs, "num_hidden_layers": 2},
                moe_intermediate_size=48,
                n_routed_experts=4,
                num_experts_per_tok=2,
                n_shared_experts=1,
                first_k_dense_replace=1,
            )
        )
    elif architecture == "next":
        model = Qwen3NextForCausalLM(
            Qwen3NextConfig(
                **{**kwargs, "num_hidden_layers": 2},
                moe_intermediate_size=48,
                num_experts=4,
                num_experts_per_tok=2,
                shared_expert_intermediate_size=48,
                layer_types=["linear_attention", "full_attention"],
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
            )
        )
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**kwargs))
    model.eval().save_pretrained(path)
    raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, **{f"t{i}": i for i in range(2, 32)}}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        pad_token="[PAD]",
        chat_template="{% for message in messages %}{{ message['content'] }} {% endfor %}",
    )
    tokenizer.save_pretrained(path)
    return model


def run(command):
    return subprocess.run(command, env=CPU_ENV, capture_output=True, text=True, timeout=120, check=False)


def read_tensors(path):
    tensors = {}
    for file in path.glob("*.safetensors"):
        tensors.update(load_file(file))
    return tensors


@pytest.mark.parametrize("architecture", ["dense", "moe", "glm", "next"])
@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_local_calibration_then_ascend_export(tmp_path, architecture, method):
    source, compressed, ascend = tmp_path / "source", tmp_path / "compressed", tmp_path / "ascend"
    create_source(source, architecture)
    command = [
        sys.executable,
        str(EXAMPLE / "quantize_w8a8.py"),
        "--model-path",
        str(source),
        "--save-path",
        str(compressed),
        "--offload-dir",
        str(tmp_path / "offload"),
        "--method",
        method,
        "--device",
        "cpu",
        "--dtype",
        "bfloat16",
        "--cpu-memory-gib",
        "1",
    ]
    if method == "gptq":
        data = tmp_path / "calibration.jsonl"
        records = [{"messages": [{"role": "user", "content": "t2 t3 t4 t5 t6 t7 t8 t9"}]}] * 4
        data.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        command += ["--calibration-data", str(data), "--samples", "4", "--sequence-length", "8"]
    result = run(command)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((compressed / "calibration_manifest.json").read_text())
    assert manifest["method"] == method
    assert manifest["calibration_samples"] == (4 if method == "gptq" else 0)
    assert manifest["device"] == "cpu"
    before = read_tensors(compressed)
    int8_keys = {key for key, tensor in before.items() if key.endswith(".weight") and tensor.dtype == torch.int8}
    assert len(int8_keys) == {"dense": 7, "moe": 16, "glm": 26, "next": 37}[architecture]
    if architecture == "moe":
        assert len([k for k in int8_keys if ".experts." in k]) == 12
        assert not any("gate_up_proj" in k for k in before)
        assert before["model.layers.0.mlp.gate.weight"].dtype == torch.bfloat16
    if architecture == "glm":
        assert len([k for k in int8_keys if ".experts." in k]) == 12
        assert before["model.layers.1.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    if architecture == "next":
        assert len([k for k in int8_keys if ".experts." in k]) == 24
        assert before["model.layers.0.linear_attn.conv1d.weight"].dtype == torch.bfloat16
        assert "model.layers.0.linear_attn.A_log" in before
        assert "model.layers.0.linear_attn.dt_bias" in before
    result = run(
        [
            str(Path(sys.executable).parent / "msmodelslim"),
            "quant",
            "--device",
            "cpu",
            "--model_path",
            str(compressed),
            "--save_path",
            str(ascend),
            "--config",
            str(EXAMPLE / "w8a8_dynamic.yaml"),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    after = read_tensors(ascend)
    for key, tensor in before.items():
        if key.endswith(".weight_zero_point"):
            assert key not in after
        else:
            expected = tensor.float() if key.endswith((".weight_scale", ".bias")) else tensor
            torch.testing.assert_close(after[key], expected, rtol=0, atol=0)
    description = json.loads((ascend / "quant_model_description.json").read_text())
    assert all(description[k] == "W8A8_DYNAMIC" for k in int8_keys)
    assert (ascend / "calibration_manifest.json").read_bytes() == (
        compressed / "calibration_manifest.json"
    ).read_bytes()
    report = verify(compressed, ascend, check_values=True, chunk_rows=7)
    assert report["quantized_linears"] == len(int8_keys)
    assert report["validation"] == "values"


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_disk_offload_matches_cpu_quantization(tmp_path, method):
    source, compressed, offload = tmp_path / "source", tmp_path / "compressed", tmp_path / "offload"
    create_source(source, num_layers=4)
    source_bytes = {p.name: p.read_bytes() for p in source.glob("*.safetensors")}
    extra = []
    if method == "gptq":
        data = tmp_path / "calibration.jsonl"
        data.write_text('\n'.join(json.dumps({"text": "t2 t3 t4 t5 t6 t7 t8 t9"}) for _ in range(4)))
        extra = ["--calibration-data", str(data), "--samples", "4", "--sequence-length", "8"]
    result = run(
        [
            sys.executable,
            str(EXAMPLE / "quantize_w8a8.py"),
            "--model-path",
            str(source),
            "--save-path",
            str(compressed),
            "--offload-dir",
            str(offload),
            "--method",
            method,
            "--device",
            "cpu",
            "--cpu-memory-gib",
            "0.00004",
            *extra,
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((compressed / "calibration_manifest.json").read_text())
    assert manifest["disk_offloaded_modules"]
    tensors = read_tensors(compressed)
    assert len([k for k, t in tensors.items() if k.endswith(".weight") and t.dtype == torch.int8]) == 28
    # CPU/disk placement must not change quantization results. This also catches
    # dtype casts accidentally dropped while preserving Parameter subclasses.
    memory_output = tmp_path / "memory_output"
    result = run(
        [
            sys.executable,
            str(EXAMPLE / "quantize_w8a8.py"),
            "--model-path",
            str(source),
            "--save-path",
            str(memory_output),
            "--offload-dir",
            str(tmp_path / "memory_offload"),
            "--method",
            method,
            "--device",
            "cpu",
            "--cpu-memory-gib",
            "1",
            *extra,
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    memory_tensors = read_tensors(memory_output)
    assert set(tensors) == set(memory_tensors)
    for name, tensor in tensors.items():
        torch.testing.assert_close(tensor, memory_tensors[name], atol=0, rtol=0)
    assert source_bytes == {p.name: p.read_bytes() for p in source.glob("*.safetensors")}


def test_disk_cache_dtype_cast_preserves_parameter(tmp_path):
    from compressed_tensors.offload.cache import DiskCache

    source = tmp_path / "source.safetensors"
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3) * 0.01
    save_file({"weight": original}, source)
    offloaded = torch.nn.Parameter(torch.empty(2, 3, device="meta", dtype=torch.bfloat16), requires_grad=True)
    offloaded.checkpoint_test_marker = "preserve metadata"
    cache = DiskCache("cpu", offload_dir=tmp_path)
    cache.index[offloaded] = {"safetensors_file": str(source), "weight_name": "weight", "dtype": "bfloat16"}
    try:
        with torch.enable_grad():
            restored = cache.onload(offloaded)
        assert isinstance(restored, torch.nn.Parameter)
        assert restored.is_leaf and restored.requires_grad
        assert restored.checkpoint_test_marker == "preserve metadata"
        torch.testing.assert_close(restored, original.bfloat16(), atol=0, rtol=0)
    finally:
        del cache.index[offloaded]


def test_moe_loading_preserves_logits(tmp_path):
    source = tmp_path / "source"
    create_source(source, "moe")
    # A fresh process keeps Transformers' conversion mappings isolated from
    # other fixtures. Unequal hidden/expert dimensions catch projection swaps.
    code = """
import sys
import torch
from transformers import AutoModelForCausalLM
from llmcompressor.utils import load_context
torch.set_grad_enabled(False)
original = AutoModelForCausalLM.from_pretrained(sys.argv[1], local_files_only=True, dtype=torch.float32).eval()
tokens = torch.tensor([[2, 4, 6, 8, 10, 12]])
expected = original(tokens).logits
with load_context():
    unpacked = AutoModelForCausalLM.from_pretrained(sys.argv[1], local_files_only=True,
        dtype=torch.float32, device_map='auto_offload', max_memory={'cpu': 1024**3}).eval()
torch.testing.assert_close(unpacked(tokens).logits, expected, atol=1e-6, rtol=1e-5)
assert len([n for n, p in unpacked.named_parameters() if '.experts.' in n and n.endswith('.weight')]) == 12
"""
    result = run([sys.executable, "-c", code, str(source)])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "case,message",
    [
        ("missing_data", "GPTQ requires"),
        ("bad_data", "expected nonempty text"),
        ("short_data", "found 1"),
        ("nonobject", "expected an object"),
        ("output_exists", "directory must be empty"),
        ("overlap", "non-overlapping"),
        ("quantized_source", "already quantized"),
        ("bad_budget", "finite positive"),
        ("rtn_data", "RTN is data-free"),
    ],
)
def test_rejects_bad_calibration_jobs_before_loading_model(tmp_path, case, message):
    source, output, offload = tmp_path / "source", tmp_path / "output", tmp_path / "offload"
    source.mkdir()
    # Deliberately invalid tensor payload: validation must precede model loading.
    (source / "model.safetensors").write_bytes(b"do not load")
    (source / "config.json").write_text('{}')
    data = tmp_path / "calibration.jsonl"
    data.write_text('{"text": "one sample"}\n')
    extra = ["--calibration-data", str(data), "--samples", "1"]
    if case == "missing_data":
        extra = []
    elif case == "bad_data":
        data.write_text('{"text": ""}\n')
    elif case == "short_data":
        extra[-1] = "2"
    elif case == "nonobject":
        data.write_text('[]\n')
    elif case == "output_exists":
        output.mkdir()
        (output / "keep").write_text("original")
    elif case == "overlap":
        offload = source / "offload"
    elif case == "quantized_source":
        (source / "config.json").write_text('{"quantization_config": {"quant_method": "compressed-tensors"}}')
    elif case == "bad_budget":
        extra += ["--cpu-memory-gib", "nan"]
    elif case == "rtn_data":
        extra += ["--method", "rtn"]
    result = run(
        [
            sys.executable,
            str(EXAMPLE / "quantize_w8a8.py"),
            "--model-path",
            str(source),
            "--save-path",
            str(output),
            "--offload-dir",
            str(offload),
            *extra,
        ]
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert "get_main_device" not in result.stdout + result.stderr
    assert not offload.exists()
    assert (source / "model.safetensors").read_bytes() == b"do not load"
    if case == "output_exists":
        assert (output / "keep").read_text() == "original"
    else:
        assert not output.exists()
