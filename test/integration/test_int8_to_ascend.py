"""Real CPU checkpoint conversion: no NPU/security/saver mocks."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from msmodelslim.core.quant_service.modelslim_convert.config_mapper import spec_to_convert_config
from msmodelslim.core.quant_service.modelslim_convert.factory import create_convert_application


def make_source(path, *, sharded=False):
    path.mkdir(mode=0o700)
    scheme = {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True},
        "output_activations": None,
    }
    config = {
        "model_type": "qwen3",
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "config_groups": {"group_0": scheme},
            "ignore": ["lm_head"],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text('{"chat_template": "test template"}')
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[127, -128, 0], [4, 5, -6]], dtype=torch.int8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.tensor([[0.01], [0.03]], dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight_zero_point": torch.zeros(2, 1, dtype=torch.int8),
        "model.layers.0.self_attn.q_proj.bias": torch.tensor([0.5, -0.25], dtype=torch.float16),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.tensor([[1, -2, 3], [-5, 8, 2]], dtype=torch.int8),
        "model.layers.0.mlp.experts.0.up_proj.weight_scale": torch.tensor([[0.125], [0.25]]),
        "model.norm.weight": torch.ones(3, dtype=torch.float16),
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.float16).reshape(4, 3),
        "lm_head.weight": torch.ones(4, 3, dtype=torch.float16),
    }
    if sharded:
        # Put a scale in another shard to exercise dependency-based loading.
        items = list(tensors.items())
        shards = {
            "model-00001-of-00002.safetensors": dict(items[::2]),
            "model-00002-of-00002.safetensors": dict(items[1::2]),
        }
        for name, values in shards.items():
            save_file(values, path / name)
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {key: shard for shard, values in shards.items() for key in values},
                }
            )
        )
    else:
        save_file(tensors, path / "model.safetensors")
    return config, tensors


def convert(source, destination, workers=1):
    config = spec_to_convert_config(
        {
            "linears": [{"match": ["*"], "target": "W8A8_DYNAMIC", "route": "auto"}],
            "save": [{"type": "ascend_v1", "part_file_size": 1}],
            "parallel": {"workers": workers, "worker_device": "cpu", "max_group_size": 1},
        },
        str(source),
        str(destination),
    )
    create_convert_application().run(config)


@pytest.mark.parametrize("sharded,workers", [(False, 1), (True, 1), (True, 2)])
def test_import_preserves_weights_scales_bias_and_float_layers(tmp_path, sharded, workers):
    source, destination = tmp_path / "source", tmp_path / "output"
    _, original = make_source(source, sharded=sharded)
    convert(source, destination, workers)
    exported = {}
    for file in destination.glob("*.safetensors"):
        exported.update(load_file(file))
    description = json.loads((destination / "quant_model_description.json").read_text())
    config = json.loads((destination / "config.json").read_text())
    assert "quantization_config" not in config
    assert (destination / "tokenizer_config.json").read_bytes() == (source / "tokenizer_config.json").read_bytes()
    assert description["model_quant_type"] == "W8A8_DYNAMIC"
    expected = set(original) - {key for key in original if key.endswith(".weight_zero_point")}
    for key, tensor in original.items():
        if key.endswith(".weight_zero_point"):
            assert key not in exported
            continue
        assert torch.equal(tensor, exported[key])
        if key.endswith(".weight") and tensor.dtype == torch.int8:
            prefix = key.removesuffix(".weight")
            scale = exported[prefix + ".weight_scale"]
            assert scale.shape == (tensor.shape[0], 1)
            assert scale.dtype == torch.float32
            assert description[key] == "W8A8_DYNAMIC"
            expected.add(prefix + ".weight_offset")
            assert torch.count_nonzero(exported[prefix + ".weight_offset"]) == 0
            # Independent dequantized linear reference, including nonzero bias.
            x = torch.tensor([[1.0, -2.0, 0.5]])
            before = torch.nn.functional.linear(
                x,
                tensor.float() * original[prefix + ".weight_scale"],
                original.get(prefix + ".bias", torch.zeros(tensor.shape[0])).float(),
            )
            after = torch.nn.functional.linear(
                x, exported[key].float() * scale, exported.get(prefix + ".bias", torch.zeros(tensor.shape[0])).float()
            )
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert set(exported) == expected
    assert description["model.norm.weight"] == "FLOAT"


@pytest.mark.parametrize(
    "change,message",
    [
        ("asymmetric", "symmetric static"),
        ("grouped", "per-channel"),
        ("static_activation", "per-token"),
        ("fp8", "static INT8"),
        ("fake_quant", "compressed int-quantized"),
        ("kv_cache", "KV-cache"),
        ("runtime_transform", "runtime transforms"),
    ],
)
def test_rejects_unsupported_config_before_writing(tmp_path, change, message):
    source, destination = tmp_path / "source", tmp_path / "output"
    config, _ = make_source(source)
    quant = config["quantization_config"]
    group = quant["config_groups"]["group_0"]
    if change == "asymmetric":
        group["weights"]["symmetric"] = False
    elif change == "grouped":
        group["weights"]["strategy"] = "group"
    elif change == "static_activation":
        group["input_activations"]["dynamic"] = False
    elif change == "fp8":
        group["weights"]["type"] = "float"
    elif change == "fake_quant":
        quant["quantization_status"] = "frozen"
    elif change == "kv_cache":
        quant["kv_cache_scheme"] = {"num_bits": 8}
    elif change == "runtime_transform":
        quant["transform_config"] = {"rotation": {}}
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        convert(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "change,message",
    [
        ("missing_scale", "missing or invalid"),
        ("bad_scale_shape", "missing or invalid"),
        ("fused_experts", "2D linear"),
        ("nan_scale", "finite and positive"),
        ("negative_scale", "finite and positive"),
        ("zero_point", "symmetric zero-point-free"),
    ],
)
def test_rejects_invalid_tensors(tmp_path, change, message):
    source, destination = tmp_path / "source", tmp_path / "output"
    _, tensors = make_source(source)
    prefix = "model.layers.0.self_attn.q_proj"
    if change == "missing_scale":
        del tensors[prefix + ".weight_scale"]
    elif change == "bad_scale_shape":
        tensors[prefix + ".weight_scale"] = torch.ones(1)
    elif change == "fused_experts":
        tensors[prefix + ".weight"] = tensors[prefix + ".weight"].unsqueeze(0)
    elif change == "nan_scale":
        tensors[prefix + ".weight_scale"][0] = float("nan")
    elif change == "negative_scale":
        tensors[prefix + ".weight_scale"][0] = -1
    elif change == "zero_point":
        tensors[prefix + ".weight_zero_point"][0] = 1
    save_file(tensors, source / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match=message):
        convert(source, destination)


@pytest.mark.parametrize("destination_kind", ["same", "child", "parent", "nonempty"])
def test_rejects_overlapping_or_nonempty_output_without_changing_weights(tmp_path, destination_kind):
    source = tmp_path / "source"
    make_source(source)
    original = (source / "model.safetensors").read_bytes()
    destination = {"same": source, "child": source / "out", "parent": tmp_path, "nonempty": tmp_path / "out"}[
        destination_kind
    ]
    if destination_kind == "nonempty":
        destination.mkdir()
        (destination / "previous.safetensors").write_bytes(b"keep previous output")
    with pytest.raises(ValueError, match="non-overlapping|empty destination"):
        convert(source, destination)
    assert (source / "model.safetensors").read_bytes() == original
    if destination_kind == "nonempty":
        assert (destination / "previous.safetensors").read_bytes() == b"keep previous output"


@pytest.mark.parametrize(
    "change,message",
    [
        ("orphan", "Orphan"),
        ("runtime_index", "Unsupported quantization tensor"),
        ("unnamed_fused", "Unsupported INT8 tensor layout"),
        ("float8", "Unsupported source weight dtype"),
        ("scale_dtype", "floating-point per-output-channel"),
    ],
)
def test_rejects_unhandled_quantization_tensors(tmp_path, change, message):
    source, destination = tmp_path / "source", tmp_path / "output"
    _, tensors = make_source(source)
    prefix = "model.layers.0.self_attn.q_proj"
    if change == "orphan":
        tensors["other.weight_scale"] = torch.ones(2, 1)
    elif change == "runtime_index":
        tensors[prefix + ".weight_g_idx"] = torch.arange(3)
    elif change == "unnamed_fused":
        tensors["model.layers.0.mlp.experts.gate_up_proj"] = torch.ones(2, 2, 3, dtype=torch.int8)
    elif change == "float8":
        tensors[prefix + ".weight"] = tensors[prefix + ".weight"].to(torch.float8_e4m3fn)
    elif change == "scale_dtype":
        tensors[prefix + ".weight_scale"] = torch.ones(2, 1, dtype=torch.int32)
    save_file(tensors, source / "model.safetensors")
    with pytest.raises((ValueError, RuntimeError), match=message):
        convert(source, destination)


def test_nested_metadata_and_vector_scales(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "output"
    config, tensors = make_source(source)
    config["text_config"] = {"quantization_config": config.pop("quantization_config")}
    (source / "config.json").write_text(json.dumps(config))
    prefix = "model.layers.0.self_attn.q_proj"
    tensors[prefix + ".weight_scale"] = tensors[prefix + ".weight_scale"].flatten().half()
    del tensors[prefix + ".weight_zero_point"]
    save_file(tensors, source / "model.safetensors")
    convert(source, destination)
    exported = {}
    for file in destination.glob("*.safetensors"):
        exported.update(load_file(file))
    torch.testing.assert_close(
        exported[prefix + ".weight_scale"], tensors[prefix + ".weight_scale"].float().unsqueeze(-1), rtol=0, atol=0
    )
    assert "quantization_config" not in json.loads((destination / "config.json").read_text())["text_config"]
