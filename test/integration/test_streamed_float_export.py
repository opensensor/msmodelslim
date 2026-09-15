"""Exercise the real Ascend saver while tracking live passthrough payloads."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from msmodelslim.core.convert.config import ConvertConfig
from msmodelslim.core.convert.protocol import ConvertContext
from msmodelslim.core.convert.types import IRKind, SourceIR, TensorRef
from msmodelslim.core.quant_service.modelslim_convert.impl.save_adapter import SaveProcessorAdapter
from msmodelslim.core.quant_service.modelslim_convert.virtual_module import PassthroughModule, set_submodule_by_path


def setup_export(tmp_path, count=8):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen4_exp","dtype":"float16"}')
    tree = nn.Module()
    expected = {}
    # The parent also owns a tensor: shallow-leaf export must not consume its
    # uninitialized descendants through the saver's recursive memo.
    names = ["ple.weight", *[f"ple.shard_{index}.weight" for index in range(count)], "mtp.experts.down_proj"]
    for index, key in enumerate(names):
        name = key.removesuffix(".weight")
        logical = "weight" if key.endswith(".weight") else "down_proj"
        expected[key] = torch.full((8, 16), index, dtype=torch.float16)
        module = PassthroughModule(
            full_name=name,
            tensor_bindings={
                logical: TensorRef(
                    logical_name=logical, key=key, shard="weights.safetensors", dtype="float16", shape=(8, 16), meta={}
                )
            },
            source_ir=SourceIR(kind=IRKind.FLOAT),
        )
        set_submodule_by_path(tree, name, module)

    class Reader:
        def __init__(self):
            self.loads = []
            self.shard_handle_cache = object()

        def load_tensors(self, mapping, device):
            live = sum(p.numel() * p.element_size() for p in tree.parameters())
            self.loads.append(live)
            return {key: expected[key].clone() for keys in mapping.values() for key in keys}

    reader = Reader()
    context = ConvertContext(
        config=ConvertConfig(
            model_path=str(source),
            save_path=str(output),
            dst_format="ascendv1",
            part_file_size=1,
        )
    )
    context.reader = reader
    return context, tree, reader, expected


@pytest.mark.parametrize("streaming", [True, False])
def test_passthrough_payload_released_and_nested_names_preserved(tmp_path, streaming):
    context, tree, reader, expected = setup_export(tmp_path)
    original_cache = reader.shard_handle_cache
    adapter = SaveProcessorAdapter()
    if streaming:
        adapter.begin(context, tree)
        adapter.finalize()
    else:
        adapter.save(context, tree)
    assert len(reader.loads) == len(expected)
    assert max(reader.loads) == 0, "previous FLOAT modules remained resident during the next load"
    assert not list(tree.parameters())
    assert reader.shard_handle_cache is original_cache
    output = Path(context.save_path)
    saved = {}
    for path in output.glob("*.safetensors"):
        saved.update(load_file(path))
    assert saved.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
    description = json.loads((output / "quant_model_description.json").read_text())
    assert all(description[key] == "FLOAT" for key in expected)


def test_failed_float_write_restores_reader_and_releases_payload(tmp_path, monkeypatch):
    context, tree, reader, _ = setup_export(tmp_path)
    original_cache = reader.shard_handle_cache
    adapter = SaveProcessorAdapter()
    adapter.begin(context, tree)

    def fail(_request):
        raise OSError("simulated disk failure")

    saver = adapter._session.bundle.saver
    monkeypatch.setattr(saver, "postprocess", fail)
    with pytest.raises(OSError, match="disk failure"):
        adapter.finalize()
    assert reader.shard_handle_cache is original_cache
    assert not list(tree.parameters())
    assert not saver.processed_modules
    assert not (Path(context.save_path) / "quant_model_description.json").exists()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.finalize()


def test_failed_partial_load_releases_payload_and_removes_temporary_cache(tmp_path, monkeypatch):
    context, tree, reader, _ = setup_export(tmp_path)
    del reader.shard_handle_cache
    adapter = SaveProcessorAdapter()
    adapter.begin(context, tree)
    module = tree.ple

    def partial_load(*_args, **_kwargs):
        module.register_parameter("weight", nn.Parameter(torch.ones(8, 16), requires_grad=False))
        raise OSError("simulated incomplete shard")

    monkeypatch.setattr(module, "lazy_init", partial_load)
    with pytest.raises(OSError, match="incomplete shard"):
        adapter.finalize()
    assert not list(tree.parameters())
    assert not hasattr(reader, "shard_handle_cache")
    assert not (Path(context.save_path) / "quant_model_description.json").exists()
