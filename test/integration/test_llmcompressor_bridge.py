"""Optional CPU smoke test using the actual LLM-Compressor serializer and CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

llmcompressor = pytest.importorskip("llmcompressor")
from datasets import Dataset
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_real_qwen3_quantization_and_cli_export(tmp_path, method):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    source, destination = tmp_path / "compressed", tmp_path / "ascend"
    model = (
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                attention_bias=True,
                tie_word_embeddings=False,
            )
        )
        .cpu()
        .eval()
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    modifier = QuantizationModifier if method == "rtn" else GPTQModifier
    calibration = {}
    if method == "gptq":
        model.to(torch.bfloat16)
        calibration = {
            "dataset": Dataset.from_dict(
                {
                    "input_ids": torch.randint(2, 32, (4, 16)).tolist(),
                    "attention_mask": [[1] * 16 for _ in range(4)],
                }
            ),
            "num_calibration_samples": 4,
            "max_seq_length": 16,
            "pipeline": "basic",
        }
    llmcompressor.oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=modifier(scheme="W8A8", targets=["Linear"], ignore=["lm_head"]),
        output_dir=str(source),
        save_compressed=True,
        clear_sparse_session=True,
        **calibration,
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "msmodelslim"),
            "quant",
            "--model_path",
            str(source),
            "--save_path",
            str(destination),
            "--config",
            str(root / "example/convert/llmcompressor_to_ascend/w8a8_dynamic.yaml"),
            "--device",
            "cpu",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    original, exported = {}, {}
    for file in source.glob("*.safetensors"):
        original.update(load_file(file))
    for file in destination.glob("*.safetensors"):
        exported.update(load_file(file))
    description = json.loads((destination / "quant_model_description.json").read_text())
    quantized = 0
    for key, tensor in original.items():
        if key.endswith(".weight_zero_point"):
            assert torch.count_nonzero(tensor) == 0
            assert key not in exported
            continue
        expected = tensor.float() if key.endswith((".weight_scale", ".bias")) else tensor
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)
        if tensor.dtype == torch.int8 and key.endswith(".weight"):
            quantized += 1
            assert description[key] == "W8A8_DYNAMIC"
            prefix = key.removesuffix(".weight")
            assert exported[prefix + ".weight_scale"].shape == (tensor.shape[0], 1)
            assert torch.count_nonzero(exported[prefix + ".weight_offset"]) == 0
    assert quantized == 7
    assert description["lm_head.weight"] == "FLOAT"
    assert "quantization_config" not in json.loads((destination / "config.json").read_text())
    # The CLI historically clears old shards; this import must reject a rerun
    # before deleting any previous checkpoint payload.
    previous_files = {p.name: p.read_bytes() for p in destination.glob("*.safetensors")}
    repeated = subprocess.run(
        result.args,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert repeated.returncode != 0
    assert "empty destination" in repeated.stdout + repeated.stderr
    assert previous_files == {p.name: p.read_bytes() for p in destination.glob("*.safetensors")}
