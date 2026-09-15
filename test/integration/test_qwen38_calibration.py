"""Real Qwen4Exp primitives, fused checkpoint IO, offload, GPTQ and bridge tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example/convert/llmcompressor_to_ascend"
sys.path.insert(0, str(EXAMPLE))

from qwen38.checkpoint import Checkpoint, MappedEmbedding
from qwen38.configuration import Qwen4ExpTextConfig
from qwen38.export import save_checkpoint
from qwen38.model import Qwen38ForCalibration, load_model
from qwen38.reference import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule


def tiny_config():
    return Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        layer_types=['linear_attention', 'full_attention'],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=2,
        pad_token_id=0,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        rope_parameters={
            'rope_type': 'default',
            'rope_theta': 10000.0,
            'partial_rotary_factor': 0.5,
            'mrope_section': [1, 1, 0],
        },
        output_gate_type='sigmoid',
    )


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = tiny_config()
    model = Qwen38ForCalibration(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.08)
    weights = {}
    for name, value in model.state_dict().items():
        if name.endswith("ple.ple_embedding.ngram_embedding.weight"):
            for index, part in enumerate(value.chunk(config.split_ngram_parts)):
                weights[name.removesuffix(".weight") + f".shard_{index}.weight"] = part.clone()
        else:
            weights[name] = value.clone()
    # Explicitly excluded components must survive the text-only export.
    weights["model.visual.test.weight"] = torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16)
    weights["mtp.test.weight"] = torch.tensor([[3.5]], dtype=torch.bfloat16)
    weights["model.visual.counter"] = torch.tensor([7], dtype=torch.int64)
    directory = tmp_path / "source"
    directory.mkdir()
    save_file(weights, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(weights, "model.safetensors"),
                "metadata": {"total_size": sum(x.numel() * x.element_size() for x in weights.values())},
            }
        )
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": config.to_dict(),
            }
        )
    )
    return directory, model, weights


@pytest.mark.parametrize("expert", [0, 2])
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_fused_slice_exact_order(source, expert, projection):
    directory, _, weights = source
    prefix = "model.language_model.layers.0.mlp.experts"
    if projection == "down_proj":
        expected = weights[prefix + ".down_proj"][expert]
    else:
        gate, up = weights[prefix + ".gate_up_proj"][expert].chunk(2)
        expected = gate if projection == "gate_proj" else up
    actual = Checkpoint(directory).read(f"{prefix}.{expert}.{projection}.weight")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


@pytest.mark.parametrize("budget", [0, 2**30])
def test_full_text_parity_with_disk_and_cpu(source, tmp_path, budget):
    directory, original, _ = source
    loaded = load_model(directory, tmp_path / "offload", dtype=torch.float32, cpu_budget_bytes=budget)
    ids = torch.tensor([[3, 4, 2, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        expected = original(ids).logits
        actual = loaded(ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert loaded.cpu_weight_bytes <= budget
    assert not any(p.is_meta for p in loaded.parameters())
    assert not any("ngram_embedding" in n for n, _ in loaded.named_parameters())


def test_ple_row_boundaries_and_repetition(source):
    directory, original, _ = source
    table = original.model.language_model.layers[0].ple.ple_embedding.ngram_embedding.weight
    mapped = MappedEmbedding(
        Checkpoint(directory),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding",
        *table.shape,
        torch.float32,
    )
    edge = mapped.rows_per_shard
    ids = torch.tensor([[0, edge - 1, edge, table.shape[0] - 1, edge, 0]])
    torch.testing.assert_close(mapped(ids), table[ids], rtol=0, atol=0)
    assert mapped(torch.empty((1, 0), dtype=torch.long)).shape == (1, 0, table.shape[1])
    with pytest.raises(ValueError, match="bounds"):
        mapped(torch.tensor([table.shape[0]]))


def test_causality_across_qsa_and_gdn(source):
    _, model, _ = source
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([11, 12, 13, 14])
    with torch.no_grad():
        first, second = model(ids).logits, model(changed).logits
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=1e-5, atol=1e-6)


def test_chunk_gdn_matches_recurrent_reference():
    torch.manual_seed(12)
    q, k, v = [torch.randn(1, 17, 2, 8) for _ in range(3)]
    g = -torch.rand(1, 17, 2)
    beta = torch.rand(1, 17, 2)
    a, sa = torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    b, sb = torch_recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(sa, sb, rtol=2e-5, atol=1e-6)


def test_reject_bad_expert_and_quantized_source(source):
    directory, _, weights = source
    checkpoint = Checkpoint(directory)
    with pytest.raises(ValueError, match="index"):
        checkpoint.read("model.language_model.layers.0.mlp.experts.3.gate_proj.weight")
    weights["model.language_model.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(3, 8, 16)
    save_file(weights, str(directory / "model.safetensors"))
    with pytest.raises(ValueError, match="shape"):
        checkpoint.read("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
    cfg = json.loads((directory / "config.json").read_text())
    cfg["quantization_config"] = {"format": "fp8"}
    (directory / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="floating"):
        Checkpoint(directory)


def test_incomplete_checkpoint_fails_before_offload(source, tmp_path):
    directory, _, _ = source
    (directory / "model.safetensors").rename(directory / "incomplete")
    with pytest.raises(FileNotFoundError, match="download incomplete"):
        load_model(directory, tmp_path / "offload")
    assert not (tmp_path / "offload").exists()


def test_reject_unconsumed_text_tensor(source, tmp_path):
    directory, _, weights = source
    name = "model.language_model.unexpected.weight"
    weights[name] = torch.ones(1)
    save_file(weights, str(directory / "model.safetensors"))
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"][name] = "model.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unconsumed"):
        load_model(directory, tmp_path / "offload")


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_calibration_cli(source, tmp_path, method):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    directory, _, _ = source
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 3, "b": 4, "c": 5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.save_pretrained(directory)
    output = tmp_path / "compressed"
    arguments = [
        sys.executable,
        str(EXAMPLE / "quantize_qwen38.py"),
        "--model-path",
        str(directory),
        "--save-path",
        str(output),
        "--offload-dir",
        str(tmp_path / "offload"),
        "--method",
        method,
        "--device",
        "cpu",
        "--dtype",
        "float16",
        "--target",
        "atlas-300i-duo",
        "--cpu-memory-gib",
        "0.000001",
    ]
    if method == "gptq":
        data = tmp_path / "calibration.jsonl"
        data.write_text('{"text": "a b c a b c"}\n')
        arguments.extend(["--calibration-data", str(data), "--samples", "1", "--sequence-length", "8"])
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "calibration_manifest.json").read_text())
    assert report["quantized_projections"] == 18
    assert report["method"] == method
    assert report["cuda_peak_allocated_bytes"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sequential_gptq_export_and_native_bridge(source, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    directory, _, weights = source
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = load_model(directory, tmp_path / "offload", dtype=dtype, device=device, cpu_budget_bytes=0)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=GPTQModifier(scheme="W8A8", targets=[r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"]),
        dataset=Dataset.from_dict(
            {"input_ids": [[3, 4, 2, 5, 6, 7, 8, 9], [5, 6, 7, 8, 3, 4, 5, 6]], "attention_mask": [[1] * 8] * 2}
        ),
        num_calibration_samples=2,
        max_seq_length=8,
        pipeline="sequential",
        sequential_targets=["Qwen4ExpTextDecoderLayer"],
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=True,
        output_dir=None,
        clear_sparse_session=True,
    )
    compressed = tmp_path / "compressed"
    report = save_checkpoint(model, compressed, shard_bytes=20000, float_dtype=dtype)
    assert report["quantized_projections"] == 18
    saved = {}
    for path in compressed.glob("*.safetensors"):
        saved.update(load_file(path))
    assert sum(t.dtype == torch.int8 for t in saved.values()) == 18
    for name, weight in weights.items():
        if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            continue
        expected = weight.to(dtype) if weight.dtype == torch.bfloat16 else weight
        torch.testing.assert_close(saved[name], expected, rtol=0, atol=0)
    native = tmp_path / "ascend"
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "msmodelslim"),
            "quant",
            "--model_path",
            str(compressed),
            "--save_path",
            str(native),
            "--config",
            str(EXAMPLE / "w8a8_dynamic.yaml"),
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
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "verify_w8a8.py"),
            "--source",
            str(compressed),
            "--export",
            str(native),
            "--check-values",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with pytest.raises(ValueError, match="empty destination"):
        save_checkpoint(model, compressed)
