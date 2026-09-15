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
from qwen38.quality import check_quantization, compare_reports, evaluate, require_held_out
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


def test_exact_prepared_ple_cache(source, tmp_path):
    from qwen38.model import build_config
    from qwen38.ple_cache import CachedEmbedding, prepare_ple_cache

    directory, original, _ = source
    rows = [{"input_ids": [3, 4, 2, 5, 6, 7, 8, 9], "attention_mask": [1] * 8}]
    checkpoint = Checkpoint(directory)
    caches = prepare_ple_cache(
        checkpoint, build_config(checkpoint.text_config), [(rows, 12)], tmp_path / "ple", torch.float32, block_rows=7
    )
    loaded = load_model(directory, tmp_path / "offload", dtype=torch.float32, cpu_budget_bytes=0, ple_cache=caches)
    ids = torch.tensor([rows[0]["input_ids"] + [0] * 4])
    mask = torch.tensor([[1] * 8 + [0] * 4])
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(ids, attention_mask=mask).logits, original(ids, attention_mask=mask).logits, rtol=1e-5, atol=1e-6
        )
    cached = next(module for module in loaded.modules() if isinstance(module, CachedEmbedding))
    with pytest.raises(ValueError, match="cache miss"):
        cached(torch.tensor([cached.weight.shape[0]]))
    assert (
        prepare_ple_cache(
            checkpoint, build_config(checkpoint.text_config), [(rows, 12)], tmp_path / "ple", torch.float32
        )
        == caches
    )
    with pytest.raises(ValueError, match="identity/checksum"):
        prepare_ple_cache(
            checkpoint, build_config(checkpoint.text_config), [(rows, 12)], tmp_path / "ple", torch.float16
        )


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


@pytest.mark.parametrize("method", ["rtn", "gptq", "resumable"])
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
        "gptq" if method == "resumable" else method,
        "--device",
        "cpu",
        "--dtype",
        "float16",
        "--target",
        "atlas-300i-duo",
        "--cpu-memory-gib",
        "0.000001",
    ]
    if method != "rtn":
        data = tmp_path / "calibration.jsonl"
        data.write_text('{"text": "a b c a b c"}\n')
        arguments.extend(["--calibration-data", str(data), "--samples", "1", "--sequence-length", "8"])
    if method == "resumable":
        arguments.extend(["--checkpoint-dir", str(tmp_path / "stages"), "--ple-cache-dir", str(tmp_path / "ple")])
    quality = tmp_path / "quality.jsonl"
    quality.write_text('{"text": "c b a c b a"}\n')
    reports = tmp_path / "quality"
    arguments.extend(["--quality-data", str(quality), "--quality-report-dir", str(reports), "--quality-samples", "1"])
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "calibration_manifest.json").read_text())
    assert report["quantized_projections"] == 18
    assert report["method"] == ("gptq" if method == "resumable" else method)
    assert report["cuda_peak_allocated_bytes"] == 0
    comparison = json.loads((reports / "comparison.json").read_text())
    assert comparison == report["quality"]
    assert comparison["before"]["quantized_projections"] == 0
    assert comparison["after"]["quantized_projections"] == 18
    assert comparison["before"]["predicted_tokens"] == comparison["after"]["predicted_tokens"] == 5
    if method == "resumable":
        arguments.extend(["--resume"])
        for option, directory in (
            ("--save-path", "resumed"),
            ("--offload-dir", "new-offload"),
            ("--quality-report-dir", "new-quality"),
        ):
            arguments[arguments.index(option) + 1] = str(tmp_path / directory)
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Reusing validated floating baseline" in result.stdout
        assert "Calibrating:" not in result.stderr
        expected = {key: value for path in output.glob("*.safetensors") for key, value in load_file(path).items()}
        actual = {
            key: value
            for path in (tmp_path / "resumed").glob("*.safetensors")
            for key, value in load_file(path).items()
        }
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


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
    held_out = [{"input_ids": [9, 8, 7, 6, 5, 4], "attention_mask": [1] * 6}]
    before = evaluate(model, held_out, device=device, chunk_size=2)
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
    after = evaluate(model, held_out, device=device, chunk_size=2, quantized=True)
    assert compare_reports(before, after)["after"]["quantized_projections"] == 18
    expert = model.model.language_model.layers[0].mlp.experts[0].gate_proj
    expert.quantization_enabled = False
    with pytest.raises(ValueError, match="enabled and frozen"):
        check_quantization(model, True)
    expert.quantization_enabled = True
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


@pytest.mark.parametrize("interruption", ["committed", "uncommitted", "final"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resumed_gptq_matches_uninterrupted(source, tmp_path, monkeypatch, interruption, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.core import create_session
    from llmcompressor.modifiers.gptq import GPTQModifier
    from qwen38 import resume
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    directory, _, _ = source
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    dataset = Dataset.from_dict(
        {"input_ids": [[3, 4, 2, 5, 6, 7, 8, 9], [5, 6, 7, 8, 3, 4, 5, 6]], "attention_mask": [[1] * 8] * 2}
    )

    def run(name, store=None):
        model = load_model(
            directory,
            tmp_path / name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
            cpu_budget_bytes=0,
        )
        if store:
            model.stage_store = store
        with create_session():
            oneshot(
                model=model,
                tokenizer=tokenizer,
                recipe=GPTQModifier(
                    scheme="W8A8", targets=[r"re:.*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)"]
                ),
                dataset=dataset,
                num_calibration_samples=2,
                max_seq_length=8,
                pipeline="qwen38_resumable" if store else "sequential",
                sequential_targets=["Qwen4ExpTextDecoderLayer"],
                sequential_offload_device="cpu",
                moe_calibrate_all_experts=True,
                shuffle_calibration_samples=False,
                output_dir=None,
                clear_sparse_session=True,
            )
        check_quantization(model, True)
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    expected = run("uninterrupted")
    store = resume.StageStore(tmp_path / "stages", {"test": 1})
    save = resume.StageStore.save
    save_file_original = resume.save_file

    def interrupted_save_file(tensors, filename, **kwargs):
        save_file_original(tensors, filename, **kwargs)
        if "stage-0001.tmp" in str(filename):
            raise RuntimeError("simulated interrupted write")

    def interrupted_save(self, model, graph, cache, index):
        save(self, model, graph, cache, index)
        if index == (2 if interruption == "final" else 1):
            raise RuntimeError("simulated interrupted process")

    if interruption == "uncommitted":
        monkeypatch.setattr(resume, "save_file", interrupted_save_file)
    else:
        monkeypatch.setattr(resume.StageStore, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        run("interrupted", store)
    count = len(store.progress["stages"])
    assert count == {"committed": 2, "uncommitted": 1, "final": 3}[interruption]
    store.close()
    monkeypatch.setattr(resume.StageStore, "save", save)
    monkeypatch.setattr(resume, "save_file", save_file_original)
    restored = resume.StageStore(tmp_path / "stages", {"test": 1}, resume=True)
    actual = run("resumed", restored)
    assert len(restored.progress["stages"]) == 3
    restored.close()
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def test_resume_fingerprint_and_corruption_rejected(tmp_path):
    from qwen38.resume import StageStore

    store = StageStore(tmp_path / "stages", {"source": "original"})
    with pytest.raises(BlockingIOError):
        StageStore(tmp_path / "stages", {"source": "original"}, resume=True)
    store.close()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        StageStore(tmp_path / "stages", {"source": "changed"}, resume=True)
    store = StageStore(tmp_path / "stages", {"source": "original"}, resume=True)
    path = store.path / "stage-0000"
    path.mkdir()
    (path / "weights.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.checked_file({"directory": path.name, "files": {"weights.safetensors": "invalid"}}, "weights.safetensors")
    store.close()


def test_baseline_reuse_validation(source, tmp_path):
    from qwen38.quality import validate_baseline

    directory, _, _ = source
    model = load_model(directory, tmp_path / "offload", dtype=torch.float32, cpu_budget_bytes=0)
    rows = [{"input_ids": [3, 4, 5, 6], "attention_mask": [1] * 4}]
    before = evaluate(model, rows, chunk_size=2)
    validate_baseline(before, model, rows, "cpu", 2)
    with pytest.raises(ValueError, match="baseline mismatch: logit_chunk_size"):
        validate_baseline(before, model, rows, "cpu", 3)
    with pytest.raises(ValueError, match="baseline mismatch: token_ids_sha256"):
        validate_baseline(before, model, [{"input_ids": [3, 5, 4, 6]}], "cpu", 2)
    with pytest.raises(ValueError, match="aggregate mismatch"):
        validate_baseline({**before, "mean_nll": before["mean_nll"] + 1}, model, rows, "cpu", 2)


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
def test_quality_matches_full_logits_with_disk_offload(source, tmp_path, chunk_size):
    directory, original, _ = source
    loaded = load_model(directory, tmp_path / "offload", dtype=torch.float32, cpu_budget_bytes=0)
    rows = [
        {"input_ids": [3, 4, 5, 6, 7, 8], "attention_mask": [1] * 6},
        {"input_ids": [8, 7, 6, 5], "attention_mask": [1] * 4},
    ]
    expected = 0
    with torch.no_grad():
        for row in rows:
            inputs = torch.tensor([row["input_ids"]])
            expected += torch.nn.functional.cross_entropy(
                original(inputs).logits[0, :-1].float(), inputs[0, 1:], reduction="sum"
            ).item()
    sizes = []
    hook = loaded.lm_head.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    loaded.train()
    report = evaluate(loaded, rows, chunk_size=chunk_size)
    hook.remove()
    assert loaded.training
    assert max(sizes) <= chunk_size
    assert report["predicted_tokens"] == 8
    assert report["mean_nll"] == pytest.approx(expected / 8, rel=1e-6)
    mismatched = {**report, "quantized_projections": 18, "token_ids_sha256": "different"}
    with pytest.raises(ValueError, match="token_ids_sha256"):
        compare_reports(report, mismatched)
    with pytest.raises(ValueError, match="expected 18"):
        evaluate(loaded, rows, quantized=True)
    with pytest.raises(ValueError, match="unpadded"):
        evaluate(loaded, [{"input_ids": [3, 4], "attention_mask": [1, 0]}])
    assert loaded.training


def test_quality_rejects_nonfinite_loss(source):
    _, model, _ = source
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite evaluation loss"):
        evaluate(model, [{"input_ids": [3, 4, 5], "attention_mask": [1] * 3}])


def test_quality_rejects_token_leakage():
    calibration = [{"input_ids": [1, 2, 3]}]
    for ids in ([1, 2, 3], [1, 2], [1, 2, 3, 4]):
        with pytest.raises(ValueError, match="duplicate"):
            require_held_out(calibration, [{"input_ids": ids}])
    require_held_out(calibration, [{"input_ids": [1, 2, 4]}])
    with pytest.raises(ValueError, match="duplicate"):
        require_held_out([], calibration * 2)


def test_quality_cli_validates_before_loading(source, tmp_path):
    from quantize_qwen38 import parse_args

    directory, _, _ = source
    data = tmp_path / "test.jsonl"
    data.write_text('{"text": "held out text"}\n')
    arguments = [
        "--model-path",
        str(directory),
        "--save-path",
        str(tmp_path / "output"),
        "--offload-dir",
        str(tmp_path / "offload"),
        "--method",
        "rtn",
        "--quality-data",
        str(data),
        "--quality-samples",
        "1",
    ]
    with pytest.raises(SystemExit):
        parse_args(arguments)
    with pytest.raises(SystemExit):
        parse_args([*arguments, "--quality-report-dir", str(directory / "quality")])
    with pytest.raises(SystemExit):
        parse_args([*arguments, "--quality-report-dir", str(tmp_path / "quality"), "--quality-samples", "2"])
    args = parse_args([*arguments, "--quality-report-dir", str(tmp_path / "quality")])
    assert args.quality_records == [{"text": "held out text"}]
    assert not args.offload_dir.exists()


def test_moe_probe_disk_parity_and_gptq_observation(source, tmp_path):
    from accelerate import init_empty_weights
    from datasets import Dataset
    from llmcompressor import oneshot
    from probe_qwen38_moe import CapturedMoe, MeasuredGPTQ, load_part
    from qwen38.model import LinearExperts, build_config
    from qwen38.reference import Qwen4ExpTextSparseMoeBlock
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    directory, original, _ = source
    checkpoint = Checkpoint(directory)
    config = build_config(checkpoint.text_config)
    with init_empty_weights():
        moe = Qwen4ExpTextSparseMoeBlock(config)
        moe.experts = LinearExperts(config)
    offload = tmp_path / "offload"
    offload.mkdir()
    moe = load_part(checkpoint, moe, "model.language_model.layers.0.mlp", device="cpu", offload_dir=offload)
    features = torch.randn(12, config.hidden_size, dtype=torch.float16)
    model = CapturedMoe(features, moe).eval()
    with torch.no_grad():
        expected = original.model.language_model.layers[0].mlp.half()(features[None])
        torch.testing.assert_close(model(torch.arange(12)[None]), expected, atol=1e-3, rtol=1e-3)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    recipe = MeasuredGPTQ(scheme="W8A8", targets=[r"re:.*experts\.\d+\.(gate_proj|up_proj|down_proj)"])
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        dataset=Dataset.from_dict({"input_ids": [list(range(8))], "attention_mask": [[1] * 8]}),
        num_calibration_samples=1,
        max_seq_length=8,
        pipeline="sequential",
        sequential_targets=["Qwen4ExpTextSparseMoeBlock"],
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=True,
        output_dir=None,
        clear_sparse_session=True,
    )
    assert recipe._observations == [{"matrices": 9, "hessian_bytes": 3 * (2 * 16**2 + 8**2) * 4, "minimum_samples": 1}]
    with torch.no_grad():
        assert torch.isfinite(model(torch.arange(8, 12)[None])).all()


def test_moe_probe_rejects_first_layer_ple(source):
    from probe_qwen38_moe import capture_inputs
    from qwen38.model import build_config

    directory, _, _ = source
    checkpoint = Checkpoint(directory)
    with pytest.raises(ValueError, match="first-layer GDN without PLE"):
        capture_inputs(checkpoint, build_config(checkpoint.text_config), [], "cpu")


def test_moe_probe_rejects_cpu_with_visible_accelerator(monkeypatch):
    from types import SimpleNamespace

    from probe_qwen38_moe import probe

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    with pytest.raises(ValueError, match="accelerator isolation"):
        probe(SimpleNamespace(device="cpu"))
