"""Measure held-out all-token perplexity with floating-point or CT reference execution.

This evaluates Transformers/compressed-tensors QDQ semantics, not Ascend kernels.
The complete decompressed model must fit on the selected device.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from quantize_w8a8 import positive_int, read_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True, help="Held-out JSONL; not calibration records")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=positive_int, default=32)
    parser.add_argument("--sequence-length", type=positive_int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    records = read_calibration(args)
    config = json.loads((args.model_path / "config.json").read_text())
    if (args.model_path / "quant_model_description.json").exists():
        parser.error("evaluate the compressed-tensors intermediate; native Ascend inference requires the NPU runtime")
    quant_config = config.get("quantization_config")
    if quant_config and quant_config.get("quant_method") != "compressed-tensors":
        parser.error("only floating-point or compressed-tensors checkpoints are supported")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Device isolation must precede accelerator-aware imports.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

    started = time.monotonic()
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    rows = []
    for record in records:
        text = record.get("text")
        chat = not isinstance(text, str) or not text.strip()
        if chat:
            text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
        row = tokenizer(text, add_special_tokens=not chat, truncation=True, max_length=args.sequence_length)[
            "input_ids"
        ]
        if len(row) < 2:
            raise ValueError("each evaluation sample must contain at least two tokens")
        rows.append(row)
    kwargs = {"quantization_config": CompressedTensorsConfig(run_compressed=False)} if quant_config else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float16,
        device_map=args.device,
        **kwargs,
    ).eval()
    quantized = [m for m in model.modules() if getattr(m, "quantization_scheme", None) is not None]
    if quant_config:
        if not quantized or any(not getattr(m, "quantization_enabled", True) for m in quantized):
            raise ValueError("quantized reference execution is not enabled")
        for module in quantized:
            scheme = module.quantization_scheme
            activation = scheme.input_activations
            if activation is None or not activation.dynamic or activation.num_bits != 8:
                raise ValueError("expected dynamic INT8 activation reference execution")
    nll, tokens = 0.0, 0
    for index, row in enumerate(rows):
        inputs = torch.tensor([row], device=args.device)
        logits = model(input_ids=inputs, use_cache=False).logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            inputs[:, 1:].reshape(-1),
            reduction="sum",
        )
        value = loss.item()
        if not math.isfinite(value):
            raise ValueError(f"non-finite loss for evaluation sample {index}")
        nll += value
        tokens += len(row) - 1
        print(f"Evaluated {index + 1}/{len(rows)}", flush=True)
    report = {
        "model_path": str(args.model_path.resolve()),
        "device": args.device,
        "dtype": "float16",
        "samples": len(rows),
        "predicted_tokens": tokens,
        "sequence_length": args.sequence_length,
        "token_ids_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "quantized_modules": len(quantized),
        "mean_nll": nll / tokens,
        "perplexity": math.exp(nll / tokens),
        "elapsed_seconds": time.monotonic() - started,
        "metric": "all-token next-token perplexity; includes user and assistant tokens",
        "execution": "compressed-tensors QDQ reference" if quant_config else "floating-point reference",
        "ascend_inference_validated": False,
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
