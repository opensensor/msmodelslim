# Qwen3-8B RTX conversion pilot for Atlas 300I Duo

Measured on September 14, 2026 using an RTX PRO 6000 Blackwell Workstation
Edition and 128 GB system RAM. Conversion, native export and exact tensor
verification passed. **Ascend loading, kernel accuracy and throughput remain
unvalidated.** The target is a 96 GB Duo with two independent 48 GB 310P chips.

## Inputs and configuration

- Original model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B),
  revision `b968826d9c46dd6066d109eabc6255188de91218`. All five original
  safetensors shards were verified against the Hub's SHA-256 metadata.
- GPTQ: W8A8, FP16 loading, CPU weight budget 64 GiB, sequential CUDA execution,
  seed 42, 128 records, maximum 1,024 tokens per record. No disk offload was needed.
- Calibration: first 128 `train_sft` records from
  [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
  revision `8049631c405ae6576f93f445c6b8166f76f5505a`.
- Evaluation: first 32 `test_sft` records at that same revision, maximum 1,024
  tokens each; 27,478 next-token predictions. Both reports have token fingerprint
  `a6d76709502bae03c785045709cc1852d21e0503c1c9e72673f3bdd9beafa4d3`.
- Software: PyTorch `2.11.0+cu130`, Transformers `5.9.0`, the dependency
  revisions and compressed-tensors disk-cache patch in [README.md](README.md).

## Measurements

| Measurement | Result |
| --- | ---: |
| GPTQ load, calibration and save wall time | 356.72 s |
| Peak PyTorch CUDA allocated memory | 3,128,169,984 bytes (2.91 GiB) |
| Peak PyTorch CUDA reserved memory | 4,320,133,120 bytes (4.02 GiB) |
| Calibration peak process RSS | 32,998,036 KiB (31.47 GiB) |
| CPU native export wall time | 58.86 s |
| CPU native export peak process RSS | 2,790,328 KiB (2.66 GiB) |
| Complete value verification wall time | 3.73 s |
| Compressed-tensors tensor payload | 9,438,504,960 bytes |
| Native Ascend tensor payload | 9,446,909,952 bytes (9.45 GB / 8.80 GiB) |
| Quantized linear modules | 252 |
| Verified native tensors | 903 |
| FP16 reference perplexity | 5.355612 |
| Compressed-tensors QDQ reference perplexity | 5.347962 |

Memory figures are conversion-host measurements, not Ascend runtime estimates.
GPU counters exclude non-PyTorch allocations. Disk timings include filesystem
caching and overlapping reference evaluation, so they are not cold-disk
benchmarks. The native tensor payload excludes metadata and filesystem overhead.

The perplexity change is approximately -0.14%, effectively unchanged on this
small English conversation sample. It does not demonstrate a quality improvement
or broad task accuracy. Both references load in FP16; the metric includes user
and assistant tokens. Quantized reference execution enables dynamic INT8 QDQ on
the GPU; it does not exercise the 310P dynamic INT8 kernel.

## First Duo runtime check

Start with one NPU device, FP16, 4,096-token context and one concurrent request.
The model's FP16 KV payload is 147,456 bytes per token, or 603,979,776 bytes
(576 MiB) for one 4K sequence. Weights, NZ layout/padding, KV block rounding,
activations and runtime workspaces require additional memory. Measure those
against actual available memory on that chip before increasing context or
concurrency. Then compare two-device parallelism against independent replicas.

The export is `W8A8_DYNAMIC`, not W8A8SC. Existing 310P dynamic INT8 accuracy
limitations still require native generation and numerical checks. This pilot
establishes an 8B conversion path; it does not establish large MoE model support.
