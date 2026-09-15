# Qwen3.8-Flash-Next calibration adapter

`quantize_qwen38.py` adds a local text calibration path without replacing the
pinned Transformers installation. It uses Apache-2.0 upstream Qwen4Exp PyTorch
operations; [provenance and modifications](qwen38/UPSTREAM.md) and the upstream
license are included. It does not provide Ascend serving support.

## What is implemented

- Source expert banks `[experts, 2*intermediate, hidden]` and
  `[experts, hidden, intermediate]` are sliced one projection at a time. Gate is
  the first half; up is the second. No complete expert bank is materialized.
- The model is constructed with meta parameters. Each loaded projection is
  assigned to CPU or disk under an explicit CPU weight-placement budget, with
  execution on the requested CPU/CUDA device. Buffers and unconsumed source
  tensors are checked, and an incomplete download fails before loading starts.
- PLE tables remain in source safetensors files. Lookup gathers selected rows
  across checkpoint shard boundaries and transfers those rows to the execution
  device. It does not stage the 102.4 GB table into RAM or GPU memory.
- The text reference includes GDN, QSA and its indexer, four-stream gated
  residuals, PLE hashing/convolution, routing and shared experts. Calibration is
  eager and uncached. The wrapper deliberately rejects generation/cache and
  multimodal arguments.
- Sequential GPTQ targets only routed expert projections. The all-experts
  calibration option supplies samples to rarely selected experts too; routed
  output accumulation remains unchanged. Attention, routers, shared experts,
  embeddings and vision remain floating point.
- Export writes compressed-tensors INT8 experts in bounded shards and copies
  every remaining original tensor, including vision, PLE and MTP. BF16 tensors
  are converted to the requested dtype; explicit FP32 and integer source tensors
  remain in their original dtype. Nonfinite conversions are rejected. MTP is
  preserved in floating point, not quantized or enabled, so its 5.214 GB payload
  is larger than the optional quantized-MTP estimate in the planning table.

This path creates many individual expert modules (73,728 projections for the
main model). A CPU weight budget is not a process RSS cap: activation caches,
GPTQ Hessians, loader temporaries and file-backed pages require additional space.
The exporter buffers approximately one output shard plus the current tensor;
a single large floating-point tensor may exceed the configured shard target.
Full-size memory and HDD throughput have not been measured. The existing native
saver also initializes floating-point passthrough modules before finalization;
its memory behavior with the full PLE payload needs attention before a full
native export on the 128 GB conversion workstation.

## Run after the source download completes

The original source revision is `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Use fresh output and offload directories. Prefer NVMe for offload when available.

```bash
.venv-cuda/bin/python example/convert/llmcompressor_to_ascend/quantize_qwen38.py \
  --model-path /path/to/original/Qwen3.8-Flash-Next \
  --save-path /path/to/fresh/compressed-qwen38 \
  --offload-dir /path/to/fresh/offload-qwen38 \
  --device cuda --dtype float16 --target atlas-300i-duo \
  --cpu-memory-gib 64 --method gptq \
  --calibration-data /path/to/train_sft.jsonl --samples 128 --sequence-length 1024
```

RTN is available with `--method rtn` and no calibration file. It verifies a
data-free conversion path, not the quality of GPTQ calibration. Do not substitute
the current IQ4_XS GGUF or an FP8 checkpoint for the floating-point source.
`calibration_manifest.json` records method, input-token hash, package versions,
device, memory observations and exported tensor counts. The compressed checkpoint
is an intermediate for the ModelSlim bridge, not a claim of serving compatibility.

## Validation on 2026-09-14

The real checkpoint's complete header inventory matches all 1,163 non-PLE
main-model parameter names and shapes in the reference construction. PLE shard
dimensions are validated separately by the loader.

The integration tests exercise exact expert slicing, CPU/disk forward parity,
PLE shard boundaries and repeated rows, causality through GDN/QSA, chunked versus
recurrent GDN, invalid checkpoints, CPU RTN/GPTQ commands, and sequential GPTQ on
CPU and CUDA followed by native export and exact tensor-value verification.
The CUDA test uses FP16 with forced disk offload on the RTX PRO 6000 Blackwell.

A separate real-weight smoke test ran the first decoder layer's attention and
gated residuals on eight UltraChat conversations, 128 tokens each. The router
selected expert 349 for 145 of those 1,024 tokens. That expert was calibrated
using its actual routed inputs:

| Observation | Result |
| --- | ---: |
| Fused versus separate FP16 projection maximum absolute difference | 0.000244140625 |
| Quantized expert output RMSE on the calibration tokens | 0.00123556 |
| Reference expert output RMS | 0.0293775 |
| Peak PyTorch CUDA allocation | 322,886,144 bytes |
| Peak process RSS, including Python startup | 2,698,616 KiB |
| Exported INT8 expert tensor payload | 4,922,880 bytes |

These are component checks on calibration inputs, not held-out model quality
results. The pilot does not validate full-model calibration or Ascend kernels.
Its script, log, commands, tensor artifact and result JSON are retained under
`models/conversion-runs/qwen3.8-flash-next-real-expert-pilot` on the 4 TB drive.

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/pytest -q \
  test/integration/test_qwen38_calibration.py
OMP_NUM_THREADS=2 .venv-cuda/bin/pytest -q \
  test/integration/test_qwen38_calibration.py -k 'sequential and cuda'
```
