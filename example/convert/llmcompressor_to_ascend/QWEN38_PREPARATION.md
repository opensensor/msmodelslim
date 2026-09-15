# Full-run preparation

## Native FLOAT export memory

The old saver initialized every unconverted module before its final traversal.
For Qwen3.8-Flash-Next this included all 102.4 GB of PLE embeddings. Native
Ascend export now loads, saves and releases each model-free module separately.
It also closes cached safetensors handles after each module to release their
file mappings. The output writer retains only its current shard buffer; a single
module can still exceed that shard target. Use a positive `part_file_size`;
zero explicitly selects the unbounded single-file writer.

A shallow leaf view preserves each module's checkpoint naming without letting
the saver's recursive traversal mark unloaded children as processed. Nested
parent/child parameters, fused floating-point MTP names, failure cleanup and
reader-cache restoration are covered by regression tests. The HF/compressed-
tensors saver retains its existing behavior.

Measured on 2026-09-15 with two already verified original checkpoint shards:

| Observation | Result |
| --- | ---: |
| Actual BF16 PLE tensor payload | 6,400,030,720 bytes |
| Number of PLE modules | 8 |
| Largest module | 800,003,840 bytes |
| Peak process RSS during export | 1,079,172 KiB (1.03 GiB) |
| Peak RSS including subsequent value verification | 1,859,996 KiB |
| Tensor payload retained by the module tree afterward | 0 bytes |
| Chunked value comparison | Exact |

This measures real tensor export, not Ascend inference or a full-model memory
limit. The benchmark used cached source data; its elapsed time is not a disk
throughput estimate. Script, log and results are retained on the 4 TB drive under
`models/conversion-runs/qwen3.8-flash-next-float-export-memory-v2`.

## Prepared local run

The run directory on the 4 TB drive is
`models/conversion-runs/qwen3.8-flash-next-full-gptq`.
It contains an exact `commands.json`, a `run.py` launcher and `readiness.json`.
The launcher defaults to checking readiness. It does not start automatically.

Prepared stages:

1. CUDA sequential GPTQ, FP16, 128 calibration conversations of at most 1,024
   tokens, 64 GiB CPU weight budget, routed experts only. Paired perplexity on
   32 held-out conversations runs immediately before and after quantization.
2. CPU ModelSlim W8A8_DYNAMIC export, one worker and 1 GB output shards.
3. Chunked exact comparison of all exported values against the compressed source.

Weight offload uses the mounted NVMe drive at
`/run/media/matteius/ai-drive/ascend-conversion/qwen38-offload`.
Preflight requires 250 GB of free space there and 500 GB for the two output
checkpoints on the 4 TB drive. These are conservative disk allowances, not
measured consumption or filesystem reservations. Both volumes had enough space
when checked. The deployment topology remains two 300I Duo cards, four independent
48 GB chips, with 256 GB host RAM; the conversion workstation has 128 GB RAM.

The launcher checks that source shards exist, the downloader recorded completion
and matching SHA-256 provenance at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, sizes still match that record, output
directories are fresh, disk space is sufficient and at least 64 GiB of GPU memory
is free. It does not rehash the entire original checkpoint. The GPU threshold is
a preflight allowance, not proof that full-model GPTQ fits.

Run `python /absolute/path/to/run.py` to refresh readiness. Once ready, the same
command with `--run` executes the three stages in order with separate logs and a
status JSON. Failed stages stop the sequence; partial directories require review
and a fresh run path before retrying. No full-model calibration, held-out quality
comparison or 310P serving validation has been completed by this preparation.

## Held-out quality check

`quantize_qwen38.py` accepts these additional arguments:

```bash
--quality-data /path/to/test_sft.jsonl \
--quality-report-dir /path/to/fresh-quality-directory \
--quality-samples 32 \
--quality-sequence-length 1024 \
--quality-logit-chunk-size 128
```

The prepared run uses the existing UltraChat `test_sft.jsonl`; calibration uses
`train_sft.jsonl`. Both came from dataset revision
`8049631c405ae6576f93f445c6b8166f76f5505a`. Tokenized duplicate inputs and shared
prefixes truncated to either sequence limit are rejected before model loading,
including duplicates within the test set. This is exact token checking, not
semantic deduplication or a claim that the dataset was absent from pretraining.
Preflight with the released tokenizer found 28,245 predicted test tokens across
the 32 conversations and no rejected duplicates. The token hash and sample counts
are recorded locally in `quality-data-preflight.json` beside the run commands.

The evaluator reuses the CPU/disk-offloaded model and scores uncached sequences
one at a time. It computes the vocabulary projection and cross-entropy in chunks
of at most 128 positions rather than allocating full-sequence logits. It still
executes a complete model pass per conversation: 32 samples before and after
calibration add 64 forward passes and substantial disk reads. This setting limits
logit memory, not attention/intermediate memory or total process RAM.

The report directory must be fresh and separate from source, checkpoint output
and offload directories. `before.json` is saved before quantization so it survives
a later calibration failure. `after.json` and `comparison.json` are saved before
checkpoint export. Reports include the exact token hash, token counts, per-sample
negative log likelihood, aggregate perplexity, dtype, device and timing. The
comparison is also embedded in `calibration_manifest.json` after successful export.

The baseline must have no quantization attached. The quantized pass requires all
routed expert projections to have enabled, frozen symmetric W8A8 quantization
with dynamic INT8 inputs. Non-finite loss stops the run. Reports show the measured
change without imposing an unvalidated quality threshold.

This measures all-token text perplexity, including user and assistant tokens,
using the local eager reference and compressed-tensors quantize/dequantize
execution. It does not validate exported Ascend execution, long-context behavior,
vision, MTP, generation quality or parity with the current IQ4_XS GGUF baseline.
Tiny-model CPU/CUDA tests exercise the measurement and conversion path; full-model
quality results remain pending the download and conversion run.

## Full MoE component probe

`probe_qwen38_moe.py` exercises all 512 routed experts in the first MoE layer,
using real token embeddings, GDN attention and gated residuals to obtain its
inputs. It also loads the real router and shared expert. It needs only the
checkpoint shards containing those components, so it can run before the entire
download finishes. The command rejects a first layer with PLE or non-GDN attention.
For CPU-only diagnostics, hide accelerators before launching Python (for CUDA,
set `CUDA_VISIBLE_DEVICES=''`); LLM-Compressor otherwise selects an accelerator
for its sequential pipeline even if input weights initially reside on CPU.

```bash
.venv-cuda/bin/python example/convert/llmcompressor_to_ascend/probe_qwen38_moe.py \
  --model-path /path/to/original/Qwen3.8-Flash-Next \
  --offload-dir /path/on/nvme/fresh-probe-offload \
  --report /path/to/run/result.json \
  --calibration-data /path/to/train_sft.jsonl \
  --quality-data /path/to/test_sft.jsonl \
  --samples 8 --quality-samples 2 --sequence-length 1024 --device cuda
```

Every MoE weight is placed on disk and loaded to the execution device on demand.
Sequential GPTQ uses the same W8A8 recipe and all-experts calibration behavior as
the full converter, targeting this single MoE block. The probe records the actual
number and storage size of live Hessian matrices immediately before compression,
GPU allocation/reservation peaks, process RSS and calibration time. A small
subclass observes the pinned LLM-Compressor internals without changing GPTQ math.

For 512 experts with hidden size 2,560 and intermediate size 640, the theoretical
FP32 Hessian payload is `512 * (2 * 2560**2 + 640**2) * 4` = 27,682,406,400 bytes.
This excludes weights, activations, inverse-matrix temporaries and allocator
overhead. Matrix size alone does not establish the full converter's memory fit.

Two held-out conversations compare the complete MoE output before and after
quantization, including routing and the shared expert. The error is a local
activation RMSE, not perplexity or a predicted percentage of model quality loss.
The probe does not exercise later-layer PLE/QSA, export a checkpoint or run Ascend
kernels. Captured activations use a small calibration dataset and the probe
does not retain the full model's 64 GiB CPU weight placement.

### Measured result on 2026-09-15

The real layer completed on the RTX PRO 6000 Blackwell with forced NVMe offload:

| Observation | Result |
| --- | ---: |
| Routed experts / quantized projections | 512 / 1,536 |
| Calibration conversations / tokens | 8 / 7,336 |
| Held-out conversations / tokens | 2 / 2,048 |
| Observed live FP32 Hessian payload | 27,682,406,400 bytes |
| Minimum calibration samples per projection | 8 |
| Peak PyTorch CUDA allocation | 34,336,519,680 bytes (31.98 GiB) |
| Peak PyTorch CUDA reservation | 34,378,612,736 bytes (32.02 GiB) |
| Peak process RSS | 2,848,908 KiB (2.72 GiB) |
| Sequential calibration/compression time | 465.6 seconds |
| Whole probe time, excluding Python startup | 500.1 seconds |
| Held-out MoE output RMSE | 0.00117161 |
| Held-out floating-point MoE output RMS | 0.04526633 |

The matrix payload exactly matches the architectural calculation. This removes
the uncertainty about whether all experts' GPTQ matrices fit together on this
GPU. It does not measure full-model peak memory with 128 calibration conversations,
all decoder components, PLE lookup and the remaining weights resident in CPU RAM.
The existing 64 GiB free-GPU preflight requirement remains unchanged.

At this small sample count, repeating the measured calibration stage 48 times
would already take about 6.2 hours. That is an illustrative component extrapolation,
not a full-run ETA; full calibration uses more samples and adds attention, PLE,
held-out evaluation, loading and two checkpoint writes.

Commands, logs, results, package versions, implementation hashes and verified
source-shard records are retained under
`models/conversion-runs/qwen3.8-flash-next-full-moe-probe` on the 4 TB drive.
`probe-executed.py` preserves the exact executed script. The committed script
additionally rejects CPU mode when an accelerator remains visible; the CUDA
calculation is unchanged. Regression coverage includes fused/linear MoE parity,
forced disk offload, real GPTQ matrix observations and unsupported-mode rejection.
