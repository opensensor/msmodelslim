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
