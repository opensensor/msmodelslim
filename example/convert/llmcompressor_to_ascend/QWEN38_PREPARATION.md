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
   tokens, 64 GiB CPU weight budget, routed experts only.
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
