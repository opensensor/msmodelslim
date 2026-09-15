# Qwen3.8-Flash-Next: four-device 310P target

The selected comparison baseline is the user's three-part GGUF checkpoint:
`model_dir/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`.
The path is recorded as supplied; its absolute location, file hashes, exact
per-tensor GGUF types and serving settings have not been captured. Higher
precision must be compared on the same tasks, prompts and context lengths.

The deployment host will have **256 GB DDR4-3200 RAM**. This makes host FP16
PLE storage the preferred first target: preserve the embedding precision and
implement host lookup before adding embedding quantization. This is a deployment
host specification, separate from the 128 GB RTX conversion workstation used
for the Qwen3-8B pilot.

## Source audit

Audited the original [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` on 2026-09-14.
All 1,658 tensor headers across 131 shards match the source shard index,
dtype/shape payload sizes and complete shard spans. Header reads total 229,760
bytes. Weight payloads have not been downloaded or SHA-256 verified in this audit.

| Source group | BF16/I64 payload, GB (decimal) |
| --- | ---: |
| Main routed experts | 241.591910400 |
| N-gram embedding tables (PLE) | 102.400491520 |
| Other language-model tensors | 9.895397400 |
| Vision tensors | 0.897862112 |
| MTP tensors | 5.214301696 |
| Total | 359.999963128 |

The checkpoint stores experts as fused 3D `gate_up_proj` and `down_proj`
parameters. PLE comprises 128 tables of shape `[2500012, 160]`, totaling
51,200,245,760 elements. This is separate from ordinary token embeddings.

## Proposed higher-precision layouts

Initial policy: routed experts use per-channel INT8 weights and dynamic INT8
activations (`W8A8_DYNAMIC`); attention, shared experts, routers, gated residuals,
vision and other floating-point tensors retain FP16/required FP32 precision.
The existing GGUF runtime may have different activation semantics, so more weight
bits alone do not prove higher end-to-end accuracy.

The table below excludes MTP. Expert accounting includes FP32 scale and zero
offset per output channel. Proposed INT8/INT4 PLE storage includes one FP32 scale
per 160-element row, with selected rows dequantized for computation.

| PLE placement/storage | NPU tensor payload GB | Host table GB | Assessment |
| --- | ---: | ---: | --- |
| On-device FP16 | 234.745 | 0 | Exceeds nominal capacity |
| On-device INT8 | 184.824 | 0 | Too little runtime headroom |
| **Host FP16** | **132.344** | **102.400** | **Preferred with the specified 256 GB host** |
| Host INT8 | 132.344 | 52.480 | Optional later memory/bandwidth optimization |
| On-device INT4 | 159.224 | 0 | Alternative if host lookup latency is unacceptable |

MTP adds approximately 2.713 GB under the same expert policy. Initial validation
should isolate the main model, then add MTP after correctness is established.
The host FP16 layout still stores about 234.745 GB of checkpoint tensor payload;
moving tables to RAM does not reduce disk storage. Host figures assume a single
distributed table, not four full private copies.
Host FP16 tables leave approximately 153.6 GB of the nominal 256 GB for the OS,
workers and loading transients before those allocations are measured. DDR4-3200
alone does not specify achievable bandwidth: channel population, NUMA placement,
random lookup behavior and PCIe transfers need measurement. Shard table ownership
across ranks and avoid serializing the main decode loop on host lookups.

Four 48 GB devices are independent memory domains. The planner's default 6 GiB
reserve per device is an assumption, not a measurement. Its ideal average is
only a starting estimate: QSA has 24 query heads and 2 KV heads, so TP4 introduces
KV replication; other parameters, NZ padding, rank imbalance, recurrent state,
KV cache and runtime buffers must be accounted for on the actual host.

## Implementation findings and next gates

Audited upstream vLLM at `381c61e008682945d43d6c8808ed633477364b4b` in a new sibling
checkout. Its Qwen4Exp package selects NVIDIA or AMD implementations and has no
Ascend branch. The NVIDIA PLE offload uses pinned host memory, CUDA UVA,
`torch.cuda.Stream` and a Triton lookup kernel. QSA depends on its own indexer,
cache handling and FlashAttention backend. A generic offload flag cannot supply
a 310P implementation of these paths.

The installed Transformers remains pinned. A dedicated local
[text calibration adapter](QWEN38_CALIBRATION.md) now provides eager Qwen4Exp
operations, fused-to-linear expert loading and file-backed PLE lookup. It feeds
ordinary 2D expert projections to the existing bridge. Native Ascend kernels and
quantized embedding tables are separate work; this calibration reference does
not implement either.

1. Bounded expert unpacking and text reference/calibration loading are implemented.
   Small CPU/CUDA GPTQ and native export tests pass, as does a real first-layer
   expert pilot. Complete the source download, then measure full-model calibration
   memory and output fidelity. The native FLOAT saver now streams its payload;
   a real 6.4 GB PLE export passed with 1.03 GiB peak export RSS.
2. Implement and numerically test sharded FP16 host PLE lookup with transfer and
   prefetch appropriate to Ascend. Check loading transients against host RAM,
   as well as steady-state table size. INT8 row storage is an optional later path.
3. Adapt Qwen4Exp dispatch, QSA/indexer/cache, gated residuals, PLE convolution
   and GDN to 310P; reuse existing 310P operations where their semantics match.
4. Validate real-weight text and image requests with eager execution, bounded
   context and one request. Then validate TP/EP, graph execution, MTP and
   concurrency. No feature is marked supported by this header audit.
5. Compare against the captured UD-IQ4_XS baseline: task success, first-token
   latency, decode throughput, long-context behavior and per-chip peak memory.

Full-model conversion, quantized PLE formats, 310P inference and quality remain
**unvalidated**. The outputs of the accounting commands below are planning
artifacts, not deployable checkpoints. The new calibration command has a separate
runbook and validation record.

## Reproduce the accounting

```bash
.venv-cpu/bin/python example/convert/llmcompressor_to_ascend/fetch_qwen38_headers.py \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 --output /path/to/audit
.venv-cpu/bin/python example/convert/llmcompressor_to_ascend/plan_qwen38.py \
  --inventory /path/to/audit/tensor_headers.json \
  --config /path/to/audit/metadata/config.json --host-memory-gb 256 \
  --report /path/to/fresh-plan.json
```

The header fetcher resumes within a pinned revision, validates byte-range
responses before reading and rejects an accidental full-file response. Its
cache checks are structural, not a verification of the remote tensor values.
The planner reports both MTP variants and accepts explicit device count,
nominal GB per device and reserve GiB per device. It never declares runtime fit.
