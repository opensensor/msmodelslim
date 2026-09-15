# LLM-Compressor W8A8 to Ascend ModelSlim

This development bridge imports a compressed-tensors `int-quantized` checkpoint
into the native ModelSlim `W8A8_DYNAMIC` layout. It reuses ModelSlim's existing
checkpoint reader, parallel converter, dynamic INT8 IR and Ascend V1 saver.
The export runs on CPU without loading a Transformers model or using CANN.

The intended workflow is:

1. Quantize original floating-point weights with LLM-Compressor on the RTX GPU.
   Use symmetric INT8 per-output-channel weights and dynamic, symmetric INT8
   per-token activations (`scheme="W8A8"`). GPTQ and round-to-nearest are both
   covered by the tiny CPU tests. Real GPTQ calibration needs representative data.
2. Save a compressed-tensors checkpoint (`save_compressed=True`).
3. Run the CPU exporter below into a fresh directory.
4. Validate loading, accuracy and generation on the intended vLLM-Ascend/310P stack.

The [full-size Qwen3-8B RTX pilot](QWEN3_8B_PILOT.md) completed GPTQ calibration,
native export, exact verification and a held-out reference quality comparison.
It produced 9.45 GB of native tensor payload; actual Duo inference remains pending.

## Supported checkpoint contract

| Source | Ascend output |
| --- | --- |
| `*.weight`, INT8, shape `[out, in]` | Same integer values and layout |
| `*.weight_scale`, FP32/FP16/BF16, `[out, 1]` or `[out]` | FP32 `[out, 1]`, preserving represented values |
| Optional all-zero `*.weight_zero_point` | FP32 all-zero `*.weight_offset` |
| Optional floating-point linear bias | FP32 bias |
| Floating-point exclusions, embedding and normalization weights | Preserved |
| compressed-tensors `quantization_config` | Native `quant_model_description.json` |

Every INT8 linear must be selected. Unfused expert projections with ordinary
two-dimensional weights are supported at the file-format level. Input metadata
and headers are checked before tensor conversion. Tensor values are validated
when loaded; a late error can leave partial output, which must not be deployed.
Use another fresh directory after failure.

The bridge rejects static activation quantization, asymmetric/group/block
quantization, KV-cache quantization, runtime transforms, packed or fused expert
tensors, nonzero zero points, unsupported quantization tensors and preprocessing
rules. GPTQ `static`/`weight` activation ordering is accepted because the exporter
restores column order; runtime `weight_g_idx` permutations are rejected.

This does **not** produce W8A8SC, W4A8, GGUF, FP8, MXFP4 or NVFP4 checkpoints.
It does not add support for a model architecture. Preservation of serialized
weights does not establish equality between NVIDIA and Ascend activation kernels.
The local 310P dynamic INT8 implementation documents accuracy limitations; NPU
loading and numerical validation remain required. NZ packing is performed by
the 310P loader after loading, rather than by this exporter.

## Reproduce the CPU development environment

Run from this repository, with `uv` available. The sibling clones used here are:

| Tool | Source | Tested revision |
| --- | --- | --- |
| ModelSlim base | <https://gitcode.com/Ascend/msmodelslim> | `7848226928475c77a064488ef94f7f87b67c2621` |
| LLM-Compressor | <https://github.com/vllm-project/llm-compressor> | `dc61a672066741a3f4bca353eb3b0b5dadf8bc2b` (tag `0.12.0.1`) |
| compressed-tensors | <https://github.com/vllm-project/compressed-tensors> | `c18a0fa8b969d789f33b7912eef57cd69c56bd9e` (tag `0.17.1`) |

For a fresh workspace, clone the two dependencies beside this repository and
check out the revisions above. Existing clones need not be replaced.

```bash
git clone --branch 0.12.0.1 https://github.com/vllm-project/llm-compressor.git ../llm-compressor
git clone --branch 0.17.1 https://github.com/vllm-project/compressed-tensors.git ../compressed-tensors
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -r example/convert/llmcompressor_to_ascend/requirements-cpu.txt
uv pip install --python .venv-cpu/bin/python --no-deps -e . -e ../llm-compressor -e ../compressed-tensors
git -C ../compressed-tensors apply ../msmodelslim/example/convert/llmcompressor_to_ascend/patches/compressed-tensors-disk-dtype.patch
ln -s ../config msmodelslim/config
ln -s ../lab_calib msmodelslim/lab_calib
ln -s ../lab_practice msmodelslim/lab_practice
```

Skip the last three commands if the package data links already exist. Editable
installs need these links because the upstream legacy setup hook does not create
them under PEP 660. Editable dependency versions may have `.dev0` suffixes even
at release tags; the Git revisions above identify the tested code.

The bundled compressed-tensors patch restores parameter metadata **after** a disk
cache dtype cast. Without it, loading FP32 source shards as BF16 can lose the
`Parameter` type and fail when saving a disk-offloaded model. The local dependency
checkout includes this patch; skip `git apply` if already applied. A regression
test checks parameter type and metadata, and forced-disk runs are compared with CPU-resident
quantization. The patch applies to the exact revision listed above.

This environment intentionally has CPU-only PyTorch and torchvision. Transformers
5.9's MoE patch discovery imports image-processing aliases even for text models,
so torchvision is needed for this loading path. Optional AutoRound features are
not installed. Use a separate CUDA environment for eventual RTX calibration. Python 3.12 avoids a
Python 3.14/Pydantic annotation failure encountered when importing LLM-Compressor.

## Calibrate local floating-point weights

### RTX conversion for Atlas 300I Duo

The deployment target for this fork is the **96 GB Atlas 300I Duo: two Ascend
310P chips, each with nominally 48 GB**. Each chip has its own memory budget.
Use the actual available device memory reported on the Ascend host, then reserve
space for KV cache, activations, communication, NZ padding and runtime workspaces.
Do not divide checkpoint bytes by total card memory and call that a runtime fit.
Tensor-parallel ranks refer to NPU devices: one Duo has two devices, two cards
have four, and three cards have six. Valid parallel sizes also depend on the
model's attention heads, expert layout and supported communication topology.

Build a separate CUDA environment for the RTX conversion host:

```bash
uv venv --python 3.12 .venv-cuda
uv pip install --python .venv-cuda/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv-cuda/bin/python -r example/convert/llmcompressor_to_ascend/requirements-cpu.txt
uv pip install --python .venv-cuda/bin/python --no-deps -e . -e ../llm-compressor -e ../compressed-tensors
```

Use the same pinned dependency checkouts, disk-cache patch and package-data links
as the CPU setup. The requirements file provides the shared Python dependencies;
PyTorch and torchvision come from the CUDA index here. CUDA's Triton package is
for the conversion host only; it is not a dependency to install on the Duo.

```bash
OMP_NUM_THREADS=8 .venv-cuda/bin/python \
  example/convert/llmcompressor_to_ascend/quantize_w8a8.py \
  --model-path /path/to/original-floating-point-model \
  --save-path /path/to/fresh-compressed-w8a8 \
  --offload-dir /path/to/fresh-scratch \
  --method gptq --device cuda --dtype float16 --target atlas-300i-duo \
  --cpu-memory-gib 64 --calibration-data /path/to/calibration.jsonl \
  --samples 128 --sequence-length 1024
```

This target profile requires explicit FP16 loading for floating-point runtime
tensors. It records the intended hardware; it does not change dynamic W8A8 into
W8A8SC or certify a particular model/kernel. Both RTN and GPTQ explicitly execute
on the selected device while retaining CPU/disk weight storage. The manifest
records elapsed time, GPU identity and PyTorch peak allocated/reserved bytes.
Those GPU figures exclude driver and non-PyTorch allocations and are not Ascend
runtime memory estimates. The existing CPU command still runs with CUDA hidden.

Run the GPU integration checks only when the GPU is available for this work:

```bash
OMP_NUM_THREADS=2 .venv-cuda/bin/python -m pytest -q \
  test/integration/test_calibration_workflow.py -k cuda
```

After conversion, use the CPU export and value verifier below. Initial Duo
deployment should explicitly select FP16, a bounded context length and low
concurrency. Measure single-chip execution first for models that fit, then
compare two-chip parallelism with independent replicas. Acceptance requires
actual 310P loading, held-out numerical/quality checks, prefill and decode
throughput, and per-chip peak memory. NVIDIA conversion success alone does not
satisfy those checks. The local dynamic INT8 kernel's documented accuracy issue
remains a reason to compare against existing W8A8SC checkpoints on the NPU.

References: [official Ascend Duo memory discussion](https://www.hiascend.com/developer/techArticles/20251212-1?envFlag=1),
[vLLM-Ascend 310P deployment guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/tutorials/hardwares/310p.html).

### Shared calibration options

`quantize_w8a8.py` prepares the compressed-tensors input for the exporter. It loads
local safetensors and tokenizer files without remote code, uses CPU/disk weight
placement with a configurable budget, and invokes sequential GPTQ calibration.
The default device is CPU; CUDA requires an explicit `--device cuda` and a
separate environment with CUDA-enabled PyTorch. CPU mode hides CUDA devices
before importing accelerator-aware libraries.

```bash
CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python \
  example/convert/llmcompressor_to_ascend/quantize_w8a8.py \
  --model-path /path/to/original-floating-point-model \
  --save-path /path/to/fresh-compressed-w8a8 \
  --offload-dir /path/to/fresh-nvme-scratch \
  --method gptq --device cpu --dtype bfloat16 \
  --cpu-memory-gib 64 \
  --calibration-data /path/to/representative.jsonl \
  --samples 512 --sequence-length 2048
```

The first `--samples` nonblank JSONL records must contain either `{"text": "..."}`
or `{"messages": [{"role": "user", "content": "..."}]}`. Chat records use the
tokenizer's chat template without adding special tokens twice. Samples are
truncated to the requested sequence length; fewer records or empty tokenized
samples are rejected. RTN uses `--method rtn` and no calibration file.

The command keeps the language-model head and conventional MoE router/gate names
in floating point; `--ignore` adds exclusions. These defaults require review for
each new architecture. `--sequential-target` can select a decoder-layer class
when automatic tracing needs guidance. LLM-Compressor performs supported expert
linearization during loading. Tiny Qwen3, Qwen3-MoE, GLM-4-MoE and Qwen3-Next tests
cover both RTN and sequential GPTQ. An independent check compares logits before
and after Qwen3-MoE linearization. These are CPU format/calibration checks, not
validation of full-size model quality or 310P execution.

| Tested family | CPU fixture coverage |
| --- | --- |
| Qwen3 | Dense attention and MLP linears |
| Qwen3-MoE | Unfused expert projections and floating-point router |
| GLM-4-MoE (GLM-4.5 family) | Dense first layer, routed/shared experts, FP32 router correction tensor |
| Qwen3-Next | Both linear/full attention, routed/shared experts, convolution and recurrent parameters |

The GLM fixture uncovered a dropped router correction tensor when a floating-point
virtual linear stored auxiliary tensors as buffers. The converter now preserves
those tensors through the native saver's parameter enumeration. Regression tests
also use nonzero corrections so preservation is checked beyond initialized zeros.

`--cpu-memory-gib` budgets **weight placement**, not total process RAM. Calibration
activations, Hessian matrices, loading transients and the active layer/expert
need additional space. For models requiring post-load expert splitting, temporary
fused tensors may also be large. Disk offload is exercised on tiny models; this
does not establish a full-size model's RAM/VRAM ceiling or acceptable HDD speed.

Output and offload directories must be fresh and separate from the source. A
successful save writes `calibration_manifest.json` with package versions, settings,
sample count, a tokenized-data fingerprint and the modules initially offloaded to
disk. Keep intermediate weights and scratch files until validation is complete.
The manifest records calibration completion, not Ascend inference compatibility.

## Verify exported checkpoints

### Check reference quality

`evaluate_w8a8.py` measures all-token next-token perplexity on local held-out
JSONL records using Transformers. Run it once on the original weights and once
on the compressed-tensors intermediate, with the same records and limits:

```bash
OMP_NUM_THREADS=8 .venv-cuda/bin/python \
  example/convert/llmcompressor_to_ascend/evaluate_w8a8.py \
  --model-path /path/to/compressed-w8a8 \
  --calibration-data /path/to/heldout.jsonl --report /path/to/fresh-quality.json \
  --device cuda --samples 32 --sequence-length 1024
```

Despite the shared `--calibration-data` option name, use records excluded from
calibration here. Compare only reports with identical `token_ids_sha256` and
`predicted_tokens`. The metric includes both user and assistant tokens; it is
not an assistant-only benchmark. Compressed checkpoints run through
compressed-tensors quantize/dequantize reference operations, with dynamic INT8
activation quantization enabled. The entire decompressed model must fit on the
selected device. This command currently uses FP16 and does not use the offload
calibration path. Native Ascend output is rejected: quality measurements here
do not establish 310P kernel parity or validate unrelated model architectures.

### Check export integrity

Every W8A8 import now validates its completed checkpoint inventory, shard index,
tensor shapes/dtypes, per-tensor quantization tags and model configuration before
reporting success. `conversion_report.json` records this **header-only** check,
tensor counts, and source/export tensor payload sizes in bytes. These are serialized
tensor sizes, not runtime RAM/VRAM requirements or filesystem allocation sizes.

For a separate comparison of every tensor value:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python \
  example/convert/llmcompressor_to_ascend/verify_w8a8.py \
  --source /path/to/compressed-w8a8 \
  --export /path/to/ascend-output \
  --check-values --chunk-rows 1024
```

The command prints a JSON report and returns a nonzero status for mismatches.
Without `--check-values`, it reads headers only. Value checks compare integer
weights and floating-point exclusions exactly, account for widening of scales and
quantized-linear biases to FP32, and verify zero offsets/zero points and positive
finite scales. Reads are chunked along the first tensor dimension; reduce
`--chunk-rows` to reduce comparison memory. A complete value check reads both
checkpoints in full and can be slow on HDDs. Neither mode certifies Ascend kernel
accuracy, model architecture support, or language-model quality.

## Export and test

```bash
CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/msmodelslim quant \
  --model_path /path/to/compressed-w8a8 \
  --save_path /path/to/fresh-ascend-output \
  --config example/convert/llmcompressor_to_ascend/w8a8_dynamic.yaml \
  --device cpu

CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python -m pytest -q test/integration
CUDA_VISIBLE_DEVICES='' .venv-cpu/bin/python -m pytest -q \
  test/cases/core/convert test/cases/processor/convert \
  test/cases/core/quant_service/modelslim_convert
```

Run the two pytest commands separately: upstream unit-test fixtures replace some
modules with mocks, whereas the integration tests exercise real serialization.
The LLM-Compressor tests construct a tiny random Qwen3 locally and perform both
RTN and GPTQ quantization before invoking the real CLI. They require no model
download, credentials or accelerator. Synthetic calibration verifies plumbing,
not useful language-model quality.

The calibration workflow tests also cover Qwen3-MoE, sequential GPTQ, forced disk
offload, unchanged source files and exact agreement with CPU-resident quantization.
The bundled disk-cache regression runs on CPU. The upstream compressed-tensors
offload suite currently assumes an accelerator during fixture collection and
cannot be collected in this CPU-only environment.

The example uses one CPU worker, one linear per conversion group, and 1 GB output
shards. This limits concurrent conversion payloads but is not a hard RAM ceiling:
the largest tensor, saver buffers and floating-point leftovers also consume RAM.
Two CPU workers and source shards with scales stored separately are tested.
Full-size memory use and HDD throughput have not yet been measured.
