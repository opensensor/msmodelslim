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
uv pip install --python .venv-cpu/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -r example/convert/llmcompressor_to_ascend/requirements-cpu.txt
uv pip install --python .venv-cpu/bin/python --no-deps -e . -e ../llm-compressor -e ../compressed-tensors
ln -s ../config msmodelslim/config
ln -s ../lab_calib msmodelslim/lab_calib
ln -s ../lab_practice msmodelslim/lab_practice
```

Skip the last three commands if the package data links already exist. Editable
installs need these links because the upstream legacy setup hook does not create
them under PEP 660. Editable dependency versions may have `.dev0` suffixes even
at release tags; the Git revisions above identify the tested code.

This environment intentionally has CPU-only PyTorch and a small dependency set
for RTN/GPTQ tests; optional AutoRound/media features are not installed. Use a
separate CUDA environment for eventual RTX calibration. Python 3.12 avoids a
Python 3.14/Pydantic annotation failure encountered when importing LLM-Compressor.

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

The example uses one CPU worker, one linear per conversion group, and 1 GB output
shards. This limits concurrent conversion payloads but is not a hard RAM ceiling:
the largest tensor, saver buffers and floating-point leftovers also consume RAM.
Two CPU workers and source shards with scales stored separately are tested.
Full-size memory use and HDD throughput have not yet been measured.
