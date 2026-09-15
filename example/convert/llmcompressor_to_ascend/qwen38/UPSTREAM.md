# Reference provenance

Text operations and text configuration are adapted from Apache-2.0 licensed
[huggingface/transformers](https://github.com/huggingface/transformers/tree/bd15bc95a89e728bbc1224084eb3b5829428c353/src/transformers/models/qwen4_exp),
commit `bd15bc95a89e728bbc1224084eb3b5829428c353`.
Copyright 2026 The Qwen Team and The HuggingFace Inc. team.

The calibration copy retains the upstream eager GDN, QSA indexer, gated
residual, router and PLE math. Optional Hub kernel decorators and Accelerate
hooks are removed, imports target the pinned installed Transformers, and the
layer type spelling `qwen_sparse_attention` becomes `full_attention` for
Transformers 5.9 config validation. No installed Transformers files are changed.
Only text primitives and text configuration are included. KV-cache generation,
vision execution, MTP and Ascend inference are outside this calibration adapter.

The resumable sequential pipeline in `resume.py` follows the Apache-2.0 licensed
[vllm-project/llm-compressor sequential pipeline](https://github.com/vllm-project/llm-compressor/blob/dc61a672066741a3f4bca353eb3b0b5dadf8bc2b/src/llmcompressor/pipelines/sequential/pipeline.py),
commit `dc61a672066741a3f4bca353eb3b0b5dadf8bc2b`. It retains the calibration,
compression and propagation order for this adapter's GPTQ configuration and adds
atomic stage persistence and restoration. It is registered separately and does
not patch the installed pipeline.
