"""Text-only, uncached Qwen4Exp reference for post-training calibration."""

from pathlib import Path
from typing import ClassVar

import torch
from accelerate import init_empty_weights
from compressed_tensors.offload import offload_module
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .checkpoint import Checkpoint, MappedEmbedding
from .configuration import Qwen4ExpTextConfig
from .reference import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextExperts,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextMLP,
    Qwen4ExpTextRotaryEmbedding,
)


class LinearExperts(nn.ModuleList):
    """Checkpoint gate/up ordering with normal Linear modules for GPTQ hooks."""

    def __init__(self, config):
        super().__init__(Qwen4ExpTextMLP(config, config.moe_intermediate_size) for _ in range(config.num_experts))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for index, expert in enumerate(self):
            tokens, slots = torch.where(top_k_index == index)
            if get_calibrate_all_experts_flag():
                values = expert(hidden_states)[tokens]
            elif tokens.numel():
                values = expert(hidden_states[tokens])
            else:
                continue
            output.index_add_(0, tokens, (values * top_k_weights[tokens, slots, None]).to(output.dtype))
        return output


class TextBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Qwen4ExpTextDecoderLayer(config, index) for index in range(config.num_hidden_layers)
        )
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(config)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        hidden = self.embed_tokens(input_ids)
        input_ids = input_ids.to(hidden.device)
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("empty sequences are not supported")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        attention_mask = attention_mask.to(hidden.device)
        positions = torch.arange(length, device=hidden.device).view(1, 1, -1).expand(3, batch, -1)
        rotary = self.rotary_emb(hidden, positions)
        visible = torch.ones((length, length), device=hidden.device, dtype=torch.bool).tril()
        visible = visible[None, None] & attention_mask[:, None, None, :].bool()
        mask = torch.zeros((batch, 1, length, length), device=hidden.device, dtype=hidden.dtype)
        mask.masked_fill_(~visible, torch.finfo(hidden.dtype).min)
        eos = self.config.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        ple_ids = torch.where(attention_mask.bool(), input_ids, eos) if self.config.ple_layer_ids else input_ids
        hidden = hidden.repeat(1, 1, self.config.hc_count)
        for layer in self.layers:
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=mask,
                conv_mask=attention_mask,
                past_key_values=None,
                ple_input_ids=ple_ids,
            )
        return self.hyper_connection_mixer(hidden)


class TextContainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = TextBackbone(config)

    def forward(self, input_ids, attention_mask=None):
        return self.language_model(input_ids, attention_mask)


class Qwen38ForCalibration(PreTrainedModel):
    config_class = Qwen4ExpTextConfig
    _no_split_modules: ClassVar[list[str]] = ["Qwen4ExpTextDecoderLayer"]
    _supports_sdpa = False
    _supports_flash_attn = False

    def __init__(self, config):
        config._attn_implementation = "eager"
        config.use_cache = False
        super().__init__(config)
        self.model = TextContainer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        if use_cache or any(value is not None for value in kwargs.values()):
            raise ValueError("this adapter accepts uncached text input_ids/attention_mask only")
        hidden = self.model(input_ids, attention_mask)
        return CausalLMOutputWithPast(logits=self.lm_head(hidden))


def build_config(source):
    values = dict(source)
    # The upstream config's default is 512; the released checkpoint uses 128.
    if values.get("ple_layer_ids") and "split_ngram_parts" not in values:
        raise ValueError("checkpoint must specify split_ngram_parts for PLE loading")
    return Qwen4ExpTextConfig(**values)


def load_model(path, offload_dir, dtype=torch.float16, device="cpu", cpu_budget_bytes=64 * 1024**3):
    """Build on meta; place each projection separately under a CPU weight budget."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("only floating calibration dtypes are supported")
    if cpu_budget_bytes < 0:
        raise ValueError("CPU weight budget cannot be negative")
    checkpoint = Checkpoint(path)
    checkpoint.require_complete()
    offload_dir = Path(offload_dir).resolve()
    if (
        offload_dir == checkpoint.path
        or offload_dir in checkpoint.path.parents
        or checkpoint.path in offload_dir.parents
    ):
        raise ValueError("offload directory must be separate from the checkpoint")
    if offload_dir.exists() and (not offload_dir.is_dir() or any(offload_dir.iterdir())):
        raise ValueError("offload directory must be empty")
    offload_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(checkpoint.text_config)
    config._name_or_path = str(checkpoint.path)
    with init_empty_weights():
        model = Qwen38ForCalibration(config)
        for name, module in list(model.named_modules()):
            if isinstance(module, Qwen4ExpTextExperts):
                model.set_submodule(name, LinearExperts(config))
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Embedding) and name.endswith("ple.ple_embedding.ngram_embedding"):
            model.set_submodule(
                name,
                MappedEmbedding(checkpoint, name, module.num_embeddings, module.embedding_dim, dtype),
            )
    placed = 0
    loaded = set()
    for name, module in model.named_modules():
        if isinstance(module, MappedEmbedding):
            continue
        for key, param in list(module.named_parameters(recurse=False)):
            full_name = f"{name}.{key}" if name else key
            tensor = checkpoint.read(full_name, dtype=dtype)
            if tensor.shape != param.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid tensor or nonfinite dtype conversion: {full_name}")
            module._parameters[key] = nn.Parameter(tensor, requires_grad=False)
            loaded.add(full_name)
        for key, buffer in list(module.named_buffers(recurse=False)):
            full_name = f"{name}.{key}" if name else key
            if full_name in checkpoint.weight_map:
                value = checkpoint.read(full_name)
                if value.shape != buffer.shape:
                    raise ValueError(f"unexpected buffer shape: {full_name}")
                module._buffers[key] = value
                loaded.add(full_name)
        size = sum(t.numel() * t.element_size() for t in module.parameters(recurse=False))
        size += sum(t.numel() * t.element_size() for t in module.buffers(recurse=False))
        if size:
            storage = "cpu" if placed + size <= cpu_budget_bytes else "disk"
            offload_module(module, device, storage, **({"offload_dir": str(offload_dir)} if storage == "disk" else {}))
            if storage == "cpu":
                placed += size
    # Check all original text tensors were accounted for; never quietly retain
    # random parameters or omit an architecture-specific checkpoint tensor.
    for name in checkpoint.weight_map:
        is_text = name.startswith("model.language_model.") or name == "lm_head.weight"
        is_fused = name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
        is_ple = ".ple.ple_embedding.ngram_embedding.shard_" in name
        if is_text and not is_fused and not is_ple and name not in loaded:
            raise ValueError(f"unconsumed text tensor: {name}")
    model.checkpoint = checkpoint
    model.cpu_weight_bytes = placed
    return model.eval()
