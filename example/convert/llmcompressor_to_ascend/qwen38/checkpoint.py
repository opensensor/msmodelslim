"""Slice fused experts and gather PLE rows without materializing their banks."""

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

EXPERT_WEIGHT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        if self.config.get("model_type") != "qwen4_exp" or self.config.get("quantization_config"):
            raise ValueError("expected an original floating-point qwen4_exp checkpoint")
        self.text_config = self.config["text_config"]
        if self.text_config.get("quantization_config"):
            raise ValueError("quantized text checkpoints are not supported")
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        for filename in set(self.weight_map.values()):
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"unsafe checkpoint shard: {filename}")

    def open(self, name):
        return safe_open(self.path / self.weight_map[name], framework="pt", device="cpu")

    def require_complete(self):
        missing = sorted(name for name in set(self.weight_map.values()) if not (self.path / name).is_file())
        if missing:
            raise FileNotFoundError(
                f"checkpoint download incomplete: {len(missing)} missing shards; first: {missing[0]}"
            )

    def shape(self, name):
        with self.open(name) as handle:
            return tuple(handle.get_slice(name).get_shape())

    def read(self, name, dtype=None):
        """Return an owned tensor, reading at most one expert projection."""
        match = EXPERT_WEIGHT.fullmatch(name)
        if match:
            prefix, expert, projection = match.groups()
            expert = int(expert)
            hidden = self.text_config["hidden_size"]
            middle = self.text_config["moe_intermediate_size"]
            count = self.text_config["num_experts"]
            if not 0 <= expert < count:
                raise ValueError(f"expert index out of range: {expert}")
            down = projection == "down_proj"
            source = prefix + (".down_proj" if down else ".gate_up_proj")
            expected = (count, hidden, middle) if down else (count, 2 * middle, hidden)
            with self.open(source) as handle:
                tensor_slice = handle.get_slice(source)
                if tuple(tensor_slice.get_shape()) != expected:
                    raise ValueError(f"unexpected fused expert shape: {source}")
                if down:
                    value = tensor_slice[expert, :, :]
                else:
                    start = middle if projection == "up_proj" else 0
                    value = tensor_slice[expert, start : start + middle, :]
                return value.to(dtype=dtype or value.dtype, copy=True).contiguous()
        with self.open(name) as handle:
            value = handle.get_tensor(name)
            return value.to(dtype=dtype or value.dtype, copy=True)


class MappedEmbedding(nn.Module):
    """Only gathered rows enter RAM/GPU; the source table stays file backed.

    The meta weight is deliberately not a parameter or buffer. Export copies the
    original PLE shards, and must never serialize this calibration placeholder.
    """

    def __init__(self, checkpoint, prefix, rows, width, dtype):
        super().__init__()
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.weight = torch.empty((rows, width), dtype=dtype, device="meta")
        count = checkpoint.text_config["split_ngram_parts"]
        if rows % count:
            raise ValueError("PLE rows must divide evenly into checkpoint shards")
        self.rows_per_shard = rows // count
        self.names = [f"{prefix}.shard_{index}.weight" for index in range(count)]
        for name in self.names:
            if checkpoint.shape(name) != (self.rows_per_shard, width):
                raise ValueError(f"unexpected PLE shard shape: {name}")

    def forward(self, indices):
        flat = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.weight.shape[0]):
            raise ValueError("PLE row index out of bounds")
        output = torch.empty((flat.numel(), self.weight.shape[1]), dtype=self.dtype)
        shard_ids = flat // self.rows_per_shard
        for shard_id in shard_ids.unique().tolist():
            positions = (shard_ids == shard_id).nonzero().flatten()
            rows = flat[positions] % self.rows_per_shard
            name = self.names[shard_id]
            with self.checkpoint.open(name) as handle:
                # safe_open maps the tensor; advanced indexing copies only rows.
                output[positions] = handle.get_tensor(name)[rows].to(self.dtype)
        return output.reshape(*indices.shape, self.weight.shape[1]).to(indices.device)
