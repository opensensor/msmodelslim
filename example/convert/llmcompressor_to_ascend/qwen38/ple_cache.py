"""Cache exact PLE rows for fixed calibration/evaluation inputs.

Hash with the reference implementation, then scan each source table once in
bounded contiguous blocks. Subsequent forwards perform only compact RAM gathers.
The full original PLE tables remain unchanged for export.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from torch import nn

from .checkpoint import MappedEmbedding
from .reference import Qwen4ExpTextNGramEmbedding
from .resume import atomic_json, digest, sync_directory


class RowsCollected(Exception):
    pass


class RowCollector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.empty(0, device="meta")
        self.rows = []

    def forward(self, indices):
        self.rows.append(indices.flatten().unique())
        raise RowsCollected


def required_rows(checkpoint, config, layer, ple_index, datasets):
    with init_empty_weights():
        embedding = Qwen4ExpTextNGramEmbedding(config, config.ple_embed_dim, layer, ple_index)
    collector = RowCollector()
    embedding.ngram_embedding = collector
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding"
    for key, value in list(embedding.named_buffers(recurse=False)):
        name = f"{prefix}.{key}"
        if name in checkpoint.weight_map:
            embedding._buffers[key] = checkpoint.read(name)
        elif value.is_meta:
            raise ValueError(f"missing PLE hash buffer: {name}")
    for rows, sequence_length in datasets:
        for row in rows:
            ids = list(row["input_ids"])
            mask = row["attention_mask"]
            ids = [token if active else embedding.eos_token_id for token, active in zip(ids, mask, strict=True)]
            # Include unpadded and right-padded inputs. EOS resets the n-gram
            # history, so extra padding contributes no new rows after the reset.
            ids += [embedding.eos_token_id] * max(0, sequence_length - len(ids))
            try:
                embedding(torch.tensor([ids], dtype=torch.long), None)
            except RowsCollected:
                pass
    if not collector.rows:
        raise ValueError("PLE row caching requires tokenized calibration or quality data")
    return torch.cat(collector.rows).unique(sorted=True)


def prepare_ple_cache(checkpoint, config, datasets, directory, dtype, block_rows=32768):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for ple_index, layer_id in enumerate(config.ple_layer_ids):
        layer = layer_id - 1
        prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"
        ids = required_rows(checkpoint, config, layer, ple_index, datasets)
        filename = directory / f"layer-{layer:04d}.safetensors"
        manifest = filename.with_suffix(".json")
        names = [f"{prefix}.shard_{part}.weight" for part in range(config.split_ngram_parts)]
        identity = {
            "source": str(checkpoint.path),
            "config": digest(checkpoint.path / "config.json"),
            "index": digest(checkpoint.path / "model.safetensors.index.json"),
            "dtype": str(dtype),
            "ids": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "files": {
                name: [checkpoint.path.joinpath(name).stat().st_size, checkpoint.path.joinpath(name).stat().st_mtime_ns]
                for name in sorted({checkpoint.weight_map[name] for name in names})
            },
        }
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["identity"] != identity or digest(filename) != saved["sha256"]:
                raise ValueError("PLE cache identity/checksum mismatch")
        else:
            shard_rows, width = checkpoint.shape(names[0])
            values = torch.empty((len(ids), width), dtype=dtype)
            for part, name in enumerate(names):
                lower = int(torch.searchsorted(ids, part * shard_rows))
                upper = int(torch.searchsorted(ids, (part + 1) * shard_rows))
                selected = ids[lower:upper] - part * shard_rows
                with checkpoint.open(name) as handle:
                    source = handle.get_slice(name)
                    for start in range(0, shard_rows, block_rows):
                        stop = min(start + block_rows, shard_rows)
                        first = int(torch.searchsorted(selected, start))
                        last = int(torch.searchsorted(selected, stop))
                        # Clone contiguous blocks: read source sequentially once,
                        # rather than faulting millions of scattered HDD pages.
                        block = source[start:stop].clone()
                        if last > first:
                            values[lower + first : lower + last] = block[selected[first:last] - start].to(dtype)
                        del block
                print(f"PLE cache layer {layer}: scanned shard {part + 1}/{len(names)}", flush=True)
            temporary = filename.with_suffix(".tmp")
            save_file({"ids": ids, "values": values}, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(filename)
            sync_directory(directory)
            atomic_json(manifest, {"identity": identity, "sha256": digest(filename), "rows": len(ids)})
            del values
        result[prefix] = filename
    return result


class CachedEmbedding(MappedEmbedding):
    def __init__(self, checkpoint, prefix, rows, width, dtype, cache):
        super().__init__(checkpoint, prefix, rows, width, dtype)
        tensors = load_file(cache)
        self.ids = tensors["ids"]
        self.values = tensors["values"]
        if self.ids.dtype != torch.long or self.values.dtype != dtype or self.values.shape != (len(self.ids), width):
            raise ValueError("invalid PLE cache shape/dtype")
        if not len(self.ids) or (self.ids[1:] <= self.ids[:-1]).any():
            raise ValueError("PLE cache IDs must be nonempty, sorted and unique")

    def forward(self, indices):
        flat = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        positions = torch.searchsorted(self.ids, flat)
        if (positions >= len(self.ids)).any() or not torch.equal(self.ids[positions], flat):
            raise ValueError("PLE cache miss: inputs differ from prepared calibration/evaluation tokens")
        return self.values[positions].reshape(*indices.shape, self.weight.shape[1]).to(indices.device)
