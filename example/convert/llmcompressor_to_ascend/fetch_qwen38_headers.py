"""Fetch pinned Qwen3.8-Flash-Next metadata and safetensors headers, without weights."""

import argparse
import hashlib
import json
import re
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from plan_qwen38 import storage


def read_range(url, start, end, file_size):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{file_size}":
            raise ValueError("server did not honor the exact byte range; refusing full weight download")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError("range payload length mismatch")
        return data


def validate_header(tensors, header_size, file_size):
    spans = []
    for tensor in tensors.values():
        storage(tensor)
        spans.append(tuple(tensor["data_offsets"]))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("overlapping or noncontiguous tensor payload")
        cursor = end
    if cursor + header_size + 8 != file_size:
        raise ValueError("header inventory does not cover the entire shard")


def fetch_shard(repo_id, revision, file, cache):
    name = file["name"]
    destination = cache / (name + ".json")
    if destination.exists():
        record = json.loads(destination.read_text())
    else:
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}"
        size = struct.unpack("<Q", read_range(url, 0, 7, file["bytes"]))[0]
        if not 0 < size <= min(16 * 1024**2, file["bytes"] - 8):
            raise ValueError("invalid safetensors header size")
        raw = read_range(url, 8, size + 7, file["bytes"])
        header = json.loads(raw)
        record = {
            "header_bytes": size,
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "tensors": {k: v for k, v in header.items() if k != "__metadata__"},
        }
    validate_header(record["tensors"], record["header_bytes"], file["bytes"])
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return name, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Exact 40-character Hugging Face commit SHA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an exact commit SHA")
    repo_id = "Qwen/Qwen3.8-Flash-Next"
    manifest_path = args.output / "source_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["repo_id"] != repo_id or previous["revision"] != args.revision:
            parser.error("output belongs to another source revision")
    elif args.output.exists() and any(args.output.iterdir()):
        parser.error("existing output has no source manifest")
    info = HfApi().model_info(repo_id, revision=args.revision, files_metadata=True)
    if info.sha != args.revision:
        raise ValueError("resolved revision mismatch")
    files = [{"name": f.rfilename, "bytes": f.size, "sha256": f.lfs.sha256 if f.lfs else None} for f in info.siblings]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"repo_id": repo_id, "revision": info.sha, "files": files}, indent=2) + "\n")
    for name in ("config.json", "model.safetensors.index.json", "README.md"):
        hf_hub_download(repo_id, name, revision=info.sha, local_dir=args.output / "metadata")
    index = json.loads((args.output / "metadata/model.safetensors.index.json").read_text())
    cache = args.output / "headers"
    cache.mkdir(exist_ok=True)
    shards = [f for f in files if f["name"].endswith(".safetensors")]
    if any(Path(f["name"]).name != f["name"] for f in shards):
        raise ValueError("expected flat checkpoint shard names")
    tensors = {}
    header_bytes = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = pool.map(lambda file: fetch_shard(repo_id, info.sha, file, cache), shards)
        for number, (file, record) in enumerate(results, 1):
            header_bytes += record["header_bytes"] + 8
            for name, tensor in record["tensors"].items():
                if name in tensors or index["weight_map"].get(name) != file:
                    raise ValueError("duplicate tensor or shard index mismatch")
                tensors[name] = {"file": file, **tensor}
            if number % 20 == 0 or number == len(shards):
                print(f"Headers: {number}/{len(shards)}", flush=True)
    total = sum(storage(t)[1] for t in tensors.values())
    if set(tensors) != set(index["weight_map"]) or total != index["metadata"]["total_size"]:
        raise ValueError("incomplete checkpoint inventory")
    result = {
        "repo_id": repo_id,
        "revision": info.sha,
        "header_bytes_read": header_bytes,
        "source_tensor_bytes": total,
        "tensors": tensors,
        "weight_payloads_downloaded": False,
        "weight_sha256_verified": False,
    }
    (args.output / "tensor_headers.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Inventoried {len(tensors)} tensors using {header_bytes:,} header bytes")


if __name__ == "__main__":
    main()
