"""Durable stage boundaries for the pinned LLMC sequential GPTQ pipeline.

Weights are saved in their exact floating GPTQ representation, with scales and
zero points. A committed boundary also owns the propagated activation cache and
RNG state. Temporary/incomplete stages are never considered resumable.
"""

import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
from importlib.metadata import version
from pathlib import Path

import compressed_tensors
import llmcompressor
import torch
from compressed_tensors.offload import disable_offloading, set_onload_device, update_offload_parameter
from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache, IntermediateValue
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.pipelines.sequential.pipeline import _get_batches
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets
from safetensors import safe_open
from safetensors.torch import save_file

from .checkpoint import EXPERT_WEIGHT


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def fingerprint(args, rows, quality_rows):
    """Bind resume to source identity, code, recipe, tokens and dependencies.

    Source payload hashes come from the verified download provenance when
    available. Current size/mtime are also recorded; this is not a full rehash.
    """
    source = args.model_path.resolve()
    index = source / "model.safetensors.index.json"
    filenames = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    root = Path(__file__).parent
    implementation = {path.name: digest(path) for path in sorted(root.glob("*.py"))}
    for filename in ("quantize_qwen38.py", "quantize_w8a8.py"):
        implementation[filename] = digest(root.parent / filename)
    for package in (llmcompressor, compressed_tensors):
        package_root = Path(package.__file__).parent
        implementation.update(
            {
                f"{package.__name__}/{path.relative_to(package_root)}": digest(path)
                for path in sorted(package_root.rglob("*.py"))
            }
        )
    return {
        "format": 1,
        "source": str(source),
        "source_files": {
            name: [source.joinpath(name).stat().st_size, source.joinpath(name).stat().st_mtime_ns] for name in filenames
        },
        "metadata": {
            name: digest(source / name)
            for name in ("config.json", "model.safetensors.index.json", "download_provenance.json")
            if (source / name).exists()
        },
        "code": implementation,
        "versions": {name: version(name) for name in ("torch", "transformers", "llmcompressor", "compressed-tensors")},
        "method": args.method,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "rows": rows,
        "quality_rows": quality_rows,
        "recipe": "GPTQ W8A8 routed experts; all experts; sequential propagate_error=True",
    }


def encode_cache(value):
    # Tagged plain containers keep torch.load(weights_only=True) sufficient.
    if isinstance(value, IntermediateValue):
        return ("intermediate", encode_cache(value.value), str(value.device) if value.device else None)
    if isinstance(value, torch.Tensor):
        return ("tensor", value.detach().cpu())
    if isinstance(value, dict):
        return ("dict", {key: encode_cache(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return ("tuple" if isinstance(value, tuple) else "list", [encode_cache(item) for item in value])
    if value is None or isinstance(value, (str, int, float, bool)):
        return ("scalar", value)
    raise TypeError(f"unsupported activation cache value: {type(value)}")


def decode_cache(value):
    kind, payload, *rest = value
    if kind == "intermediate":
        return IntermediateValue(decode_cache(payload), torch.device(rest[0]) if rest[0] else None)
    if kind in ("scalar", "tensor"):
        return payload
    if kind == "dict":
        return {key: decode_cache(item) for key, item in payload.items()}
    if kind in ("list", "tuple"):
        result = [decode_cache(item) for item in payload]
        return tuple(result) if kind == "tuple" else result
    raise ValueError(f"unknown cache tag: {kind}")


def graph_signature(subgraph):
    """FX subgraph construction uses sets; canonicalize placeholder/output order."""

    def encode(value):
        if isinstance(value, torch.fx.Node):
            return {"node": value.name}
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        return repr(value)

    return {
        "inputs": sorted(subgraph.input_names),
        "consumed": sorted(subgraph.consumed_names),
        "nodes": [
            {
                "name": node.name,
                "op": node.op,
                "target": encode(node.target),
                "args": encode(node.args),
                "kwargs": encode(node.kwargs),
            }
            for node in sorted(subgraph.graph.nodes, key=lambda node: node.name)
        ],
    }


class StageStore:
    def __init__(self, directory, identity, resume=False):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = (self.path / "lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = self.path / "progress.json"
        if resume:
            self.progress = json.loads(manifest.read_text())
            if self.progress["identity"] != identity:
                raise ValueError("resume fingerprint mismatch: source, code, settings or token data changed")
        else:
            if any(path.name != "lock" for path in self.path.iterdir()):
                raise ValueError("checkpoint directory must be empty unless --resume is supplied")
            self.progress = {"identity": identity, "stages": [], "graphs": None}
            atomic_json(manifest, self.progress)

    def close(self):
        self.lock.close()

    def bind_graphs(self, subgraphs):
        graphs = [graph_signature(graph) for graph in subgraphs]
        if self.progress["graphs"] is None:
            self.progress["graphs"] = graphs
            atomic_json(self.path / "progress.json", self.progress)
        elif graphs != self.progress["graphs"]:
            raise ValueError("resume graph mismatch")

    def checked_file(self, stage, name):
        if Path(name).name != name:
            raise ValueError("unsafe checkpoint filename")
        path = self.path / stage["directory"] / name
        if digest(path) != stage["files"][name]:
            raise ValueError(f"checkpoint checksum mismatch: {path}")
        return path

    def restore(self, model):
        for stage in self.progress["stages"]:
            for name in stage["weight_files"]:
                with safe_open(self.checked_file(stage, name), framework="pt", device="cpu") as handle:
                    for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                        module_name, parameter = key.rsplit(".", 1)
                        module = model.get_submodule(module_name)
                        if not EXPERT_WEIGHT.fullmatch(module_name + ".weight"):
                            raise ValueError(f"unexpected restored module: {module_name}")
                        update_offload_parameter(module, parameter, handle.get_tensor(key))
        if not self.progress["stages"]:
            return None
        stage = self.progress["stages"][-1]
        state = torch.load(self.checked_file(stage, "activations.pt"), map_location="cpu", weights_only=True)
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return IntermediatesCache(decode_cache(state["cache"]), torch.device("cpu"))

    def save(self, model, subgraph, activations, index):
        if index != len(self.progress["stages"]):
            raise ValueError("stage checkpoints must be committed in order")
        directory = f"stage-{index:04d}"
        temporary = self.path / (directory + ".tmp")
        # Only uncommitted, store-owned stage directories may be replaced.
        for path in (temporary, self.path / directory):
            if path.exists():
                shutil.rmtree(path)
        temporary.mkdir()
        relevant = subgraph.submodules(model, recurse=True)
        pending, weight_files = {}, []
        size = 0

        def flush():
            nonlocal pending, size
            if pending:
                name = f"weights-{len(weight_files):04d}.safetensors"
                save_file(pending, temporary / name)
                weight_files.append(name)
                pending, size = {}, 0

        for name, module in model.named_modules():
            if module not in relevant or not EXPERT_WEIGHT.fullmatch(name + ".weight"):
                continue
            state = dict(module.named_parameters(recurse=False))
            state.update(module.named_buffers(recurse=False))
            for key, value in state.items():
                value = value.detach().to("cpu", copy=True).contiguous()
                if size + value.nbytes > 1024**3:
                    flush()
                pending[f"{name}.{key}"] = value
                size += value.nbytes
        flush()
        state = {
            "cache": encode_cache(activations.batch_intermediates),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, temporary / "activations.pt")
        files = {}
        for path in temporary.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            files[path.name] = digest(path)
        sync_directory(temporary)
        os.replace(temporary, self.path / directory)
        sync_directory(self.path)
        stage = {"directory": directory, "files": files, "weight_files": weight_files}
        previous = self.progress["stages"][-1] if self.progress["stages"] else None
        self.progress["stages"].append(stage)
        atomic_json(self.path / "progress.json", self.progress)
        # Weights remain immutable; only the latest activation boundary is needed.
        if previous:
            old_cache = self.path / previous["directory"] / "activations.pt"
            old_cache.unlink(missing_ok=True)
            sync_directory(old_cache.parent)
        print(f"Durable checkpoint: completed stage {index + 1}/{len(self.progress['graphs'])}", flush=True)


@CalibrationPipeline.register("qwen38_resumable")
class ResumableSequentialPipeline(CalibrationPipeline):
    """Pinned sequential algorithm, with atomic writes after propagation.

    This intentionally supports only this adapter's GPTQ policy: dynamic input
    quantization, one decoder per stage, CPU activation cache, error propagation.
    """

    @staticmethod
    def __call__(model, dataloader, dataset_args):
        if not dataset_args.propagate_error or dataset_args.sequential_offload_device != "cpu":
            raise ValueError("resumable Qwen38 GPTQ requires error propagation and CPU activation storage")
        store = model.stage_store
        session = active_session()
        device = get_main_device()
        set_onload_device(model, device)
        subgraphs = trace_subgraphs(
            model,
            next(iter(dataloader)),
            infer_sequential_targets(model, dataset_args.sequential_targets),
            dataset_args.tracing_ignore,
            dataset_args.sequential_targets_per_subgraph,
        )
        store.bind_graphs(subgraphs)
        LifecycleCallbacks.calibration_epoch_start()
        activations = store.restore(model)
        start = len(store.progress["stages"])
        print(f"Resumable GPTQ: starting stage {start + 1}/{len(subgraphs)}", flush=True)
        session.state.loss_masks = None
        session.state.sequential_prefetch = False
        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            stack.enter_context(DisableQuantization(model))
            if activations is None:
                activations = IntermediatesCache.from_dataloader(dataloader, device, torch.device("cpu"))
            for index in range(start, len(subgraphs)):
                graph = subgraphs[index]
                with disable_offloading():
                    for batch, inputs in _get_batches(
                        activations, len(dataloader), graph.input_names, f"({index + 1}/{len(subgraphs)}): Calibrating"
                    ):
                        session.state.current_batch_idx = batch
                        graph.forward(model, **inputs)
                    LifecycleCallbacks.sequential_epoch_end(list(graph.submodules(model)))
                    with HooksMixin.disable_hooks():
                        for batch, inputs in _get_batches(
                            activations,
                            len(dataloader),
                            graph.input_names,
                            f"({index + 1}/{len(subgraphs)}): Propagating",
                        ):
                            outputs = graph.forward(model, **inputs)
                            if index < len(subgraphs) - 1:
                                activations.update(batch, outputs)
                                activations.delete(batch, graph.consumed_names)
                store.save(model, graph, activations, index)
            LifecycleCallbacks.calibration_epoch_end()
