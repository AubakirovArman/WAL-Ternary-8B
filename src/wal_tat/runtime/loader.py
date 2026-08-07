"""Model resolution and backend loading for the public CLI.

The validated V18 and native CPU V4 modules are bundled in the wheel. A source
checkout remains a development fallback. The portable correctness runtime is
loaded from the model artifact itself.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

from .platform import detect_platform


def resolve_checkpoint(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    if candidate.exists():
        raise ValueError(f"checkpoint is not a directory: {candidate}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "A Hugging Face model id requires `pip install 'wal-tat[runtime]'`"
        ) from error
    return Path(snapshot_download(str(value))).resolve()


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_runtime(checkpoint: Path):
    candidates = (
        checkpoint / "packed_runtime.py",
        checkpoint / "code" / "qwen_v77_packed_reference_runtime.py",
    )
    for path in candidates:
        if path.is_file():
            return _load_file(f"wal_checkpoint_runtime_{abs(hash(str(path)))}", path)
    raise ValueError(
        "model artifact has no packed_runtime.py; download WAL-Ternary-8B v0.2 "
        "or pass a complete release directory"
    )


def _source_root() -> Path | None:
    configured = os.environ.get("WAL_RUNTIME_SOURCE")
    candidates = [Path(configured)] if configured else []
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for root in candidates:
        if (root / "experiments" / "qwen_v77_direct_packed_triton_runtime_v18.py").is_file():
            return root.resolve()
    return None


def development_module(name: str):
    bundled = Path(__file__).resolve().parent / "_v18"
    if (bundled / f"{name}.py").is_file():
        value = str(bundled)
        if value not in sys.path:
            sys.path.insert(0, value)
        return __import__(name)
    root = _source_root()
    if root is None:
        return None
    experiments = root / "experiments"
    value = str(experiments)
    if value not in sys.path:
        sys.path.insert(0, value)
    return __import__(name)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    info = detect_platform()
    if info.cuda_available:
        return "cuda:0"
    # The current direct-packed Apple path is the native ARM64/NEON CPU
    # backend.  The portable MPS path is available explicitly with
    # ``--device mps`` but is not yet the fastest default.
    if info.mps_available and not info.apple_silicon:
        return "mps"
    return "cpu"


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accounting = manifest.get("accounting", {})
    release = manifest.get("release", {})
    runtime_files = [str(path.relative_to(checkpoint)) for path in (
        checkpoint / "packed_runtime.py",
        checkpoint / "code" / "qwen_v77_packed_reference_runtime.py",
    ) if path.is_file()]
    return {
        "checkpoint": str(checkpoint),
        "variant": manifest.get("variant"),
        "release": release,
        "unique_parameters": accounting.get("unique_parameters"),
        "maximum_whole_bpw": accounting.get("maximum_whole_bpw"),
        "packed_matrices": manifest.get("base", {}).get("matrix_count", 252),
        "overlays": len(manifest.get("overlays", [])),
        "runtime_files": runtime_files,
    }


def load_model(
    checkpoint: Path,
    *,
    device: str,
    hardware_cache: Path | None,
    backend: str = "auto",
    verify_hashes: bool = True,
):
    selected = backend
    if backend == "auto":
        selected = "cuda-v18" if device.startswith("cuda") else (
            "cpu-v4" if device == "cpu" else "portable"
        )
    if selected == "cuda-v18":
        module = development_module("qwen_v77_direct_packed_triton_runtime_v18")
        if module is None:
            if backend != "auto":
                raise RuntimeError(
                    "cuda-v18 backend is unavailable in this installation"
                )
            selected = "portable"
        else:
            if hardware_cache is None:
                raise ValueError("cuda-v18 requires --hardware-cache")
            model, report = module.load_direct_model(
                checkpoint, device=device, verify_hashes=verify_hashes,
                hardware_cache=hardware_cache,
                verify_hardware_cache_hashes=verify_hashes,
            )
            return model, report, selected
    if selected == "cpu-v4":
        module = development_module("qwen_v77_direct_packed_native_cpu_runtime_v1")
        if module is None:
            if backend != "auto":
                raise RuntimeError(
                    "cpu-v4 backend is unavailable in this installation"
                )
            selected = "portable"
        else:
            if hardware_cache is None:
                raise ValueError("cpu-v4 requires --hardware-cache")
            model, report = module.load_model(checkpoint, hardware_cache)
            return model, report, selected
    if selected != "portable":
        raise ValueError(f"unknown backend: {selected}")
    module = checkpoint_runtime(checkpoint)
    dtype = torch.float32 if device == "cpu" else (
        torch.float16 if device == "mps" else torch.bfloat16
    )
    model, report = module.load_packed_model(
        checkpoint, device=device, dtype=dtype, row_chunk=256,
        verify_hashes=verify_hashes,
    )
    return model, report, "portable"
