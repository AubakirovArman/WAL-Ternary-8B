"""V17 exact gate: V12 reductions with paired BF16 activation loads."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

import qwen_v77_direct_packed_cuda_operator_v12 as v12


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "experiments/cuda/wal_weight_reuse_decode_v1.cu"
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="wal_weight_reuse_decode_v17",
            sources=[str(CUDA_SOURCE)],
            extra_cuda_cflags=[
                "-O3", "-lineinfo", "-DWAL_SINGLE_BARRIER_REDUCTION=1",
                "-DWAL_VECTOR_ACTIVATION2=1",
            ],
            verbose=False,
        )
    return _EXTENSION


def direct_wal_linear_v17(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    return v12._direct_wal_linear_with_extension(
        value, base_cache, paths, extension(),
    )
