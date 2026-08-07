"""CUDA C++ T3+sparse-k8 base with the accepted packed WALB2 overlay."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

import qwen_v77_direct_packed_triton_operator_v2 as operator_v2


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "experiments/cuda/wal_t3_sparse_gemv_v1.cu"
_EXTENSION = None


def extension():
    """Build once (or load the torch extension cache) and reuse thereafter."""
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="wal_t3_sparse_gemv_r5",
            sources=[str(CUDA_SOURCE)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXTENSION


def direct_t3_sparse_gemv_cuda(
    value: torch.Tensor,
    cache: tuple[torch.Tensor, ...],
    *,
    threads: int = 128,
) -> torch.Tensor:
    """Execute the canonical packed T3+sparse-k8 bytes without dense weights."""
    t3, positions, sign_bits, alpha, beta, rows, columns = cache
    rows, columns = int(rows), int(columns)
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape != (1, columns):
        raise ValueError("CUDA T3 kernel requires batch one with matching width")
    result = extension().t3_sparse_gemv(
        flat.reshape(-1), t3, positions, sign_bits, alpha, beta,
        rows, columns, threads,
    )
    return result.reshape(*value.shape[:-1], rows)


def direct_wal_linear_v5(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
    *,
    block_n: int,
    block_bytes: int,
    num_warps: int,
) -> torch.Tensor:
    """CUDA base plus the accepted two-path packed-byte WALB2 implementation."""
    result = direct_t3_sparse_gemv_cuda(value, base_cache, threads=128)
    if not paths:
        return result
    if len(paths) != 2:
        raise ValueError("v5 currently requires zero or two WALB2 paths")
    return operator_v2.direct_walb2_pair_gemv_v2(
        value, result, paths,
        block_n=block_n, block_bytes=block_bytes, num_warps=num_warps,
    )
