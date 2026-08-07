"""Batch-one V18: exact paired activation loads for T3+k8 and WALB2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
T3_SOURCE = ROOT / "experiments/cuda/wal_t3_sparse_gemv_v1.cu"
WALB2_SOURCE = ROOT / "experiments/cuda/wal_binary_pair_gemv_v1.cu"
_T3_EXTENSION = None
_WALB2_EXTENSION = None


def t3_extension():
    global _T3_EXTENSION
    if _T3_EXTENSION is None:
        _T3_EXTENSION = load(
            name="wal_t3_sparse_gemv_v18",
            sources=[str(T3_SOURCE)],
            extra_cuda_cflags=["-O3", "-lineinfo", "-DWAL_VECTOR_ACTIVATION2=1"],
            verbose=False,
        )
    return _T3_EXTENSION


def walb2_extension():
    global _WALB2_EXTENSION
    if _WALB2_EXTENSION is None:
        _WALB2_EXTENSION = load(
            name="wal_binary_pair_gemv_v18",
            sources=[str(WALB2_SOURCE)],
            extra_cuda_cflags=["-O3", "-lineinfo", "-DWAL_VECTOR_ACTIVATION2=1"],
            verbose=False,
        )
    return _WALB2_EXTENSION


def direct_t3_sparse_v18(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    t3, positions, sign_bits, alpha, beta, rows, columns = base_cache
    rows, columns = int(rows), int(columns)
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape != (1, columns):
        raise ValueError("V18 requires batch one with matching input width")
    result = t3_extension().t3_sparse_gemv(
        flat.reshape(-1), t3, positions, sign_bits, alpha, beta,
        rows, columns, 128,
    )
    return result.reshape(*value.shape[:-1], rows)


def direct_wal_linear_v18(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    if not paths:
        return direct_t3_sparse_v18(value, base_cache)
    if len(paths) != 2:
        raise ValueError("V18 requires zero or two WALB2 paths")
    path0, path1 = paths
    dims0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dims1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dims0 != dims1:
        raise ValueError("V18 WALB2 path dimensions differ")
    rows, columns, rank = dims0
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape != (1, columns):
        raise ValueError("V18 requires batch one with matching input width")
    extension = walb2_extension()
    hidden0, hidden1 = extension.binary_pair_v(
        flat.reshape(-1), path0["v"], path1["v"],
        path0["latent"], path1["latent"], path0["column"], path1["column"],
        rank, columns, 128,
    )
    t3, positions, sign_bits, alpha, beta, base_rows, base_columns = base_cache
    if int(base_rows) != rows or int(base_columns) != columns:
        raise ValueError("V18 base and overlay dimensions differ")
    result = extension.t3_sparse_walb2_u(
        flat.reshape(-1), t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, path0["u"], path1["u"], path0["row"], path1["row"],
        rows, columns, rank, 128,
    )
    return result.reshape(*value.shape[:-1], rows)
