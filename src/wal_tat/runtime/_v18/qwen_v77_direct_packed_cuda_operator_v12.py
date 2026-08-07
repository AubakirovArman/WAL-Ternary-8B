"""V12 decode operator: V10 weight reuse with one-barrier batch reductions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "experiments/cuda/wal_weight_reuse_decode_v1.cu"
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="wal_weight_reuse_decode_v12",
            sources=[str(CUDA_SOURCE)],
            extra_cuda_cflags=[
                "-O3", "-lineinfo", "-DWAL_SINGLE_BARRIER_REDUCTION=1",
            ],
            verbose=False,
        )
    return _EXTENSION


def direct_wal_linear_v12(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    return _direct_wal_linear_with_extension(
        value, base_cache, paths, extension(),
    )


def _direct_wal_linear_with_extension(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
    ext,
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    batch, columns = flat.shape
    if not 2 <= batch <= 16:
        raise ValueError("V12 one-barrier path requires batch 2..16")
    t3, positions, sign_bits, alpha, beta, rows, base_columns = base_cache
    rows, base_columns = int(rows), int(base_columns)
    if columns != base_columns:
        raise ValueError("V12 base activation width mismatch")
    if not paths:
        output = ext.t3_sparse_reuse(
            flat, t3, positions, sign_bits, alpha, beta,
            rows, columns, batch,
        )
        return output.reshape(*value.shape[:-1], rows)
    if len(paths) != 2:
        raise ValueError("V12 requires zero or two WALB2 paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1 or dimensions0[:2] != (rows, columns):
        raise ValueError("V12 base/overlay dimensions differ")
    rank = dimensions0[2]
    hidden0, hidden1 = ext.binary_pair_v_reuse(
        flat, path0["v"], path1["v"], path0["latent"], path1["latent"],
        path0["column"], path1["column"], rank, columns, batch,
    )
    output = ext.t3_sparse_walb2_u_reuse(
        flat, t3, positions, sign_bits, alpha, beta, hidden0, hidden1,
        path0["u"], path1["u"], path0["row"], path1["row"],
        rows, columns, rank, batch,
    )
    return output.reshape(*value.shape[:-1], rows)
