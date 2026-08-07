"""V10 decode operator: reuse each packed weight across the active batch."""
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
            name="wal_weight_reuse_decode_v10",
            sources=[str(CUDA_SOURCE)], extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXTENSION


def direct_wal_linear_v10(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    batch, columns = flat.shape
    if not 2 <= batch <= 16:
        raise ValueError("V10 weight-reuse path requires batch 2..16")
    t3, positions, sign_bits, alpha, beta, rows, base_columns = base_cache
    rows, base_columns = int(rows), int(base_columns)
    if columns != base_columns:
        raise ValueError("V10 base activation width mismatch")
    if not paths:
        output = extension().t3_sparse_reuse(
            flat, t3, positions, sign_bits, alpha, beta,
            rows, columns, batch,
        )
        return output.reshape(*value.shape[:-1], rows)
    if len(paths) != 2:
        raise ValueError("V10 requires zero or two WALB2 paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1 or dimensions0[:2] != (rows, columns):
        raise ValueError("V10 base/overlay dimensions differ")
    rank = dimensions0[2]
    hidden0, hidden1 = extension().binary_pair_v_reuse(
        flat, path0["v"], path1["v"], path0["latent"], path1["latent"],
        path0["column"], path1["column"], rank, columns, batch,
    )
    output = extension().t3_sparse_walb2_u_reuse(
        flat, t3, positions, sign_bits, alpha, beta, hidden0, hidden1,
        path0["u"], path1["u"], path0["row"], path1["row"],
        rows, columns, rank, batch,
    )
    return output.reshape(*value.shape[:-1], rows)
