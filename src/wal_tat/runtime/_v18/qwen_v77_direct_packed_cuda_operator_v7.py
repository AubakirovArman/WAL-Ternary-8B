"""Two-kernel corrected path: WALB2-V then fused T3+sparse-k8+WALB2-U."""
from __future__ import annotations

from typing import Any

import torch

import qwen_v77_direct_packed_cuda_operator_v5 as operator_v5
import qwen_v77_direct_packed_cuda_operator_v6 as operator_v6


def direct_wal_linear_v7(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    if not paths:
        return operator_v5.direct_t3_sparse_gemv_cuda(value, base_cache, threads=128)
    if len(paths) != 2:
        raise ValueError("v7 requires zero or two WALB2 paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1:
        raise ValueError("WALB2 path dimensions differ")
    rows, columns, rank = dimensions0
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape != (1, columns):
        raise ValueError("v7 requires batch one with matching input width")
    hidden0, hidden1 = operator_v6.extension().binary_pair_v(
        flat.reshape(-1), path0["v"], path1["v"],
        path0["latent"], path1["latent"], path0["column"], path1["column"],
        rank, columns, 128,
    )
    t3, positions, sign_bits, alpha, beta, base_rows, base_columns = base_cache
    if int(base_rows) != rows or int(base_columns) != columns:
        raise ValueError("base and overlay dimensions differ")
    result = operator_v6.extension().t3_sparse_walb2_u(
        flat.reshape(-1), t3, positions, sign_bits, alpha, beta,
        hidden0, hidden1, path0["u"], path1["u"], path0["row"], path1["row"],
        rows, columns, rank, 128,
    )
    return result.reshape(*value.shape[:-1], rows)
