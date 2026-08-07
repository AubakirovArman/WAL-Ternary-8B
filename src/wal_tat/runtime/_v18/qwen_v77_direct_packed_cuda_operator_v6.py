"""CUDA C++ execution of both packed T3+sparse-k8 and paired WALB2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

import qwen_v77_direct_packed_cuda_operator_v5 as operator_v5


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "experiments/cuda/wal_binary_pair_gemv_v1.cu"
_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="wal_binary_pair_gemv_r6",
            sources=[str(CUDA_SOURCE)],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    return _EXTENSION


def direct_walb2_pair_cuda(
    value: torch.Tensor,
    output: torch.Tensor,
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    if len(paths) != 2:
        raise ValueError("CUDA WALB2 requires exactly two paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1:
        raise ValueError("WALB2 path dimensions differ")
    rows, columns, rank = dimensions0
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    flat_output = output.reshape(-1, output.shape[-1]).contiguous()
    if flat.shape != (1, columns) or flat_output.shape != (1, rows):
        raise ValueError("CUDA WALB2 requires batch one with matching dimensions")
    hidden0, hidden1 = extension().binary_pair_v(
        flat.reshape(-1), path0["v"], path1["v"],
        path0["latent"], path1["latent"], path0["column"], path1["column"],
        rank, columns, 128,
    )
    result = extension().binary_pair_u_accumulate(
        hidden0, hidden1, path0["u"], path1["u"],
        path0["row"], path1["row"], flat_output.reshape(-1), rows, rank, 128,
    )
    return result.reshape_as(output)


def direct_wal_linear_v6(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    result = operator_v5.direct_t3_sparse_gemv_cuda(value, base_cache, threads=128)
    if not paths:
        return result
    return direct_walb2_pair_cuda(value, result, paths)
