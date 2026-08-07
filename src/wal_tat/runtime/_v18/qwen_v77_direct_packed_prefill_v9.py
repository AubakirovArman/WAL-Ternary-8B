"""V9 experimental prefill: unpack one operator, use tensor-core GEMM, discard it.

The canonical and persistent representation remains T3+sparse-k8+WALB2.  For
large-M prefill, repeatedly decoding the same packed byte in every M tile is
wasteful.  V9 expands only the currently executing operator into temporary
workspace, applies cuBLAS GEMMs, and immediately releases the dense views.
No complete Transformer body, INT8 body, or effective weight tree is cached.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_t3_scaled_kernel(
    t3_ptr, alpha_ptr, weight_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    groups: tl.constexpr, bytes_per_row: tl.constexpr,
    BLOCK_N: tl.constexpr, PACKED_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    group = tl.program_id(1)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ob = tl.arange(0, PACKED_K)
    mask_n = on < n_size
    packed = tl.load(
        t3_ptr + on[:, None] * bytes_per_row + group * PACKED_K + ob[None, :],
        mask=mask_n[:, None], other=0,
    ).to(tl.int32)
    scale = tl.load(
        alpha_ptr + on * groups + group, mask=mask_n, other=0.0,
    ).to(tl.float32)
    for lane in range(0, 4):
        k = group * 128 + ob * 4 + lane
        code = ((packed >> (lane * 2)) & 3) - 1
        tl.store(
            weight_ptr + on[:, None] * k_size + k[None, :],
            code.to(tl.float32) * scale[:, None], mask=mask_n[:, None],
        )


@triton.jit
def _add_sparse_k8_kernel(
    pos_ptr, sign_ptr, beta_ptr, weight_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    group = tl.program_id(1)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    slot = tl.arange(0, 8)
    mask_n = on < n_size
    record = (on * groups + group)[:, None]
    position = tl.load(
        pos_ptr + record * 8 + slot[None, :], mask=mask_n[:, None], other=0,
    ).to(tl.int32)
    sign_byte = tl.load(
        sign_ptr + on * groups + group, mask=mask_n, other=0,
    ).to(tl.int32)
    sign = (((sign_byte[:, None] >> slot[None, :]) & 1) * 2 - 1).to(tl.float32)
    beta = tl.load(
        beta_ptr + on * groups + group, mask=mask_n, other=0.0,
    ).to(tl.float32)
    offset = on[:, None] * k_size + group * 128 + position
    prior = tl.load(weight_ptr + offset, mask=mask_n[:, None], other=0.0).to(tl.float32)
    tl.store(
        weight_ptr + offset, prior + beta[:, None] * sign,
        mask=mask_n[:, None],
    )


@triton.jit
def _unpack_binary_fp32_kernel(
    bits_ptr, column_ptr, weight_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    bytes_per_row: tl.constexpr, has_column: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_BYTES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    byte = pid_b * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    mask_n = on < n_size
    mask_b = byte < bytes_per_row
    packed = tl.load(
        bits_ptr + on[:, None] * bytes_per_row + byte[None, :],
        mask=mask_n[:, None] & mask_b[None, :], other=0,
    ).to(tl.int32)
    for lane in range(0, 8):
        k = byte * 8 + lane
        mask = mask_n[:, None] & (k[None, :] < k_size)
        code = (((packed >> lane) & 1) * 2 - 1).to(tl.float32)
        if has_column:
            column = tl.load(
                column_ptr + k, mask=k < k_size, other=0.0,
            ).to(tl.float32)
            code *= column[None, :]
        tl.store(
            weight_ptr + on[:, None] * k_size + k[None, :], code, mask=mask,
        )


def unpack_t3_sparse_bf16(
    cache: tuple[torch.Tensor, ...], device: torch.device,
) -> torch.Tensor:
    t3, positions, sign_bits, alpha, beta, rows, columns = cache
    rows, columns = int(rows), int(columns)
    groups = columns // 128
    weight = torch.empty((rows, columns), device=device, dtype=torch.bfloat16)
    block_n = 32
    grid = (triton.cdiv(rows, block_n), groups)
    _unpack_t3_scaled_kernel[grid](
        t3, alpha, weight, rows, columns, groups, columns // 4,
        BLOCK_N=block_n, PACKED_K=32, num_warps=8,
    )
    _add_sparse_k8_kernel[grid](
        positions, sign_bits, beta, weight, rows, columns, groups,
        BLOCK_N=block_n, num_warps=4,
    )
    return weight


def unpack_binary_fp32(
    bits: torch.Tensor, rows: int, columns: int,
    column: torch.Tensor | None,
) -> torch.Tensor:
    weight = torch.empty((rows, columns), device=bits.device, dtype=torch.float32)
    block_n, block_bytes = 16, 64
    grid = (
        triton.cdiv(rows, block_n), triton.cdiv(columns // 8, block_bytes),
    )
    dummy = bits
    _unpack_binary_fp32_kernel[grid](
        bits, dummy if column is None else column, weight,
        rows, columns, columns // 8, has_column=column is not None,
        BLOCK_N=block_n, BLOCK_BYTES=block_bytes, num_warps=8,
    )
    return weight


@torch.no_grad()
def direct_wal_linear_prefill_v9(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
    *, minimum_m: int = 17,
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    rows, columns = int(base_cache[-2]), int(base_cache[-1])
    if flat.shape[0] < minimum_m or flat.shape[1] != columns:
        raise ValueError(
            f"V9 prefill requires M>={minimum_m} and a matching input width"
        )

    base_weight = unpack_t3_sparse_bf16(base_cache, value.device)
    result = torch.mm(flat, base_weight.transpose(0, 1), out_dtype=torch.float32)
    del base_weight

    if paths:
        if len(paths) != 2:
            raise ValueError("V9 expects zero or two WALB2 paths")
        flat_fp32 = flat.float()
        for path in paths:
            if (int(path["out"]), int(path["in"])) != (rows, columns):
                raise ValueError("V9 base/overlay dimensions differ")
            rank = int(path["rank"])
            v_weight = unpack_binary_fp32(
                path["v"], rank, columns, path["column"],
            )
            hidden = torch.mm(flat_fp32, v_weight.transpose(0, 1))
            hidden.mul_(path["latent"].float().unsqueeze(0))
            del v_weight
            u_weight = unpack_binary_fp32(path["u"], rows, rank, None)
            correction = torch.mm(hidden, u_weight.transpose(0, 1))
            correction.mul_(path["row"].float().unsqueeze(0))
            result.add_(correction)
            del hidden, u_weight, correction
        del flat_fp32
    return result.reshape(*value.shape[:-1], rows)
