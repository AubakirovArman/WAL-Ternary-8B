"""Decode-optimized batch-one kernels for the V77 direct-packed runtime.

The v1 kernels index one logical weight per Triton lane.  Four T3 weights or
eight WALB2 signs share one physical byte, so that formulation can issue
duplicate packed-byte loads and keeps unnecessarily wide temporary tensors.
These kernels make the packed byte the K-lane and unpack its symbols in a
small compile-time loop.  The checkpoint and hardware-cache ABI are unchanged.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _t3_sparse_k8_packed_gemv_kernel(
    x_ptr, t3_ptr, pos_ptr, sign_bits_ptr, alpha_ptr, beta_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
):
    """T3+sparse-k8 GEMV with one load per physical T3 byte."""
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ob = tl.arange(0, PACKED_K)
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for group in range(0, groups):
        byte = group * PACKED_K + ob
        packed = tl.load(
            t3_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        dense = tl.zeros((BLOCK_N,), tl.float32)
        for lane in range(0, 4):
            k = group * 128 + ob * 4 + lane
            value = tl.load(x_ptr + k).to(tl.float32)
            code = ((packed >> (lane * 2)) & 3) - 1
            dense += tl.sum(code.to(tl.float32) * value[None, :], axis=1)

        sign_byte = tl.load(
            sign_bits_ptr + on * groups + group, mask=mask_n, other=0,
        ).to(tl.int32)
        sparse = tl.zeros((BLOCK_N,), tl.float32)
        for slot in range(0, 8):
            index = (on * groups + group) * 8 + slot
            position = tl.load(
                pos_ptr + index, mask=mask_n, other=0,
            ).to(tl.int32)
            sign = (((sign_byte >> slot) & 1) * 2 - 1).to(tl.float32)
            sparse += tl.load(
                x_ptr + group * 128 + position, mask=mask_n, other=0.0,
            ).to(tl.float32) * sign

        alpha = tl.load(
            alpha_ptr + on * groups + group, mask=mask_n, other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_ptr + on * groups + group, mask=mask_n, other=0.0,
        ).to(tl.float32)
        acc += dense * alpha + sparse * beta
    tl.store(out_ptr + on, acc, mask=mask_n)


@triton.jit
def _binary_pair_v_packed_gemv_kernel(
    x_ptr, bits0_ptr, bits1_ptr,
    row0_ptr, row1_ptr, column0_ptr, column1_ptr,
    out0_ptr, out1_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Both WALB2 V paths with one load per physical sign byte."""
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ob = tl.arange(0, BLOCK_BYTES)
    mask_n = on < n_size
    acc0 = tl.zeros((BLOCK_N,), tl.float32)
    acc1 = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, bytes_per_row, BLOCK_BYTES):
        byte = start + ob
        mask_byte = byte < bytes_per_row
        packed0 = tl.load(
            bits0_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None] & mask_byte[None, :], other=0,
        ).to(tl.int32)
        packed1 = tl.load(
            bits1_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None] & mask_byte[None, :], other=0,
        ).to(tl.int32)
        for lane in range(0, 8):
            k = byte * 8 + lane
            mask_k = k < k_size
            value = tl.load(x_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
            column0 = tl.load(
                column0_ptr + k, mask=mask_k, other=0.0,
            ).to(tl.float32)
            column1 = tl.load(
                column1_ptr + k, mask=mask_k, other=0.0,
            ).to(tl.float32)
            code0 = (((packed0 >> lane) & 1) * 2 - 1).to(tl.float32)
            code1 = (((packed1 >> lane) & 1) * 2 - 1).to(tl.float32)
            acc0 += tl.sum(
                code0 * (value * column0)[None, :], axis=1,
            )
            acc1 += tl.sum(
                code1 * (value * column1)[None, :], axis=1,
            )
    row0 = tl.load(row0_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    row1 = tl.load(row1_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out0_ptr + on, acc0 * row0, mask=mask_n)
    tl.store(out1_ptr + on, acc1 * row1, mask=mask_n)


@triton.jit
def _binary_pair_u_packed_accumulate_gemv_kernel(
    hidden0_ptr, hidden1_ptr, bits0_ptr, bits1_ptr,
    row0_ptr, row1_ptr, output_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    """Both WALB2 U paths with packed-byte K lanes and fused accumulation."""
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ob = tl.arange(0, BLOCK_BYTES)
    mask_n = on < n_size
    acc0 = tl.zeros((BLOCK_N,), tl.float32)
    acc1 = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, bytes_per_row, BLOCK_BYTES):
        byte = start + ob
        mask_byte = byte < bytes_per_row
        packed0 = tl.load(
            bits0_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None] & mask_byte[None, :], other=0,
        ).to(tl.int32)
        packed1 = tl.load(
            bits1_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None] & mask_byte[None, :], other=0,
        ).to(tl.int32)
        for lane in range(0, 8):
            k = byte * 8 + lane
            mask_k = k < k_size
            hidden0 = tl.load(
                hidden0_ptr + k, mask=mask_k, other=0.0,
            ).to(tl.float32)
            hidden1 = tl.load(
                hidden1_ptr + k, mask=mask_k, other=0.0,
            ).to(tl.float32)
            code0 = (((packed0 >> lane) & 1) * 2 - 1).to(tl.float32)
            code1 = (((packed1 >> lane) & 1) * 2 - 1).to(tl.float32)
            acc0 += tl.sum(code0 * hidden0[None, :], axis=1)
            acc1 += tl.sum(code1 * hidden1[None, :], axis=1)
    row0 = tl.load(row0_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    row1 = tl.load(row1_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    prior = tl.load(output_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(output_ptr + on, prior + acc0 * row0 + acc1 * row1, mask=mask_n)


def direct_t3_sparse_gemv_v2(
    value: torch.Tensor,
    cache: tuple[torch.Tensor, ...],
    *,
    block_n: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Evaluate a batch-one T3+sparse-k8 matrix without changing the ABI."""
    t3, positions, sign_bits, alpha, beta, n_size, k_size = cache
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape[0] != 1 or flat.shape[1] != int(k_size):
        raise ValueError("v2 T3 kernel requires batch one with matching width")
    n_size, k_size = int(n_size), int(k_size)
    if k_size % 128:
        raise ValueError("v2 T3 kernel requires a multiple-of-128 input width")
    output = torch.empty((n_size,), device=value.device, dtype=torch.float32)
    _t3_sparse_k8_packed_gemv_kernel[(triton.cdiv(n_size, block_n),)](
        flat, t3, positions, sign_bits, alpha, beta, output,
        n_size, k_size, k_size // 128, k_size // 4,
        BLOCK_N=block_n, PACKED_K=32, num_warps=num_warps,
    )
    return output.reshape(*value.shape[:-1], n_size)


def direct_walb2_pair_gemv_v2(
    value: torch.Tensor,
    output: torch.Tensor,
    paths: tuple[dict[str, Any], ...],
    *,
    block_n: int = 32,
    block_bytes: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    """Evaluate the two-path WALB2 overlay using packed-byte K lanes."""
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    flat_output = output.reshape(-1, output.shape[-1]).contiguous()
    if flat.shape[0] != 1 or flat_output.shape[0] != 1 or len(paths) != 2:
        raise ValueError("v2 WALB2 kernel requires batch one and two paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1:
        raise ValueError("v2 WALB2 path dimensions differ")
    out_features, in_features, rank = dimensions0
    if in_features % 8 or rank % 8:
        raise ValueError("v2 WALB2 widths must be divisible by eight")
    if flat.shape[1] != in_features or flat_output.shape[1] != out_features:
        raise ValueError("v2 WALB2 activation shape mismatch")
    hidden0 = torch.empty((rank,), device=value.device, dtype=torch.float32)
    hidden1 = torch.empty((rank,), device=value.device, dtype=torch.float32)
    _binary_pair_v_packed_gemv_kernel[(triton.cdiv(rank, block_n),)](
        flat, path0["v"], path1["v"],
        path0["latent"], path1["latent"], path0["column"], path1["column"],
        hidden0, hidden1, rank, in_features, in_features // 8,
        BLOCK_N=block_n, BLOCK_BYTES=block_bytes, num_warps=num_warps,
    )
    _binary_pair_u_packed_accumulate_gemv_kernel[
        (triton.cdiv(out_features, block_n),)
    ](
        hidden0, hidden1, path0["u"], path1["u"], path0["row"], path1["row"],
        flat_output, out_features, rank, rank // 8,
        BLOCK_N=block_n, BLOCK_BYTES=block_bytes, num_warps=num_warps,
    )
    return flat_output.reshape_as(output)


def direct_wal_linear_v2(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
    *,
    block_n: int = 32,
    block_bytes: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    """Batch-one v2 fast path; callers retain v1 for prefill and fallbacks."""
    if value.reshape(-1, value.shape[-1]).shape[0] != 1:
        raise ValueError("direct_wal_linear_v2 is a decode-only path")
    result = direct_t3_sparse_gemv_v2(
        value, base_cache, block_n=block_n, num_warps=num_warps,
    )
    if not paths:
        return result
    if len(paths) != 2:
        raise ValueError("direct_wal_linear_v2 currently requires zero or two paths")
    return direct_walb2_pair_gemv_v2(
        value, result, paths, block_n=block_n,
        block_bytes=block_bytes, num_warps=num_warps,
    )
