"""Direct-packed Triton proof for one complete V77 WAL linear operator.

This prototype deliberately keeps no logical INT8 weight matrices.  The T3
base is held as four 2-bit symbols per byte, sparse-k8 positions are byte
indices with one packed sign byte per group, and WALB2 U/V factors stay as
one-bit signs.  Triton expands symbols only into registers while evaluating
the matrix product.

The archival checkpoint remains the canonical radix-3 representation.  A
hardware cache is built a row chunk at a time; it is smaller than the current
logical-INT8 cache and is the proposed v1 compute ABI for CPU/CUDA kernels.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import torch
import triton
import triton.language as tl

import qwen_v77_packed_reference_runtime as ref
import qwen_v77_packed_triton_runtime as logical


DEFAULT_CHECKPOINT = ref.DEFAULT_CHECKPOINT
WALB2_HEADER = struct.Struct("<8sIIIIIQQQQI")
HARDWARE_BASE_HEADER = struct.Struct("<8sIIIIQQQQQ")
HARDWARE_BASE_MAGIC = b"WALHW001"


@triton.jit
def _direct_t3_sparse_k8_kernel(
    x_ptr, t3_ptr, pos_ptr, sign_bits_ptr, alpha_ptr, beta_ptr, out_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    groups: tl.constexpr, bytes_per_row: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_m = om < m_size
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for group in range(0, groups):
        k = group * BLOCK_K + ok
        xv = tl.load(
            x_ptr + om[:, None] * k_size + k[None, :],
            mask=mask_m[:, None], other=0.0,
        ).to(tl.float32)
        packed = tl.load(
            t3_ptr + on[:, None] * bytes_per_row + k[None, :] // 4,
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        shift = (k % 4) * 2
        code = ((packed >> shift[None, :]) & 3) - 1
        dot = tl.dot(xv, tl.trans(code.to(tl.float32)), input_precision="tf32")
        a = tl.load(alpha_ptr + on * groups + group,
                    mask=mask_n, other=0.0).to(tl.float32)
        sparse = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        sign_byte = tl.load(
            sign_bits_ptr + on * groups + group,
            mask=mask_n, other=0,
        ).to(tl.int32)
        for slot in range(0, 8):
            index = (on * groups + group) * 8 + slot
            p = tl.load(pos_ptr + index, mask=mask_n, other=0).to(tl.int32)
            sign = (((sign_byte >> slot) & 1) * 2 - 1).to(tl.float32)
            xs = tl.load(
                x_ptr + om[:, None] * k_size + group * BLOCK_K + p[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0.0,
            ).to(tl.float32)
            sparse += xs * sign[None, :]
        b = tl.load(beta_ptr + on * groups + group,
                    mask=mask_n, other=0.0).to(tl.float32)
        acc += dot * a[None, :] + sparse * b[None, :]
    tl.store(
        out_ptr + om[:, None] * n_size + on[None, :], acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _direct_t3_sparse_k8_gemv_kernel(
    x_ptr, t3_ptr, pos_ptr, sign_bits_ptr, alpha_ptr, beta_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for group in range(0, groups):
        k = group * BLOCK_K + ok
        xv = tl.load(x_ptr + k).to(tl.float32)
        packed = tl.load(
            t3_ptr + on[:, None] * bytes_per_row + k[None, :] // 4,
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        code = ((packed >> (((k % 4) * 2)[None, :])) & 3) - 1
        dense = tl.sum(code.to(tl.float32) * xv[None, :], axis=1)
        sparse = tl.zeros((BLOCK_N,), tl.float32)
        sign_byte = tl.load(
            sign_bits_ptr + on * groups + group, mask=mask_n, other=0
        ).to(tl.int32)
        for slot in range(0, 8):
            index = (on * groups + group) * 8 + slot
            position = tl.load(pos_ptr + index, mask=mask_n, other=0).to(tl.int32)
            sign = (((sign_byte >> slot) & 1) * 2 - 1).to(tl.float32)
            sparse += tl.load(x_ptr + group * BLOCK_K + position,
                              mask=mask_n, other=0.0).to(tl.float32) * sign
        alpha = tl.load(alpha_ptr + on * groups + group,
                        mask=mask_n, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + on * groups + group,
                       mask=mask_n, other=0.0).to(tl.float32)
        acc += dense * alpha + sparse * beta
    tl.store(out_ptr + on, acc, mask=mask_n)


@triton.jit
def _direct_binary_sign_kernel(
    x_ptr, bits_ptr, row_ptr, column_ptr, out_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    HAS_COLUMN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_m = om < m_size
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, k_size, BLOCK_K):
        k = start + ok
        mask_k = k < k_size
        xv = tl.load(
            x_ptr + om[:, None] * k_size + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        if HAS_COLUMN:
            column = tl.load(column_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
            xv *= column[None, :]
        bit_index = on[:, None] * k_size + k[None, :]
        packed = tl.load(
            bits_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        code = (((packed >> (bit_index % 8)) & 1) * 2 - 1).to(tl.float32)
        acc += tl.dot(xv, tl.trans(code), input_precision="tf32")
    row = tl.load(row_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    acc *= row[None, :]
    tl.store(
        out_ptr + om[:, None] * n_size + on[None, :], acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _direct_binary_sign_gemv_kernel(
    x_ptr, bits_ptr, row_ptr, column_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, HAS_COLUMN: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, k_size, BLOCK_K):
        k = start + ok
        mask_k = k < k_size
        value = tl.load(x_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        if HAS_COLUMN:
            column = tl.load(column_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
            value *= column
        bit_index = on[:, None] * k_size + k[None, :]
        packed = tl.load(
            bits_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        code = (((packed >> (bit_index % 8)) & 1) * 2 - 1).to(tl.float32)
        acc += tl.sum(code * value[None, :], axis=1)
    row = tl.load(row_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out_ptr + on, acc * row, mask=mask_n)


@triton.jit
def _direct_binary_pair_v_gemv_kernel(
    x_ptr, bits0_ptr, bits1_ptr,
    row0_ptr, row1_ptr, column0_ptr, column1_ptr,
    out0_ptr, out1_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Evaluate both WALB2 V paths while reading the input activation once."""
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_n = on < n_size
    acc0 = tl.zeros((BLOCK_N,), tl.float32)
    acc1 = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, k_size, BLOCK_K):
        k = start + ok
        mask_k = k < k_size
        value = tl.load(x_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        column0 = tl.load(column0_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        column1 = tl.load(column1_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        bit_index = on[:, None] * k_size + k[None, :]
        packed0 = tl.load(
            bits0_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        packed1 = tl.load(
            bits1_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        shift = bit_index % 8
        code0 = (((packed0 >> shift) & 1) * 2 - 1).to(tl.float32)
        code1 = (((packed1 >> shift) & 1) * 2 - 1).to(tl.float32)
        acc0 += tl.sum(code0 * (value * column0)[None, :], axis=1)
        acc1 += tl.sum(code1 * (value * column1)[None, :], axis=1)
    row0 = tl.load(row0_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    row1 = tl.load(row1_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out0_ptr + on, acc0 * row0, mask=mask_n)
    tl.store(out1_ptr + on, acc1 * row1, mask=mask_n)


@triton.jit
def _direct_binary_pair_u_accumulate_gemv_kernel(
    hidden0_ptr, hidden1_ptr, bits0_ptr, bits1_ptr,
    row0_ptr, row1_ptr, output_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Evaluate both WALB2 U paths and add them directly to the base result."""
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_n = on < n_size
    acc0 = tl.zeros((BLOCK_N,), tl.float32)
    acc1 = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, k_size, BLOCK_K):
        k = start + ok
        mask_k = k < k_size
        hidden0 = tl.load(hidden0_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        hidden1 = tl.load(hidden1_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
        bit_index = on[:, None] * k_size + k[None, :]
        packed0 = tl.load(
            bits0_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        packed1 = tl.load(
            bits1_ptr + bit_index // 8,
            mask=mask_n[:, None] & mask_k[None, :], other=0,
        ).to(tl.int32)
        shift = bit_index % 8
        code0 = (((packed0 >> shift) & 1) * 2 - 1).to(tl.float32)
        code1 = (((packed1 >> shift) & 1) * 2 - 1).to(tl.float32)
        acc0 += tl.sum(code0 * hidden0[None, :], axis=1)
        acc1 += tl.sum(code1 * hidden1[None, :], axis=1)
    row0 = tl.load(row0_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    row1 = tl.load(row1_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    prior = tl.load(output_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(output_ptr + on, prior + acc0 * row0 + acc1 * row1, mask=mask_n)


def _launch_shape(m_size: int) -> tuple[int, int, int]:
    return (32, 32, 4) if m_size >= 256 else (16, 64, 8)


def direct_t3_sparse_mm(value: torch.Tensor, cache: tuple[torch.Tensor, ...]) -> torch.Tensor:
    t3, positions, sign_bits, alpha, beta, n_size, k_size = cache
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    n_size = int(n_size)
    k_size = int(k_size)
    groups = k_size // 128
    output = torch.empty((flat.shape[0], n_size), device=value.device, dtype=torch.float32)
    if flat.shape[0] == 1:
        block_n = 64
        _direct_t3_sparse_k8_gemv_kernel[(triton.cdiv(n_size, block_n),)](
            flat, t3, positions, sign_bits, alpha, beta, output,
            n_size, k_size, groups, k_size // 4,
            BLOCK_N=block_n, BLOCK_K=128, num_warps=4,
        )
        return output.reshape(*value.shape[:-1], n_size)
    block_m, block_n, warps = _launch_shape(flat.shape[0])
    grid = (triton.cdiv(flat.shape[0], block_m), triton.cdiv(n_size, block_n))
    _direct_t3_sparse_k8_kernel[grid](
        flat, t3, positions, sign_bits, alpha, beta, output,
        flat.shape[0], n_size, k_size, groups, k_size // 4,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
    )
    return output.reshape(*value.shape[:-1], n_size)


def direct_binary_mm(
    value: torch.Tensor,
    bits: torch.Tensor,
    rows: int,
    columns: int,
    row_scale: torch.Tensor,
    column_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    output = torch.empty((flat.shape[0], rows), device=value.device, dtype=torch.float32)
    if flat.shape[0] == 1:
        block_n = 64
        _direct_binary_sign_gemv_kernel[(triton.cdiv(rows, block_n),)](
            flat, bits, row_scale, row_scale if column_scale is None else column_scale,
            output, rows, columns, HAS_COLUMN=column_scale is not None,
            BLOCK_N=block_n, BLOCK_K=128, num_warps=4,
        )
        return output.reshape(*value.shape[:-1], rows)
    block_m, block_n, warps = _launch_shape(flat.shape[0])
    grid = (triton.cdiv(flat.shape[0], block_m), triton.cdiv(rows, block_n))
    dummy = row_scale
    _direct_binary_sign_kernel[grid](
        flat, bits, row_scale, dummy if column_scale is None else column_scale, output,
        flat.shape[0], rows, columns, HAS_COLUMN=column_scale is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
    )
    return output.reshape(*value.shape[:-1], rows)


def direct_walb2_pair_gemv(
    value: torch.Tensor,
    output: torch.Tensor,
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    """Fused two-path WALB2 decode; no persistent dense weight is created."""
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    flat_output = output.reshape(-1, output.shape[-1]).contiguous()
    if flat.shape[0] != 1 or flat_output.shape[0] != 1 or len(paths) != 2:
        raise ValueError("paired WALB2 GEMV requires batch one and exactly two paths")
    path0, path1 = paths
    dimensions0 = (int(path0["out"]), int(path0["in"]), int(path0["rank"]))
    dimensions1 = (int(path1["out"]), int(path1["in"]), int(path1["rank"]))
    if dimensions0 != dimensions1:
        raise ValueError("paired WALB2 paths have incompatible dimensions")
    out_features, in_features, rank = dimensions0
    if flat.shape[1] != in_features or flat_output.shape[1] != out_features:
        raise ValueError("paired WALB2 activation shape mismatch")
    hidden0 = torch.empty((rank,), device=value.device, dtype=torch.float32)
    hidden1 = torch.empty((rank,), device=value.device, dtype=torch.float32)
    block_n = 64
    _direct_binary_pair_v_gemv_kernel[(triton.cdiv(rank, block_n),)](
        flat, path0["v"], path1["v"],
        path0["latent"], path1["latent"], path0["column"], path1["column"],
        hidden0, hidden1, rank, in_features,
        BLOCK_N=block_n, BLOCK_K=128, num_warps=4,
    )
    _direct_binary_pair_u_accumulate_gemv_kernel[(triton.cdiv(out_features, block_n),)](
        hidden0, hidden1, path0["u"], path1["u"], path0["row"], path1["row"],
        flat_output, out_features, rank,
        BLOCK_N=block_n, BLOCK_K=128, num_warps=4,
    )
    return flat_output.reshape_as(output)


def _pack_two_bit_ternary(base: torch.Tensor) -> torch.Tensor:
    """Map {-1,0,+1} to {0,1,2}; four symbols per byte."""
    symbols = (base.reshape(base.shape[0], -1).to(torch.int16) + 1).to(torch.uint8)
    if symbols.shape[1] % 4:
        raise ValueError("T3 row width must be divisible by four")
    lanes = symbols.reshape(symbols.shape[0], -1, 4)
    return (lanes[..., 0] | (lanes[..., 1] << 2) |
            (lanes[..., 2] << 4) | (lanes[..., 3] << 6)).contiguous()


def build_hardware_base_cache(
    endpoint: Any, path: Path, device: torch.device, row_chunk: int = 256,
) -> tuple[torch.Tensor, ...]:
    header = endpoint._read_matrix_header(path)
    n_size, k_size = int(header["rows"]), int(header["columns"])
    packed_parts, position_parts, sign_parts, alpha_parts, beta_parts = [], [], [], [], []
    for _, base, sparse, alpha, beta in ref._iter_t3_sparse_rows(
        endpoint, path, row_chunk=row_chunk
    ):
        rows, groups, width = sparse.shape
        positions = sparse.ne(0).nonzero(as_tuple=False)[:, 2].reshape(rows, groups, 8)
        signs = sparse.gather(-1, positions.long()).gt(0).to(torch.uint8)
        sign_bits = torch.zeros((rows, groups), dtype=torch.uint8)
        for bit in range(8):
            sign_bits |= signs[..., bit] << bit
        packed_parts.append(_pack_two_bit_ternary(base))
        position_parts.append(positions.to(torch.uint8))
        sign_parts.append(sign_bits)
        alpha_parts.append(alpha)
        beta_parts.append(beta)
    tensors = tuple(torch.cat(parts, dim=0).contiguous().to(device) for parts in (
        packed_parts, position_parts, sign_parts, alpha_parts, beta_parts
    ))
    return (*tensors, n_size, k_size)


def write_hardware_base_cache(path: Path, cache: tuple[torch.Tensor, ...]) -> int:
    """Serialize one compute-ready base without changing the V77 checkpoint."""
    tensors = tuple(item.detach().contiguous().cpu() for item in cache[:5])
    n_size, k_size = int(cache[-2]), int(cache[-1])
    groups = k_size // 128
    expected_shapes = (
        (n_size, k_size // 4),
        (n_size, groups, 8),
        (n_size, groups),
        (n_size, groups),
        (n_size, groups),
    )
    expected_dtypes = (
        torch.uint8, torch.uint8, torch.uint8, torch.float16, torch.float16
    )
    for tensor, shape, dtype in zip(tensors, expected_shapes, expected_dtypes):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError("hardware base tensor schema mismatch")
    lengths = tuple(t.numel() * t.element_size() for t in tensors)
    header = HARDWARE_BASE_HEADER.pack(
        HARDWARE_BASE_MAGIC, 1, n_size, k_size, groups, *lengths
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444
    )
    try:
        os.write(descriptor, header)
        for tensor in tensors:
            raw = memoryview(tensor.view(torch.uint8).numpy()).cast("B")
            while raw:
                written = os.write(descriptor, raw)
                raw = raw[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    return target.stat().st_size


def load_hardware_base_cache(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Load the compute ABI directly; no logical INT8 tensor is created."""
    raw = Path(path).read_bytes()
    if len(raw) < HARDWARE_BASE_HEADER.size:
        raise ValueError("short WAL hardware base")
    magic, version, n_size, k_size, groups, *lengths = HARDWARE_BASE_HEADER.unpack_from(raw)
    if magic != HARDWARE_BASE_MAGIC or version != 1 or groups != k_size // 128:
        raise ValueError("unsupported WAL hardware base")
    expected_lengths = (
        n_size * (k_size // 4),
        n_size * groups * 8,
        n_size * groups,
        n_size * groups * 2,
        n_size * groups * 2,
    )
    if tuple(lengths) != expected_lengths or len(raw) != HARDWARE_BASE_HEADER.size + sum(lengths):
        raise ValueError("WAL hardware base byte accounting mismatch")
    shapes = (
        (n_size, k_size // 4), (n_size, groups, 8), (n_size, groups),
        (n_size, groups), (n_size, groups),
    )
    dtypes = (torch.uint8, torch.uint8, torch.uint8, torch.float16, torch.float16)
    tensors = []
    offset = HARDWARE_BASE_HEADER.size
    for length, shape, dtype in zip(lengths, shapes, dtypes):
        tensor = torch.frombuffer(bytearray(raw[offset:offset + length]), dtype=dtype)
        tensors.append(tensor.reshape(shape).contiguous().to(device))
        offset += length
    return (*tensors, int(n_size), int(k_size))


def load_packed_overlay(path: Path, device: torch.device) -> tuple[dict[str, Any], ...]:
    raw = path.read_bytes()
    if len(raw) < WALB2_HEADER.size:
        raise ValueError("short WALB2 file")
    (magic, version, out_features, in_features, rank, paths, u_bytes, v_bytes,
     scale_bytes, payload_bytes, reserved) = WALB2_HEADER.unpack_from(raw)
    if magic != b"WALLB2\0\0" or version != 1 or reserved != 0:
        raise ValueError("unsupported WALB2 header")
    if len(raw) != WALB2_HEADER.size + payload_bytes:
        raise ValueError("WALB2 byte accounting mismatch")
    expected_scale = 2 * (out_features + rank + in_features)
    if u_bytes != (out_features * rank + 7) // 8 or v_bytes != (rank * in_features + 7) // 8:
        raise ValueError("WALB2 code byte accounting mismatch")
    if scale_bytes != expected_scale:
        raise ValueError("WALB2 scale byte accounting mismatch")
    offset = WALB2_HEADER.size
    result = []
    for _ in range(paths):
        u = torch.frombuffer(bytearray(raw[offset:offset + u_bytes]), dtype=torch.uint8)
        offset += u_bytes
        v = torch.frombuffer(bytearray(raw[offset:offset + v_bytes]), dtype=torch.uint8)
        offset += v_bytes
        row = torch.frombuffer(bytearray(raw[offset:offset + 2 * out_features]), dtype=torch.float16)
        offset += 2 * out_features
        latent = torch.frombuffer(bytearray(raw[offset:offset + 2 * rank]), dtype=torch.float16)
        offset += 2 * rank
        column = torch.frombuffer(bytearray(raw[offset:offset + 2 * in_features]), dtype=torch.float16)
        offset += 2 * in_features
        result.append({
            "u": u.contiguous().to(device), "v": v.contiguous().to(device),
            "row": row.contiguous().to(device), "latent": latent.contiguous().to(device),
            "column": column.contiguous().to(device), "out": out_features,
            "in": in_features, "rank": rank,
        })
    if offset != len(raw):
        raise ValueError("WALB2 trailing bytes")
    return tuple(result)


def direct_wal_linear(
    value: torch.Tensor,
    base_cache: tuple[torch.Tensor, ...],
    paths: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    result = direct_t3_sparse_mm(value, base_cache)
    if value.reshape(-1, value.shape[-1]).shape[0] == 1 and len(paths) == 2:
        return direct_walb2_pair_gemv(value, result, paths)
    for path in paths:
        hidden = direct_binary_mm(
            value, path["v"], path["rank"], path["in"], path["latent"], path["column"]
        )
        result.add_(direct_binary_mm(
            hidden, path["u"], path["out"], path["rank"], path["row"]
        ))
    return result


def tensor_bytes(values: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> int:
    return sum(item.numel() * item.element_size() for item in values)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    endpoint, codec = ref.bundled_runtime(checkpoint)
    _, manifest, _, _ = ref.load_manifests(checkpoint)
    overlay_record = manifest["overlays"][args.ordinal]
    if int(overlay_record["ordinal"]) != args.ordinal:
        raise ValueError("overlay ordinal mismatch")
    base_path = checkpoint / overlay_record["base_t3_file"]
    overlay_path = checkpoint / overlay_record["file"]
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    started = time.monotonic()
    base_cache = build_hardware_base_cache(endpoint, base_path, device)
    paths = load_packed_overlay(overlay_path, device)
    build_seconds = time.monotonic() - started
    n_size, k_size = int(base_cache[-2]), int(base_cache[-1])
    x = torch.randn((args.batch, k_size), device=device, dtype=torch.bfloat16)

    # Existing logical-INT8 implementation is the numerical oracle here.
    base_parts, pos_parts, sign_parts, alpha_parts, beta_parts = [], [], [], [], []
    for _, base, sparse, alpha, beta in ref._iter_t3_sparse_rows(endpoint, base_path, row_chunk=256):
        rows, groups, width = sparse.shape
        pos = sparse.ne(0).nonzero(as_tuple=False)[:, 2].reshape(rows, groups, 8).to(torch.uint8)
        base_parts.append(base.reshape(rows, groups * width).to(torch.int8))
        pos_parts.append(pos)
        sign_parts.append(sparse.gather(-1, pos.long()).to(torch.int8))
        alpha_parts.append(alpha)
        beta_parts.append(beta)
    logical_cache = tuple(torch.cat(parts).contiguous().to(device) for parts in (
        base_parts, pos_parts, sign_parts, alpha_parts, beta_parts
    ))
    logical_paths = []
    bundle = codec.read_binary_lowrank_bundle(overlay_path)
    for item in bundle.paths:
        logical_paths.append(tuple(t.to(device) for t in (
            item.u_codes, item.v_codes, item.row_scales_fp16,
            item.latent_scales_fp16, item.column_scales_fp16,
        )))

    def oracle() -> torch.Tensor:
        base, pos, signs, alpha, beta = logical_cache
        out = logical.t3_sparse_mm(x, base, pos, signs, alpha, beta)
        for u, v, row, latent, column in logical_paths:
            hidden = logical.scaled_sign_mm(x, v, latent, column)
            out.add_(logical.scaled_sign_mm(hidden, u, row))
        return out

    for _ in range(2):
        direct_wal_linear(x, base_cache, paths)
        oracle()
    torch.cuda.synchronize()
    direct_value = direct_wal_linear(x, base_cache, paths)
    oracle_value = oracle()
    torch.cuda.synchronize()
    delta = (direct_value - oracle_value).abs()

    def timed(callable_: Any) -> float:
        samples = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(); callable_(); end.record(); end.synchronize()
            samples.append(begin.elapsed_time(end))
        return float(torch.tensor(samples).median().item())

    direct_ms = timed(lambda: direct_wal_linear(x, base_cache, paths))
    oracle_ms = timed(oracle)
    hardware_tensors = list(base_cache[:5])
    for path in paths:
        hardware_tensors.extend(path[key] for key in ("u", "v", "row", "latent", "column"))
    logical_tensors = list(logical_cache)
    for path in logical_paths:
        logical_tensors.extend(path)
    report = {
        "schema": "wal-v77-direct-packed-triton-operator-v1",
        "checkpoint": str(checkpoint),
        "matrix": overlay_record["name"],
        "ordinal": args.ordinal,
        "shape": [n_size, k_size],
        "batch": args.batch,
        "archival_base_bytes": base_path.stat().st_size,
        "archival_overlay_bytes": overlay_path.stat().st_size,
        "hardware_cache_bytes": tensor_bytes(hardware_tensors),
        "logical_int8_cache_bytes": tensor_bytes(logical_tensors),
        "hardware_vs_logical_ratio": tensor_bytes(hardware_tensors) / tensor_bytes(logical_tensors),
        "build_seconds": build_seconds,
        "direct_median_ms": direct_ms,
        "logical_int8_median_ms": oracle_ms,
        "direct_over_logical_time_ratio": direct_ms / oracle_ms,
        "mean_abs_error": float(delta.mean().item()),
        "max_abs_error": float(delta.max().item()),
        "mean_abs_reference": float(oracle_value.abs().mean().item()),
        "allclose_atol_0_02_rtol_0_02": bool(torch.allclose(
            direct_value, oracle_value, atol=0.02, rtol=0.02
        )),
        "no_dense_int8_or_fp_weight_cache": True,
        "compute_abi": {
            "t3": "2-bit symbols, four per byte, register decode",
            "sparse_k8": "8 uint8 positions plus one sign byte per g128",
            "walb2": "original 1-bit U/V factors, register decode",
            "scales": "original FP16 alpha/beta and row/latent/column scales",
        },
    }
    del logical_cache, logical_paths, oracle_value, direct_value, x
    gc.collect(); torch.cuda.empty_cache()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ordinal", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = benchmark(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
