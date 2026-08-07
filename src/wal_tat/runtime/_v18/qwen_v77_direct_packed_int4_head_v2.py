"""Packed-byte INT4 LM-head GEMV for batch-one V77 decode."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import qwen_v77_direct_packed_triton_runtime_v1 as runtime_v1


@triton.jit
def _int4_head_packed_byte_gemv_kernel(
    x_ptr, code_ptr, scale_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
):
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ob = tl.arange(0, PACKED_K)
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for group in range(0, groups):
        byte = group * PACKED_K + ob
        packed = tl.load(
            code_ptr + on[:, None] * bytes_per_row + byte[None, :],
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        dense = tl.zeros((BLOCK_N,), tl.float32)
        for lane in range(0, 2):
            k = group * 128 + ob * 2 + lane
            value = tl.load(x_ptr + k).to(tl.float32)
            code = ((packed >> (lane * 4)) & 15) - 7
            dense += tl.sum(code.to(tl.float32) * value[None, :], axis=1)
        scale = tl.load(
            scale_ptr + on * groups + group, mask=mask_n, other=0.0,
        ).to(tl.float32)
        acc += dense * scale
    tl.store(out_ptr + on, acc, mask=mask_n)


def direct_int4_head_gemv_v2(
    value: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    block_n: int = 16,
    num_warps: int = 4,
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if flat.shape != (1, in_features):
        raise ValueError("packed-byte INT4 head requires batch one")
    output = torch.empty((out_features,), device=value.device, dtype=torch.float32)
    _int4_head_packed_byte_gemv_kernel[(triton.cdiv(out_features, block_n),)](
        flat, codes, scales, output,
        out_features, in_features, in_features // 128, in_features // 2,
        BLOCK_N=block_n, PACKED_K=64, num_warps=num_warps,
    )
    return output.reshape(*value.shape[:-1], out_features)


class DirectPackedINT4HeadV2(runtime_v1.DirectPackedINT4Head):
    """Use packed-byte decode for M=1 and retain v1 for prefill."""

    block_n = 16
    num_warps = 2

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.reshape(-1, value.shape[-1]).shape[0] != 1:
            return super().forward(value)
        codes, scales = self._packed(value.device)
        return direct_int4_head_gemv_v2(
            value, codes, scales,
            out_features=self.out_features, in_features=self.in_features,
            block_n=self.block_n, num_warps=self.num_warps,
        ).to(self.output_dtype)
