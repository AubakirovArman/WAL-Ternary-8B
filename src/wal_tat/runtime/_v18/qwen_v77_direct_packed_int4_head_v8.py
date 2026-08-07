"""Packed INT4 LM head for decode batches up to sixteen."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import qwen_v77_direct_packed_int4_head_v2 as head_v2


@triton.jit
def _int4_head_packed_byte_batched_kernel(
    x_ptr, code_ptr, scale_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
):
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = tl.program_id(1)
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
            activation = tl.load(x_ptr + batch * k_size + k).to(tl.float32)
            code = ((packed >> (lane * 4)) & 15) - 7
            dense += tl.sum(code.to(tl.float32) * activation[None, :], axis=1)
        scale = tl.load(
            scale_ptr + on * groups + group, mask=mask_n, other=0.0,
        ).to(tl.float32)
        acc += dense * scale
    tl.store(out_ptr + batch * n_size + on, acc, mask=mask_n)


def direct_int4_head_batched_v8(
    value: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
    *, out_features: int, in_features: int,
) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    batch = flat.shape[0]
    if not 2 <= batch <= 16 or flat.shape[1] != in_features:
        raise ValueError("V8 INT4 head requires batch 2..16")
    output = torch.empty(
        (batch, out_features), device=value.device, dtype=torch.float32,
    )
    block_n = 16
    _int4_head_packed_byte_batched_kernel[
        (triton.cdiv(out_features, block_n), batch)
    ](
        flat, codes, scales, output,
        out_features, in_features, in_features // 128, in_features // 2,
        BLOCK_N=block_n, PACKED_K=64, num_warps=2,
    )
    return output.reshape(*value.shape[:-1], out_features)


class DirectPackedINT4HeadV8(head_v2.DirectPackedINT4HeadV2):
    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        flattened = value.reshape(-1, value.shape[-1]).shape[0]
        if flattened == 1:
            return super().forward(value)
        if not 2 <= flattened <= 16:
            return head_v2.runtime_v1.DirectPackedINT4Head.forward(self, value)
        codes, scales = self._packed(value.device)
        return direct_int4_head_batched_v8(
            value, codes, scales,
            out_features=self.out_features, in_features=self.in_features,
        ).to(self.output_dtype)
