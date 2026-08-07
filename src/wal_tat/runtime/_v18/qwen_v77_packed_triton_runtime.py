"""Triton GPU runtime for WAL-Ternary-8B V77 logical low-bit codes.

The immutable radix-3 checkpoint is decoded once to an inference cache made of
INT8 ternary codes plus compact sparse-k8 support/sign arrays.  Triton kernels
consume those low-bit logical codes directly; no BF16/FP16 Transformer matrix
is ever constructed or retained.  This is the first performance runtime tier,
between the streaming correctness reference and a future direct-radix kernel.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
import triton
import triton.language as tl

import qwen_v77_packed_reference_runtime as ref


@triton.jit
def _t3_sparse_k8_kernel(
    x_ptr, base_ptr, pos_ptr, sign_ptr, alpha_ptr, beta_ptr, out_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    groups: tl.constexpr,
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
        xv = tl.load(x_ptr + om[:, None] * k_size + k[None, :],
                     mask=mask_m[:, None], other=0.0).to(tl.float32)
        bc = tl.load(base_ptr + on[:, None] * k_size + k[None, :],
                     mask=mask_n[:, None], other=0).to(tl.float32)
        dot = tl.dot(xv, tl.trans(bc), input_precision="tf32")
        a = tl.load(alpha_ptr + on * groups + group,
                    mask=mask_n, other=0.0).to(tl.float32)
        sparse = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for slot in range(0, 8):
            index = (on * groups + group) * 8 + slot
            p = tl.load(pos_ptr + index, mask=mask_n, other=0).to(tl.int32)
            s = tl.load(sign_ptr + index, mask=mask_n, other=0).to(tl.float32)
            xs = tl.load(x_ptr + om[:, None] * k_size +
                         group * BLOCK_K + p[None, :],
                         mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            sparse += xs * s[None, :]
        b = tl.load(beta_ptr + on * groups + group,
                    mask=mask_n, other=0.0).to(tl.float32)
        acc += dot * a[None, :] + sparse * b[None, :]
    tl.store(out_ptr + om[:, None] * n_size + on[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _scaled_sign_kernel(
    x_ptr, code_ptr, row_ptr, column_ptr, out_ptr,
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
        xv = tl.load(x_ptr + om[:, None] * k_size + k[None, :],
                     mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        if HAS_COLUMN:
            col = tl.load(column_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
            xv *= col[None, :]
        code = tl.load(code_ptr + on[:, None] * k_size + k[None, :],
                       mask=mask_n[:, None] & mask_k[None, :], other=0).to(tl.float32)
        acc += tl.dot(xv, tl.trans(code), input_precision="tf32")
    row = tl.load(row_ptr + on, mask=mask_n, other=0.0).to(tl.float32)
    acc *= row[None, :]
    tl.store(out_ptr + om[:, None] * n_size + on[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _group_code_kernel(
    x_ptr, code_ptr, scale_ptr, out_ptr,
    m_size: tl.constexpr, n_size: tl.constexpr, k_size: tl.constexpr,
    groups: tl.constexpr,
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
        xv = tl.load(x_ptr + om[:, None] * k_size + k[None, :],
                     mask=mask_m[:, None], other=0.0).to(tl.float32)
        code = tl.load(code_ptr + on[:, None] * k_size + k[None, :],
                       mask=mask_n[:, None], other=0).to(tl.float32)
        dot = tl.dot(xv, tl.trans(code), input_precision="tf32")
        scale = tl.load(scale_ptr + on * groups + group,
                        mask=mask_n, other=0.0).to(tl.float32)
        acc += dot * scale[None, :]
    tl.store(out_ptr + om[:, None] * n_size + on[None, :], acc,
             mask=mask_m[:, None] & mask_n[None, :])


def t3_sparse_mm(value, base, positions, signs, alpha, beta):
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    n_size, k_size = base.shape
    groups = k_size // 128
    out = torch.empty((flat.shape[0], n_size), device=value.device, dtype=torch.float32)
    block_m = 32
    block_n = 32 if flat.shape[0] >= 256 else 64
    warps = 4 if flat.shape[0] >= 256 else 8
    grid = (triton.cdiv(flat.shape[0], block_m), triton.cdiv(n_size, block_n))
    _t3_sparse_k8_kernel[grid](
        flat, base, positions, signs, alpha, beta, out,
        flat.shape[0], n_size, k_size, groups,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
    )
    return out.reshape(*value.shape[:-1], n_size)


def scaled_sign_mm(value, codes, row_scales, column_scales=None):
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    n_size, k_size = codes.shape
    out = torch.empty((flat.shape[0], n_size), device=value.device, dtype=torch.float32)
    block_m = 32
    block_n = 32 if flat.shape[0] >= 256 else 64
    warps = 4 if flat.shape[0] >= 256 else 8
    grid = (triton.cdiv(flat.shape[0], block_m), triton.cdiv(n_size, block_n))
    dummy = row_scales
    _scaled_sign_kernel[grid](
        flat, codes, row_scales, dummy if column_scales is None else column_scales, out,
        flat.shape[0], n_size, k_size, HAS_COLUMN=column_scales is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
    )
    return out.reshape(*value.shape[:-1], n_size)


def group_code_mm(value, codes, scales, group_size: int):
    if group_size != 128:
        raise ValueError("Triton endpoint currently requires g128")
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    n_size, k_size = codes.shape
    groups = k_size // group_size
    out = torch.empty((flat.shape[0], n_size), device=value.device, dtype=torch.float32)
    block_m = 32
    block_n = 32 if flat.shape[0] >= 256 else 64
    warps = 4 if flat.shape[0] >= 256 else 8
    grid = (triton.cdiv(flat.shape[0], block_m), triton.cdiv(n_size, block_n))
    _group_code_kernel[grid](
        flat, codes, scales, out, flat.shape[0], n_size, k_size, groups,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
    )
    return out.reshape(*value.shape[:-1], n_size)


class TritonWALLinear(ref.PackedWALLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code_cache: dict[str, tuple[torch.Tensor, ...]] = {}

    def _codes(self, device: torch.device):
        key = str(device)
        if key in self._code_cache:
            return self._code_cache[key]
        base_parts, pos_parts, sign_parts, alpha_parts, beta_parts = [], [], [], [], []
        for _, base, sparse, alpha, beta in ref._iter_t3_sparse_rows(
            self.endpoint, self.base_path, row_chunk=max(self.row_chunk, 256)
        ):
            rows, groups, width = sparse.shape
            nz = sparse.ne(0).nonzero(as_tuple=False)
            if nz.shape[0] != rows * groups * 8:
                raise ValueError("sparse-k8 cardinality drift")
            positions = nz[:, 2].reshape(rows, groups, 8).to(torch.uint8)
            signs = sparse.gather(-1, positions.long()).to(torch.int8)
            base_parts.append(base.reshape(rows, groups * width).to(torch.int8))
            pos_parts.append(positions)
            sign_parts.append(signs)
            alpha_parts.append(alpha)
            beta_parts.append(beta)
        cache = tuple(torch.cat(parts, dim=0).contiguous().to(device)
                      for parts in (base_parts, pos_parts, sign_parts,
                                    alpha_parts, beta_parts))
        self._code_cache[key] = cache
        return cache

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base, positions, signs, alpha, beta = self._codes(value.device)
        result = t3_sparse_mm(value, base, positions, signs, alpha, beta)
        for u_codes, v_codes, row, latent, column in self._overlay(value.device):
            hidden = scaled_sign_mm(value, v_codes, latent, column)
            result.add_(scaled_sign_mm(hidden, u_codes, row))
        return result.to(self.output_dtype)


class TritonEndpointLinear(ref.PackedEndpointLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._code_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _codes(self, device: torch.device):
        key = str(device)
        if key in self._code_cache:
            return self._code_cache[key]
        code_parts, scale_parts = [], []
        for _, codes, scales in self.endpoint.iter_quantized_endpoint_rows(
            self.path, row_chunk=max(self.row_chunk, 512)
        ):
            code_parts.append(codes.reshape(codes.shape[0], -1).to(torch.int8))
            scale_parts.append(scales)
        cached = (torch.cat(code_parts).contiguous().to(device),
                  torch.cat(scale_parts).contiguous().to(device))
        self._code_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        codes, scales = self._codes(value.device)
        return group_code_mm(value, codes, scales,
                             self.endpoint._read_endpoint_header(self.path)["payload"].group_size).to(self.output_dtype)


def load_triton_model(checkpoint: str | Path, *, device: str = "cuda:0",
                      dtype: torch.dtype = torch.bfloat16, verify_hashes: bool = True):
    return ref.load_packed_model(
        checkpoint, device=device, dtype=dtype, row_chunk=256,
        verify_hashes=verify_hashes, wal_linear_class=TritonWALLinear,
        endpoint_linear_class=TritonEndpointLinear,
    )


@torch.no_grad()
def smoke(checkpoint: str | Path, *, device: str, prompt: str) -> dict[str, Any]:
    from transformers import AutoTokenizer
    started = time.monotonic()
    model, report = load_triton_model(checkpoint, device=device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    logits = model(**inputs, use_cache=False).logits[:, -1].float()
    torch.cuda.synchronize()
    report["forward"] = {
        "seconds": time.monotonic() - begin,
        "input_tokens": int(inputs.input_ids.numel()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "argmax_token_id": int(logits.argmax(-1).item()),
        "argmax_token": tokenizer.decode(logits.argmax(-1).tolist()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }
    cache_bytes = 0
    for module in model.modules():
        for cache_name in ("_code_cache", "_overlay_cache"):
            for cached in getattr(module, cache_name, {}).values():
                stack = list(cached)
                while stack:
                    item = stack.pop()
                    if isinstance(item, torch.Tensor):
                        cache_bytes += item.numel() * item.element_size()
                    elif isinstance(item, (tuple, list)):
                        stack.extend(item)
    report["logical_lowbit_cache_bytes"] = cache_bytes
    report["total_seconds"] = time.monotonic() - started
    report["runtime_tier"] = "triton-logical-lowbit-v1"
    return report


def operator_check(checkpoint: str | Path, *, device: str, matrix_name: str | None,
                   seed: int, repeats: int, batch: int = 2) -> dict[str, Any]:
    root, manifest, base_root, base_manifest = ref.load_manifests(checkpoint)
    endpoint, codec = ref.bundled_runtime(root)
    bases = {str(x["name"]): x for x in base_manifest["matrices"]}
    overlays = {str(x["name"]): x for x in manifest["overlays"]}
    if matrix_name is None:
        matrix_name = next(name for name in bases if name in overlays)
    row = bases[matrix_name]
    shape = tuple(int(x) for x in row["shape"])
    module = TritonWALLinear(
        base_root / row["file"], endpoint=endpoint, codec=codec,
        overlay_path=None if matrix_name not in overlays else root / overlays[matrix_name]["file"],
        in_features=shape[1], out_features=shape[0], row_chunk=256,
        output_dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn((batch, shape[1]), generator=generator).to(device)
    reference = ref.PackedWALLinear(
        base_root / row["file"], endpoint=endpoint, codec=codec,
        overlay_path=None if matrix_name not in overlays else root / overlays[matrix_name]["file"],
        in_features=shape[1], out_features=shape[0], row_chunk=256,
        output_dtype=torch.float32,
    )
    expected = reference(value)
    actual = module(value)
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(repeats):
        actual = module(value)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    error = (actual - expected).abs()
    cache_bytes = sum(x.numel() * x.element_size() for x in module._codes(value.device))
    cache_bytes += sum(x.numel() * x.element_size()
                       for path in module._overlay(value.device) for x in path)
    return {"status": "ok", "matrix": matrix_name, "shape": list(shape),
            "device": device, "batch": batch, "repeats": repeats,
            "mean_seconds": elapsed / repeats, "logical_cache_bytes": cache_bytes,
            "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
            "argmax_agreement": bool(actual.argmax(-1).equal(expected.argmax(-1))),
            "full_bf16_body_materialized": False,
            "compute_format": "INT8 ternary + sparse-k8 support/sign + FP16 scales"}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("operator-check")
    check.add_argument("checkpoint", nargs="?", default=str(ref.DEFAULT_CHECKPOINT))
    check.add_argument("--device", default="cuda:0")
    check.add_argument("--matrix")
    check.add_argument("--seed", type=int, default=20260806)
    check.add_argument("--repeats", type=int, default=10)
    check.add_argument("--batch", type=int, default=2)
    run = sub.add_parser("smoke")
    run.add_argument("checkpoint", nargs="?", default=str(ref.DEFAULT_CHECKPOINT))
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--prompt", default="Hello")
    run.add_argument("--output")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("this runtime tier requires CUDA")
    if args.command == "operator-check":
        result = operator_check(args.checkpoint, device=args.device,
                                matrix_name=args.matrix, seed=args.seed,
                                repeats=args.repeats, batch=args.batch)
    else:
        result = smoke(args.checkpoint, device=args.device, prompt=args.prompt)
        if args.output:
            raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444)
            try:
                offset = 0
                while offset < len(raw):
                    offset += os.write(fd, raw[offset:])
                os.fchmod(fd, 0o444)
                os.fsync(fd)
            finally:
                os.close(fd)
    print(json.dumps(result, indent=2, allow_nan=False))
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
