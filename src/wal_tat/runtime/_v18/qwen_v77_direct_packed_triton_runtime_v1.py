"""Whole-model V77 runtime using the direct-packed Transformer operator.

The Transformer body uses the hardware-packed implementation from
``qwen_v77_direct_packed_triton_operator_v1``. Embedding remains random-access
packed INT3 and the LM head consumes its original packed INT4 stream directly.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
import triton
import triton.language as tl

import qwen_v77_packed_reference_runtime as ref
from qwen_v77_direct_packed_triton_operator_v1 import (
    build_hardware_base_cache,
    direct_binary_mm,
    direct_t3_sparse_mm,
    direct_wal_linear,
    direct_walb2_pair_gemv,
    load_hardware_base_cache,
    load_packed_overlay,
)


class DirectPackedWALLinear(ref.PackedWALLinear):
    """Lazy hardware-packed T3+sparse-k8+WALB2 CUDA linear."""

    hardware_cache_root: Path | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._direct_cache: dict[str, tuple[tuple[torch.Tensor, ...], tuple[dict, ...]]] = {}

    def _direct(self, device: torch.device):
        key = str(device)
        cached = self._direct_cache.get(key)
        if cached is None:
            hardware = None if self.hardware_cache_root is None else (
                self.hardware_cache_root / (self.base_path.name + ".walhw")
            )
            base = (
                load_hardware_base_cache(hardware, device)
                if hardware is not None and hardware.is_file()
                else build_hardware_base_cache(
                    self.endpoint, self.base_path, device,
                    row_chunk=max(self.row_chunk, 256),
                )
            )
            paths = () if self.overlay_path is None else load_packed_overlay(
                self.overlay_path, device
            )
            cached = (base, paths)
            self._direct_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base, paths = self._direct(value.device)
        if os.environ.get("WAL_NVTX_PROFILE") != "1":
            return direct_wal_linear(value, base, paths).to(self.output_dtype)
        torch.cuda.nvtx.range_push(getattr(self, "profile_name", self.base_path.name))
        try:
            torch.cuda.nvtx.range_push("t3_sparse_k8")
            try:
                result = direct_t3_sparse_mm(value, base)
            finally:
                torch.cuda.nvtx.range_pop()
            if len(paths) == 2 and value.reshape(-1, value.shape[-1]).shape[0] == 1:
                torch.cuda.nvtx.range_push("walb2_pair")
                try:
                    return direct_walb2_pair_gemv(value, result, paths).to(
                        self.output_dtype
                    )
                finally:
                    torch.cuda.nvtx.range_pop()
            for ordinal, path in enumerate(paths):
                torch.cuda.nvtx.range_push(f"walb2_path{ordinal}")
                try:
                    hidden = direct_binary_mm(
                        value, path["v"], path["rank"], path["in"],
                        path["latent"], path["column"],
                    )
                    result.add_(direct_binary_mm(
                        hidden, path["u"], path["out"], path["rank"], path["row"]
                    ))
                finally:
                    torch.cuda.nvtx.range_pop()
            return result.to(self.output_dtype)
        finally:
            torch.cuda.nvtx.range_pop()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def validate_hardware_cache(
    checkpoint: Path,
    hardware_cache: Path,
    *,
    verify_matrix_hashes: bool = True,
) -> dict:
    """Fail closed if a derived compute cache is stale or unauthenticated."""
    attestation_path = hardware_cache / "attestation.json"
    manifest_path = hardware_cache / "manifest.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if attestation.get("schema") != "wal-v77-hardware-cache-attestation-v1":
        raise ValueError("unsupported hardware-cache attestation")
    layout = attestation.get("layout", {})
    if (
        layout.get("endianness") != "little"
        or sys.byteorder != "little"
        or layout.get("magic") != "WALHW001"
        or int(layout.get("version", -1)) != 1
        or attestation.get("abi_version") != "wal-direct-packed-runtime-abi-v1"
        or attestation.get("cache_depends_on_compile_flags") is not False
    ):
        raise ValueError("unsupported hardware-cache layout contract")
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_sha = _sha256(checkpoint_manifest_path)
    attested_parent_sha = attestation["canonical_checkpoint_manifest_sha256"]
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    # The public v0.2 manifest is a metadata-only successor of the immutable
    # V77 candidate.  Its release record carries the exact parent manifest
    # digest.  Accept that explicit lineage, then still verify every matrix
    # source hash and (optionally) every derived cache member below.
    release_parent_sha = checkpoint_manifest.get("release", {}).get(
        "candidate_manifest_sha256"
    )
    if checkpoint_sha != attested_parent_sha and release_parent_sha != attested_parent_sha:
        raise ValueError("hardware cache belongs to a different checkpoint")
    cache_manifest_sha = _sha256(manifest_path)
    if cache_manifest_sha != attestation["cache_manifest_sha256"]:
        raise ValueError("hardware-cache manifest digest mismatch")
    _, _, _, base_manifest = ref.load_manifests(checkpoint)
    sources = {str(row["name"]): str(row["sha256"]) for row in base_manifest["matrices"]}
    records = []
    payload_bytes = 0
    for ordinal, row in enumerate(cache_manifest["matrices"]):
        if int(row["ordinal"]) != ordinal or sources.get(str(row["name"])) != row["source_sha256"]:
            raise ValueError("hardware-cache matrix lineage mismatch")
        member = hardware_cache / str(row["file"])
        size = member.stat().st_size
        if size != int(row["bytes"]):
            raise ValueError("hardware-cache matrix size mismatch")
        if verify_matrix_hashes and _sha256(member) != row["sha256"]:
            raise ValueError("hardware-cache matrix digest mismatch")
        payload_bytes += size
        records.append({
            "ordinal": ordinal,
            "name": row["name"],
            "file": row["file"],
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "source_sha256": row["source_sha256"],
        })
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    matrix_root = hashlib.sha256(canonical).hexdigest()
    if (
        len(records) != int(attestation["matrix_count"])
        or payload_bytes != int(attestation["matrix_payload_bytes"])
        or matrix_root != attestation["matrix_manifest_root_sha256"]
    ):
        raise ValueError("hardware-cache aggregate attestation mismatch")
    identity = {
        "abi_version": attestation["abi_version"],
        "cache_manifest_sha256": cache_manifest_sha,
        # Cache identity is anchored to the immutable candidate manifest.  A
        # public metadata successor is reported separately and never changes
        # the derived-cache identity.
        "canonical_checkpoint_manifest_sha256": attested_parent_sha,
        "compute_abi_sha256": attestation["compute_abi_sha256"],
        "converter_sha256": attestation["converter_sha256"],
        "endianness": layout["endianness"],
        "layout_magic": layout["magic"],
        "layout_version": int(layout["version"]),
        "matrix_manifest_root_sha256": matrix_root,
    }
    canonical_identity = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode()
    cache_identity = hashlib.sha256(canonical_identity).hexdigest()
    if cache_identity != attestation["cache_identity_sha256"]:
        raise ValueError("hardware-cache identity digest mismatch")
    return {
        "status": "verified",
        "checkpoint_manifest_sha256": checkpoint_sha,
        "attested_parent_manifest_sha256": attested_parent_sha,
        "metadata_successor_lineage": checkpoint_sha != attested_parent_sha,
        "cache_manifest_sha256": cache_manifest_sha,
        "matrix_manifest_root_sha256": matrix_root,
        "matrix_count": len(records),
        "matrix_payload_bytes": payload_bytes,
        "matrix_hashes_verified": verify_matrix_hashes,
        "cache_identity_sha256": cache_identity,
        "layout": layout,
        "target_backends": attestation["target_backends"],
        "converter_sha256": attestation["converter_sha256"],
        "compute_abi_sha256": attestation["compute_abi_sha256"],
    }


@triton.jit
def _direct_int4_head_kernel(
    x_ptr, code_ptr, scale_ptr, out_ptr,
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
            code_ptr + on[:, None] * bytes_per_row + k[None, :] // 2,
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        unsigned = (packed >> ((k % 2) * 4)[None, :]) & 15
        code = unsigned - 7
        dot = tl.dot(xv, tl.trans(code.to(tl.float32)), input_precision="tf32")
        scale = tl.load(
            scale_ptr + on * groups + group, mask=mask_n, other=0.0
        ).to(tl.float32)
        acc += dot * scale[None, :]
    tl.store(
        out_ptr + om[:, None] * n_size + on[None, :], acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _direct_int4_head_gemv_kernel(
    x_ptr, code_ptr, scale_ptr, out_ptr,
    n_size: tl.constexpr, k_size: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    mask_n = on < n_size
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for group in range(0, groups):
        k = group * BLOCK_K + ok
        value = tl.load(x_ptr + k).to(tl.float32)
        packed = tl.load(
            code_ptr + on[:, None] * bytes_per_row + k[None, :] // 2,
            mask=mask_n[:, None], other=0,
        ).to(tl.int32)
        code = ((packed >> (((k % 2) * 4)[None, :])) & 15) - 7
        dot = tl.sum(code.to(tl.float32) * value[None, :], axis=1)
        scale = tl.load(scale_ptr + on * groups + group,
                        mask=mask_n, other=0.0).to(tl.float32)
        acc += dot * scale
    tl.store(out_ptr + on, acc, mask=mask_n)


class DirectPackedINT4Head(nn.Module):
    """LM head whose original INT4 codes are decoded only in registers."""

    def __init__(self, path: Path, *, endpoint, row_chunk: int, output_dtype):
        super().__init__()
        self.path = Path(path)
        self.endpoint = endpoint
        header = endpoint._read_endpoint_header(self.path)
        self.spec = header["spec"]
        self.payload = header["payload"]
        if self.spec.name != "int4-symmetric" or int(self.spec.code_bits) != 4:
            raise ValueError("DirectPackedINT4Head requires int4-symmetric")
        self.in_features = int(self.payload.columns)
        self.out_features = int(self.payload.rows)
        self.output_dtype = output_dtype
        self._packed_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _packed(self, device: torch.device):
        key = str(device)
        cached = self._packed_cache.get(key)
        if cached is None:
            header_bytes = self.endpoint.ENDPOINT_HEADER.size
            code_bytes = int(self.payload.code_bits) // 8
            metadata_bytes = int(self.payload.metadata_bits) // 8
            with self.path.open("rb") as handle:
                handle.seek(header_bytes)
                raw_codes = handle.read(code_bytes)
                raw_scales = handle.read(metadata_bytes)
                if len(raw_codes) != code_bytes or len(raw_scales) != metadata_bytes:
                    raise ValueError("truncated direct INT4 endpoint")
                if handle.read(1):
                    raise ValueError("direct INT4 endpoint has trailing bytes")
            codes = torch.frombuffer(bytearray(raw_codes), dtype=torch.uint8).to(device)
            scales = torch.frombuffer(bytearray(raw_scales), dtype=torch.float16).to(device)
            if not torch.isfinite(scales).all() or torch.any(scales <= 0):
                raise ValueError("invalid direct INT4 endpoint scales")
            cached = (codes.contiguous(), scales.contiguous())
            self._packed_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        codes, scales = self._packed(value.device)
        flat = value.reshape(-1, value.shape[-1]).contiguous()
        output = torch.empty(
            (flat.shape[0], self.out_features), device=value.device, dtype=torch.float32
        )
        groups = self.in_features // int(self.payload.group_size)
        if flat.shape[0] == 1:
            block_n = 64
            _direct_int4_head_gemv_kernel[(triton.cdiv(self.out_features, block_n),)](
                flat, codes, scales, output,
                self.out_features, self.in_features, groups, self.in_features // 2,
                BLOCK_N=block_n, BLOCK_K=128, num_warps=4,
            )
            return output.reshape(*value.shape[:-1], self.out_features).to(self.output_dtype)
        block_m = 32 if flat.shape[0] >= 256 else 16
        block_n = 32 if flat.shape[0] >= 256 else 64
        warps = 4 if flat.shape[0] >= 256 else 8
        grid = (
            triton.cdiv(flat.shape[0], block_m),
            triton.cdiv(self.out_features, block_n),
        )
        _direct_int4_head_kernel[grid](
            flat, codes, scales, output,
            flat.shape[0], self.out_features, self.in_features,
            groups, self.in_features // 2,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=128, num_warps=warps,
        )
        return output.reshape(*value.shape[:-1], self.out_features).to(self.output_dtype)


def load_direct_model(
    checkpoint: str | Path,
    *,
    device: str = "cuda:0",
    verify_hashes: bool = True,
    hardware_cache: str | Path | None = None,
    verify_hardware_cache_hashes: bool = True,
):
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    cache_path = None if hardware_cache is None else Path(hardware_cache).resolve(strict=True)
    cache_report = None if cache_path is None else validate_hardware_cache(
        checkpoint_path, cache_path, verify_matrix_hashes=verify_hardware_cache_hashes
    )
    DirectPackedWALLinear.hardware_cache_root = cache_path
    model, report = ref.load_packed_model(
        checkpoint,
        device=device,
        dtype=torch.bfloat16,
        row_chunk=256,
        verify_hashes=verify_hashes,
        wal_linear_class=DirectPackedWALLinear,
        endpoint_linear_class=DirectPackedINT4Head,
    )
    report["hardware_cache_attestation"] = cache_report
    return model, report


def cache_accounting(model: torch.nn.Module) -> dict[str, int]:
    body = overlay = endpoint_packed = 0
    for module in model.modules():
        if isinstance(module, DirectPackedWALLinear):
            for base, paths in module._direct_cache.values():
                body += sum(t.numel() * t.element_size() for t in base[:5])
                for path in paths:
                    overlay += sum(
                        path[key].numel() * path[key].element_size()
                        for key in ("u", "v", "row", "latent", "column")
                    )
        elif isinstance(module, DirectPackedINT4Head):
            for codes, scales in module._packed_cache.values():
                endpoint_packed += codes.numel() * codes.element_size()
                endpoint_packed += scales.numel() * scales.element_size()
    return {
        "direct_body_base_bytes": body,
        "direct_body_overlay_bytes": overlay,
        "direct_packed_int4_head_bytes": endpoint_packed,
        "total_persistent_weight_cache_bytes": body + overlay + endpoint_packed,
    }


@torch.no_grad()
def smoke(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    started = time.monotonic()
    model, report = load_direct_model(
        args.checkpoint, device="cuda:0", verify_hashes=not args.skip_hashes,
        hardware_cache=args.hardware_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, trust_remote_code=False
    )
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    logits = model(**inputs, use_cache=False).logits[:, -1].float()
    torch.cuda.synchronize()
    cold_seconds = time.monotonic() - begin
    warm_begin = time.monotonic()
    warm_logits = model(**inputs, use_cache=False).logits[:, -1].float()
    torch.cuda.synchronize()
    warm_seconds = time.monotonic() - warm_begin
    report.update({
        "runtime_tier": "direct-packed-transformer-and-int4-head-v1",
        "forward": {
            "seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "input_tokens": int(inputs.input_ids.numel()),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "argmax_token_id": int(logits.argmax(-1).item()),
            "argmax_token": tokenizer.decode(logits.argmax(-1).tolist()),
            "warm_argmax_matches": bool(warm_logits.argmax(-1).equal(logits.argmax(-1))),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "cache": cache_accounting(model),
        "total_seconds": time.monotonic() - started,
        "transformer_int8_or_fp_weight_cache": False,
        "endpoint_int8_or_fp_weight_cache": False,
    })
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=str(ref.DEFAULT_CHECKPOINT))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--output")
    parser.add_argument("--hardware-cache")
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = smoke(args)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.output:
        raw = (payload + "\n").encode()
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
