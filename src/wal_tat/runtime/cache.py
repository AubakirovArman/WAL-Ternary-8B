"""Portable readers for the WAL compute-cache and WALB2 formats.

This module intentionally has no Triton/CUDA dependency so it can be used by
x86-64 and Apple Silicon native CPU backends.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import struct
import sys
import time
from typing import Any

import torch


WALB2_HEADER = struct.Struct("<8sIIIIIQQQQI")
HARDWARE_BASE_HEADER = struct.Struct("<8sIIIIQQQQQ")
HARDWARE_BASE_MAGIC = b"WALHW001"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_hardware_base_cache(
    path: Path, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    raw = Path(path).read_bytes()
    if len(raw) < HARDWARE_BASE_HEADER.size:
        raise ValueError("short WAL hardware base")
    magic, version, rows, columns, groups, *lengths = (
        HARDWARE_BASE_HEADER.unpack_from(raw)
    )
    if magic != HARDWARE_BASE_MAGIC or version != 1 or groups != columns // 128:
        raise ValueError("unsupported WAL hardware base")
    expected = (
        rows * (columns // 4), rows * groups * 8, rows * groups,
        rows * groups * 2, rows * groups * 2,
    )
    if tuple(lengths) != expected or len(raw) != HARDWARE_BASE_HEADER.size + sum(lengths):
        raise ValueError("WAL hardware base byte accounting mismatch")
    shapes = (
        (rows, columns // 4), (rows, groups, 8), (rows, groups),
        (rows, groups), (rows, groups),
    )
    dtypes = (torch.uint8, torch.uint8, torch.uint8, torch.float16, torch.float16)
    tensors = []
    offset = HARDWARE_BASE_HEADER.size
    for length, shape, dtype in zip(lengths, shapes, dtypes):
        value = torch.frombuffer(
            bytearray(raw[offset:offset + length]), dtype=dtype,
        )
        tensors.append(value.reshape(shape).contiguous().to(device))
        offset += length
    return (*tensors, int(rows), int(columns))


def load_packed_overlay(
    path: Path, device: torch.device,
) -> tuple[dict[str, Any], ...]:
    raw = Path(path).read_bytes()
    if len(raw) < WALB2_HEADER.size:
        raise ValueError("short WALB2 file")
    (magic, version, out_features, in_features, rank, paths, u_bytes, v_bytes,
     scale_bytes, payload_bytes, reserved) = WALB2_HEADER.unpack_from(raw)
    if magic != b"WALLB2\0\0" or version != 1 or reserved != 0:
        raise ValueError("unsupported WALB2 header")
    if len(raw) != WALB2_HEADER.size + payload_bytes:
        raise ValueError("WALB2 byte accounting mismatch")
    expected_scale = 2 * (out_features + rank + in_features)
    if u_bytes != (out_features * rank + 7) // 8:
        raise ValueError("WALB2 U byte accounting mismatch")
    if v_bytes != (rank * in_features + 7) // 8 or scale_bytes != expected_scale:
        raise ValueError("WALB2 V/scale byte accounting mismatch")
    result = []
    offset = WALB2_HEADER.size
    for _ in range(paths):
        def take(length: int, dtype: torch.dtype) -> torch.Tensor:
            nonlocal offset
            value = torch.frombuffer(
                bytearray(raw[offset:offset + length]), dtype=dtype,
            ).contiguous().to(device)
            offset += length
            return value
        result.append({
            "u": take(u_bytes, torch.uint8),
            "v": take(v_bytes, torch.uint8),
            "row": take(2 * out_features, torch.float16),
            "latent": take(2 * rank, torch.float16),
            "column": take(2 * in_features, torch.float16),
            "out": out_features, "in": in_features, "rank": rank,
        })
    if offset != len(raw):
        raise ValueError("WALB2 trailing bytes")
    return tuple(result)


def _pack_two_bit_ternary(base: torch.Tensor) -> torch.Tensor:
    symbols = (base.reshape(base.shape[0], -1).to(torch.int16) + 1).to(torch.uint8)
    if symbols.shape[1] % 4:
        raise ValueError("T3 row width must be divisible by four")
    lanes = symbols.reshape(symbols.shape[0], -1, 4)
    return (
        lanes[..., 0] | (lanes[..., 1] << 2) |
        (lanes[..., 2] << 4) | (lanes[..., 3] << 6)
    ).contiguous()


def build_hardware_base_cache(
    reference: Any, endpoint: Any, path: Path, *, row_chunk: int = 256,
) -> tuple[torch.Tensor, ...]:
    header = endpoint._read_matrix_header(path)
    rows, columns = int(header["rows"]), int(header["columns"])
    packed_parts, position_parts, sign_parts, alpha_parts, beta_parts = (
        [], [], [], [], [],
    )
    for _, base, sparse, alpha, beta in reference._iter_t3_sparse_rows(
        endpoint, path, row_chunk=row_chunk,
    ):
        chunk_rows, groups, _ = sparse.shape
        positions = sparse.ne(0).nonzero(as_tuple=False)[:, 2].reshape(
            chunk_rows, groups, 8,
        )
        signs = sparse.gather(-1, positions.long()).gt(0).to(torch.uint8)
        sign_bits = torch.zeros((chunk_rows, groups), dtype=torch.uint8)
        for bit in range(8):
            sign_bits |= signs[..., bit] << bit
        packed_parts.append(_pack_two_bit_ternary(base))
        position_parts.append(positions.to(torch.uint8))
        sign_parts.append(sign_bits)
        alpha_parts.append(alpha.to(torch.float16))
        beta_parts.append(beta.to(torch.float16))
    tensors = tuple(torch.cat(values, dim=0).contiguous() for values in (
        packed_parts, position_parts, sign_parts, alpha_parts, beta_parts,
    ))
    return (*tensors, rows, columns)


def write_hardware_base_cache(path: Path, cache: tuple[torch.Tensor, ...]) -> int:
    tensors = tuple(value.detach().contiguous().cpu() for value in cache[:5])
    rows, columns = int(cache[-2]), int(cache[-1])
    groups = columns // 128
    lengths = tuple(value.numel() * value.element_size() for value in tensors)
    header = HARDWARE_BASE_HEADER.pack(
        HARDWARE_BASE_MAGIC, 1, rows, columns, groups, *lengths,
    )
    payload = bytearray(header)
    for value in tensors:
        payload.extend(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    _write_new(path, bytes(payload))
    return path.stat().st_size


def build_hardware_cache(
    checkpoint: Path, output: Path, reference: Any,
    *, progress: Any = None,
) -> dict[str, Any]:
    """Build and attest the portable `.walhw` cache without Triton/CUDA."""
    checkpoint = checkpoint.resolve(strict=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root, _, base_root, base_manifest = reference.load_manifests(checkpoint)
    endpoint, _ = reference.bundled_runtime(root)
    records = []
    for ordinal, row in enumerate(base_manifest["matrices"]):
        source = base_root / str(row["file"])
        target = output / (Path(str(row["file"])).name + ".walhw")
        begin = time.monotonic()
        cache = build_hardware_base_cache(reference, endpoint, source)
        size = write_hardware_base_cache(target, cache)
        # Parse before publication; this catches schema and byte-count errors.
        load_hardware_base_cache(target, torch.device("cpu"))
        record = {
            "ordinal": ordinal, "name": row["name"],
            "source": str(row["file"]), "source_sha256": row["sha256"],
            "file": target.name, "bytes": size, "sha256": _sha256(target),
            "seconds": time.monotonic() - begin,
        }
        records.append(record)
        if progress:
            progress(ordinal + 1, len(base_manifest["matrices"]), record)
    report = {
        "schema": "wal-v77-hardware-cache-v1",
        "checkpoint": str(root),
        "checkpoint_manifest_sha256": _sha256(root / "manifest.json"),
        "format": "T3-2bit + sparse positions/sign bits + FP16 alpha/beta",
        "matrices": records, "matrix_count": len(records),
        "payload_bytes": sum(row["bytes"] for row in records),
        "seconds": time.monotonic() - started,
    }
    manifest_path = output / "manifest.json"
    _write_new(
        manifest_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    canonical_records = [{key: row[key] for key in (
        "ordinal", "name", "file", "bytes", "sha256", "source_sha256",
    )} for row in records]
    matrix_root = hashlib.sha256(json.dumps(
        canonical_records, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    checkpoint_manifest = json.loads((root / "manifest.json").read_text())
    parent_sha = checkpoint_manifest.get("release", {}).get(
        "candidate_manifest_sha256", report["checkpoint_manifest_sha256"],
    )
    converter_sha = _sha256(Path(__file__))
    layout = {
        "endianness": sys.byteorder, "magic": "WALHW001", "version": 1,
        "t3_symbols_per_byte": 4, "t3_bits_per_symbol": 2,
        "sparse_positions_per_g128": 8,
        "sparse_position_dtype": "uint8", "scale_dtype": "fp16",
    }
    compute_abi_sha = hashlib.sha256(json.dumps(
        layout, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    identity = {
        "abi_version": "wal-direct-packed-runtime-abi-v1",
        "cache_manifest_sha256": _sha256(manifest_path),
        "canonical_checkpoint_manifest_sha256": parent_sha,
        "compute_abi_sha256": compute_abi_sha,
        "converter_sha256": converter_sha,
        "endianness": layout["endianness"], "layout_magic": layout["magic"],
        "layout_version": layout["version"],
        "matrix_manifest_root_sha256": matrix_root,
    }
    attestation = {
        "schema": "wal-v77-hardware-cache-attestation-v1",
        "abi_version": identity["abi_version"],
        "canonical_checkpoint_manifest_sha256": parent_sha,
        "converter_sha256": converter_sha,
        "compute_abi_sha256": compute_abi_sha,
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "matrix_manifest_root_sha256": matrix_root,
        "cache_identity_sha256": hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "matrix_count": len(records),
        "matrix_payload_bytes": report["payload_bytes"], "layout": layout,
        "target_backends": ["x86_64", "arm64-neon", "cuda"],
        "cache_depends_on_compile_flags": False,
        "kernel_arithmetic_contract": {
            "decode_accumulator": "fp32", "activation_storage": "bf16/fp32",
            "block_k": 128,
        },
        "policy": "fail closed on parent, manifest, matrix lineage, byte count, or digest mismatch",
    }
    _write_new(
        output / "attestation.json",
        (json.dumps(attestation, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {**report, "attestation": attestation}
