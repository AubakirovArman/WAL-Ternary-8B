"""Bit-exact reference codec for binary low-rank residual bundles.

The format stores one or more paths of the form

``diag(row) @ U_binary @ diag(latent) @ V_binary @ diag(column)``

without retaining floating-point ``U``/``V`` tensors.  Scales are canonical
positive FP16 values; any sign normalization must happen before packing.
The implementation is deliberately a storage/reference decoder, not a claim
of a packed inference kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import stat
import struct
from typing import Sequence

import torch

from .binary import pack_binary_codes, unpack_binary_codes


MAGIC = b"WALLB2\0\0"
VERSION = 1
# magic, version, out, in, rank, paths, U bytes/path, V bytes/path,
# scale bytes/path, total payload bytes, reserved zero.
HEADER = struct.Struct("<8sIIIIIQQQQI")
MAX_PATHS = 8
MAX_DIMENSION = 1 << 20
# A bundle is one matrix-local residual, not a whole-model container.  This cap
# rejects malicious headers before they can trigger multi-terabyte allocations.
MAX_SERIALIZED_BYTES = 64 << 20


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    if not data:
        return torch.empty(0, dtype=dtype)
    return torch.frombuffer(bytearray(data), dtype=dtype).clone()


def _packed_bytes(count: int) -> int:
    if count <= 0:
        raise ValueError("binary code count must be positive")
    return (count + 7) // 8


def _validate_codes(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if tuple(value.shape) != shape:
        raise ValueError(f"binary factor shape is {tuple(value.shape)}, expected {shape}")
    detached = value.detach().contiguous().cpu()
    if not torch.all((detached == -1) | (detached == 1)):
        raise ValueError("binary low-rank factors must contain only {-1,+1}")
    return detached.to(torch.int8).clone()


def _validate_scale(value: torch.Tensor, length: int, label: str) -> torch.Tensor:
    if tuple(value.shape) != (length,):
        raise ValueError(f"{label} scale shape is {tuple(value.shape)}, expected {(length,)}")
    result = value.detach().to(torch.float16).contiguous().cpu().clone()
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError(f"{label} scales must be finite and strictly positive")
    return result


@dataclass(frozen=True)
class BinaryLowRankPath:
    """One canonical binary factor path with positive FP16 scales."""

    u_codes: torch.Tensor
    v_codes: torch.Tensor
    row_scales_fp16: torch.Tensor
    latent_scales_fp16: torch.Tensor
    column_scales_fp16: torch.Tensor


@dataclass(frozen=True)
class BinaryLowRankBundle:
    """A shape-bound collection of equal-rank binary residual paths."""

    shape: tuple[int, int]
    rank: int
    paths: tuple[BinaryLowRankPath, ...]

    def __post_init__(self) -> None:
        out_features, in_features = self.shape
        if (
            out_features <= 0
            or in_features <= 0
            or self.rank <= 0
            or out_features > MAX_DIMENSION
            or in_features > MAX_DIMENSION
            or self.rank > MAX_DIMENSION
        ):
            raise ValueError("binary low-rank dimensions are invalid")
        if not (1 <= len(self.paths) <= MAX_PATHS):
            raise ValueError("binary low-rank bundle path count is invalid")
        for path in self.paths:
            if tuple(path.u_codes.shape) != (out_features, self.rank):
                raise ValueError("U factor shape does not match bundle")
            if tuple(path.v_codes.shape) != (self.rank, in_features):
                raise ValueError("V factor shape does not match bundle")
            if tuple(path.row_scales_fp16.shape) != (out_features,):
                raise ValueError("row scale shape does not match bundle")
            if tuple(path.latent_scales_fp16.shape) != (self.rank,):
                raise ValueError("latent scale shape does not match bundle")
            if tuple(path.column_scales_fp16.shape) != (in_features,):
                raise ValueError("column scale shape does not match bundle")
            if path.u_codes.dtype != torch.int8 or path.v_codes.dtype != torch.int8:
                raise TypeError("binary factors must use int8 logical codes")
            if not torch.all((path.u_codes == -1) | (path.u_codes == 1)):
                raise ValueError("U factor is not binary")
            if not torch.all((path.v_codes == -1) | (path.v_codes == 1)):
                raise ValueError("V factor is not binary")
            for label, scale in (
                ("row", path.row_scales_fp16),
                ("latent", path.latent_scales_fp16),
                ("column", path.column_scales_fp16),
            ):
                if scale.dtype != torch.float16:
                    raise TypeError(f"{label} scales must use FP16")
                if not torch.isfinite(scale).all() or torch.any(scale <= 0):
                    raise ValueError(f"{label} scales must be finite and positive")
        if self.serialized_nbytes > MAX_SERIALIZED_BYTES:
            raise ValueError("binary low-rank bundle exceeds the serialized size cap")

    @property
    def path_count(self) -> int:
        return len(self.paths)

    @property
    def u_bytes_per_path(self) -> int:
        return _packed_bytes(self.shape[0] * self.rank)

    @property
    def v_bytes_per_path(self) -> int:
        return _packed_bytes(self.rank * self.shape[1])

    @property
    def scale_bytes_per_path(self) -> int:
        return 2 * (self.shape[0] + self.rank + self.shape[1])

    @property
    def payload_nbytes(self) -> int:
        return self.path_count * (
            self.u_bytes_per_path
            + self.v_bytes_per_path
            + self.scale_bytes_per_path
        )

    @property
    def serialized_nbytes(self) -> int:
        return HEADER.size + self.payload_nbytes

    def physical_bpw(self, *, include_header: bool = True) -> float:
        size = self.serialized_nbytes if include_header else self.payload_nbytes
        return size * 8 / (self.shape[0] * self.shape[1])


def make_binary_lowrank_path(
    u_codes: torch.Tensor,
    v_codes: torch.Tensor,
    row_scales: torch.Tensor,
    latent_scales: torch.Tensor,
    column_scales: torch.Tensor,
) -> BinaryLowRankPath:
    """Validate and detach one canonical path from training tensors."""

    if u_codes.ndim != 2 or v_codes.ndim != 2:
        raise ValueError("binary factors must be rank-2 matrices")
    out_features, rank = map(int, u_codes.shape)
    v_rank, in_features = map(int, v_codes.shape)
    if rank != v_rank:
        raise ValueError("U/V latent ranks differ")
    return BinaryLowRankPath(
        u_codes=_validate_codes(u_codes, (out_features, rank)),
        v_codes=_validate_codes(v_codes, (rank, in_features)),
        row_scales_fp16=_validate_scale(row_scales, out_features, "row"),
        latent_scales_fp16=_validate_scale(latent_scales, rank, "latent"),
        column_scales_fp16=_validate_scale(column_scales, in_features, "column"),
    )


def make_binary_lowrank_bundle(
    paths: Sequence[BinaryLowRankPath],
) -> BinaryLowRankBundle:
    if not paths:
        raise ValueError("binary low-rank bundle requires at least one path")
    first = paths[0]
    if first.u_codes.ndim != 2 or first.v_codes.ndim != 2:
        raise ValueError("binary factors must be rank-2 matrices")
    out_features, rank = map(int, first.u_codes.shape)
    v_rank, in_features = map(int, first.v_codes.shape)
    if rank != v_rank:
        raise ValueError("U/V latent ranks differ")
    return BinaryLowRankBundle(
        shape=(out_features, in_features), rank=rank, paths=tuple(paths)
    )


def _validate_canonical_padding(packed: torch.Tensor, count: int) -> None:
    remainder = count % 8
    if remainder == 0:
        return
    invalid_mask = (0xFF << remainder) & 0xFF
    if int(packed[-1].item()) & invalid_mask:
        raise ValueError("binary factor has non-zero non-canonical padding bits")


def _decode_codes(data: bytes, count: int, shape: tuple[int, int]) -> torch.Tensor:
    packed = _tensor_from_bytes(data, torch.uint8)
    if packed.numel() != _packed_bytes(count):
        raise ValueError("binary factor payload length is invalid")
    _validate_canonical_padding(packed, count)
    return unpack_binary_codes(packed, count).view(shape)


def decode_binary_lowrank_bundle(
    bundle: BinaryLowRankBundle,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dense decode; no packed-kernel or speed claim is implied."""

    out_features, in_features = bundle.shape
    result = torch.zeros((out_features, in_features), dtype=torch.float32)
    for path in bundle.paths:
        left = (
            path.u_codes.float()
            * path.row_scales_fp16.float().view(-1, 1)
            * path.latent_scales_fp16.float().view(1, -1)
        )
        right = path.v_codes.float() * path.column_scales_fp16.float().view(1, -1)
        result.add_(left @ right)
    return result.to(dtype)


def write_binary_lowrank_bundle(
    path: str | Path,
    bundle: BinaryLowRankBundle,
) -> int:
    """Create and fsync one canonical bundle without an overwrite window."""

    # Frozen dataclasses do not make their tensor storage immutable.  Snapshot and
    # revalidate every tensor immediately before serialization so a caller cannot
    # mutate a previously validated bundle into a non-canonical artifact.
    snapshot = make_binary_lowrank_bundle(
        tuple(
            make_binary_lowrank_path(
                path_value.u_codes,
                path_value.v_codes,
                path_value.row_scales_fp16,
                path_value.latent_scales_fp16,
                path_value.column_scales_fp16,
            )
            for path_value in bundle.paths
        )
    )
    if snapshot.shape != bundle.shape or snapshot.rank != bundle.rank:
        raise ValueError("binary low-rank bundle tensors changed shape")

    output = Path(path)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        snapshot.shape[0],
        snapshot.shape[1],
        snapshot.rank,
        snapshot.path_count,
        snapshot.u_bytes_per_path,
        snapshot.v_bytes_per_path,
        snapshot.scale_bytes_per_path,
        snapshot.payload_nbytes,
        0,
    )
    if not output.name:
        raise ValueError("binary low-rank output filename is invalid")

    # Publish only after the complete inode has been fsynced.  A hard link gives
    # create-only atomic publication: an existing final path is never replaced,
    # and a write failure can leave at most an unadvertised random staging file.
    parent_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    parent_descriptor = os.open(output.parent, parent_flags)
    staging_name = f".walb2-{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    opened: os.stat_result | None = None
    staging_present = False
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            staging_name, flags, 0o600, dir_fd=parent_descriptor
        )
        staging_present = True
        opened = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb", buffering=0, closefd=False) as handle:
                _write_exact(handle, header, "header")
                for ordinal, path_value in enumerate(snapshot.paths):
                    _write_exact(
                        handle,
                        _raw_bytes(pack_binary_codes(path_value.u_codes)),
                        f"path {ordinal} U",
                    )
                    _write_exact(
                        handle,
                        _raw_bytes(pack_binary_codes(path_value.v_codes)),
                        f"path {ordinal} V",
                    )
                    _write_exact(
                        handle,
                        _raw_bytes(path_value.row_scales_fp16),
                        f"path {ordinal} row scales",
                    )
                    _write_exact(
                        handle,
                        _raw_bytes(path_value.latent_scales_fp16),
                        f"path {ordinal} latent scales",
                    )
                    _write_exact(
                        handle,
                        _raw_bytes(path_value.column_scales_fp16),
                        f"path {ordinal} column scales",
                    )
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
            descriptor = None

        staged = os.stat(
            staging_name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(staged.st_mode)
            or (staged.st_dev, staged.st_ino) != (opened.st_dev, opened.st_ino)
            or staged.st_size != snapshot.serialized_nbytes
        ):
            raise RuntimeError("binary low-rank staging inode changed or is incomplete")
        os.link(
            staging_name,
            output.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.fsync(parent_descriptor)
        published = os.stat(
            output.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino) != (opened.st_dev, opened.st_ino)
            or published.st_size != snapshot.serialized_nbytes
        ):
            raise RuntimeError("binary low-rank published inode is inconsistent")
        os.unlink(staging_name, dir_fd=parent_descriptor)
        staging_present = False
        os.fsync(parent_descriptor)
        return int(published.st_size)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if staging_present and opened is not None:
            try:
                current = os.stat(
                    staging_name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
                    os.unlink(staging_name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve the original serialization/publication failure.  The
                # random staging name is never treated as a valid artifact.
                pass
        raise
    finally:
        os.close(parent_descriptor)


def _write_exact(handle, value: bytes, label: str) -> None:
    remaining = memoryview(value)
    while remaining:
        written = handle.write(remaining)
        if written is None or written <= 0:
            raise OSError(f"binary low-rank {label} payload could not be written")
        remaining = remaining[written:]


def _read_exact(handle, length: int, label: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise ValueError(f"binary low-rank {label} payload is truncated")
    return value


def read_binary_lowrank_bundle(path: str | Path) -> BinaryLowRankBundle:
    """Authenticate structure, canonical padding, codes and FP16 scales."""

    source = Path(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("binary low-rank source is not a regular file")
        if before.st_size > MAX_SERIALIZED_BYTES:
            raise ValueError("binary low-rank source exceeds the serialized size cap")
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as handle:
            raw_header = _read_exact(handle, HEADER.size, "header")
            (
                magic,
                version,
                out_features,
                in_features,
                rank,
                path_count,
                u_bytes,
                v_bytes,
                scale_bytes,
                payload_bytes,
                reserved,
            ) = HEADER.unpack(raw_header)
            if magic != MAGIC or version != VERSION:
                raise ValueError("unsupported binary low-rank format")
            if reserved != 0:
                raise ValueError("binary low-rank reserved header field is non-zero")
            if (
                out_features <= 0
                or in_features <= 0
                or rank <= 0
                or out_features > MAX_DIMENSION
                or in_features > MAX_DIMENSION
                or rank > MAX_DIMENSION
                or not (1 <= path_count <= MAX_PATHS)
            ):
                raise ValueError("binary low-rank header dimensions are invalid")
            expected_u = _packed_bytes(out_features * rank)
            expected_v = _packed_bytes(rank * in_features)
            expected_scales = 2 * (out_features + rank + in_features)
            expected_payload = path_count * (expected_u + expected_v + expected_scales)
            if (
                HEADER.size + expected_payload > MAX_SERIALIZED_BYTES
                or
                u_bytes != expected_u
                or v_bytes != expected_v
                or scale_bytes != expected_scales
                or payload_bytes != expected_payload
                or before.st_size != HEADER.size + expected_payload
            ):
                raise ValueError("binary low-rank header accounting is inconsistent")

            paths: list[BinaryLowRankPath] = []
            for ordinal in range(path_count):
                u_codes = _decode_codes(
                    _read_exact(handle, expected_u, f"path {ordinal} U"),
                    out_features * rank,
                    (out_features, rank),
                )
                v_codes = _decode_codes(
                    _read_exact(handle, expected_v, f"path {ordinal} V"),
                    rank * in_features,
                    (rank, in_features),
                )
                row_bytes = 2 * out_features
                latent_bytes = 2 * rank
                column_bytes = 2 * in_features
                row = _tensor_from_bytes(
                    _read_exact(handle, row_bytes, f"path {ordinal} row scales"),
                    torch.float16,
                )
                latent = _tensor_from_bytes(
                    _read_exact(handle, latent_bytes, f"path {ordinal} latent scales"),
                    torch.float16,
                )
                column = _tensor_from_bytes(
                    _read_exact(handle, column_bytes, f"path {ordinal} column scales"),
                    torch.float16,
                )
                paths.append(
                    make_binary_lowrank_path(u_codes, v_codes, row, latent, column)
                )
            if handle.read(1):
                raise ValueError("binary low-rank file contains trailing bytes")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError("binary low-rank file changed while decoding")
    finally:
        os.close(descriptor)
    return BinaryLowRankBundle(
        shape=(out_features, in_features), rank=rank, paths=tuple(paths)
    )
