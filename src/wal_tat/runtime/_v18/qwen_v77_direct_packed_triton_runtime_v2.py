"""V77 runtime with packed-byte decode kernels and the frozen v1 prefill path."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import triton
import triton.language as tl

import qwen_v77_direct_packed_triton_operator_v2 as operator_v2
import qwen_v77_direct_packed_triton_runtime_v1 as runtime_v1
import qwen_v77_packed_reference_runtime as reference


# Frozen from the representative shape sweep.  The same conservative launch
# geometry is used for all decode matrices until a per-shape policy is proven.
DECODE_BLOCK_N = 16
DECODE_BLOCK_BYTES = 16
DECODE_NUM_WARPS = 4


def decode_launch_policy(
    base: tuple[torch.Tensor, ...], paths: tuple[dict, ...],
) -> tuple[int, int, int]:
    """Frozen H200 policy from the four representative shape sweeps."""
    out_features, in_features = int(base[-2]), int(base[-1])
    if not paths:
        return 16, 32, 2
    if out_features == 12288 and in_features == 4096:
        return 16, 32, 2
    return DECODE_BLOCK_N, DECODE_BLOCK_BYTES, DECODE_NUM_WARPS


@triton.jit
def _direct_int3_embedding_kernel(
    ids_ptr, codes_ptr, scales_ptr, out_ptr,
    embedding_dim: tl.constexpr, groups: tl.constexpr,
    bytes_per_row: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Random-access INT3 row decode entirely on the GPU."""
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = column < embedding_dim
    token_id = tl.load(ids_ptr + row).to(tl.int64)
    bit = column * 3
    byte = bit // 8
    shift = bit % 8
    base = token_id * bytes_per_row
    low = tl.load(codes_ptr + base + byte, mask=mask, other=0).to(tl.int32)
    high = tl.load(
        codes_ptr + base + byte + 1,
        mask=mask & (shift > 5), other=0,
    ).to(tl.int32)
    unsigned = ((low | (high << 8)) >> shift) & 7
    code = unsigned - 3
    scale = tl.load(
        scales_ptr + token_id * groups + column // 128,
        mask=mask, other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + row * embedding_dim + column,
        code.to(tl.float32) * scale,
        mask=mask,
    )


class DirectPackedINT3Embedding(nn.Module):
    """Persistent packed INT3 endpoint with graph-safe GPU row lookup."""

    def __init__(self, path: Path, *, endpoint, output_dtype: torch.dtype):
        super().__init__()
        self.path = Path(path)
        self.endpoint = endpoint
        header = endpoint._read_endpoint_header(self.path)
        self.spec = header["spec"]
        self.payload = header["payload"]
        if self.spec.name != "int3-symmetric" or int(self.spec.code_bits) != 3:
            raise ValueError("DirectPackedINT3Embedding requires int3-symmetric")
        self.num_embeddings = int(self.payload.rows)
        self.embedding_dim = int(self.payload.columns)
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
                    raise ValueError("truncated direct INT3 embedding")
                if handle.read(1):
                    raise ValueError("direct INT3 embedding has trailing bytes")
            codes = torch.frombuffer(bytearray(raw_codes), dtype=torch.uint8).to(device)
            scales = torch.frombuffer(bytearray(raw_scales), dtype=torch.float16).to(device)
            if not torch.isfinite(scales).all() or torch.any(scales <= 0):
                raise ValueError("invalid direct INT3 embedding scales")
            cached = (codes.contiguous(), scales.contiguous())
            self._packed_cache[key] = cached
        return cached

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        codes, scales = self._packed(input_ids.device)
        flat_ids = input_ids.reshape(-1).contiguous()
        output = torch.empty(
            (flat_ids.numel(), self.embedding_dim),
            device=input_ids.device, dtype=self.output_dtype,
        )
        groups = self.embedding_dim // int(self.payload.group_size)
        block_k = 256
        _direct_int3_embedding_kernel[
            (flat_ids.numel(), triton.cdiv(self.embedding_dim, block_k))
        ](
            flat_ids, codes, scales, output,
            self.embedding_dim, groups, self.embedding_dim * 3 // 8,
            BLOCK_K=block_k, num_warps=4,
        )
        return output.reshape(*input_ids.shape, self.embedding_dim)


class DirectPackedWALLinearV2(runtime_v1.DirectPackedWALLinear):
    """Use byte-lane v2 kernels for M=1 and retain the proven v1 fallback."""

    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.reshape(-1, value.shape[-1]).shape[0] != 1:
            return super().forward(value)
        base, paths = self._direct(value.device)
        block_n, block_bytes, num_warps = decode_launch_policy(base, paths)
        result = operator_v2.direct_wal_linear_v2(
            value, base, paths,
            block_n=block_n,
            block_bytes=block_bytes,
            num_warps=num_warps,
        )
        return result.to(self.output_dtype)


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
    cache_report = None if cache_path is None else runtime_v1.validate_hardware_cache(
        checkpoint_path, cache_path,
        verify_matrix_hashes=verify_hardware_cache_hashes,
    )
    DirectPackedWALLinearV2.hardware_cache_root = cache_path
    model, report = reference.load_packed_model(
        checkpoint_path,
        device=device,
        dtype=torch.bfloat16,
        row_chunk=256,
        verify_hashes=verify_hashes,
        wal_linear_class=DirectPackedWALLinearV2,
        endpoint_linear_class=runtime_v1.DirectPackedINT4Head,
    )
    _, _, base_root, base_manifest = reference.load_manifests(checkpoint_path)
    endpoint, _ = reference.bundled_runtime(checkpoint_path)
    endpoints = {str(row["name"]): row for row in base_manifest["endpoints"]}
    embedding_row = endpoints["model.embed_tokens.weight"]
    model.model.embed_tokens = DirectPackedINT3Embedding(
        base_root / str(embedding_row["file"]),
        endpoint=endpoint,
        output_dtype=torch.bfloat16,
    )
    report["hardware_cache_attestation"] = cache_report
    report["decode_kernel"] = {
        "version": "packed-byte-v2",
        "block_n": DECODE_BLOCK_N,
        "block_bytes": DECODE_BLOCK_BYTES,
        "num_warps": DECODE_NUM_WARPS,
        "shape_policy": {
            "uncorrected": [16, 32, 2],
            "gate_up_12288x4096": [16, 32, 2],
            "other_corrected": [16, 16, 4],
        },
        "prefill_fallback": "direct-packed-v1",
    }
    return model, report


def cache_accounting(model: torch.nn.Module) -> dict[str, int]:
    result = runtime_v1.cache_accounting(model)
    embedding = 0
    for module in model.modules():
        if isinstance(module, DirectPackedINT3Embedding):
            for codes, scales in module._packed_cache.values():
                embedding += codes.numel() * codes.element_size()
                embedding += scales.numel() * scales.element_size()
    result["direct_packed_int3_embedding_bytes"] = embedding
    result["total_persistent_weight_cache_bytes"] += embedding
    return result
