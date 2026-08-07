"""Runtime V8: V7 batch-one kernels plus true CUDA batch 2..16 decode."""
from __future__ import annotations

from pathlib import Path

import torch

import qwen_v77_direct_packed_cuda_operator_v7 as operator_v7
import qwen_v77_direct_packed_cuda_operator_v8 as operator_v8
import qwen_v77_direct_packed_int4_head_v8 as head_v8
import qwen_v77_direct_packed_triton_runtime_v1 as runtime_v1
import qwen_v77_direct_packed_triton_runtime_v2 as runtime_v2
import qwen_v77_packed_reference_runtime as reference


class DirectPackedWALLinearV8(runtime_v2.DirectPackedWALLinearV2):
    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        flattened = value.reshape(-1, value.shape[-1]).shape[0]
        base, paths = self._direct(value.device)
        if flattened == 1:
            return operator_v7.direct_wal_linear_v7(value, base, paths).to(self.output_dtype)
        if 2 <= flattened <= 16:
            return operator_v8.direct_wal_linear_v8(value, base, paths).to(self.output_dtype)
        return runtime_v1.DirectPackedWALLinear.forward(self, value)


def load_direct_model(
    checkpoint: str | Path,
    *, device: str = "cuda:0", verify_hashes: bool = True,
    hardware_cache: str | Path | None = None,
    verify_hardware_cache_hashes: bool = True,
):
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    cache_path = None if hardware_cache is None else Path(hardware_cache).resolve(strict=True)
    cache_report = None if cache_path is None else runtime_v1.validate_hardware_cache(
        checkpoint_path, cache_path,
        verify_matrix_hashes=verify_hardware_cache_hashes,
    )
    DirectPackedWALLinearV8.hardware_cache_root = cache_path
    model, report = reference.load_packed_model(
        checkpoint_path, device=device, dtype=torch.bfloat16, row_chunk=256,
        verify_hashes=verify_hashes, wal_linear_class=DirectPackedWALLinearV8,
        endpoint_linear_class=head_v8.DirectPackedINT4HeadV8,
    )
    _, _, base_root, base_manifest = reference.load_manifests(checkpoint_path)
    endpoint, _ = reference.bundled_runtime(checkpoint_path)
    endpoints = {str(row["name"]): row for row in base_manifest["endpoints"]}
    embedding_row = endpoints["model.embed_tokens.weight"]
    model.model.embed_tokens = runtime_v2.DirectPackedINT3Embedding(
        base_root / str(embedding_row["file"]), endpoint=endpoint,
        output_dtype=torch.bfloat16,
    )
    report["hardware_cache_attestation"] = cache_report
    report["decode_kernel"] = {
        "version": "cuda-cpp-batched-body-v8",
        "batch_one": "accepted-fused-v7",
        "batch_2_to_16": "cuda-cpp-2d-grid-t3-walb2 plus batched-int4-head",
        "prefill_fallback": "direct-packed-v1",
    }
    return model, report


def cache_accounting(model: torch.nn.Module) -> dict[str, int]:
    return runtime_v2.cache_accounting(model)
