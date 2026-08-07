"""Runtime V17: V12 scheduling with exact paired activation loads."""
from __future__ import annotations

from pathlib import Path

import torch

import qwen_v77_direct_packed_cuda_operator_v17 as operator_v17
import qwen_v77_direct_packed_triton_runtime_v12 as runtime_v12


class DirectPackedWALLinearV17(runtime_v12.DirectPackedWALLinearV12):
    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        flattened = value.reshape(-1, value.shape[-1]).shape[0]
        if 2 <= flattened <= 16:
            base, paths = self._direct(value.device)
            return operator_v17.direct_wal_linear_v17(
                value, base, paths,
            ).to(self.output_dtype)
        return super().forward(value)


def load_direct_model(
    checkpoint: str | Path,
    *, device: str = "cuda:0", verify_hashes: bool = True,
    hardware_cache: str | Path | None = None,
    verify_hardware_cache_hashes: bool = True,
):
    model, report = runtime_v12.load_direct_model(
        checkpoint, device=device, verify_hashes=verify_hashes,
        hardware_cache=hardware_cache,
        verify_hardware_cache_hashes=verify_hardware_cache_hashes,
    )
    replaced = 0
    for module in model.modules():
        if type(module) is runtime_v12.DirectPackedWALLinearV12:
            module.__class__ = DirectPackedWALLinearV17
            replaced += 1
    report["decode_kernel"] = {
        "version": "cuda-v17-paired-activation-loads",
        "batch_one": "accepted-fused-v7",
        "batch_2_to_16": "V12 reductions plus exact bf16x2/half2 activation loads",
        "prefill_m_gt_16": "temporary current-operator unpack V9",
        "persistent_dense_body": False,
        "replaced_matrices": replaced,
    }
    return model, report


def cache_accounting(model: torch.nn.Module) -> dict[str, int]:
    return runtime_v12.cache_accounting(model)
