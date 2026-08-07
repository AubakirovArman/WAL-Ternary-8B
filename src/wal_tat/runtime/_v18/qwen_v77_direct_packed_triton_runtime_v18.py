"""Runtime V18: exact activation-pair kernels for single and batched decode."""
from __future__ import annotations

from pathlib import Path

import torch

import qwen_v77_direct_packed_cuda_operator_v18 as operator_v18
import qwen_v77_direct_packed_triton_runtime_v17 as runtime_v17


class DirectPackedWALLinearV18(runtime_v17.DirectPackedWALLinearV17):
    @torch.no_grad()
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        flattened = value.reshape(-1, value.shape[-1]).shape[0]
        if flattened == 1:
            base, paths = self._direct(value.device)
            return operator_v18.direct_wal_linear_v18(
                value, base, paths,
            ).to(self.output_dtype)
        return super().forward(value)


def load_direct_model(
    checkpoint: str | Path,
    *, device: str = "cuda:0", verify_hashes: bool = True,
    hardware_cache: str | Path | None = None,
    verify_hardware_cache_hashes: bool = True,
):
    model, report = runtime_v17.load_direct_model(
        checkpoint, device=device, verify_hashes=verify_hashes,
        hardware_cache=hardware_cache,
        verify_hardware_cache_hashes=verify_hardware_cache_hashes,
    )
    replaced = 0
    for module in model.modules():
        if type(module) is runtime_v17.DirectPackedWALLinearV17:
            module.__class__ = DirectPackedWALLinearV18
            replaced += 1
    report["decode_kernel"] = {
        "version": "cuda-v18-exact-paired-activation-loads",
        "batch_one": "V7 fusion plus exact bf16x2/half2 activation loads",
        "batch_2_to_16": "V17 exact paired activation loads",
        "prefill_m_gt_16": "temporary current-operator unpack V9",
        "persistent_dense_body": False,
        "replaced_matrices": replaced,
    }
    return model, report


def cache_accounting(model: torch.nn.Module) -> dict[str, int]:
    return runtime_v17.cache_accounting(model)
