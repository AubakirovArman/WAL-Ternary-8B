"""Runtime backend discovery without importing optional accelerator stacks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
from pathlib import Path
import shutil

import torch


@dataclass(frozen=True)
class RuntimePlatform:
    machine: str
    system: str
    processor: str
    cpu_count: int
    recommended_cpu_threads: int
    torch_version: str
    cuda_available: bool
    cuda_device: str | None
    cuda_capability: tuple[int, int] | None
    mps_available: bool
    apple_silicon: bool
    avx2: bool
    avx512: bool
    compiler: str | None
    ninja: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def _cpu_flags() -> set[str]:
    if platform.system() != "Linux":
        return set()
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return set()
    for line in path.read_text(errors="ignore").splitlines():
        if line.lower().startswith(("flags", "features")) and ":" in line:
            return set(line.split(":", 1)[1].split())
    return set()


def detect_platform() -> RuntimePlatform:
    machine = platform.machine().lower()
    system = platform.system()
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    flags = _cpu_flags()
    logical = os.cpu_count() or 1
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        # Apple does not expose SMT; use all performance+efficiency cores and
        # let ATen's work stealing balance the row-parallel kernels.
        recommended_threads = logical
    else:
        # Packed GEMV is bandwidth-heavy. One thread per physical core is a
        # better default than SMT, and the measured 4-socket Xeon optimum is
        # capped at 64 until a NUMA-partitioned backend is available.
        recommended_threads = max(1, min(64, logical // 2 if logical > 4 else logical))
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
    return RuntimePlatform(
        machine=machine,
        system=system,
        processor=platform.processor(),
        cpu_count=logical,
        recommended_cpu_threads=recommended_threads,
        torch_version=torch.__version__,
        cuda_available=cuda,
        cuda_device=torch.cuda.get_device_name(0) if cuda else None,
        cuda_capability=torch.cuda.get_device_capability(0) if cuda else None,
        mps_available=mps,
        apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
        avx2="avx2" in flags,
        avx512=any(flag.startswith("avx512") for flag in flags),
        compiler=compiler,
        ninja=shutil.which("ninja"),
    )
