"""Build the reusable 2-bit compute cache for V77 Transformer matrices."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

import qwen_v77_packed_reference_runtime as ref
from qwen_v77_direct_packed_triton_operator_v1 import (
    build_hardware_base_cache,
    load_hardware_base_cache,
    write_hardware_base_cache,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def build(checkpoint: Path, output: Path) -> dict:
    started = time.monotonic()
    root, manifest, base_root, base_manifest = ref.load_manifests(checkpoint)
    endpoint, _ = ref.bundled_runtime(root)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(base_manifest["matrices"]):
        source = base_root / row["file"]
        target = output / (Path(row["file"]).name + ".walhw")
        begin = time.monotonic()
        cache = build_hardware_base_cache(endpoint, source, torch.device("cpu"), 256)
        size = write_hardware_base_cache(target, cache)
        # Fail closed by parsing the just-published cache before recording it.
        check = load_hardware_base_cache(target, torch.device("cpu"))
        expected = sum(t.numel() * t.element_size() for t in check[:5])
        if size != expected + 64:
            raise RuntimeError("hardware cache serialized size mismatch")
        records.append({
            "ordinal": index,
            "name": row["name"],
            "source": str(row["file"]),
            "source_sha256": row["sha256"],
            "file": target.name,
            "bytes": size,
            "sha256": sha256(target),
            "seconds": time.monotonic() - begin,
        })
        print(json.dumps({"event": "matrix", "completed": index + 1,
                          "total": len(base_manifest["matrices"]),
                          "name": row["name"], "bytes": size}), flush=True)
    report = {
        "schema": "wal-v77-hardware-cache-v1",
        "checkpoint": str(root),
        "checkpoint_manifest_sha256": sha256(root / "manifest.json"),
        "format": "T3-2bit + sparse positions/sign bits + FP16 alpha/beta",
        "matrices": records,
        "matrix_count": len(records),
        "payload_bytes": sum(x["bytes"] for x in records),
        "seconds": time.monotonic() - started,
    }
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    manifest_path = output / "manifest.json"
    descriptor = os.open(
        manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o444
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=str(ref.DEFAULT_CHECKPOINT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build(Path(args.checkpoint).resolve(strict=True), Path(args.output).resolve())
    print(json.dumps({key: report[key] for key in (
        "schema", "matrix_count", "payload_bytes", "seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
