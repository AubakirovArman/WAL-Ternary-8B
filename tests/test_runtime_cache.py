from __future__ import annotations

import struct

import torch

from wal_tat.runtime.cache import (
    HARDWARE_BASE_HEADER, HARDWARE_BASE_MAGIC, build_hardware_cache,
    load_hardware_base_cache,
)


def test_load_minimal_hardware_cache(tmp_path):
    rows, columns, groups = 1, 128, 1
    payloads = (
        bytes(columns // 4), bytes(groups * 8), bytes(groups),
        torch.ones((rows, groups), dtype=torch.float16).numpy().tobytes(),
        torch.zeros((rows, groups), dtype=torch.float16).numpy().tobytes(),
    )
    header = HARDWARE_BASE_HEADER.pack(
        HARDWARE_BASE_MAGIC, 1, rows, columns, groups,
        *(len(value) for value in payloads),
    )
    path = tmp_path / "matrix.walhw"
    path.write_bytes(header + b"".join(payloads))
    loaded = load_hardware_base_cache(path, torch.device("cpu"))
    assert loaded[-2:] == (rows, columns)
    assert loaded[0].shape == (rows, columns // 4)
    assert loaded[3].item() == 1.0


def test_rejects_truncated_hardware_cache(tmp_path):
    path = tmp_path / "bad.walhw"
    path.write_bytes(struct.pack("<Q", 1))
    try:
        load_hardware_base_cache(path, torch.device("cpu"))
    except ValueError as error:
        assert "short" in str(error)
    else:
        raise AssertionError("truncated cache accepted")


def test_build_and_attest_minimal_cache(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    base = checkpoint / "base"
    base.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}")
    source = base / "m.wal"
    source.write_bytes(b"source")

    class Endpoint:
        @staticmethod
        def _read_matrix_header(path):
            return {"rows": 1, "columns": 128}

    class Reference:
        @staticmethod
        def load_manifests(root):
            return root, {}, base, {"matrices": [{
                "name": "m.weight", "file": "m.wal", "sha256": "source-sha",
            }]}

        @staticmethod
        def bundled_runtime(root):
            return Endpoint(), object()

        @staticmethod
        def _iter_t3_sparse_rows(endpoint, path, row_chunk):
            ternary = torch.zeros((1, 128), dtype=torch.int8)
            sparse = torch.zeros((1, 1, 128), dtype=torch.int8)
            sparse[0, 0, :8] = 1
            yield 0, ternary, sparse, torch.ones((1, 1)), torch.ones((1, 1))

    output = tmp_path / "cache"
    report = build_hardware_cache(checkpoint, output, Reference())
    assert report["matrix_count"] == 1
    assert (output / "manifest.json").is_file()
    assert (output / "attestation.json").is_file()
    loaded = load_hardware_base_cache(output / "m.wal.walhw", torch.device("cpu"))
    assert loaded[-2:] == (1, 128)
