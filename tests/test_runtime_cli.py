from __future__ import annotations

import json

from wal_tat.runtime import cli
from wal_tat.runtime.platform import detect_platform


def test_doctor_is_json(capsys):
    assert cli.main(["doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["machine"]
    assert payload["recommended_cpu_threads"] >= 1
    assert payload["recommended_device"] in {"cpu", "mps", "cuda:0"}


def test_inspect_minimal_checkpoint(tmp_path, capsys):
    manifest = {
        "variant": "test",
        "release": {"version": "v0"},
        "accounting": {"unique_parameters": 10, "maximum_whole_bpw": 3.0},
        "base": {"matrix_count": 2},
        "overlays": [{"name": "x"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "packed_runtime.py").write_text("# test\n")
    assert cli.main(["inspect", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["packed_matrices"] == 2
    assert payload["overlays"] == 1
    assert payload["runtime_files"] == ["packed_runtime.py"]


def test_platform_thread_default_is_bounded():
    info = detect_platform()
    assert 1 <= info.recommended_cpu_threads <= info.cpu_count
