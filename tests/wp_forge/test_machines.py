from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conversion.wp_forge.machines import Machine, load_machines, run_on, free_bytes


def test_load_machines_defaults_models_dir(machines_json: Path) -> None:
    ms = load_machines(machines_json)
    assert ms["box-a"] == Machine("box-a", None, "~/models")
    assert ms["box-b"] == Machine("box-b", "u@box-b", "/mnt/nvme")
    assert ms["box-a"].is_local and not ms["box-b"].is_local


def test_load_machines_rejects_unknown_keys(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text('{"x": {"local": true, "bogus": 1}}')
    with pytest.raises(ValueError, match="bogus"):
        load_machines(p)


def test_run_on_local_executes() -> None:
    out = run_on(Machine("l", None, "~/models"), ["echo", "hi"])
    assert out.stdout.strip() == "hi"


def test_run_on_remote_builds_ssh_argv(monkeypatch) -> None:
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_on(Machine("r", "u@h", "/m"), ["df", "-B1", "/m"])
    assert seen["argv"][:3] == ["ssh", "-o", "BatchMode=yes"] and seen["argv"][-2] == "u@h"
    assert seen["argv"][-1] == "df -B1 /m"


def test_free_bytes_local(tmp_path: Path) -> None:
    assert free_bytes(Machine("l", None, "~/models"), str(tmp_path)) > 0
