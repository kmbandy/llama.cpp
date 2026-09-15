from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def machines_json(tmp_path: Path) -> Path:
    p = tmp_path / "machines.json"
    p.write_text('{"box-a": {"local": true}, "box-b": {"ssh": "u@box-b", "models_dir": "/mnt/nvme"}}')
    return p
