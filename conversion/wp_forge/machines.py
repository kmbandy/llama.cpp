"""Fleet machine config: ~/.config/mad-lab-agents/machines.json.

Entry shape (dashboard-owned): {"name": {"local": true} | {"ssh": "user@host"}}.
wp-forge adds one optional key, "models_dir" (default "~/models", expanded on
the target machine, never on the caller).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("~/.config/mad-lab-agents/machines.json").expanduser()
_ALLOWED = {"local", "ssh", "models_dir"}
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


@dataclass(frozen=True)
class Machine:
    name: str
    ssh: str | None  # "user@host", or None for local
    models_dir: str  # absolute, or "~" path to be expanded on that machine

    @property
    def is_local(self) -> bool:
        return self.ssh is None


def load_machines(path: Path | None = None) -> dict[str, Machine]:
    path = path if path is not None else DEFAULT_PATH
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"machines.json {path}: top level must be an object")
    out: dict[str, Machine] = {}
    for name, spec in raw.items():
        extra = set(spec) - _ALLOWED
        if extra:
            raise ValueError(f"machines.json {name}: unknown keys {sorted(extra)}")
        if bool(spec.get("local")) == bool(spec.get("ssh")):
            raise ValueError(f"machines.json {name}: exactly one of local/ssh required")
        out[name] = Machine(name, spec.get("ssh"), spec.get("models_dir", "~/models"))
    return out


def run_on(
    m: Machine, argv: list[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    """Run argv on m: exec locally, or shlex-join it as a single remote ssh command."""
    if m.is_local:
        cmd = argv
    else:
        cmd = SSH_BASE + [m.ssh, " ".join(shlex.quote(a) for a in argv)]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def expand_remote_home(m: Machine, path: str) -> str:
    """Expand a leading ~ against the home of m's user (never the caller's)."""
    if not path.startswith("~"):
        return path
    if m.is_local:
        return os.path.expanduser(path)
    home = run_on(m, ["sh", "-c", "echo ~"]).stdout.strip()
    return home + path[1:]


def free_bytes(m: Machine, path: str) -> int:
    path = expand_remote_home(m, path)
    # nearest existing ancestor: df on a not-yet-created dir fails
    probe = path
    script = f"p={shlex.quote(probe)}; while [ ! -d \"$p\" ]; do p=$(dirname \"$p\"); done; df -B1 --output=avail \"$p\" | tail -1"
    out = run_on(m, ["sh", "-c", script])
    return int(out.stdout.strip())
