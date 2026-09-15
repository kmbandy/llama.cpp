from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from conversion.wp_forge.machines import Machine
from conversion.wp_forge.sink import LocalSink, SSHSink, sink_for


# ssh target for the remote-sink param; localhost by default, or a fleet box that
# accepts key auth from here (e.g. WP_FORGE_TEST_SSH_HOST=mad-lab-2026 on mad-lab-main).
SSH_HOST = os.environ.get("WP_FORGE_TEST_SSH_HOST", "localhost")


def _ssh_ok() -> bool:
    return (
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", SSH_HOST, "true"],
            capture_output=True,
        ).returncode
        == 0
    )


@pytest.fixture(params=["local", "ssh"])
def sink(request, tmp_path: Path):
    root = tmp_path / "root"
    if request.param == "local":
        yield LocalSink(str(root))
        return
    if not _ssh_ok():
        pytest.skip(f"ssh -o BatchMode=yes {SSH_HOST} true failed; ssh param unavailable")
    # remote root lives under /tmp on the target, not the caller's tmp_path
    remote_root = f"/tmp/wp-forge-test-{os.getpid()}-{tmp_path.name}"
    yield SSHSink(SSH_HOST, remote_root)
    subprocess.run(["ssh", "-o", "BatchMode=yes", SSH_HOST, f"rm -rf {remote_root}"], capture_output=True)


def test_write_close_renames_and_hashes(sink) -> None:
    sink.mkdir()
    data = os.urandom(3 * 1024 * 1024 + 17)
    with sink.open("a/b.wpb") as w:
        w.write(data[:1_000_000])
        w.write(data[1_000_000:])
        n, h = w.close()
    assert n == len(data) and h == hashlib.sha256(data).hexdigest()
    assert sink.exists("a/b.wpb") and not sink.exists("a/b.wpb.part")
    assert sink.size("a/b.wpb") == len(data) and sink.sha256("a/b.wpb") == h


def test_abort_removes_part(sink) -> None:
    sink.mkdir()
    w = sink.open("x.wpb")
    w.write(b"abc")
    w.abort()
    assert not sink.exists("x.wpb") and not sink.exists("x.wpb.part")


def test_open_refuses_existing(sink) -> None:
    sink.mkdir()
    sink.put_text("hi", "m.json")
    with pytest.raises(FileExistsError):
        sink.open("m.json")


def test_put_file(sink, tmp_path: Path) -> None:
    sink.mkdir()
    f = tmp_path / "src.bin"
    f.write_bytes(os.urandom(70_000))
    n, h = sink.put_file(f, "d/src.bin")
    assert n == 70_000
    assert sink.sha256("d/src.bin") == h == hashlib.sha256(f.read_bytes()).hexdigest()


def test_sink_for() -> None:
    assert isinstance(sink_for(Machine("l", None, "~/models"), "/tmp/x"), LocalSink)
    assert isinstance(sink_for(Machine("r", "u@h", "/m"), "/m/x"), SSHSink)
