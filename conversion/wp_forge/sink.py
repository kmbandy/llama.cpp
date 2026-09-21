"""Where output bytes land. Local or over ssh, same interface, sha256 computed on the way through.

Every blob is written as <path>.part and renamed on close, after the byte count on the
far side is checked against what we sent. Existing finals are never overwritten.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import Protocol

from .machines import Machine, SSH_BASE

CHUNK = 8 * 1024 * 1024


class BlobWriter(Protocol):
    def write(self, b: bytes | memoryview) -> None: ...
    def close(self) -> tuple[int, str]: ...
    def abort(self) -> None: ...
    def __enter__(self) -> "BlobWriter": ...
    def __exit__(self, et: object, ev: object, tb: object) -> None: ...


class Sink(Protocol):
    root: str
    def open(self, relpath: str) -> BlobWriter: ...
    def exists(self, relpath: str) -> bool: ...
    def size(self, relpath: str) -> int | None: ...
    def sha256(self, relpath: str) -> str: ...
    def put_file(self, local: Path, relpath: str) -> tuple[int, str]: ...
    def put_text(self, text: str, relpath: str) -> None: ...
    def get_text(self, relpath: str) -> str | None: ...
    def mkdir(self) -> None: ...


class _Base:
    def __enter__(self) -> _Base:
        return self

    def __exit__(self, et: object, ev: object, tb: object) -> None:
        if et is not None:
            self.abort()


class _LocalWriter(_Base):
    def __init__(self, final: Path):
        self.final = final
        self.part = final.with_name(final.name + ".part")
        self.part.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.part, "wb")
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b: bytes | memoryview) -> None:
        self.f.write(b)
        self.h.update(b)
        self.n += len(b)

    def close(self) -> tuple[int, str]:
        self.f.close()
        got = self.part.stat().st_size
        if got != self.n:
            raise IOError(f"{self.part}: wrote {self.n} bytes, file has {got}")
        os.replace(self.part, self.final)
        return self.n, self.h.hexdigest()

    def abort(self) -> None:
        try:
            self.f.close()
        finally:
            self.part.unlink(missing_ok=True)


class _SSHWriter(_Base):
    def __init__(self, host: str, final: str):
        self.host, self.final, self.part = host, final, final + ".part"
        d = shlex.quote(os.path.dirname(final))
        cmd = f"mkdir -p {d} && cat > {shlex.quote(self.part)}"
        self.p = subprocess.Popen(
            SSH_BASE + [host, cmd], stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b: bytes | memoryview) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(b)
        self.h.update(b)
        self.n += len(b)

    def close(self) -> tuple[int, str]:
        assert self.p.stdin is not None
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            err = self.p.stderr.read().decode() if self.p.stderr else ""
            raise IOError(f"ssh {self.host} cat > {self.part}: rc={rc} {err[-500:]}")
        got = int(_ssh(self.host, f"stat -c %s {shlex.quote(self.part)}").strip())
        if got != self.n:
            raise IOError(f"{self.host}:{self.part}: sent {self.n} bytes, remote has {got}")
        _ssh(
            self.host,
            f"mv -n {shlex.quote(self.part)} {shlex.quote(self.final)} && test -e {shlex.quote(self.final)}",
        )
        return self.n, self.h.hexdigest()

    def abort(self) -> None:
        try:
            if self.p.stdin:
                self.p.stdin.close()
            self.p.wait()
        finally:
            _ssh(self.host, f"rm -f {shlex.quote(self.part)}", check=False)


def _ssh(host: str, cmd: str, check: bool = True) -> str:
    r = subprocess.run(SSH_BASE + [host, cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise IOError(f"ssh {host} {cmd!r}: rc={r.returncode} {r.stderr[-500:]}")
    return r.stdout


class LocalSink:
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)

    def _p(self, rel: str) -> Path:
        return Path(self.root) / rel

    def mkdir(self) -> None:
        Path(self.root).mkdir(parents=True, exist_ok=True)

    def open(self, relpath: str) -> BlobWriter:
        p = self._p(relpath)
        if p.exists():
            raise FileExistsError(str(p))
        return _LocalWriter(p)

    def exists(self, relpath: str) -> bool:
        return self._p(relpath).exists()

    def size(self, relpath: str) -> int | None:
        p = self._p(relpath)
        return p.stat().st_size if p.exists() else None

    def sha256(self, relpath: str) -> str:
        h = hashlib.sha256()
        with open(self._p(relpath), "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    def put_file(self, local: Path, relpath: str) -> tuple[int, str]:
        with self.open(relpath) as w:
            with open(local, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    w.write(chunk)
            return w.close()

    def put_text(self, text: str, relpath: str) -> None:
        with self.open(relpath) as w:
            w.write(text.encode())
            w.close()


    def get_text(self, relpath: str) -> str | None:
        try:
            return self._p(relpath).read_text()
        except OSError:
            return None


class SSHSink:
    def __init__(self, host: str, root: str):
        self.host, self.root = host, root

    def _p(self, rel: str) -> str:
        return f"{self.root.rstrip('/')}/{rel}"

    def mkdir(self) -> None:
        _ssh(self.host, f"mkdir -p {shlex.quote(self.root)}")

    def open(self, relpath: str) -> BlobWriter:
        if self.exists(relpath):
            raise FileExistsError(f"{self.host}:{self._p(relpath)}")
        return _SSHWriter(self.host, self._p(relpath))

    def exists(self, relpath: str) -> bool:
        return (
            _ssh(self.host, f"test -e {shlex.quote(self._p(relpath))} && echo y || echo n").strip()
            == "y"
        )

    def size(self, relpath: str) -> int | None:
        out = _ssh(self.host, f"stat -c %s {shlex.quote(self._p(relpath))} 2>/dev/null || true").strip()
        return int(out) if out else None

    def sha256(self, relpath: str) -> str:
        return _ssh(self.host, f"sha256sum {shlex.quote(self._p(relpath))}").split()[0]

    def put_file(self, local: Path, relpath: str) -> tuple[int, str]:
        with self.open(relpath) as w:
            with open(local, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    w.write(chunk)
            return w.close()

    def put_text(self, text: str, relpath: str) -> None:
        with self.open(relpath) as w:
            w.write(text.encode())
            w.close()


    def get_text(self, relpath: str) -> str | None:
        if not self.exists(relpath):
            return None
        return _ssh(self.host, f"cat {shlex.quote(self._p(relpath))}")


def sink_for(m: Machine, root: str) -> Sink:
    if m.is_local:
        return LocalSink(root)
    return SSHSink(m.ssh, root)
