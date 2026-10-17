"""wp-forge merge-sets: local sink only, everything under tmp_path.

The descriptor tool is faked (it just writes a file); the rest is real code
path through cli.main.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from conversion.dsv41_experts_from_hf import aggregate_identity
from conversion.wp_forge.cli import main
from conversion.wp_forge.tools import Tools

INTO_BASE = "dsv41"
FROM_BASE = "dsv41-head"
WANT = [f"{INTO_BASE}-experts-{i:05d}-of-00005.wpb" for i in range(1, 6)]


@dataclass
class Env:
    root: Path
    bundle: Path
    machines: Path
    spine: Path
    dir_a: Path
    dir_b: Path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_set(d: Path, *, sid: str, role: str, base: str, spine: Path, layer_list: list[tuple[int, int]],
                layers: tuple[int, int]) -> dict:
    """One expert set: a blob + index per shard, a manifest, an old descriptor."""
    d.mkdir(parents=True)
    n = len(layer_list)
    shards, blobs = [], []
    for i, (lo, hi) in enumerate(layer_list):
        name = f"{base}-experts-{i + 1:05d}-of-{n:05d}.wpb"
        payload = f"{name}-payload".encode() * 16
        (d / name).write_bytes(payload)
        ih = _sha(payload)
        idx_name = name[: -len(".wpb")] + ".wpi.json"
        (d / idx_name).write_text(json.dumps({
            "blob_file": name, "shard_index": i, "shard_count": n,
            "layer_first": lo, "layer_last": hi, "group_count": 2, "blob_bytes": len(payload),
            "content_hash": {"algorithm": "sha256", "value": ih},
            "model_files": [str(spine)], "groups": [],
        }, indent=2))
        shards.append({
            "blob_file": name, "index_file": idx_name, "shard_index": i,
            "layer_first": lo, "layer_last": hi, "group_count": 2, "blob_bytes": len(payload),
            "content_hash": {"algorithm": "sha256", "value": ih},
        })
        blobs.append({"file": name, "bytes": len(payload), "sha256": _sha(payload)})
    man_name = f"{base}-experts-manifest.json"
    man = {
        "format": "llama.cpp.weight-pager.expert-shard-manifest", "version": 1,
        "sharding_mode": "layer-ranges", "layer_ranges": [f"{layers[0]}-{layers[1]}"],
        "allow_partial": True, "input_model": "fake/model", "model_files": [str(spine)],
        "retained_expert_range": {"first": 0, "last": 3}, "expert_ggml_type": "q8_0",
        "content_hash": aggregate_identity(shards),
        "total_group_count": 2 * n, "total_blob_bytes": sum(b["bytes"] for b in blobs),
        "shard_count": n, "shards": shards, "note": f"set {sid}",
    }
    (d / man_name).write_text(json.dumps(man, indent=2))
    desc_name = f"{base}-experts-manifest.expert-descriptor.json"
    (d / desc_name).write_text("old-descriptor")
    return {
        "id": sid, "role": role, "layers": list(layers), "width": None, "n_ff_exp": 4,
        "machine": "box-a", "path": str(d), "manifest": f"{d}/{man_name}",
        "descriptor": f"{d}/{desc_name}", "bytes": sum(b["bytes"] for b in blobs), "blobs": blobs,
    }


@pytest.fixture
def env(tmp_path: Path) -> Env:
    spine = tmp_path / "spine.gguf"
    spine.write_bytes(b"GGUF-fake-spine")
    dir_a, dir_b = tmp_path / "box-a" / "experts-0-2", tmp_path / "box-a" / "head-3-4"
    a = _write_set(dir_a, sid="a", role="experts", base=INTO_BASE, spine=spine,
                   layer_list=[(0, 0), (1, 1), (2, 2)], layers=(0, 2))
    b = _write_set(dir_b, sid="b", role="spec_head", base=FROM_BASE, spine=spine,
                   layer_list=[(3, 3), (4, 4)], layers=(3, 4))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "bundle": "t", "version": 1, "arch": "deepseek41",
        "expert_sets": [a, b], "dispatch_order": ["a", "b"],
    }, indent=2))
    machines = tmp_path / "machines.json"
    machines.write_text('{"box-a": {"local": true}}')
    return Env(tmp_path, bundle, machines, spine, dir_a, dir_b)


class _FakeTools:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def descriptor(self, spine: Path, manifest: Path) -> Path:
        self.calls.append((Path(spine), Path(manifest)))
        out = manifest.parent / (manifest.stem + ".expert-descriptor.json")
        out.write_text("fresh-descriptor")
        return out


@pytest.fixture
def fake_tools(monkeypatch) -> _FakeTools:
    fake = _FakeTools()
    monkeypatch.setattr(Tools, "discover", classmethod(lambda cls, *a, **k: fake))
    return fake


def _argv(env: Env, src: str = "b", *extra: str) -> list[str]:
    return ["merge-sets", str(env.bundle), "--into", "a", "--from", src,
            "--machines", str(env.machines), *extra]


def _snapshot(env: Env) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(env.root.rglob("*")) if p.is_file()}


def _events(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_merge_renames_and_rewrites(env, fake_tools, capsys) -> None:
    assert main(_argv(env)) == 0

    # blobs: 1..5 of 5 in into's dir; from's dir gone (it was empty after the move)
    assert sorted(p.name for p in env.dir_a.glob("*.wpb")) == WANT
    assert not env.dir_b.exists()

    # indexes renumbered; old -of-00003 indexes deleted
    for i, name in enumerate(WANT):
        idx = json.loads((env.dir_a / (name[: -len(".wpb")] + ".wpi.json")).read_text())
        assert idx["blob_file"] == name
        assert idx["shard_index"] == i and idx["shard_count"] == 5
        assert idx["model_files"] == [str(env.spine)]
    assert json.loads((env.dir_a / (WANT[4][: -len(".wpb")] + ".wpi.json")).read_text())["layer_first"] == 4
    assert not list(env.dir_a.glob("*-of-00003.wpi.json"))

    # manifest
    man = json.loads((env.dir_a / f"{INTO_BASE}-experts-manifest.json").read_text())
    assert man["layer_ranges"] == ["0-4"]
    assert man["shard_count"] == 5
    assert [s["blob_file"] for s in man["shards"]] == WANT
    assert man["content_hash"] == aggregate_identity(man["shards"])
    assert man["total_blob_bytes"] == sum(s["blob_bytes"] for s in man["shards"])
    assert man["note"].endswith("merged b (layers 3-4) via wp-forge merge-sets")
    assert man["retained_expert_range"] == {"first": 0, "last": 3}

    # descriptor regenerated from the merged manifest; spine passed through
    assert (env.dir_a / f"{INTO_BASE}-experts-manifest.expert-descriptor.json").read_text() == "fresh-descriptor"
    assert [c[0] for c in fake_tools.calls] == [env.spine]

    # bundle: one set covering 0-4, from removed, sha256 carried over
    b = json.loads(env.bundle.read_text())
    assert [s["id"] for s in b["expert_sets"]] == ["a"]
    assert b["expert_sets"][0]["layers"] == [0, 4]
    assert b["dispatch_order"] == ["a"]
    assert [x["file"] for x in b["expert_sets"][0]["blobs"]] == WANT
    assert b["expert_sets"][0]["bytes"] == sum(x["bytes"] for x in b["expert_sets"][0]["blobs"])
    old = json.loads((env.bundle.parent / "bundle.json.bak-merge").read_text())
    assert [s["id"] for s in old["expert_sets"]] == ["a", "b"]
    assert {x["sha256"] for x in b["expert_sets"][0]["blobs"]} == {
        x["sha256"] for s in old["expert_sets"] for x in s["blobs"]}

    # backup of the manifest exists and holds the original 3-shard manifest
    bak = json.loads((env.dir_a / f"{INTO_BASE}-experts-manifest.json.bak-merge").read_text())
    assert bak["shard_count"] == 3 and bak["layer_ranges"] == ["0-2"]

    steps = [e["step"] for e in _events(capsys.readouterr().out) if e.get("kind") == "merge"]
    assert steps == ["start", "backup", "rename_blobs", "indexes", "manifest", "cleanup", "bundle"]


def test_dry_run_changes_nothing(env, fake_tools, capsys) -> None:
    before = _snapshot(env)
    assert main(_argv(env, "b", "--dry-run")) == 0
    assert _snapshot(env) == before
    assert fake_tools.calls == []

    summary = json.loads(capsys.readouterr().out)
    assert summary["shard_count"] == 5
    assert len(summary["renames"]) == 10  # 5 blobs + 5 indexes
    assert summary["manifest"]["layer_ranges"] == ["0-4"]
    assert summary["bundle"]["layers"] == [0, 4]


def test_missing_descriptor_tool_fails_before_writes(env, monkeypatch, capsys) -> None:
    def no_tool(cls, *a, **k):
        raise FileNotFoundError("no binaries")

    monkeypatch.setattr(Tools, "discover", classmethod(no_tool))
    before = _snapshot(env)
    assert main(_argv(env)) == 2
    assert _snapshot(env) == before


def test_missing_spine_is_an_error(env, fake_tools) -> None:
    env.spine.unlink()
    before = _snapshot(env)
    assert main(_argv(env)) == 2
    assert _snapshot(env) == before


def test_missing_set_exits_2(env, fake_tools) -> None:
    before = _snapshot(env)
    assert main(_argv(env, "nope")) == 2
    assert _snapshot(env) == before


def _edit_bundle(env: Env, fn) -> None:
    b = json.loads(env.bundle.read_text())
    fn(b)
    env.bundle.write_text(json.dumps(b, indent=2))


def _edit_json(path: Path, **kw) -> None:
    m = json.loads(path.read_text())
    m.update(kw)
    path.write_text(json.dumps(m, indent=2))


def _set_b(b: dict) -> dict:
    return next(s for s in b["expert_sets"] if s["id"] == "b")


CASES = {
    "non-contiguous": lambda e: _edit_bundle(e, lambda b: _set_b(b).update(layers=[5, 6])),
    "machine-mismatch": lambda e: _edit_bundle(e, lambda b: _set_b(b).update(machine="box-b")),
    "type-mismatch": lambda e: _edit_json(e.dir_b / f"{FROM_BASE}-experts-manifest.json",
                                          expert_ggml_type="q4_0"),
    "sharding-mode": lambda e: _edit_json(e.dir_a / f"{INTO_BASE}-experts-manifest.json",
                                          sharding_mode="expert-slice"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_precondition_errors_exit_2(env, fake_tools, case, capsys) -> None:
    CASES[case](env)
    before = _snapshot(env)
    assert main(_argv(env)) == 2
    assert _snapshot(env) == before
    assert fake_tools.calls == []
    assert "merge-sets:" in capsys.readouterr().err
