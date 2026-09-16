"""bundle.json: write_bundle / place_bundle / verify_bundle (Task 12).

Machines a/b are both local with distinct models_dir, so verify_bundle can
read the spine and sinks directly.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import gguf
import numpy as np
import pytest

from conversion.wp_forge import bundle as B
from conversion.wp_forge.arch import ARCHS
from conversion.wp_forge.machines import Machine
from conversion.wp_forge.plan import (LayerRange, Placement, Plan, ResolvedPlan,
                                      SetSpec, StageSpec)

ARCH = ARCHS["deepseek41"]
HPARAMS = {"moe_intermediate_size": 64, "num_hidden_layers": 4}
FAKE_SHA = "f" * 64


def _build_rplan(tmp_path, sidecar_paths=None):
    a = Machine("a", None, str(tmp_path / "a"))
    b = Machine("b", None, str(tmp_path / "b"))
    d = str(tmp_path / "a" / "m")
    db = str(tmp_path / "b" / "m")
    s0 = SetSpec(id="L0-1-w0", role="experts", layers=LayerRange(0, 1), slice_index=0,
                 widths=[32, 32], machine="a", dir=f"{d}/L0-1-w0",
                 output_base="m-L0-1-w0", quant="q8_0", est_bytes=0)
    s1 = SetSpec(id="L0-1-w1", role="experts", layers=LayerRange(0, 1), slice_index=1,
                 widths=[32, 32], machine="a", dir=f"{d}/L0-1-w1",
                 output_base="m-L0-1-w1", quant="q8_0", est_bytes=0)
    s2 = SetSpec(id="L2-3", role="experts", layers=LayerRange(2, 3), slice_index=None,
                 widths=None, machine="b", dir=f"{db}/L2-3",
                 output_base="m-L2-3", quant="q8_0", est_bytes=0)
    plan = Plan(name="m", source="hf:x/y", quant={"experts": "q8_0", "dense": "q8_0"},
                machines_path=None, spine=Placement("a", None), sidecars=Placement("a", None),
                experts=[StageSpec(layers=LayerRange(0, 1), machine="a", path=None, widths=[32, 32]),
                         StageSpec(layers=LayerRange(2, 3), machine="b", path=None, widths=None)],
                spec_head="none", allow_partial=False)
    return ResolvedPlan(plan=plan, arch=ARCH, hparams=HPARAMS, machines={"a": a, "b": b},
                        spine_path=f"{d}/m-spine.gguf",
                        sidecar_paths=sidecar_paths or {}, sets=[s0, s1, s2], bundle_dir=d)


@pytest.fixture()
def rplan(tmp_path):
    return _build_rplan(tmp_path)


def _blob_bytes(n: int, seed: int) -> bytes:
    return bytes((i * 31 + seed * 17 + 5) % 256 for i in range(n))


def _make_bundle(rplan, spine):
    sizes = {"L0-1-w0": [100, 64], "L0-1-w1": [72], "L2-3": [48]}
    sets = {}
    for i, s in enumerate(rplan.sets):
        blobs = []
        for j, n in enumerate(sizes[s.id]):
            data = _blob_bytes(n, i * 10 + j)
            blobs.append((f"{s.id}-blob-{j}.gguf", n, hashlib.sha256(data).hexdigest()))
        sets[s.id] = B.ExpertSetResult(
            manifest_rel="manifest.json",
            descriptor_rel=None if s.id == "L2-3" else "descriptor.json",
            blobs=blobs)
    return B.write_bundle(rplan, spine=spine, sidecars={}, sets=sets,
                          forge_commit="a8aa8d928", created="2026-07-30T00:00:00+00:00")


def _write_spine(path: str, with_experts: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(p), arch="deepseek41")
    w.add_name("wp-forge-test-spine")
    w.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=np.float32))
    if with_experts:
        w.add_tensor("blk.0.ffn_gate_exps.weight", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lay_sets(rplan, bundle) -> None:
    for i, s in enumerate(bundle["expert_sets"]):
        d = Path(s["path"])
        d.mkdir(parents=True, exist_ok=True)
        (d / os.path.basename(s["manifest"])).write_text("{}")
        if s["descriptor"] is not None:
            (d / os.path.basename(s["descriptor"])).write_text("{}")
        for j, blob in enumerate(s["blobs"]):
            (d / blob["file"]).write_bytes(_blob_bytes(blob["bytes"], i * 10 + j))


def test_write_bundle_shape(rplan, tmp_path):
    b = _make_bundle(rplan, spine=(999, FAKE_SHA))
    assert set(b) == {"bundle", "version", "arch", "source", "created", "forge_commit",
                      "quant", "spine", "sidecars", "expert_sets", "dispatch_order"}
    assert b["bundle"] == "m"
    assert b["version"] == 1
    assert b["arch"] == ARCH.name
    assert b["source"] == "hf:x/y"
    assert b["created"] == "2026-07-30T00:00:00+00:00"
    assert b["forge_commit"] == "a8aa8d928"
    assert b["quant"] == {"experts": "q8_0", "dense": "q8_0"}
    assert b["sidecars"] == []
    assert b["dispatch_order"] == ["L0-1-w0", "L0-1-w1", "L2-3"]
    assert b["spine"] == {"machine": "a", "path": rplan.spine_path,
                          "bytes": 999, "sha256": FAKE_SHA}
    sets = {s["id"]: s for s in b["expert_sets"]}
    assert list(sets) == ["L0-1-w0", "L0-1-w1", "L2-3"]
    for s in b["expert_sets"]:
        assert set(s) == {"id", "role", "layers", "width", "n_ff_exp", "machine", "path",
                          "manifest", "descriptor", "bytes", "blobs"}
        for blob in s["blobs"]:
            assert set(blob) == {"file", "bytes", "sha256"}
    d = rplan.bundle_dir
    assert [s["layers"] for s in b["expert_sets"]] == [[0, 1], [0, 1], [2, 3]]
    assert [s["width"] for s in b["expert_sets"]] == [[0, 32], [32, 64], None]
    assert [s["n_ff_exp"] for s in b["expert_sets"]] == [64, 64, 64]
    assert [s["machine"] for s in b["expert_sets"]] == ["a", "a", "b"]
    assert [s["bytes"] for s in b["expert_sets"]] == [164, 72, 48]
    assert sets["L0-1-w0"]["path"] == f"{d}/L0-1-w0"
    assert sets["L0-1-w0"]["manifest"] == f"{d}/L0-1-w0/manifest.json"
    assert sets["L0-1-w0"]["descriptor"] == f"{d}/L0-1-w0/descriptor.json"
    assert sets["L2-3"]["descriptor"] is None
    assert sets["L0-1-w0"]["blobs"][0] == {
        "file": "L0-1-w0-blob-0.gguf", "bytes": 100,
        "sha256": hashlib.sha256(_blob_bytes(100, 0)).hexdigest()}

    # created=None -> utc now, iso8601
    b2 = B.write_bundle(rplan, spine=(999, FAKE_SHA), sidecars={},
                        sets={s.id: B.ExpertSetResult("m.json", None, []) for s in rplan.sets},
                        forge_commit="x")
    assert b2["created"] != b["created"]
    dt = datetime.fromisoformat(b2["created"])
    assert dt.tzinfo is not None
    assert b2["expert_sets"][0]["bytes"] == 0

    # sidecar entry shape (separate plan so sidecar_paths is populated)
    rp2 = _build_rplan(tmp_path, sidecar_paths={"ffn": f"{d}/ffn.gguf"})
    b3 = B.write_bundle(rp2, spine=(1, FAKE_SHA),
                        sidecars={"ffn": (5, "0" * 64)},
                        sets={s.id: B.ExpertSetResult("m.json", None, []) for s in rp2.sets},
                        forge_commit="x", created="2026-07-30T00:00:00+00:00")
    assert b3["sidecars"] == [{"class": "ffn", "machine": "a", "path": f"{d}/ffn.gguf",
                               "bytes": 5, "sha256": "0" * 64}]


def test_place_bundle(rplan):
    b = _make_bundle(rplan, spine=(999, FAKE_SHA))
    d = rplan.bundle_dir
    other = str(Path(rplan.sets[2].dir).parent)
    expected = json.dumps(b, indent=2)

    first = B.place_bundle(rplan, b)
    assert first == [f"a:{d}", f"b:{other}"]
    assert (Path(d) / "bundle.json").read_text() == expected
    assert (Path(other) / "bundle.json").read_text() == expected

    second = B.place_bundle(rplan, b)
    assert second == [f"a:{d} (existing kept)", f"b:{other} (existing kept)"]
    assert (Path(d) / "bundle.json").read_text() == expected
    assert (Path(other) / "bundle.json").read_text() == expected
    assert (Path(d) / "bundle.json.new").read_text() == expected
    assert (Path(other) / "bundle.json.new").read_text() == expected


def test_verify_bundle(rplan):
    _write_spine(rplan.spine_path)
    info = (os.path.getsize(rplan.spine_path), _sha(rplan.spine_path))
    b = _make_bundle(rplan, spine=info)
    _lay_sets(rplan, b)

    rep = B.verify_bundle(b, rplan.machines, deep=True, hparams=HPARAMS, arch=ARCH)
    assert rep.ok, rep.summary()
    assert rep.summary() == f"all {len(rep.checks)} checks ok"

    # the placed bundle.json is readable via Path too
    placed = B.place_bundle(rplan, b)
    rep = B.verify_bundle(placed[0].split(":", 1)[1] + "/bundle.json",
                          rplan.machines, hparams=HPARAMS, arch=ARCH)
    assert rep.ok, rep.summary()

    # flip one byte in one blob: size still ok, deep sha256 fails naming it
    s0 = b["expert_sets"][0]
    blob = s0["blobs"][0]
    bp = Path(s0["path"]) / blob["file"]
    raw = bytearray(bp.read_bytes())
    raw[0] ^= 0xFF
    bp.write_bytes(bytes(raw))
    rep = B.verify_bundle(b, rplan.machines, deep=True, hparams=HPARAMS, arch=ARCH)
    assert not rep.ok
    by = {(c.target, c.name): c for c in rep.checks}
    assert by[(f"L0-1-w0:{blob['file']}", "size")].ok
    assert not by[(f"L0-1-w0:{blob['file']}", "sha256")].ok
    assert blob["file"] in rep.summary()
    # shallow run: the flip is invisible without deep
    rep_nd = B.verify_bundle(b, rplan.machines, hparams=HPARAMS, arch=ARCH)
    assert rep_nd.ok, rep_nd.summary()
    bp.write_bytes(_blob_bytes(blob["bytes"], 0))  # restore

    # delete a descriptor -> descriptor_present fails for that set only
    Path(s0["descriptor"]).unlink()
    rep = B.verify_bundle(b, rplan.machines, hparams=HPARAMS, arch=ARCH)
    by = {(c.target, c.name): c for c in rep.checks}
    assert not by[("L0-1-w0", "descriptor_present")].ok
    assert by[("L0-1-w1", "descriptor_present")].ok
    Path(s0["descriptor"]).write_text("{}")

    # descriptor=None -> ok with "none recorded"
    assert by[("L2-3", "descriptor_present")].ok
    assert by[("L2-3", "descriptor_present")].detail == "none recorded"

    # drop a set -> coverage fails (experts no longer tile 0..3)
    b2 = json.loads(json.dumps(b))
    b2["expert_sets"] = [s for s in b2["expert_sets"] if s["id"] != "L2-3"]
    rep = B.verify_bundle(b2, rplan.machines, hparams=HPARAMS, arch=ARCH)
    cov = [c for c in rep.checks if c.name == "coverage"][0]
    assert not cov.ok
    # ...but without hparams, L0-1 alone tiles 0..1 exactly once
    rep3 = B.verify_bundle(b2, rplan.machines)
    assert [c for c in rep3.checks if c.name == "coverage"][0].ok

    # a spine holding expert tensors -> spine_no_experts fails
    bad = str(Path(rplan.bundle_dir) / "bad-spine.gguf")
    _write_spine(bad, with_experts=True)
    info2 = (os.path.getsize(bad), _sha(bad))
    b3 = json.loads(json.dumps(b))
    b3["spine"] = {"machine": "a", "path": bad, "bytes": info2[0], "sha256": info2[1]}
    rep = B.verify_bundle(b3, rplan.machines, hparams=HPARAMS, arch=ARCH)
    sn = [c for c in rep.checks if c.name == "spine_no_experts"][0]
    assert not sn.ok
    assert "blk.0.ffn_gate_exps.weight" in sn.detail
    # spine_size/spine_sha256 still pass for the bad spine (sizes match)
    by = {(c.target, c.name): c for c in rep.checks}
    assert by[("spine", "spine_size")].ok
    rep_deep = B.verify_bundle(b3, rplan.machines, deep=True, hparams=HPARAMS, arch=ARCH)
    by = {(c.target, c.name): c for c in rep_deep.checks}
    assert by[("spine", "spine_sha256")].ok
