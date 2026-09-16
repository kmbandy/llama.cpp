"""End-to-end tests for the wp-forge Job driver + CLI.

Real CPU tools (build-cpu/bin), synthetic DeepSeek-V4.1-shaped HF repo, two
local "machines" (tmp/a, tmp/b). Covers: CLI --dry-run JSON shape, a full
run (spine + engram sidecar + 3 sets + bundle + deep verify + event log),
resume (crashed second expert group, rerun skips done stages), and the
preflight space check.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conversion.wp_forge import bundle
from conversion.wp_forge import job as job_mod
from conversion.wp_forge import machines as machines_mod
from conversion.wp_forge.cli import main
from conversion.wp_forge.job import Job, PreflightError
from conversion.wp_forge.source import HFSource
from conversion.wp_forge.stages import ExpertStage
from conversion.wp_forge.sink import LocalSink
from conversion.wp_forge.tools import Tools
from synth import SyntheticSpineBuilder, make_synthetic_hf_repo

PLAN_YAML = """\
name: m
source: hf:fake/repo
quant:
  experts: q8_0
  dense: q8_0
  engram: q8_0
spine: a
experts:
  - layers: 0-1
    machine: a
    widths: [32, 32]
spec_head:
  layers: 2-2
  machine: b
"""


@pytest.fixture
def env(tmp_path: Path):
    try:
        tools = Tools.discover()
    except FileNotFoundError:
        pytest.skip("wp-forge C++ tools not built (Tools.discover raised)")
    hub = make_synthetic_hf_repo(tmp_path / "hub")
    mj = tmp_path / "machines.json"
    mj.write_text(json.dumps({
        "a": {"local": True, "models_dir": str(tmp_path / "a")},
        "b": {"local": True, "models_dir": str(tmp_path / "b")},
    }))
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN_YAML)
    wd = tmp_path / "work"

    def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    return dict(
        tools=tools, hub=hub, mj=mj, plan=plan, wd=wd, tmp_path=tmp_path,
        fetch=fetch,
        a=tmp_path / "a", b=tmp_path / "b",
        source=lambda cache=wd / "hf-cache": HFSource("fake/repo", cache, fetch=fetch),
    )


def _rplan(env, source):
    from conversion.wp_forge import plan as plan_mod
    plan = plan_mod.load_plan(env["plan"])
    machines = machines_mod.load_machines(env["mj"])
    return plan_mod.resolve(plan, source.hparams(), machines, source_is_gguf=source.is_gguf)


def _events(wd: Path) -> list[dict]:
    log = wd / "forge.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def test_cli_dry_run(env, monkeypatch, capsys):
    monkeypatch.setenv("WP_FORGE_HF_LOCAL_HUB", str(env["hub"]))
    rc = main(["run", str(env["plan"]), "--dry-run", "--machines", str(env["mj"]),
               "--workdir", str(env["wd"])])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["name"] == "m"
    assert out["spine_path"] == f"{env['a']}/m/m-spine.gguf"
    assert out["bundle_dir"] == f"{env['a']}/m"
    assert [s["id"] for s in out["sets"]] == ["L0-1-w0", "L0-1-w1", "L2-2"]
    assert all(s["est_bytes"] > 0 for s in out["sets"])
    assert out["dispatch_order"] == ["L0-1-w0", "L0-1-w1", "L2-2"]
    assert set(out["sidecar_paths"]) == {"engram"}


def test_full_run(env):
    wd = env["wd"]
    source = env["source"]()
    rplan = _rplan(env, source)
    b = Job(rplan, source, env["tools"], workdir=wd,
            builder=SyntheticSpineBuilder()).run()

    assert (env["a"] / "m" / "m-spine.gguf").is_file()
    assert (env["a"] / "m" / "m-engram.gguf").is_file()
    for sid in ("L0-1-w0", "L0-1-w1"):
        d = env["a"] / "m" / sid
        assert (d / f"m-{sid}-experts-manifest.json").is_file()
    assert (env["b"] / "m" / "L2-2" / "m-L2-2-experts-manifest.json").is_file()
    assert (env["a"] / "m" / "bundle.json").is_file()
    assert (env["b"] / "m" / "bundle.json").is_file()

    machines = machines_mod.load_machines(env["mj"])
    report = bundle.verify_bundle(
        env["a"] / "m" / "bundle.json", machines, deep=True,
        hparams=source.hparams(), arch=rplan.arch,
    )
    assert report.ok, report.summary()
    assert b["bundle"] == "m" and b["forge_commit"]

    events = _events(wd)
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "plan_resolved"
    assert all("ts" in e and "job_id" in e for e in events)
    assert sum(1 for e in events if e["kind"] == "stage_done") >= 3
    assert sum(1 for e in events if e["kind"] == "layer_done") == 3
    assert any(e["kind"] == "verify" and e.get("ok") is True for e in events)
    assert (env["a"] / "m" / "forge.jsonl").is_file()


def test_resume_after_crash(env, monkeypatch):
    wd = env["wd"]
    source = env["source"]()
    rplan = _rplan(env, source)

    real_run = ExpertStage.run

    def flaky(self):
        if self.stage_id == "L2-2":
            raise RuntimeError("boom")
        return real_run(self)

    monkeypatch.setattr(ExpertStage, "run", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        Job(rplan, source, env["tools"], workdir=wd,
            builder=SyntheticSpineBuilder()).run()

    events1 = _events(wd)
    assert any(e["kind"] == "error" and "boom" in e["message"] for e in events1)

    # second run: resume
    monkeypatch.setattr(ExpertStage, "run", real_run)
    b = Job(rplan, source, env["tools"], workdir=wd, resume=True,
            builder=SyntheticSpineBuilder()).run()

    events2 = _events(wd)[len(events1):]
    pr = next(e for e in events2 if e["kind"] == "plan_resolved")
    assert pr["resumed"] is True
    starts = [e["stage"] for e in events2 if e["kind"] == "stage_start"]
    assert "spine" not in starts
    assert "engram" not in starts
    assert "L0-1" not in starts
    assert "L2-2" in starts
    assert any(e["kind"] == "verify" and e.get("ok") is True for e in events2)

    # recovery used the first group's manifest: its blobs verify deep-true
    machines = machines_mod.load_machines(env["mj"])
    report = bundle.verify_bundle(
        env["a"] / "m" / "bundle.json", machines, deep=True,
        hparams=source.hparams(), arch=rplan.arch,
    )
    assert report.ok, report.summary()
    del b


def test_preflight_names_machine(env, monkeypatch):
    real = machines_mod.free_bytes

    def fake(m, path):
        return 0 if m.name == "b" else real(m, path)

    monkeypatch.setattr(job_mod.machines_mod, "free_bytes", fake)
    source = env["source"]()
    rplan = _rplan(env, source)
    with pytest.raises(PreflightError, match=r"b: need .*have 0"):
        Job(rplan, source, env["tools"], workdir=env["wd"]).run()
    # preflight fails before any stage: log has at most plan_resolved absent
    events = _events(env["wd"])
    assert not any(e["kind"] == "plan_resolved" for e in events)
