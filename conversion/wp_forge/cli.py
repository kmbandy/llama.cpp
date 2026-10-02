"""wp-forge CLI: wp-forge run PLAN.yaml [--dry-run ...] | wp-forge verify BUNDLE.json.

Stdout is JSON only (the resolved-plan JSON for --dry-run, JSON event lines
for a real run, the verify summary line for verify). All logging goes to
stderr. Exit codes: 0 ok, 1 verify failed, 2 plan/preflight error.

``make_source(spec, cache_dir, fetch=None)`` is module-level so tests can
monkeypatch it. HF sources: if the fetch is not injected, the default fetch
uses the network (huggingface_hub); setting the environment variable
WP_FORGE_HF_LOCAL_HUB to a directory makes the default fetch a local copy
from that directory instead (the tests' hub stand-in).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from . import bundle as bundle_mod
from . import job as job_mod
from . import machines as machines_mod
from . import plan as plan_mod
from . import source as source_mod
from .tools import Tools


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def make_source(spec: str, cache_dir: Path | None = None, fetch=None):
    """Build the Source for a plan's source: field. spec: 'hf:<repo>' or 'gguf:<path>'."""
    if spec.startswith("hf:"):
        hub = os.environ.get("WP_FORGE_HF_LOCAL_HUB")
        if fetch is None and hub:
            def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
                dest_dir = Path(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)
                return Path(shutil.copy(Path(hub) / filename, dest_dir / filename))
        return source_mod.HFSource(spec[3:], Path(cache_dir) if cache_dir else Path("hf-cache"),
                                   fetch=fetch)
    if spec.startswith("gguf:"):
        return source_mod.GGUFSource(spec[5:])
    raise plan_mod.PlanError(f"source: {spec!r} (want 'hf:<repo>' or 'gguf:<path>')")


def _dry_run_json(rplan) -> dict:
    return {
        "name": rplan.plan.name,
        "spine_path": rplan.spine_path,
        "sidecar_paths": dict(rplan.sidecar_paths),
        "sets": [asdict(s) for s in rplan.sets],
        "bundle_dir": rplan.bundle_dir,
        "dispatch_order": rplan.dispatch_order(),
    }


def _run_convert(args) -> int:
    from . import convert as convert_mod
    from .sink import sink_for

    plan = plan_mod.load_convert_plan(args.plan)
    machines = machines_mod.load_machines(args.machines or plan.machines_path)
    rc, rules, ml8 = plan_mod.resolve_convert(plan, machines)
    workdir = Path(args.workdir)
    emit = lambda e: print(json.dumps(e), flush=True)  # noqa: E731
    local_out = Path(rc.out_path) if rc.machine.is_local else workdir / f"{plan.name}.gguf"
    kw = dict(cache_dir=Path(args.cache_dir) if args.cache_dir else None,
              workers=args.workers if args.workers > 1 else None, converter=plan.converter)

    est = convert_mod.convert(plan.source, local_out, rules, ml8, dry_run=True, events=emit, **kw)
    if args.dry_run:
        print(json.dumps({**est.summary(), "out": rc.out_path,
                          "tensors": [asdict(d) for d in est.decisions]}, indent=2))
        return 0
    need = int(est.bytes * 1.05)
    short = []
    if machines_mod.free_bytes(rc.machine, rc.out_path.rsplit("/", 1)[0]) < need:
        short.append(f"{rc.machine.name}:{rc.out_path} needs {need} bytes")
    if not rc.machine.is_local and shutil.disk_usage(workdir).free < need:
        short.append(f"workdir {workdir} needs {need} bytes")
    if short:
        raise job_mod.PreflightError("; ".join(short))
    _log(f"wp-forge: convert {plan.source} -> {rc.machine.name}:{rc.out_path} (~{est.bytes / 1e9:.2f} GB)")
    res = convert_mod.convert(plan.source, local_out, rules, ml8, events=emit,
                              stamp={"plan": plan.name, "commit": _commit()}, **kw)
    if not rc.machine.is_local:
        sink = sink_for(rc.machine, rc.out_path.rsplit("/", 1)[0])
        sink.mkdir()
        sink.put_file(local_out, rc.out_path.rsplit("/", 1)[1])
        local_out.unlink()
    emit({"kind": "stage_done", "stage": "convert", **res.summary(), "out": rc.out_path})
    return 0


def _commit() -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def _run(args) -> int:
    if plan_mod.is_convert_plan(args.plan):
        return _run_convert(args)
    plan = plan_mod.load_plan(args.plan)
    machines = machines_mod.load_machines(args.machines)
    workdir = Path(args.workdir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else workdir / "hf-cache"
    source = make_source(plan.source, cache_dir)
    rplan = plan_mod.resolve(plan, source.hparams(), machines,
                             source_is_gguf=source.is_gguf)
    if args.dry_run:
        print(json.dumps(_dry_run_json(rplan), indent=2))
        return 0
    _log(f"wp-forge: plan {plan.name!r}, {len(rplan.sets)} sets -> {rplan.bundle_dir}")
    job = job_mod.Job(
        rplan, source, Tools.discover(),
        workdir=workdir, resume=args.resume, workers=args.workers,
        verify_load=args.verify_load,
        event_sink=lambda e: print(json.dumps(e), flush=True),
    )
    job.run()
    return 0


def _verify(args) -> int:
    machines = machines_mod.load_machines(args.machines)
    report = bundle_mod.verify_bundle(Path(args.bundle), machines, deep=args.deep)
    print(report.summary())
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wp-forge", add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="forge a plan")
    p_run.add_argument("plan")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--workdir", default=".")
    p_run.add_argument("--machines", default=None)
    p_run.add_argument("--cache-dir", default=None)
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--verify-load", action="store_true")

    p_ver = sub.add_parser("verify", help="re-check a bundle against the machines")
    p_ver.add_argument("bundle")
    p_ver.add_argument("--deep", action="store_true")
    p_ver.add_argument("--machines", default=None)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "run":
            return _run(args)
        return _verify(args)
    except (plan_mod.PlanError, job_mod.PreflightError) as e:
        _log(f"wp-forge: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
