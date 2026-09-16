"""bundle.json: the router-linking record for a forged build.

write_bundle() turns a resolved plan plus per-part facts (bytes, sha256) into
the record; place_bundle() ships it to every machine that holds a part;
verify_bundle() re-checks the record against what is actually on disk.
Routers read this instead of hand-transcribed INI, and `dispatch_order` is
exactly the required --expert-dispatch order. Nothing runtime-tunable
belongs in it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import gguf

from .plan import ResolvedPlan
from .sink import sink_for

_EXPERT_TENSOR_KEYS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


class ExpertSetResult(NamedTuple):
    """What forging one expert set produced. blobs: (dst_name, bytes, sha256)."""
    manifest_rel: str
    descriptor_rel: str | None
    blobs: list[tuple[str, int, str]]


class Check(NamedTuple):
    target: str
    name: str
    ok: bool
    detail: str


class VerifyReport:
    def __init__(self, checks: list[Check]):
        self.checks = list(checks)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.ok]
        if failed:
            return "\n".join(f"{c.target}: {c.name}: {c.detail}" for c in failed)
        return f"all {len(self.checks)} checks ok"


def write_bundle(rplan: ResolvedPlan, *, spine: tuple[int, str],
                 sidecars: dict[str, tuple[int, str]],
                 sets: dict[str, ExpertSetResult], forge_commit: str,
                 created: str | None = None) -> dict:
    """Assemble bundle.json content from resolved-plan + sink facts.

    spine: (bytes, sha256) of the spine gguf; sidecars: class -> (bytes, sha256);
    sets: set id -> ExpertSetResult, in dispatch order.
    """
    plan, arch = rplan.plan, rplan.arch
    if created is None:
        created = datetime.now(timezone.utc).isoformat()
    by_id = {s.id: s for s in rplan.sets}
    expert_sets = []
    for sid, res in sets.items():
        s = by_id[sid]
        if s.slice_index is not None and s.widths:
            i = s.slice_index
            width = [sum(s.widths[:i]), sum(s.widths[:i + 1])]
        else:
            width = None
        expert_sets.append({
            "id": s.id,
            "role": s.role,
            "layers": [s.layers.first, s.layers.last],
            "width": width,
            "n_ff_exp": int(rplan.hparams[arch.n_ff_exp_key]),
            "machine": s.machine,
            "path": s.dir,
            "manifest": f"{s.dir}/{res.manifest_rel}",
            "descriptor": None if res.descriptor_rel is None else f"{s.dir}/{res.descriptor_rel}",
            "bytes": sum(n for _, n, _ in res.blobs),
            "blobs": [{"file": f, "bytes": n, "sha256": h} for f, n, h in res.blobs],
        })
    sidecar_list = [
        {"class": cls, "machine": plan.sidecars.machine, "path": rplan.sidecar_paths[cls],
         "bytes": n, "sha256": h}
        for cls, (n, h) in sidecars.items()
    ]
    return {
        "bundle": plan.name,
        "version": 1,
        "arch": arch.name,
        "source": plan.source,
        "created": created,
        "forge_commit": forge_commit,
        "quant": plan.quant,
        "spine": {"machine": plan.spine.machine, "path": rplan.spine_path,
                  "bytes": spine[0], "sha256": spine[1]},
        "sidecars": sidecar_list,
        "expert_sets": expert_sets,
        "dispatch_order": rplan.dispatch_order(),
    }


def place_bundle(rplan: ResolvedPlan, bundle: dict) -> list[str]:
    """Write bundle.json on the spine machine and on every other machine that
    holds a set. Never overwrites: an existing bundle.json becomes bundle.json.new.
    Returns ["machine:dir", ...] with " (existing kept)" for the .new case."""
    text = json.dumps(bundle, indent=2)
    dirs: list[tuple[str, str]] = [(rplan.plan.spine.machine, rplan.bundle_dir)]
    for s in rplan.sets:
        if (s.machine, os.path.dirname(s.dir)) not in dirs:
            dirs.append((s.machine, os.path.dirname(s.dir)))
    out = []
    for machine, d in dirs:
        sink = sink_for(rplan.machines[machine], d)
        sink.mkdir()
        if sink.exists("bundle.json"):
            sink.put_text(text, "bundle.json.new")
            out.append(f"{machine}:{d} (existing kept)")
        else:
            sink.put_text(text, "bundle.json")
            out.append(f"{machine}:{d}")
    return out


def _run_check(target: str, name: str, fn) -> Check:
    """A failed check never raises; fn returns (ok, detail)."""
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"error: {e}"
    return Check(target, name, bool(ok), str(detail))


def _coverage_check(b: dict, hparams: dict | None, arch) -> Check:
    def run():
        expert = [s for s in b["expert_sets"] if s["role"] == "experts"]
        head = [s for s in b["expert_sets"] if s["role"] == "spec_head"]
        # per layer: the experts a set actually holds (its slice width when
        # sliced, the full expert count when unsliced) must sum to n_ff_exp;
        # a layer is covered when the total is > 0.
        full = int(expert[0]["n_ff_exp"]) if expert else 0
        total: dict[int, int] = {}
        for s in expert:
            held = (s["width"][1] - s["width"][0]) if s["width"] else full
            for l in range(s["layers"][0], s["layers"][1] + 1):
                total[l] = total.get(l, 0) + held
        exp_max = max((s["layers"][1] for s in expert), default=-1)
        problems = []
        for l in range(exp_max + 1):
            n = total.get(l, 0)
            if n == 0:
                problems.append(f"layer {l} uncovered")
            elif n != full:
                problems.append(f"layer {l} experts sum {n}, want {full}")
        start = exp_max + 1
        hseen: dict[int, int] = {}
        for s in head:
            for l in range(s["layers"][0], s["layers"][1] + 1):
                hseen[l] = hseen.get(l, 0) + 1
        if hseen:
            if min(hseen) != start:
                problems.append(f"spec_head starts at layer {min(hseen)}, want {start}")
            holes = [l for l in range(start, max(hseen) + 1) if l not in hseen]
            if holes:
                problems.append(f"spec_head gaps at {holes}")
            dups = [l for l, n in hseen.items() if n > 1]
            if dups:
                problems.append(f"spec_head overlaps at {dups}")
        if hparams is not None and arch is not None:
            want = int(hparams[arch.n_layer_key])
            if exp_max + 1 != want:
                problems.append(f"experts cover {exp_max + 1} layers, arch has {want}")
        if problems:
            return False, "; ".join(problems)
        return True, f"experts tile 0..{exp_max}" + (", spec_head contiguous" if hseen else "")
    return _run_check("coverage", "coverage", run)


def verify_bundle(bundle: dict | Path, machines: dict, *, deep: bool = False,
                  hparams: dict | None = None, arch=None) -> VerifyReport:
    """Re-check a bundle against what is on the machines.

    Never raises on a failed check; raises only if `bundle` (a Path) cannot be
    read. deep adds sha256 re-hashes of every blob and the spine.
    """
    if not isinstance(bundle, dict):
        b = json.loads(Path(bundle).read_text())
    else:
        b = bundle
    checks: list[Check] = [_coverage_check(b, hparams, arch)]

    for s in b["expert_sets"]:
        sink = sink_for(machines[s["machine"]], s["path"])
        man = os.path.basename(s["manifest"])
        checks.append(_run_check(s["id"], "manifest_present",
                                 lambda m=man: (sink.exists(m), "present" if sink.exists(m) else "missing")))
        desc = s["descriptor"]
        if desc is None:
            checks.append(Check(s["id"], "descriptor_present", True, "none recorded"))
        else:
            d = os.path.basename(desc)
            checks.append(_run_check(s["id"], "descriptor_present",
                                     lambda x=d: (sink.exists(x), "present" if sink.exists(x) else "missing")))
        for blob in s["blobs"]:
            f = blob["file"]
            checks.append(_run_check(f"{s['id']}:{f}", "size",
                                     lambda f=f: (
                                         lambda n: (n == blob["bytes"],
                                                    f"{n} != {blob['bytes']}" if n != blob["bytes"] else "ok"))(sink.size(f))))
            if deep:
                want = blob["sha256"]
                checks.append(_run_check(f"{s['id']}:{f}", "sha256",
                                         lambda f=f, w=want: (
                                             sink.sha256(f) == w,
                                             "match" if sink.sha256(f) == w else "mismatch")))

    sp = b["spine"]
    sp_sink = sink_for(machines[sp["machine"]], os.path.dirname(sp["path"]))
    sp_rel = os.path.basename(sp["path"])
    checks.append(_run_check("spine", "spine_size",
                             lambda: (
                                 lambda n: (n == sp["bytes"],
                                            f"{n} != {sp['bytes']}" if n != sp["bytes"] else "ok"))(sp_sink.size(sp_rel))))
    if deep:
        want = sp["sha256"]
        checks.append(_run_check("spine", "spine_sha256",
                                 lambda: (sp_sink.sha256(sp_rel) == want,
                                          "match" if sp_sink.sha256(sp_rel) == want else "mismatch")))

    def spine_no_experts():
        if not machines[sp["machine"]].is_local:
            return True, "skipped: remote"
        r = gguf.GGUFReader(sp["path"])
        bad = sorted(t.name for t in r.tensors if any(k in t.name for k in _EXPERT_TENSOR_KEYS))
        if bad:
            return False, f"spine holds expert tensors: {bad}"
        return True, "no expert tensors"
    checks.append(_run_check("spine", "spine_no_experts", spine_no_experts))
    return VerifyReport(checks)
