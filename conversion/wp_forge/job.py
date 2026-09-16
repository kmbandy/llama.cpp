"""wp-forge Job: drive one plan end to end and emit a JSONL event log.

Job is the surface a future console pane (and the CLI) drives: every fact it
learns is a JSON line in <workdir>/forge.jsonl, with kinds exactly
plan_resolved / stage_start / layer_done / blob_written / verify / stage_done /
error. Stage classes emit kind/stage themselves; Job wraps their `events`
callable to stamp ts + job_id, append to the log, and forward to an optional
event_sink. blob_written is emitted by Job itself (spine, each sidecar, each
expert blob) -- the stages do not emit it.

Stage order: spine, then one SidecarStage per arch sidecar class present in
the plan's quant block (HF sources only), then one ExpertStage per (role,
layers) group of rplan.sets in plan order. After each expert stage the source
shards that only that group needed are released (experts groups from
source.layer_shards, spec-head groups from the shards holding mtp.{stage}.*
names via source.tensor_index; never a shard a later group still needs).
GGUF sources never release.

Preflight (before any stage): every machine holding sets needs
free_bytes(dir) >= 1.1 x the sum of its sets' est_bytes; the workdir needs
free >= est_local_peak_bytes, where the peak stand-in is
2 x (smallest set's est_bytes / its layer count) -- i.e. two layers of the
smallest set's single-layer footprint, which bounds the local hot path
(one-layer GGUF + repack outputs) for any group.

Resume: a stage whose stage_done is already logged in forge.jsonl is skipped
and its result recovered from the sink (spine/sidecar: size + sha of the
gguf; expert: manifest re-read, descriptor only if the file is present, blob
shas from the manifest). Expert recovery is LocalSink-only in v1 (SSH sinks
raise NotImplementedError). If recovery finds nothing durable (e.g. the crash
happened before the manifest was written) the stage simply re-runs -- its own
per-layer resume then skips the resided layers.

No retries: any exception emits one error event and re-raises.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import bundle as bundle_mod
from . import machines as machines_mod
from . import sink as sink_mod
from .bundle import ExpertSetResult
from .plan import ResolvedPlan, SetSpec
from .sink import LocalSink, Sink
from .source import Source
from .stages import (
    ConverterSpineBuilder,
    ExpertStage,
    SidecarStage,
    SpineStage,
    StageResult,
)
from .tools import Tools


class PreflightError(RuntimeError):
    """A destination or the workdir lacks the planned space; message lists every shortfall."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Job:
    def __init__(
        self,
        rplan: ResolvedPlan,
        source: Source,
        tools: Tools,
        *,
        workdir: Path,
        resume: bool = False,
        builder=None,
        workers: int = 1,
        verify_load: bool = False,
        event_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.rplan = rplan
        self.source = source
        self.tools = tools
        self.workdir = Path(workdir)
        self.resume = resume
        self.builder = builder
        self.workers = int(workers)
        self.verify_load = verify_load
        self._event_sink = event_sink
        self.job_id = uuid.uuid4().hex
        self._log_path = self.workdir / "forge.jsonl"
        self.workdir.mkdir(parents=True, exist_ok=True)

    # -- events ---------------------------------------------------------------

    def _emit(self, ev: dict) -> None:
        out = dict(ev)
        out["ts"] = _now()
        out["job_id"] = self.job_id
        with self._log_path.open("a") as f:
            f.write(json.dumps(out) + "\n")
        if self._event_sink is not None:
            self._event_sink(out)

    def _events(self, stage_name: str) -> Callable[[dict], None]:
        """events callable handed to a stage: stamp, log, forward."""
        def ev(e: dict) -> None:
            if "kind" not in e or "stage" not in e:
                e = dict(e)
                e.setdefault("stage", stage_name)
            self._emit(e)
        return ev

    def _log_events(self) -> list[dict]:
        if not self._log_path.is_file():
            return []
        return [json.loads(line) for line in self._log_path.read_text().splitlines()
                if line.strip()]

    @staticmethod
    def _completed(log: list[dict]) -> set[str]:
        return {e["stage"] for e in log if e.get("kind") == "stage_done" and "stage" in e}

    # -- sinks ------------------------------------------------------------------

    def _spine_sink(self) -> Sink:
        return sink_mod.sink_for(self.rplan.machines[self.rplan.plan.spine.machine],
                                 self.rplan.bundle_dir)

    def _sidecar_sink(self) -> Sink:
        path = list(self.rplan.sidecar_paths.values())[0]
        scm = self.rplan.machines[self.rplan.plan.sidecars.machine]
        return sink_mod.sink_for(scm, os.path.dirname(path))

    # -- preflight -----------------------------------------------------------------

    def _est_local_peak_bytes(self) -> int:
        if not self.rplan.sets:
            return 0
        smallest = min(self.rplan.sets, key=lambda s: s.est_bytes)
        n_layers = len(smallest.layers.layers())
        return int(2 * (smallest.est_bytes / n_layers))

    def _preflight(self, log: list[dict]) -> None:
        peak = self._est_local_peak_bytes()
        free = {
            name: machines_mod.free_bytes(m, m.models_dir)
            for name, m in self.rplan.machines.items()
        }
        planned: dict[str, int] = {}
        for s in self.rplan.sets:
            planned[s.machine] = planned.get(s.machine, 0) + s.est_bytes
        problems: list[str] = []
        for name, need in planned.items():
            have = free.get(name, 0)
            if have < 1.1 * need:
                problems.append(f"{name}: need {int(1.1 * need)}, have {have}")
        wd_free = machines_mod.free_bytes(
            machines_mod.Machine("workdir", None, str(self.workdir)), str(self.workdir)
        )
        if wd_free < peak:
            problems.append(f"workdir: need {peak}, have {wd_free}")
        if problems:
            raise PreflightError("; ".join(problems))
        self._emit({
            "kind": "plan_resolved",
            "resumed": bool(self.resume and log),
            "sets": [
                {"id": s.id, "machine": s.machine, "dir": s.dir, "est_bytes": s.est_bytes}
                for s in self.rplan.sets
            ],
            "free_bytes": free,
            "est_local_peak_bytes": peak,
        })

    # -- stage grouping ----------------------------------------------------------------

    def _expert_groups(self) -> list[list[SetSpec]]:
        groups: list[list[SetSpec]] = []
        for s in self.rplan.sets:
            key = (s.role, s.layers.first, s.layers.last)
            for g in groups:
                t = g[0]
                if (t.role, t.layers.first, t.layers.last) == key:
                    g.append(s)
                    break
            else:
                groups.append([s])
        return groups

    def _shards_needed(self, s: SetSpec) -> set[str]:
        if s.role == "experts":
            out: set[str] = set()
            for L in s.layers.layers():
                out.update(self.source.layer_shards(L))
            return out
        # spec head: the shards holding mtp.{stage}.* names
        idx = self.source.tensor_index()
        n_layer = self.rplan.arch.n_layer(self.rplan.hparams)
        return {
            shard for L in s.layers.layers()
            for name, shard in idx.items()
            if name.startswith(f"mtp.{L - n_layer}.")
        }

    # -- resume recovery ---------------------------------------------------------------

    def _recovered(self, stage: str, groups: list[list[SetSpec]]):
        if stage == "spine":
            sink = self._spine_sink()
            rel = self.rplan.spine_path.rsplit("/", 1)[-1]
            return sink.size(rel) or 0, sink.sha256(rel)
        if stage in self.rplan.sidecar_paths:
            sink = self._sidecar_sink()
            rel = self.rplan.sidecar_paths[stage].rsplit("/", 1)[-1]
            return sink.size(rel) or 0, sink.sha256(rel)
        for g in groups:
            if g[0].id.rsplit("-w", 1)[0] != stage:
                continue
            sinks = {s.id: sink_mod.sink_for(self.rplan.machines[s.machine], s.dir) for s in g}
            for sink in sinks.values():
                if not isinstance(sink, LocalSink):
                    raise NotImplementedError("resume of remote sets not supported in v1")
            res: dict[str, StageResult] = {}
            for s in g:
                sink = sinks[s.id]
                manifest_rel = f"{s.output_base}-experts-manifest.json"
                try:
                    manifest = json.loads(Path(sink.root, manifest_rel).read_text())
                except (OSError, json.JSONDecodeError):
                    return None  # nothing durable: let the stage re-run
                descriptor_rel = f"{manifest_rel}.expert-descriptor.json"
                if not Path(sink.root, descriptor_rel).is_file():
                    descriptor_rel = None
                # content_hash is an aggregate identity, not the blob sha:
                # re-hash the blob from the (local) sink.
                blobs = [
                    (s["blob_file"], int(s["blob_bytes"]), sink.sha256(s["blob_file"]))
                    for s in manifest.get("shards", [])
                ]
                res[s.id] = StageResult(manifest_rel, descriptor_rel, blobs)
            return res
        raise KeyError(f"no stage {stage!r} in plan")

    # -- finish ------------------------------------------------------------------------

    def _forge_commit(self) -> str:
        root = Path(__file__).resolve().parents[2]
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root,
                capture_output=True, text=True, timeout=10, check=True,
            )
            return out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _load_test(self, b: dict) -> object:
        if self.rplan.arch.name != "deepseek41":
            return "skipped: no load test for this arch"
        binary = Path(__file__).resolve().parents[2] / "build-cpu" / "bin" / "test-dsv41-spine-load"
        if not binary.is_file():
            return "skipped: no binary"
        env = dict(os.environ)
        env["DSV41_SPINE"] = b["spine"]["path"]
        env["DSV41_ENGRAM"] = next(
            (s["path"] for s in b.get("sidecars", []) if s["class"] == "engram"), ""
        )
        return subprocess.run([str(binary)], env=env, capture_output=True).returncode

    # -- driver ----------------------------------------------------------------------------

    def run(self) -> dict:
        log = self._log_events() if self.resume else []
        done = self._completed(log)
        groups = self._expert_groups()
        try:
            self._preflight(log)

            # 1. spine
            if "spine" in done:
                spine = self._recovered("spine", groups)
                self._emit({"kind": "stage_done", "stage": "spine", "resumed": True})
            else:
                builder = self.builder or ConverterSpineBuilder(self.tools, self.rplan.arch)
                spine = SpineStage(
                    self.rplan, self.source, self.tools, self._spine_sink(),
                    self._events("spine"), self.workdir, builder,
                ).run()
            self._emit({
                "kind": "blob_written", "stage": "spine",
                "file": self.rplan.spine_path.rsplit("/", 1)[-1],
                "bytes": int(spine[0]), "sha256": str(spine[1]),
            })

            # 2. sidecars (HF sources only; only classes the plan quantizes)
            sidecars: dict[str, tuple[int, str]] = {}
            for cls in self.rplan.arch.sidecars:
                if self.source.is_gguf or cls not in self.rplan.plan.quant:
                    continue
                if cls in done:
                    sc = self._recovered(cls, groups)
                    self._emit({"kind": "stage_done", "stage": cls, "resumed": True})
                else:
                    sc = SidecarStage(
                        cls, self.rplan, self.source, self.tools, self._sidecar_sink(),
                        self._events(cls), self.workdir,
                    ).run()
                sidecars[cls] = (int(sc[0]), str(sc[1]))
                self._emit({
                    "kind": "blob_written", "stage": cls,
                    "file": self.rplan.sidecar_paths[cls].rsplit("/", 1)[-1],
                    "bytes": int(sc[0]), "sha256": str(sc[1]),
                })

            # 3. expert stages in plan order, releasing shards as we go
            set_results: dict[str, StageResult] = {}
            for i, g in enumerate(groups):
                stage_id = g[0].id.rsplit("-w", 1)[0]
                sinks = {
                    s.id: sink_mod.sink_for(self.rplan.machines[s.machine], s.dir) for s in g
                }
                for sink in sinks.values():
                    sink.mkdir()
                if stage_id in done:
                    res = self._recovered(stage_id, groups)
                    if res is None:
                        res = ExpertStage(
                            g, self.rplan, self.source, self.tools, sinks,
                            self._events(stage_id), self.workdir, workers=self.workers,
                        ).run()
                    else:
                        self._emit({"kind": "stage_done", "stage": stage_id, "resumed": True})
                else:
                    res = ExpertStage(
                        g, self.rplan, self.source, self.tools, sinks,
                        self._events(stage_id), self.workdir, workers=self.workers,
                    ).run()
                set_results.update(res)
                for s in g:
                    for name, nbytes, sha in set_results[s.id].blobs:
                        self._emit({
                            "kind": "blob_written", "stage": stage_id,
                            "file": name, "bytes": int(nbytes), "sha256": sha,
                        })
                if not self.source.is_gguf and i < len(groups) - 1:
                    later: set[str] = set()
                    for later_group in groups[i + 1:]:
                        for s in later_group:
                            later |= self._shards_needed(s)
                    for shard in self._shards_needed(g[0]) - later:
                        self.source.release_shard(shard)

            # 4. bundle, place, verify
            bundle = bundle_mod.write_bundle(
                self.rplan,
                spine=(int(spine[0]), str(spine[1])),
                sidecars=sidecars,
                sets={
                    k: ExpertSetResult(v.manifest_rel, v.descriptor_rel,
                                       [(n, int(b), h) for n, b, h in v.blobs])
                    for k, v in set_results.items()
                },
                forge_commit=self._forge_commit(),
            )
            bundle_mod.place_bundle(self.rplan, bundle)
            report = bundle_mod.verify_bundle(
                bundle, self.rplan.machines, hparams=self.rplan.hparams,
                arch=self.rplan.arch,
            )
            self._emit({"kind": "verify", "ok": report.ok, "summary": report.summary()})
            if self.verify_load:
                self._emit({"kind": "verify", "load": self._load_test(bundle)})
            sink = self._spine_sink()
            if not sink.exists("forge.jsonl"):
                sink.put_text(self._log_path.read_text(), "forge.jsonl")
            return bundle
        except Exception as e:
            self._emit({"kind": "error", "stage": "job",
                        "message": str(e) or type(e).__name__})
            raise
