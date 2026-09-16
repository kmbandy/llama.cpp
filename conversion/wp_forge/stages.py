"""wp-forge stage drivers.

``SpineBuilder`` / ``ConverterSpineBuilder`` / ``SpineStage`` and
``SidecarStage`` produce the dense spine GGUF and per-class sidecar GGUFs;
``ExpertStage`` (below) owns the routed-expert stages.

``ExpertStage`` owns ALL width sets of one plan stage. The sets of a stage
share the same layers (the stage's ``layers`` range); they differ only in
``widths`` / ``slice_index`` (width-sliced sets), or there is a single unsliced
set. For each layer, in order, the stage:

  1. resolves the HF tensor names for the layer's experts (main stack via
     ``expert_tensor_names``; spec-head / MTP via ``expert_tensor_names_stage``);
  2. reads the f32 gate/up/down arrays (lossless-mxfp4 repack when the source
     already carries a scale partner in the target format, else dequant->quant);
  3. stacks them into packed uint8 rows and writes a ONE-LAYER GGUF whose KV
     carries the whole model's layer/expert geometry;
  4. runs ``llama-wp-repack`` on that GGUF (``--layer-ranges L-L --allow-partial``
     plus ``--expert-slices ... --slice-output-split`` for width-sliced sets);
  5. streams each set's per-layer blob through its sink (skipping a put when the
     blob is already on the sink at the expected size) and keeps the per-layer
     output for the later stitch;
  6. deletes the one-layer GGUF and (at the end) the raw repack outputs, and
     emits ``layer_done``.

After the last layer each set is stitched (``stitch``); its index JSONs and
manifest are uploaded, and ``llama-wp-expert-descriptor`` is run on LOCAL copies
of the manifest + index JSONs against the spine (assumed local for v1). The
sliced+layer-partial descriptor is a known C++ gap, so a ``ToolError`` there is
downgraded to a ``verify`` event with ``descriptor_rel=None`` rather than an abort.

Resume: a layer's quant+repack runs only if at least one set's blob for that
layer is missing from the sink or has the wrong size; a fully-resided layer is
skipped (no shard read, no repack) and its per-layer output is recovered from
the sink.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Protocol

import gguf
import numpy as np

from .arch import ArchSpec, classify, expert_tensor_names, expert_tensor_names_stage
from .plan import ResolvedPlan, SetSpec
from .quant import lossless_repack, quantize_expert
from .sink import Sink
from .source import Source
from .stitch import LayerOutput, layer_outputs_from_repack, stitch
from .tools import Tools, ToolError


@dataclass
class StageResult:
    manifest_rel: str
    descriptor_rel: str | None
    blobs: list[tuple[str, int, str]]  # (dst_name, bytes, sha256)


class ExpertStage:
    def __init__(
        self,
        stage_sets: list[SetSpec],
        rplan: ResolvedPlan,
        source: Source,
        tools: Tools,
        sinks: dict[str, Sink],
        events: Callable[[dict], None],
        workdir: Path,
        *,
        workers: int = 1,
    ) -> None:
        if not stage_sets:
            raise ValueError("ExpertStage needs at least one set")
        # One stage owns all width sets of one plan stage: they share layers and
        # widths (the same split applied to every width set); only slice_index
        # differs. Assert that.
        layers = tuple(stage_sets[0].layers.layers())
        widths = stage_sets[0].widths
        role = stage_sets[0].role
        for s in stage_sets:
            if tuple(s.layers.layers()) != layers:
                raise ValueError(
                    f"stage sets disagree on layers: {stage_sets[0].id}={list(layers)} "
                    f"vs {s.id}={s.layers.layers()}"
                )
            if s.widths != widths:
                raise ValueError(
                    f"stage sets disagree on widths: {stage_sets[0].id}={widths} vs {s.id}={s.widths}"
                )
            if s.role != role:
                raise ValueError(
                    f"stage sets disagree on role: {stage_sets[0].id}={role} vs {s.id}={s.role}"
                )
        for s in stage_sets:
            if s.id not in sinks:
                raise KeyError(f"no sink for set {s.id}")

        self.stage_sets = list(stage_sets)
        self.rplan = rplan
        self.source = source
        self.tools = tools
        self.sinks = sinks
        self.events = events
        self.workdir = Path(workdir)
        self.workers = int(workers)
        self.layers = list(layers)
        self.n = len(self.layers)
        self.role = role
        self.stage_id = stage_sets[0].id.rsplit("-w", 1)[0]
        self.quant = stage_sets[0].quant

        self.arch = rplan.arch
        self.hp = rplan.hparams
        self.n_layer = self.arch.n_layer(self.hp)
        self.n_ff = int(self.hp[self.arch.n_ff_exp_key])
        self.n_embd = int(self.hp[self.arch.hidden_key])
        self.n_expert_used = int(self.hp["num_experts_per_tok"])
        self.n_expert = (
            int(self.hp[self.arch.spec_head_n_expert_key])
            if role == "spec_head"
            else int(self.hp[self.arch.n_expert_key])
        )
        self.quant_dtype = gguf.GGMLQuantizationType[self.quant.upper()]
        # raw repack bases to clean up after the final stitch.
        self._raw_bases: list[int] = []

    # -- naming -----------------------------------------------------------

    def _dst_name(self, set_spec: SetSpec, pos: int) -> str:
        # pos is the 1-based index of the layer within the stage's layers.
        return f"{set_spec.output_base}-experts-{pos:05d}-of-{self.n:05d}.wpb"

    def _expected_blob_bytes(self, set_spec: SetSpec) -> int:
        """On-disk size repack produces for this set's one-layer blob.

        Used only for the resume size-compare. Exact for unsliced and for
        block-aligned quant types; a mismatch (or padding) just means "re-run",
        which is the safe direction.
        """
        block, type_size = gguf.GGML_QUANT_SIZES[self.quant_dtype]

        def row_bytes(cols: int) -> int:
            return (cols // block) * type_size

        if set_spec.widths is None:
            gate = self.n_ff * row_bytes(self.n_embd)
            up = self.n_ff * row_bytes(self.n_embd)
            down = self.n_embd * row_bytes(self.n_ff)
        else:
            w = set_spec.widths[set_spec.slice_index or 0]
            gate = w * row_bytes(self.n_embd)
            up = w * row_bytes(self.n_embd)
            down = self.n_embd * row_bytes(w)
        return self.n_expert * (gate + up + down)

    def _needs(self, set_spec: SetSpec, pos: int) -> bool:
        dst = self._dst_name(set_spec, pos)
        if not self.sinks[set_spec.id].exists(dst):
            return True
        return self.sinks[set_spec.id].size(dst) != self._expected_blob_bytes(set_spec)

    # -- source reading ---------------------------------------------------

    def _shards_for(self, L: int) -> list[str]:
        if self.role == "experts":
            return self.source.layer_shards(L)
        # Spec-head tensors are named mtp.{stage}.*; layer_shards only keys off
        # layers.{L}.*, so pick the shards that actually hold them.
        idx = self.source.tensor_index()
        stage = L - self.n_layer
        prefix = f"mtp.{stage}."
        seen: list[str] = []
        for name, shard in idx.items():
            if name.startswith(prefix) and shard not in seen:
                seen.append(shard)
        if not seen:
            raise KeyError(f"no shards hold spec-head tensors for layer {L} (mtp.{stage})")
        return seen

    def _gather(self, L: int) -> dict[str, np.ndarray]:
        """Packed per-role stacks [n_expert, rows, bytes_per_row] uint8."""
        if self.role == "experts":
            names_by_eid = [expert_tensor_names(self.arch, L, eid) for eid in range(self.n_expert)]
        else:
            stage = L - self.n_layer
            names_by_eid = [
                expert_tensor_names_stage(self.arch, stage, eid) for eid in range(self.n_expert)
            ]

        stacks: dict[str, np.ndarray] = {}
        for shard in self._shards_for(L):
            with self.source.open_shard(shard) as reader:
                keys = set(reader.keys())
                for eid, names in enumerate(names_by_eid):
                    for role in ("gate", "up", "down"):
                        wname, sname = names[role]
                        if wname not in keys:
                            continue
                        f32 = reader.get_tensor(wname)
                        scale = reader.get_tensor(sname) if (sname is not None and sname in keys) else None
                        packed = lossless_repack(f32, scale, "mxfp4", self.quant) if scale is not None else None
                        if packed is None:
                            packed = quantize_expert({role: f32}, self.quant, workers=self.workers)[role]
                        if role not in stacks:
                            stacks[role] = np.empty((self.n_expert, *packed.shape), dtype=np.uint8)
                        stacks[role][eid] = packed
        for role in ("gate", "up", "down"):
            if role not in stacks:
                raise KeyError(f"role {role} has no experts for layer {L}")
        return stacks

    def _write_gguf(self, L: int, stacks: dict[str, np.ndarray], path: Path) -> None:
        writer = gguf.GGUFWriter(str(path), arch=self.arch.name)
        writer.add_block_count(self.n_layer + self.arch.nextn_count(self.hp))
        writer.add_embedding_length(self.n_embd)
        writer.add_expert_count(self.n_expert)
        writer.add_expert_feed_forward_length(self.n_ff)
        writer.add_expert_used_count(self.n_expert_used)
        for role in ("gate", "up", "down"):
            writer.add_tensor(
                f"blk.{L}.ffn_{role}_exps.weight", stacks[role], raw_dtype=self.quant_dtype
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    # -- per-layer execution ----------------------------------------------

    def _run_layer(
        self, per_layer: Dict[str, dict[int, tuple[LayerOutput, str, int, str]]], pos: int, L: int,
        need: dict[str, bool],
    ) -> None:
        t0 = time.time()
        stacks = self._gather(L)
        gguf_path = self.workdir / f"L{L:03d}.gguf"
        self._write_gguf(L, stacks, gguf_path)

        widths = self.stage_sets[0].widths
        raw_base = self.workdir / f"raw-L{L:03d}"
        self.tools.repack(
            gguf_path,
            raw_base,
            layer_ranges=f"{L}-{L}",
            allow_partial=True,
            expert_slices=widths or None,
            slice_output_split=bool(widths),
        )
        self._raw_bases.append(L)

        for s in self.stage_sets:
            lo = layer_outputs_from_repack(raw_base, L, s.slice_index)
            dst = self._dst_name(s, pos)
            if need[s.id]:
                nbytes, sha = self.sinks[s.id].put_file(lo.blob, dst)
            else:
                nbytes = self.sinks[s.id].size(dst)
                sha = self.sinks[s.id].sha256(dst)
            per_layer[s.id][L] = (lo, dst, nbytes, sha)

        # Delete the one-layer GGUF now; raw outputs are kept for the final
        # stitch and cleaned up at the end of run().
        gguf_path.unlink(missing_ok=True)
        self.events(
            {"kind": "layer_done", "stage": self.stage_id, "layer": L,
             "bytes": sum(per_layer[s.id][L][2] for s in self.stage_sets),
             "secs": round(time.time() - t0, 3)}
        )

    def _recover(
        self, per_layer: Dict[str, dict[int, tuple[LayerOutput, str, int, str]]], pos: int, L: int,
    ) -> None:
        # Whole layer resided: no shard read, no repack. Rebuild each set's
        # per-layer output from the (local) sink.
        for s in self.stage_sets:
            sink = self.sinks[s.id]
            dst = self._dst_name(s, pos)
            blob = Path(sink.root) / dst
            index = Path(sink.root) / (dst[: -len(".wpb")] + ".wpi.json")
            per_layer[s.id][L] = (
                LayerOutput(L, index, blob), dst, sink.size(dst), sink.sha256(dst)
            )

    # -- per-set finish ----------------------------------------------------

    def _finish_set(
        self, s: SetSpec, entries: dict[int, tuple[LayerOutput, str, int, str]]
    ) -> StageResult:
        sink = self.sinks[s.id]
        sink.mkdir()
        per_layer_outputs = [entries[L][0] for L in self.layers]
        stitched = stitch(
            per_layer_outputs,
            s,
            self.rplan.spine_path,
            input_model=self.rplan.plan.source,
            expert_type=s.quant,
            n_expert=self.n_expert,
        )

        manifest_rel = f"{s.output_base}-experts-manifest.json"
        manifest_text = json.dumps(stitched.manifest, indent=2) + "\n"
        if not sink.exists(manifest_rel):
            sink.put_text(manifest_text, manifest_rel)
        for plan in stitched.blobs:
            idx_rel = plan.dst_name[: -len(".wpb")] + ".wpi.json"
            if not sink.exists(idx_rel):
                sink.put_text(plan.index_text, idx_rel)

        descriptor_rel: str | None = f"{s.output_base}-experts-manifest.expert-descriptor.json"
        spine_local = Path(self.rplan.spine_path)
        if not spine_local.exists():
            raise AssertionError(f"spine not local: {spine_local}")
        try:
            desc_dir = self.workdir / f"desc-{s.id}"
            desc_dir.mkdir(parents=True, exist_ok=True)
            (desc_dir / manifest_rel).write_text(manifest_text)
            for plan in stitched.blobs:
                (desc_dir / (plan.dst_name[: -len(".wpb")] + ".wpi.json")).write_text(plan.index_text)
            desc_local = self.tools.descriptor(spine_local, desc_dir / manifest_rel)
            if not sink.exists(descriptor_rel):
                sink.put_text(desc_local.read_text(), descriptor_rel)
            shutil.rmtree(desc_dir, ignore_errors=True)
        except ToolError:
            descriptor_rel = None
            self.events(
                {
                    "kind": "verify",
                    "set": s.id,
                    "descriptor": "skipped: tool rejected sliced+layer-partial manifest",
                }
            )

        blobs = [(entries[L][1], entries[L][2], entries[L][3]) for L in self.layers]
        return StageResult(manifest_rel=manifest_rel, descriptor_rel=descriptor_rel, blobs=blobs)

    # -- driver ------------------------------------------------------------

    def run(self) -> dict[str, StageResult]:
        t0 = time.time()
        self.events({"kind": "stage_start", "stage": self.stage_id, "layers": list(self.layers)})
        self.workdir.mkdir(parents=True, exist_ok=True)

        per_layer: Dict[str, dict[int, tuple[LayerOutput, str, int, str]]] = {
            s.id: {} for s in self.stage_sets
        }
        for pos, L in enumerate(self.layers, start=1):
            need = {s.id: self._needs(s, pos) for s in self.stage_sets}
            if any(need.values()):
                self._run_layer(per_layer, pos, L, need)
            else:
                self._recover(per_layer, pos, L)

        results: dict[str, StageResult] = {
            s.id: self._finish_set(s, per_layer[s.id]) for s in self.stage_sets
        }
        self.events({"kind": "stage_done", "stage": self.stage_id, "layers": list(self.layers)})
        self._cleanup()
        return results

    def _cleanup(self) -> None:
        for L in self._raw_bases:
            for p in self.workdir.glob(f"raw-L{L:03d}*"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            self._raw_bases = []
        for p in self.workdir.glob("desc-*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Spine + sidecar stages (Task 11)
# ---------------------------------------------------------------------------


class SpineBuilder(Protocol):
    def build(self, peel_dir: Path, out_gguf: Path, quant: dict[str, str]) -> Path: ...


class ConverterSpineBuilder:
    """Spine via the stock convert_hf_to_gguf.py + llama-quantize.

    NOTE: the plan's per-class ``spec_head.dense`` quant override is NOT wired
    for the converter path yet -- llama-quantize ``--tensor-type`` keys on the
    GGUF tensor names the stock converter emits, not on wp-forge classes, so
    only the plan's ``dense`` class (defaulted to q8_0) is applied.
    """

    def __init__(self, tools: Tools, arch: ArchSpec) -> None:
        self.tools = tools
        self.arch = arch

    def build(self, peel_dir: Path, out_gguf: Path, quant: dict[str, str]) -> Path:
        if not self.arch.converter_tolerates_missing_experts:
            raise NotImplementedError(
                f"converter path for arch {self.arch.name}: the stock converter "
                "rejects HF repos whose routed-expert tensors are missing; the "
                "zero-stub peel trick is a follow-up"
            )
        tmp_bf16 = out_gguf.parent / (out_gguf.name + ".bf16.gguf")
        try:
            self.tools.convert_hf(peel_dir, tmp_bf16)
            result = self.tools.quantize(
                tmp_bf16, out_gguf,
                default_type=quant.get("dense", "q8_0"), tensor_types={},
            )
        finally:
            tmp_bf16.unlink(missing_ok=True)
        return result


class SpineStage:
    """Produce the spine GGUF (dense tensors only) and land it on the sink.

    HF source: every shard is peeled -- tensors classified "dense" /
    "spec_head.dense" are copied, bf16, into workdir/peel/<shard filename>,
    config.json is copied, and model.safetensors.index.json is regenerated
    over the peel -- then the builder converts+quantizes the peel.

    The peel deliberately keeps the source's SHARD FILENAMES so the index and
    the weight_map a stock converter expects stay valid. Shards are NOT
    released after the peel: the expert stages still need to read the routed
    expert tensors out of the same shards.

    GGUF source: tools.dense_extract on the first shard; no peel, no builder.
    """

    def __init__(
        self,
        rplan: ResolvedPlan,
        source: Source,
        tools: Tools,
        sink: Sink,
        events: Callable[[dict], None],
        workdir: Path,
        builder: SpineBuilder,
    ) -> None:
        self.rplan = rplan
        self.source = source
        self.tools = tools
        self.sink = sink
        self.events = events
        self.workdir = Path(workdir)
        self.builder = builder

    def run(self) -> tuple[int, str]:
        spine_rel = self.rplan.spine_path.rsplit("/", 1)[-1]
        if self.sink.exists(spine_rel) and (self.sink.size(spine_rel) or 0) > 0:
            self.events({"kind": "stage_done", "stage": "spine", "resumed": True})
            return self.sink.size(spine_rel), self.sink.sha256(spine_rel)

        t0 = time.time()
        if self.source.is_gguf:
            local = self.tools.dense_extract(self.source.path, self.workdir / "spine.gguf")
            size, sha = self.sink.put_file(local, spine_rel)
            local.unlink()
            self.events(
                {"kind": "stage_done", "stage": "spine", "bytes": size,
                 "sha": sha, "secs": round(time.time() - t0, 3)}
            )
            return size, sha

        self.workdir.mkdir(parents=True, exist_ok=True)
        peel_dir = self.workdir / "peel"
        peel_bytes = self._peel(peel_dir)
        self.events(
            {"kind": "stage_start", "stage": "spine",
             "est_local_peak_bytes": 2 * peel_bytes}
        )
        out_gguf = self.workdir / "spine.gguf"
        result = self.builder.build(peel_dir, out_gguf, self.rplan.plan.quant)
        size, sha = self.sink.put_file(result, spine_rel)
        shutil.rmtree(peel_dir)
        for p in (out_gguf, result):
            p.unlink(missing_ok=True)
        self.events(
            {"kind": "stage_done", "stage": "spine", "bytes": size,
             "sha": sha, "secs": round(time.time() - t0, 3)}
        )
        return size, sha

    def _peel(self, peel_dir: Path) -> int:
        import torch
        from safetensors.torch import save_file  # noqa: F401 (API per contract)

        peel_dir.mkdir(parents=True, exist_ok=True)
        arch = self.rplan.arch
        index = self.source.tensor_index()
        shards = sorted(set(index.values()))
        weight_map: dict[str, str] = {}
        for shard in shards:
            tensors: dict[str, "torch.Tensor"] = {}
            with self.source.open_shard(shard) as reader:
                for name in reader.keys():
                    if classify(arch, name) not in ("dense", "spec_head.dense"):
                        continue
                    # reader.get_tensor f32-casts; back to bf16 for the peel
                    # (dsv41_experts_from_hf peels raw, the synthetic repo is
                    # bf16 so the round trip is lossless there)
                    tensors[name] = torch.from_numpy(reader.get_tensor(name)).to(torch.bfloat16)
                    weight_map[name] = shard
            # NOTE: no source.release_shard(shard) -- the expert stages still
            # read routed experts from these shards.
            if tensors:
                save_file(tensors, str(peel_dir / shard))
        (peel_dir / "config.json").write_text(
            (self.source.cache_dir / "config.json").read_text()
        )
        total = sum(p.stat().st_size for p in peel_dir.glob("*.safetensors"))
        (peel_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total},
                        "weight_map": weight_map}) + "\n"
        )
        return total


class SidecarStage:
    """One named sidecar class (e.g. "engram") from the HF source to the sink.

    Every tensor classified ``cls`` is written into ONE GGUF (arch = the plan's
    arch, KV is just general.name). Tensor names: for class "engram" the HF
    name ``layers.{L}.engram.*`` is rekeyed to the deepseek41 blk names the
    converter uses (engram.embed -> blk.{L}.engram_embd); any other sidecar
    class keeps its HF name unchanged. f16/bf16 plan quants store the raw
    values; quant types are packed per-tensor through quantize_expert.

    GGUF sources have no sidecars: run() is a no-op returning (0, "").
    """

    def __init__(
        self,
        cls: str,
        rplan: ResolvedPlan,
        source: Source,
        tools: Tools,
        sink: Sink,
        events: Callable[[dict], None],
        workdir: Path,
    ) -> None:
        self.cls = cls
        self.rplan = rplan
        self.source = source
        self.tools = tools
        self.sink = sink
        self.events = events
        self.workdir = Path(workdir)

    def run(self) -> tuple[int, str]:
        if self.source.is_gguf:
            return 0, ""
        sidecar_rel = self.rplan.sidecar_paths[self.cls].rsplit("/", 1)[-1]
        if self.sink.exists(sidecar_rel) and (self.sink.size(sidecar_rel) or 0) > 0:
            self.events({"kind": "stage_done", "stage": self.cls, "resumed": True})
            return self.sink.size(sidecar_rel), self.sink.sha256(sidecar_rel)

        t0 = time.time()
        arch = self.rplan.arch
        qtype = self.rplan.plan.quant[self.cls]
        out_gguf = self.workdir / f"{self.cls}.gguf"
        self.workdir.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(str(out_gguf), arch=arch.name)
        writer.add_name(self.rplan.plan.name)
        for name, shard in self.source.tensor_index().items():
            if classify(arch, name) != self.cls:
                continue
            with self.source.open_shard(shard) as reader:
                arr = reader.get_tensor(name)
            payload = self._to_gguf(arr, qtype)
            if isinstance(payload, tuple):
                packed, qt = payload
                # raw_shape is the byte shape the writer stores; gguf-py
                # derives the ggml shape from it (quants.quant_shape_from_byte_shape)
                writer.add_tensor(self._gguf_name(name), packed, raw_dtype=qt)
            else:
                writer.add_tensor(self._gguf_name(name), payload)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        size, sha = self.sink.put_file(out_gguf, sidecar_rel)
        out_gguf.unlink()
        self.events(
            {"kind": "stage_done", "stage": self.cls, "bytes": size,
             "sha": sha, "secs": round(time.time() - t0, 3)}
        )
        return size, sha

    @staticmethod
    def _gguf_name(hf_name: str) -> str:
        # dsv41_engram_from_hf naming: layers.{L}.engram.embed.weight is the
        # only sidecar tensor the arch ships; k/q/wkv are absent from this
        # source layout. Any other class keeps its HF name unchanged.
        if hf_name.rsplit(".", 2)[0].rsplit(".", 1)[-1] == "engram":
            bid = hf_name.split(".")[1]
            return gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.ENGRAM_EMBD].format(bid=bid) + ".weight"
        return hf_name

    def _to_gguf(self, arr: np.ndarray, qtype: str):
        qt = gguf.GGMLQuantizationType[qtype.upper()]
        if qt in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            # BF16 is packed through gguf-py (np has no bf16); F16 is plain
            # little-endian bytes, matching the writer's F16 handling.
            if qt is gguf.GGMLQuantizationType.BF16:
                return gguf.quantize(np.ascontiguousarray(arr), qt)
            return arr.astype(np.float16)
        if arr.ndim != 2:
            raise QuantError(
                f"sidecar {self.cls}: quant {qtype} needs a 2-D tensor, "
                f"got shape {arr.shape}"
            )
        packed = quantize_expert({"t": arr}, qtype, workers=1)["t"]
        return packed, qt
