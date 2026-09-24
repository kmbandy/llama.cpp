"""Generalized set stitcher for wp-forge.

Generalizes ``dsv41_experts_from_hf.stitch`` / ``raw_sources``: it takes the
per-layer outputs of ``llama-wp-repack`` run on one-layer GGUFs
(``--layer-ranges L-L --allow-partial``, plus ``--expert-slices ...
--slice-output-split`` for width-sliced sets) and produces the stitched
manifest plus an ordered ``BlobPlan`` list that the stage streams through the
sink. The stitcher never copies or links blobs and never touches the network.

Two manifest shapes are emitted:
  * unsliced sets: exactly today's DS4.1 ``layer-ranges`` v1 shape
    (``aggregate_identity`` imported, not copied, from the DS4.1 converter);
  * width-sliced sets: the NEW combined shape ``sharding_mode = "expert-slice"``
    PLUS ``layer_ranges`` / ``allow_partial`` / ``expert_ggml_type`` and the
    per-slice ``expert_slicing`` block with ``selected_slice`` -- the shape the
    C++ descriptor/worker task teaches them to accept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .plan import SetSpec
from conversion.dsv41_experts_from_hf import aggregate_identity

MANIFEST_FORMAT = "llama.cpp.weight-pager.expert-shard-manifest"

# Per-layer output of llama-wp-repack on a ONE-LAYER gguf.
LayerOutput = NamedTuple("LayerOutput", [("layer", int), ("index_json", Path), ("blob", Path)])

# Where one stitched shard's bytes come from: src_blob is streamed as dst_name
# and index_text is the per-layer index JSON, rewritten for the stitched set.
BlobPlan = NamedTuple("BlobPlan", [("src_blob", Path), ("dst_name", str), ("index_text", str)])


@dataclass
class StitchedSet:
    manifest: dict  # write as <output_base>-experts-manifest.json
    blobs: list[BlobPlan]


def _require_layers(spec: SetSpec, layers: list[int]) -> None:
    if not layers:
        raise ValueError("no layers to stitch")
    if layers != sorted(layers) or len(set(layers)) != len(layers):
        raise ValueError(f"layers must be ascending and unique, got {layers}")
    if layers[0] < spec.layers.first or layers[-1] > spec.layers.last:
        raise ValueError(
            f"layers {layers} do not fit within set {spec.id} layers {spec.layers}"
        )
    expected = list(range(spec.layers.first, spec.layers.last + 1))
    if sorted(layers) != expected:
        raise ValueError(
            f"layers must be contiguous within {spec.id} layers {spec.layers}, got {layers}"
        )


def _rewrite_index(
    idx: dict,
    layer: int,
    i: int,
    n: int,
    dst_blob_name: str,
    model_files: list[str],
) -> str:
    """Exactly what dsv41's stitch() does to a per-layer index JSON."""
    if idx["layer_first"] != layer or idx["layer_last"] != layer:
        raise ValueError(f"index is not layer {layer}: first={idx['layer_first']} last={idx['layer_last']}")
    idx["blob_file"] = dst_blob_name
    idx["shard_index"] = i
    idx["shard_count"] = n
    idx["model_files"] = model_files
    return json.dumps(idx, indent=2) + "\n"


def stitch(
    per_layer: list[LayerOutput],
    set_spec: SetSpec,
    spine_path: str,
    input_model: str,
    expert_type: str,
    n_expert: int,
) -> StitchedSet:
    """Stitch per-layer repack outputs into one set manifest + BlobPlan list.

    ``per_layer`` must be one LayerOutput per layer of ``set_spec.layers``,
    ascending and contiguous (ValueError otherwise). For width-sliced sets
    (``set_spec.widths`` / ``slice_index`` set) each entry must be the output
    of the set's own slice (``--slice-output-split``); the per-layer slice
    manifests' ``expert_slicing`` geometry must agree or it raises.
    """
    layers = [lo.layer for lo in per_layer]
    _require_layers(set_spec, layers)
    n = len(layers)
    model_files = [spine_path]

    def dst_name(i: int) -> str:
        return f"{set_spec.output_base}-experts-{i + 1:05d}-of-{n:05d}.wpb"

    blobs: list[BlobPlan] = []
    shards: list[dict] = []
    for i, lo in enumerate(per_layer):
        idx = json.loads(lo.index_json.read_text())
        blobs.append(
            BlobPlan(
                src_blob=lo.blob,
                dst_name=dst_name(i),
                index_text=_rewrite_index(idx, lo.layer, i, n, dst_name(i), model_files),
            )
        )
        shards.append(
            {
                "blob_file": blobs[i].dst_name,
                "index_file": blobs[i].dst_name[: -len(".wpb")] + ".wpi.json",
                "shard_index": i,
                "layer_first": lo.layer,
                "layer_last": lo.layer,
                "group_count": idx["group_count"],
                "blob_bytes": idx["blob_bytes"],
                "content_hash": idx["content_hash"],
            }
        )
    total = sum(int(s["blob_bytes"]) for s in shards)

    manifest: dict = {
        "format": MANIFEST_FORMAT,
        "version": 1,
        "sharding_mode": "layer-ranges",
        "layer_ranges": [str(set_spec.layers)],
        "allow_partial": True,
        "input_model": input_model,
        "model_files": model_files,
        "retained_expert_range": {
            "first": 0 if set_spec.expert_first is None else set_spec.expert_first,
            "last": n_expert - 1 if set_spec.expert_last is None else set_spec.expert_last,
        },
        "expert_ggml_type": expert_type,
        "content_hash": aggregate_identity(shards),
        "total_group_count": sum(s["group_count"] for s in shards),
        "total_blob_bytes": total,
        "shard_count": n,
        "shards": shards,
        "note": f"wp-forge {set_spec.id} experts layers {set_spec.layers}, {expert_type}",
    }

    if set_spec.widths is not None:
        if set_spec.slice_index is None:
            raise ValueError(f"width-sliced set {set_spec.id} has no slice_index")
        geometry = None
        # per-slice repack manifests sit next to the slice index files as
        # <slice_base>-experts-manifest.json (see layer_outputs_from_repack)
        for lo in per_layer:
            man = json.loads(_slice_manifest_for(lo).read_text())
            es = man.get("expert_slicing")
            if es is None:
                raise ValueError(f"{lo.index_json}: slice manifest has no expert_slicing block")
            geo = {"widths": es["widths"], "n_ff_exp": es["n_ff_exp"], "n_embd": es["n_embd"]}
            if geometry is None:
                geometry = geo
            elif geo != geometry:
                raise ValueError(
                    f"{lo.index_json}: expert_slicing geometry {geo} disagrees with {geometry}"
                )
        first_man = json.loads(_slice_manifest_for(per_layer[0]).read_text())
        es = dict(first_man["expert_slicing"])
        if es.get("selected_slice") != set_spec.slice_index:
            raise ValueError(
                f"{per_layer[0].index_json}: repack produced slice {es.get('selected_slice')}, "
                f"set {set_spec.id} wants slice {set_spec.slice_index}"
            )
        manifest["sharding_mode"] = "expert-slice"
        manifest["expert_slicing"] = es
        manifest["note"] = (
            f"wp-forge {set_spec.id} experts layers {set_spec.layers}, {expert_type}, "
            f"slice {set_spec.slice_index}/{es['slice_count']} widths {set_spec.widths}"
        )

    return StitchedSet(manifest=manifest, blobs=blobs)


def _slice_manifest_for(lo: LayerOutput) -> Path:
    # <...>-eslice-slice-NNNNN-experts-00001-of-00001.wpi.json
    #      -> <...>-eslice-slice-NNNNN-experts-manifest.json
    return lo.index_json.parent / (lo.index_json.name.replace("-00001-of-00001.wpi.json", "-manifest.json"))


def layer_outputs_from_repack(base: Path, layer: int, slice_index: int | None) -> LayerOutput:
    """Locate (layer, index_json, blob) from one per-layer repack base.

    ``base`` is the positional output arg passed to llama-wp-repack. For an
    unsliced run the manifest is ``<base>-experts-manifest.json``; for a
    ``--slice-output-split`` run it is
    ``<base>-eslice-slice-NNNNN-experts-manifest.json``.
    """
    if slice_index is None:
        man_path = Path(str(base) + "-experts-manifest.json")
    else:
        man_path = Path(f"{base}-eslice-slice-{slice_index:05d}-experts-manifest.json")
    man = json.loads(man_path.read_text())
    shard = man["shards"][0]
    return LayerOutput(
        layer,
        man_path.parent / shard["index_file"],
        man_path.parent / shard["blob_file"],
    )
