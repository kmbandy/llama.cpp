"""wp-forge merge-sets: fold one expert set into the adjacent set on the same machine.

Used for the DeepSeek spec-head set (layers 40-42) folded into the experts set
ending at layer 39, so one worker serves layers 0-42 from one manifest.

Everything that can fail is computed in memory first (validation, renamed
index JSON, the merged manifest, the regenerated descriptor). The sink is only
touched after that, so a PlanError leaves both sets as they were.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from conversion.dsv41_experts_from_hf import aggregate_identity

from .machines import Machine
from .plan import PlanError
from .sink import sink_for
from .tools import Tools, ToolError

_BLOB_RE = re.compile(r"^(?P<base>.+)-experts-\d{5}-of-\d{5}\.wpb$")
Emit = Callable[[dict], None]


def _set(bundle: dict, sid: str) -> dict:
    for s in bundle["expert_sets"]:
        if s["id"] == sid:
            return s
    raise PlanError(f"merge-sets: no expert set {sid!r} in bundle")


def _base(entry: dict) -> str:
    if not entry["blobs"]:
        raise PlanError(f"merge-sets: {entry['id']} has no blobs")
    bases = set()
    for b in entry["blobs"]:
        m = _BLOB_RE.match(b["file"])
        if m is None:
            raise PlanError(f"merge-sets: {entry['id']}: blob {b['file']!r} is not <base>-experts-NNNNN-of-NNNNN.wpb")
        bases.add(m["base"])
    if len(bases) != 1:
        raise PlanError(f"merge-sets: {entry['id']}: blobs do not share one output base: {sorted(bases)}")
    return bases.pop()


def merge_sets(bundle_path: Path, machines: dict[str, Machine], into_id: str, from_id: str, *,
               dry_run: bool = False, emit: Emit = lambda e: None) -> dict:
    """Merge set from_id into the set just before it (into_id). Returns the plan summary.

    Raises PlanError (CLI exit 2) on any precondition failure, before any write.
    """
    bundle_path = Path(bundle_path)
    raw_bundle = bundle_path.read_text()
    bundle = json.loads(raw_bundle)
    into, frm = _set(bundle, into_id), _set(bundle, from_id)
    if into["machine"] != frm["machine"]:
        raise PlanError(f"merge-sets: {into_id} is on {into['machine']}, {from_id} on {frm['machine']}")
    if into["machine"] not in machines:
        raise PlanError(f"merge-sets: machine {into['machine']!r} is not in machines.json")
    il, fl = into["layers"], frm["layers"]
    if il[1] + 1 != fl[0]:
        raise PlanError(f"merge-sets: {from_id} layers {fl} do not follow {into_id} layers {il}")

    m = machines[into["machine"]]

    def norm(d: str) -> str:
        return os.path.expanduser(d) if m.is_local else d

    into_dir, from_dir = norm(into["path"]), norm(frm["path"])
    # one sink rooted at the common parent, so the cross-dir moves are plain renames
    root = os.path.commonpath([into_dir, from_dir])
    sink = sink_for(m, root)

    def rel(d: str, name: str) -> str:
        return os.path.relpath(os.path.join(d, name), root)

    def read_manifest(entry: dict, d: str) -> tuple[str, dict]:
        name = os.path.basename(entry["manifest"])
        text = sink.get_text(rel(d, name))
        if text is None:
            raise PlanError(f"merge-sets: {entry['id']}: manifest {name} not found in {d}")
        return text, json.loads(text)

    into_man_raw, man_i = read_manifest(into, into_dir)
    _, man_f = read_manifest(frm, from_dir)
    for label, man in (("into", man_i), ("from", man_f)):
        if man.get("sharding_mode") != "layer-ranges":
            raise PlanError(f"merge-sets: {label} manifest sharding_mode is {man.get('sharding_mode')!r}, "
                            "want 'layer-ranges'")
    if man_i.get("expert_ggml_type") != man_f.get("expert_ggml_type"):
        raise PlanError(f"merge-sets: expert_ggml_type differs: {man_i.get('expert_ggml_type')} vs "
                        f"{man_f.get('expert_ggml_type')}")

    spine_local = Path(os.path.expanduser(man_i["model_files"][0]))
    if not spine_local.is_file():
        raise PlanError(f"merge-sets: spine {spine_local} is not on this machine; the descriptor needs it")

    def load_shards(entry: dict, d: str, man: dict) -> list[tuple[dict, dict, dict]]:
        blobs = {b["file"]: b for b in entry["blobs"]}
        out = []
        for sh in man["shards"]:
            b = blobs.get(sh["blob_file"])
            if b is None:
                raise PlanError(f"merge-sets: {entry['id']}: shard blob {sh['blob_file']} not in bundle")
            if sink.size(rel(d, sh["blob_file"])) != b["bytes"]:
                raise PlanError(f"merge-sets: {entry['id']}: {sh['blob_file']} is not {b['bytes']} bytes on disk")
            idx_text = sink.get_text(rel(d, sh["index_file"]))
            if idx_text is None:
                raise PlanError(f"merge-sets: {entry['id']}: index {sh['index_file']} missing in {d}")
            out.append((sh, b, json.loads(idx_text)))
        return out

    base = _base(into)
    entries = ([(into_dir, *t) for t in load_shards(into, into_dir, man_i)]
               + [(from_dir, *t) for t in load_shards(frm, from_dir, man_f)])
    n = len(entries)

    # new names, indexes and manifest shard list, all in memory
    new_shards, new_blobs, index_out, moves = [], [], [], []
    for k, (d, sh, b, idx) in enumerate(entries):
        nb = f"{base}-experts-{k + 1:05d}-of-{n:05d}.wpb"
        ni = nb[: -len(".wpb")] + ".wpi.json"
        idx = dict(idx, blob_file=nb, shard_index=k, shard_count=n)
        index_out.append((ni, json.dumps(idx, indent=2) + "\n"))
        new_shards.append({
            "blob_file": nb, "index_file": ni, "shard_index": k,
            "layer_first": sh["layer_first"], "layer_last": sh["layer_last"],
            "group_count": sh["group_count"], "blob_bytes": sh["blob_bytes"],
            "content_hash": sh["content_hash"],
        })
        new_blobs.append({"file": nb, "bytes": b["bytes"], "sha256": b["sha256"]})
        moves.append((d, sh["blob_file"], sh["index_file"], nb, ni))

    rot: dict = {}
    for man in (man_i, man_f):
        for key, v in (man.get("ml8_rotation") or {}).items():
            if key in rot and rot[key] != v:
                raise PlanError(f"merge-sets: ml8_rotation[{key}] disagrees between the two manifests")
            rot[key] = v

    new_man = dict(man_i)
    new_man.update({
        "layer_ranges": [f"{il[0]}-{fl[1]}"],
        "shards": new_shards,
        "shard_count": n,
        "content_hash": aggregate_identity(new_shards),
        "total_group_count": sum(int(s["group_count"]) for s in new_shards),
        "total_blob_bytes": sum(int(s["blob_bytes"]) for s in new_shards),
        "note": f"{man_i.get('note', 'wp-forge')}; merged {from_id} (layers {fl[0]}-{fl[1]}) via wp-forge merge-sets",
    })
    if rot:
        new_man["ml8_rotation"] = rot
    man_name = os.path.basename(into["manifest"])
    desc_name = (os.path.basename(into["descriptor"]) if into.get("descriptor")
                 else Path(man_name).stem + ".expert-descriptor.json")

    # preflight: nothing the merge creates may already exist
    created = [nb for _, _, _, nb, _ in moves] + [ni for _, _, _, _, ni in moves]
    created += [man_name + ".bak-merge"]
    for name in created:
        if sink.exists(rel(into_dir, name)):
            raise PlanError(f"merge-sets: {name} already exists in {into_dir}")
    if not into.get("descriptor") and sink.exists(rel(into_dir, desc_name)):
        raise PlanError(f"merge-sets: {desc_name} already exists in {into_dir}")
    bak_bundle = bundle_path.with_name(bundle_path.name + ".bak-merge")
    if bak_bundle.exists():
        raise PlanError(f"merge-sets: {bak_bundle} already exists")

    new_bundle_set = dict(into, layers=[il[0], fl[1]], bytes=sum(b["bytes"] for b in new_blobs),
                          blobs=new_blobs,
                          descriptor=f"{into['path']}/{desc_name}")
    summary = {
        "into": into_id, "from": from_id, "machine": into["machine"], "dir": into_dir,
        "shard_count": n,
        "renames": [{"from": os.path.join(d, old), "to": os.path.join(into_dir, nb)}
                    for d, old, _, nb, _ in moves]
                   + [{"from": os.path.join(d, oi), "to": os.path.join(into_dir, ni)}
                      for d, _, oi, _, ni in moves],
        "removed": [os.path.join(from_dir, os.path.basename(frm["manifest"]))]
                   + ([os.path.join(from_dir, os.path.basename(frm["descriptor"]))] if frm.get("descriptor") else []),
        "manifest": {"path": os.path.join(into_dir, man_name), "layer_ranges": new_man["layer_ranges"],
                     "shard_count": n, "content_hash": new_man["content_hash"]["value"],
                     "total_group_count": new_man["total_group_count"],
                     "total_blob_bytes": new_man["total_blob_bytes"]},
        "bundle": {"layers": new_bundle_set["layers"], "bytes": new_bundle_set["bytes"]},
    }
    if dry_run:
        return summary

    # descriptor: regenerated locally from the merged manifest and index files
    man_text = json.dumps(new_man, indent=2) + "\n"
    tmp = Path(tempfile.mkdtemp(prefix="wp-forge-merge-"))
    try:
        (tmp / man_name).write_text(man_text)
        for ni, text in index_out:
            (tmp / ni).write_text(text)
        desc_text = Path(Tools.discover().descriptor(spine_local, tmp / man_name)).read_text()
    except (ToolError, FileNotFoundError) as e:
        raise PlanError(f"merge-sets: descriptor tool failed: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # writes start here
    emit({"kind": "merge", "step": "start", "into": into_id, "from": from_id,
          "machine": into["machine"], "shards": n, "base": base})
    sink.put_text(into_man_raw, rel(into_dir, man_name + ".bak-merge"))
    shutil.copy2(bundle_path, bak_bundle)
    emit({"kind": "merge", "step": "backup", "manifest": man_name + ".bak-merge",
          "bundle": bak_bundle.name})

    for d, old, _, nb, _ in moves:
        sink.rename(rel(d, old), rel(into_dir, nb))
    emit({"kind": "merge", "step": "rename_blobs", "n": n})

    for ni, text in index_out:
        sink.put_text(text, rel(into_dir, ni))
    for d, _, oi, _, _ in moves:
        sink.remove(rel(d, oi))
    emit({"kind": "merge", "step": "indexes", "n": n})

    sink.remove(rel(into_dir, man_name))
    sink.put_text(man_text, rel(into_dir, man_name))
    sink.remove(rel(into_dir, desc_name))
    sink.put_text(desc_text, rel(into_dir, desc_name))
    emit({"kind": "merge", "step": "manifest", "manifest": man_name, "descriptor": desc_name})

    sink.remove(rel(from_dir, os.path.basename(frm["manifest"])))
    if frm.get("descriptor"):
        sink.remove(rel(from_dir, os.path.basename(frm["descriptor"])))
    if from_dir != into_dir:
        if not sink.rmdir(os.path.relpath(from_dir, root)):
            emit({"kind": "merge", "step": "keep_dir", "path": from_dir,
                  "why": "not empty, or sink cannot remove dirs"})
    emit({"kind": "merge", "step": "cleanup", "removed_set": from_id})

    new_bundle = dict(bundle)
    new_bundle["expert_sets"] = [new_bundle_set if s["id"] == into_id else s
                                 for s in bundle["expert_sets"] if s["id"] != from_id]
    new_bundle["dispatch_order"] = [x for x in bundle["dispatch_order"] if x != from_id]
    bundle_path.write_text(json.dumps(new_bundle, indent=2))
    emit({"kind": "merge", "step": "bundle", "path": str(bundle_path), "backup": bak_bundle.name,
          "layers": new_bundle_set["layers"]})
    return summary
