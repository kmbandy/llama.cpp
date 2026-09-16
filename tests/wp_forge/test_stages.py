"""End-to-end tests for the wp-forge ExpertStage driver.

Drives a real ``ExpertStage`` (quant -> one-layer GGUF -> llama-wp-repack ->
sink stream -> stitch -> llama-wp-expert-descriptor) over the synthetic
DeepSeek-V4.1-shaped HF repo from ``synth``. The C++ tools must be built
(``build-cpu/bin``); tests skip if ``Tools.discover()`` cannot find them.
"""
from __future__ import annotations

import json
import shutil
from os import environ
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from conversion.wp_forge.arch import ARCHS
from conversion.wp_forge.plan import LayerRange, SetSpec
from conversion.wp_forge.sink import LocalSink
from conversion.wp_forge.source import HFSource
from conversion.wp_forge.stages import (
    ConverterSpineBuilder,
    ExpertStage,
    SidecarStage,
    SpineStage,
)
from conversion.wp_forge.tools import Tools
from synth import SyntheticSpineBuilder, make_synthetic_hf_repo

# Expected on-disk size of one layer of experts at the test's geometry:
# n_ff=64, n_embd=32, 4 experts, q8_0 -> gate/up 64x34, down 32x68 per expert,
# 3*2176*4 = 26112 bytes.
BLOB_1LYR = 26112


def _write_spine(path: Path) -> None:
    w = gguf.GGUFWriter(str(path), arch="deepseek41")
    w.add_block_count(3)  # n_layer(2) + nextn(1)
    w.add_embedding_length(32)
    w.add_expert_count(4)
    w.add_expert_feed_forward_length(64)
    w.add_expert_used_count(2)
    w.add_name("wp-forge-test-spine")  # descriptor reads general.name
    w.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture
def env(tmp_path: Path):
    try:
        tools = Tools.discover()
    except FileNotFoundError:
        pytest.skip("wp-forge C++ tools not built (Tools.discover raised)")

    hub = make_synthetic_hf_repo(tmp_path / "hub")

    def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    source = HFSource("fake/repo", tmp_path / "cache", fetch=fetch)
    spine = tmp_path / "spine.gguf"
    _write_spine(spine)
    rplan = SimpleNamespace(
        arch=ARCHS["deepseek41"],
        hparams=source.hparams(),
        spine_path=str(spine),
        sidecar_paths={"engram": f"/models/fake-repo/synthetic-engram.gguf"},
        plan=SimpleNamespace(
            source="fake/repo",
            name="synthetic",
            quant={"dense": "q8_0", "spec_head.dense": "q8_0", "engram": "q8_0"},
        ),
    )
    return SimpleNamespace(source=source, rplan=rplan, tools=tools, tmp_path=tmp_path)


def _set(id: str, *, role="experts", first=0, last=1, slice_index=None,
         widths=None, output_base=None, quant="q8_0") -> SetSpec:
    return SetSpec(
        id=id,
        role=role,
        layers=LayerRange(first, last),
        slice_index=slice_index,
        widths=widths,
        machine="box-a",
        dir=f"/models/{id}",
        output_base=output_base or id,
        quant=quant,
        est_bytes=0,
    )


def _stage(sets, env, sinks, workdir):
    events: list[dict] = []
    stage = ExpertStage(
        sets, env.rplan, env.source, env.tools, sinks, events.append, workdir
    )
    return stage, events


def test_unsliced_layer_band(env):
    set_spec = _set("L0-1")
    sink = LocalSink(str(env.tmp_path / "L0-1"))
    sink.mkdir()
    stage, events = _stage([set_spec], env, {"L0-1": sink}, env.tmp_path / "work")
    res = stage.run()
    r = res["L0-1"]

    assert len(r.blobs) == 2
    for dst, sz, sha in r.blobs:
        assert sink.exists(dst) and sink.size(dst) == sz == BLOB_1LYR
        assert sink.sha256(dst) == sha
        assert sink.exists(dst[: -len(".wpb")] + ".wpi.json")  # 2 indexes

    manifest_rel = "L0-1-experts-manifest.json"
    desc_rel = "L0-1-experts-manifest.expert-descriptor.json"
    assert r.manifest_rel == manifest_rel and r.descriptor_rel == desc_rel
    assert sink.exists(manifest_rel) and sink.exists(desc_rel)

    manifest = json.loads((env.tmp_path / "L0-1" / manifest_rel).read_text())
    desc = json.loads((env.tmp_path / "L0-1" / desc_rel).read_text())
    assert len(desc["layers"]) == 2
    assert desc["sharding_mode"] == "layer-ranges"
    assert desc["hparams"]["n_expert"] == 4 and desc["hparams"]["n_embd"] == 32
    assert manifest["total_blob_bytes"] == sum(sink.size(b[0]) for b in r.blobs)
    assert [e["layer"] for e in events if e.get("kind") == "layer_done"] == [0, 1]


def test_width_sliced(env):
    sets = [_set("L0-1-w0", slice_index=0, widths=[32, 32]),
            _set("L0-1-w1", slice_index=1, widths=[32, 32])]
    sinks = {s.id: LocalSink(str(env.tmp_path / s.id)) for s in sets}
    for s in sets:
        sinks[s.id].mkdir()
    stage, _ = _stage(sets, env, sinks, env.tmp_path / "work-sliced")
    res = stage.run()
    for i, set_spec in enumerate(sets):
        r = res[set_spec.id]
        assert len(r.blobs) == 2
        for dst, sz, sha in r.blobs:
            assert sinks[set_spec.id].exists(dst) and sinks[set_spec.id].size(dst) == sz
        man = json.loads(
            (env.tmp_path / set_spec.id / f"{set_spec.output_base}-experts-manifest.json").read_text()
        )
        assert man["sharding_mode"] == "expert-slice"
        assert man["expert_slicing"]["selected_slice"] == i
        assert man["expert_slicing"]["widths"] == [32, 32]
        # sliced + layer-partial descriptors are produced since d0c18fe5a
        # (descriptor/worker accept the combined manifest shape)
        assert r.descriptor_rel is not None and sinks[set_spec.id].exists(r.descriptor_rel)
        desc = json.loads((env.tmp_path / set_spec.id / r.descriptor_rel).read_text())
        assert desc["sharding_mode"] == "expert-slice" and desc["layer_ranges"] == ["0-1"]
        assert desc["expert_slicing"]["selected_slice"] == i


def test_spec_head(env):
    set_spec = _set("L2-2", role="spec_head", first=2, last=2)
    sink = LocalSink(str(env.tmp_path / "L2-2"))
    sink.mkdir()
    stage, _ = _stage([set_spec], env, {"L2-2": sink}, env.tmp_path / "work-sh")
    res = stage.run()
    r = res["L2-2"]
    assert len(r.blobs) == 1
    dst, sz, sha = r.blobs[0]
    # 4 experts x q8_0 x (gate 64x32 + up 64x32 + down 32x64) = 26112, the same
    # per-layer size as the main-stack layers -- proves the mtp.0.* names were
    # resolved through expert_tensor_names_stage.
    assert sink.exists(dst) and sink.size(dst) == BLOB_1LYR == sz
    assert dst == "L2-2-experts-00001-of-00001.wpb"


def test_resume_skips_resided_layers(env):
    set_spec = _set("L0-1")
    sink = LocalSink(str(env.tmp_path / "L0-1"))
    sink.mkdir()

    calls = {"n": 0}
    real_repack = env.tools.repack

    def counting_repack(*a, **k):
        calls["n"] += 1
        return real_repack(*a, **k)

    # Tools is a frozen dataclass; bypass the guard (restored in a finally).
    object.__setattr__(env.tools, "repack", counting_repack)

    try:
        stage, _ = _stage([set_spec], env, {"L0-1": sink}, env.tmp_path / "work-resume")
        stage.run()
        assert calls["n"] == 2  # fresh: both layers repack

        blobs = sorted(str(p) for p in (env.tmp_path / "L0-1").glob("L0-1-experts-*of-*.wpb"))
        assert len(blobs) == 2
        mtimes = {p: Path(p).stat().st_mtime_ns for p in blobs}
        victim = next(p for p in blobs if "-00001-of-" in p)  # layer 0
        other = next(p for p in blobs if "-00002-of-" in p)   # layer 1
        Path(victim).unlink()

        calls["n"] = 0
        stage2, _ = _stage([set_spec], env, {"L0-1": sink}, env.tmp_path / "work-resume")
        stage2.run()
    finally:
        object.__setattr__(env.tools, "repack", real_repack)
    assert calls["n"] == 1  # only the missing layer's repack ran

    assert mtimes[other] == Path(other).stat().st_mtime_ns  # resided blob untouched
    assert Path(victim).exists() and sink.size(Path(victim).name) == BLOB_1LYR


def test_spine_stage_synthetic(env):
    sink = LocalSink(str(env.tmp_path / "spine-out"))
    sink.mkdir()
    builder = SyntheticSpineBuilder()
    stage = SpineStage(
        env.rplan, env.source, env.tools, sink, lambda e: None,
        env.tmp_path / "work-spine", builder,
    )
    size, sha = stage.run()

    spine_rel = env.rplan.spine_path.rsplit("/", 1)[-1]
    assert sink.exists(spine_rel) and sink.size(spine_rel) == size
    assert sink.sha256(spine_rel) == sha
    assert not (env.tmp_path / "work-spine" / "peel").exists()
    assert not (env.tmp_path / "work-spine" / "spine.gguf").exists()

    r = gguf.GGUFReader(str(env.tmp_path / "spine-out" / spine_rel))
    names = {t.name for t in r.tensors}
    for want in ("blk.0.attn.wq.weight",
                 "blk.0.ffn.shared_experts.w1.weight",
                 "blk.0.attn_norm.weight",
                 "mtp.0.attn.wq.weight",
                 "embed.weight"):
        assert want in names, names
    assert not any("ffn.experts" in n for n in names)
    assert not any("engram" in n for n in names)
    # engram tensors were peeled away, not written
    assert not any(n.endswith("engram.embed.weight") for n in builder.peeled)

    # resume: second run skips everything, builder not called again
    events: list[dict] = []
    builds = {"n": 0}
    real_build = builder.build

    def counting_build(*a, **k):
        builds["n"] += 1
        return real_build(*a, **k)

    builder.build = counting_build
    try:
        stage2 = SpineStage(
            env.rplan, env.source, env.tools, sink, events.append,
            env.tmp_path / "work-spine2", builder,
        )
        size2, sha2 = stage2.run()
    finally:
        builder.build = real_build
    assert builds["n"] == 0
    assert size2 == size and sha2 == sha
    done = [e for e in events if e.get("kind") == "stage_done"]
    assert done and done[0].get("resumed") is True
    assert done[0]["stage"] == "spine"


def test_sidecar_stage_engram(env):
    sink = LocalSink(str(env.tmp_path / "engram-out"))
    sink.mkdir()
    stage = SidecarStage(
        "engram", env.rplan, env.source, env.tools, sink,
        lambda e: None, env.tmp_path / "work-engram",
    )
    size, sha = stage.run()

    sidecar_rel = env.rplan.sidecar_paths["engram"].rsplit("/", 1)[-1]
    assert sink.exists(sidecar_rel) and sink.size(sidecar_rel) == size
    assert sink.sha256(sidecar_rel) == sha
    assert not (env.tmp_path / "work-engram" / "engram.gguf").exists()

    n_layer = env.rplan.arch.n_layer(env.rplan.hparams)
    r = gguf.GGUFReader(str(env.tmp_path / "engram-out" / sidecar_rel))
    assert r.tensors is not None and len(r.tensors) == n_layer
    for L, t in enumerate(r.tensors):
        assert t.name == f"blk.{L}.engram_embd.weight", t.name
        assert t.tensor_type is gguf.GGMLQuantizationType.Q8_0
        assert tuple(int(x) for x in t.shape[::-1]) == (16, 32)  # GGUFReader shape is ggml ne-order (reversed)
    assert r.get_field("general.name").contents() == "synthetic"


def test_converter_builder_env_gated(env):
    if environ.get("WP_FORGE_REAL_CONVERTER") != "1":
        pytest.skip("set WP_FORGE_REAL_CONVERTER=1 to run the stock converter")
    # Peel the synthetic repo exactly like SpineStage does, then hand it to the
    # real converter. If the stock converter demands tokenizer files this
    # fails -- by design we do NOT fake them; the failure message names what
    # it wanted.
    import torch
    from safetensors.torch import save_file

    from conversion.wp_forge.arch import classify

    rplan = env.rplan
    source = env.source
    peel_dir = env.tmp_path / "peel-real"
    peel_dir.mkdir()
    weight_map = {}
    for shard in sorted(set(source.tensor_index().values())):
        tensors = {}
        with source.open_shard(shard) as reader:
            for name in reader.keys():
                if classify(rplan.arch, name) in ("dense", "spec_head.dense"):
                    tensors[name] = torch.from_numpy(reader.get_tensor(name)).to(torch.bfloat16)
                    weight_map[name] = shard
        if tensors:
            save_file(tensors, str(peel_dir / shard))
    (peel_dir / "config.json").write_text((source.cache_dir / "config.json").read_text())
    total = sum(p.stat().st_size for p in peel_dir.glob("*.safetensors"))
    (peel_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}) + "\n"
    )

    builder = ConverterSpineBuilder(env.tools, rplan.arch)
    out = env.tmp_path / "work-real" / "spine.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = builder.build(peel_dir, out, rplan.plan.quant)
    except Exception as e:  # noqa: BLE001 - report exactly what the converter asked for
        pytest.fail(f"stock converter path failed: {type(e).__name__}: {e}")
    assert result.is_file() and result.stat().st_size > 0
    assert not out.with_name(out.name + ".bf16.gguf").exists()
