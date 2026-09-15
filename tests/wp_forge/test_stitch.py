"""Stitcher tests: real llama-wp-repack on tiny synthetic one-layer GGUFs."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT))

import gguf  # noqa: E402
from gguf import quants  # noqa: E402

from conversion.wp_forge.plan import LayerRange, SetSpec  # noqa: E402
from conversion.wp_forge.stitch import (  # noqa: E402
    LayerOutput,
    layer_outputs_from_repack,
    stitch,
)
from conversion.wp_forge.tools import Tools  # noqa: E402

# Tiny model: 2 layers, 4 experts, FFN 64, embd 32 -> each per-layer repack
# is ~26 KiB and finishes well under a second.
N_LAYERS = 2
N_EXPERT = 4
N_FF = 64
N_EMBD = 32
WIDTHS = [32, 32]  # two slices of the FFN intermediate, 32 (Q8_0 block) aligned

TOOL_ERROR = ""
try:
    _TOOLS = Tools.discover()
except Exception as e:  # FileNotFoundError from discovery, or missing build
    TOOL_ERROR = str(e)
    _TOOLS = None


@pytest.fixture(scope="module")
def tools():
    if _TOOLS is None:
        pytest.skip(f"llama-wp-repack unavailable: {TOOL_ERROR}")
    return _TOOLS


def _packed_q8(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    f = rng.standard_normal(shape, dtype=np.float32) * 0.05
    return quants.quantize(f, gguf.GGMLQuantizationType.Q8_0)


def _write_layer_gguf(path: Path, layer: int, rng: np.random.Generator) -> None:
    """Mirror dsv41_experts_from_hf.convert_layer_experts KV set + dims:
    arch deepseek41, block_count, expert_count, expert_feed_forward_length,
    embedding_length; gate/up [n_expert, n_ff, n_embd], down
    [n_expert, n_embd, n_ff] as Q8_0 (quantized along the contiguous axis)."""
    w = gguf.GGUFWriter(str(path), arch="deepseek41", use_temp_file=True)
    w.add_block_count(N_LAYERS)
    w.add_expert_count(N_EXPERT)
    w.add_expert_feed_forward_length(N_FF)
    w.add_embedding_length(N_EMBD)
    q8 = gguf.GGMLQuantizationType.Q8_0
    w.add_tensor(f"blk.{layer}.ffn_gate_exps.weight", _packed_q8(rng, (N_EXPERT, N_FF, N_EMBD)), raw_dtype=q8)
    w.add_tensor(f"blk.{layer}.ffn_up_exps.weight", _packed_q8(rng, (N_EXPERT, N_FF, N_EMBD)), raw_dtype=q8)
    w.add_tensor(f"blk.{layer}.ffn_down_exps.weight", _packed_q8(rng, (N_EXPERT, N_EMBD, N_FF)), raw_dtype=q8)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _set_spec(**kw) -> SetSpec:
    base = dict(
        id="L0-1",
        role="experts",
        layers=LayerRange(0, N_LAYERS - 1),
        slice_index=None,
        widths=None,
        machine="local",
        dir="/tmp/wp-forge-test/L0-1",
        output_base="test-set",
        quant="Q8_0",
        est_bytes=0,
    )
    base.update(kw)
    return SetSpec(**base)


@pytest.fixture(scope="module")
def repack_run(tools, tmp_path_factory):
    """Build the two one-layer GGUFs and repack each, unsliced and sliced."""
    d = tmp_path_factory.mktemp("stitch")
    rng = np.random.default_rng(1234)
    for layer in range(N_LAYERS):
        g = d / f"layer{layer}.gguf"
        _write_layer_gguf(g, layer, rng)
        # unsliced:  --layer-ranges L-L --allow-partial
        subprocess.run(
            [str(tools.repack_bin), str(g), str(d / f"raw-unsliced-L{layer}"),
             "--layer-ranges", f"{layer}-{layer}", "--allow-partial"],
            check=True, capture_output=True, text=True,
        )
        # sliced: same call + --expert-slices 32,32 --slice-output-split
        subprocess.run(
            [str(tools.repack_bin), str(g), str(d / f"raw-sliced-L{layer}"),
             "--layer-ranges", f"{layer}-{layer}", "--allow-partial",
             "--expert-slices", ",".join(str(w) for w in WIDTHS),
             "--slice-output-split"],
            check=True, capture_output=True, text=True,
        )
    return d


_UNSLICED_CACHE: dict = {}


def _stitch_unsliced(repack_run: Path) -> dict:
    """Stitch both layers of the unsliced repack run; cached across tests."""
    if "manifest" not in _UNSLICED_CACHE:
        spine = str(repack_run / "spine.gguf")
        per_layer = [
            layer_outputs_from_repack(repack_run / f"raw-unsliced-L{L}", L, None)
            for L in range(N_LAYERS)
        ]
        _UNSLICED_CACHE["manifest"] = stitch(
            per_layer, _set_spec(), spine,
            input_model=str(repack_run / "layer0.gguf"), expert_type="Q8_0", n_expert=N_EXPERT,
        )
    return _UNSLICED_CACHE["manifest"]


def test_stitch_unsliced(tools, repack_run):
    spine = str(repack_run / "spine.gguf")
    per_layer = [
        layer_outputs_from_repack(repack_run / f"raw-unsliced-L{L}", L, None)
        for L in range(N_LAYERS)
    ]
    out = _stitch_unsliced(repack_run)
    m = out.manifest
    assert m["format"] == "llama.cpp.weight-pager.expert-shard-manifest"
    assert m["sharding_mode"] == "layer-ranges"
    assert m["layer_ranges"] == ["0-1"]
    assert m["allow_partial"] is True
    assert m["model_files"] == [spine]
    assert m["expert_ggml_type"] == "Q8_0"
    assert m["retained_expert_range"] == {"first": 0, "last": N_EXPERT - 1}
    assert "expert_slicing" not in m
    assert m["shard_count"] == 2
    assert len(out.blobs) == 2

    total = 0
    for i, (lo, plan) in enumerate(zip(per_layer, out.blobs)):
        idx = json.loads(lo.index_json.read_text())
        assert plan.src_blob == lo.blob
        assert plan.dst_name == f"test-set-experts-{i + 1:05d}-of-00002.wpb"
        assert m["shards"][i]["blob_file"] == plan.dst_name
        assert m["shards"][i]["index_file"] == plan.dst_name[: -len(".wpb")] + ".wpi.json"
        assert m["shards"][i]["content_hash"] == idx["content_hash"]
        assert m["shards"][i]["layer_first"] == m["shards"][i]["layer_last"] == i
        total += idx["blob_bytes"]
        # rewritten index parses and carries the stitched fields
        rw = json.loads(plan.index_text)
        assert rw["blob_file"] == plan.dst_name
        assert rw["shard_index"] == i
        assert rw["shard_count"] == 2
        assert rw["model_files"] == [spine]

    assert m["total_blob_bytes"] == total
    assert m["total_group_count"] == sum(s["group_count"] for s in m["shards"])
    assert m["content_hash"]["algorithm"] == "sha256-of-shards"


def test_stitch_sliced(tools, repack_run):
    spine = str(repack_run / "spine.gguf")
    unsliced_total = _stitch_unsliced(repack_run).manifest["total_blob_bytes"]
    slice_totals = []
    for slice_index in (0, 1):
        spec = _set_spec(
            id=f"L0-1-w{slice_index}",
            slice_index=slice_index,
            widths=list(WIDTHS),
            dir=f"/tmp/wp-forge-test/L0-1-w{slice_index}",
            output_base=f"test-set-w{slice_index}",
        )
        per_layer = [
            layer_outputs_from_repack(repack_run / f"raw-sliced-L{L}", L, slice_index)
            for L in range(N_LAYERS)
        ]
        out = stitch(
            per_layer, spec, spine,
            input_model=str(repack_run / "layer0.gguf"), expert_type="Q8_0", n_expert=N_EXPERT,
        )
        m = out.manifest
        assert m["sharding_mode"] == "expert-slice"
        assert m["layer_ranges"] == ["0-1"]
        assert m["allow_partial"] is True
        assert m["expert_ggml_type"] == "Q8_0"
        assert m["model_files"] == [spine]
        es = m["expert_slicing"]
        assert es["selected_slice"] == slice_index
        assert es["widths"] == WIDTHS
        assert es["n_ff_exp"] == N_FF
        assert es["n_embd"] == N_EMBD
        assert es["slice_count"] == 2
        assert m["shard_count"] == 2
        for i, plan in enumerate(out.blobs):
            assert plan.dst_name == f"test-set-w{slice_index}-experts-{i + 1:05d}-of-00002.wpb"
            rw = json.loads(plan.index_text)
            assert rw["shard_index"] == i
            assert rw["shard_count"] == 2
        slice_totals.append(m["total_blob_bytes"])
        assert m["total_group_count"] == N_LAYERS * N_EXPERT
    # sliced blobs are page-padded (expert_slicing.page_alignment, zero
    # padding), so the slices carry at least the unsliced bytes, never fewer
    assert sum(slice_totals) >= unsliced_total


def _fake_slice_outputs(repack_run: Path, d: Path, disagree: str | None) -> list[LayerOutput]:
    """Copy each layer's real per-slice repack output (a BASE prefix, not a
    dir) into its own subdir of d; optionally corrupt the geometry in
    layer-1's slice manifest so the stitcher must raise."""
    out = []
    for layer in range(N_LAYERS):
        src_base = f"raw-sliced-L{layer}"
        sub = d / f"L{layer}"
        sub.mkdir(parents=True, exist_ok=True)
        # keep file names: the manifest references its index/blob by name
        for f in repack_run.glob(src_base + "*"):
            shutil.copy(f, sub / f.name)
        dst_base = sub / src_base
        if disagree and layer == 1:
            man_path = sub / f"{src_base}-eslice-slice-00000-experts-manifest.json"
            man = json.loads(man_path.read_text())
            man["expert_slicing"][disagree] = 999999
            man_path.write_text(json.dumps(man))
        out.append(layer_outputs_from_repack(dst_base, layer, 0))
    return out


def test_stitch_errors(tools, repack_run, tmp_path):
    spine = str(repack_run / "spine.gguf")
    spec = _set_spec()
    per_layer = [
        layer_outputs_from_repack(repack_run / f"raw-unsliced-L{L}", L, None)
        for L in range(N_LAYERS)
    ]
    common = dict(input_model=str(repack_run / "layer0.gguf"), expert_type="Q8_0", n_expert=N_EXPERT)

    # non-contiguous: [0, 1] within a 0-2 set
    gap = [per_layer[0], LayerOutput(2, per_layer[1].index_json, per_layer[1].blob)]
    with pytest.raises(ValueError, match="contiguous"):
        stitch(gap, _set_spec(id="L0-2", layers=LayerRange(0, 2)), spine, **common)
    # descending
    with pytest.raises(ValueError, match="ascending"):
        stitch(list(reversed(per_layer)), spec, spine, **common)
    # a layer outside the set
    off = [LayerOutput(5 + i, lo.index_json, lo.blob) for i, lo in enumerate(per_layer)]
    with pytest.raises(ValueError, match="do not fit within"):
        stitch(off, spec, spine, **common)

    # width-sliced set: geometry disagreement between the two layers
    wspec = _set_spec(
        id="L0-1-w0", slice_index=0, widths=list(WIDTHS),
        dir="/tmp/wp-forge-test/L0-1-w0", output_base="test-set-w0",
    )
    for field in ("widths", "n_ff_exp", "n_embd"):
        fake = _fake_slice_outputs(repack_run, tmp_path, disagree=field)
        with pytest.raises(ValueError, match="disagrees"):
            stitch(fake, wspec, spine, **common)
    # selected slice mismatch: feed slice-1 outputs to the slice-0 set
    other = [
        layer_outputs_from_repack(repack_run / f"raw-sliced-L{L}", L, 1)
        for L in range(N_LAYERS)
    ]
    with pytest.raises(ValueError, match="wants slice 0"):
        stitch(other, wspec, spine, **common)
