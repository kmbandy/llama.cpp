"""In-place engram GGUF (dsv41_engram_from_hf.build_engram_gguf) and SidecarStage's deepseek41 engram path.

A tiny synthetic engram checkpoint (2 layers, one safetensors shard per layer, FP8 + E8M0 scales,
BF16 k/q, FP8 wkv with 32x32 block scales) is read through the local-shard branch, so nothing
touches the network. The reference is the old path: convert_table into staging bins, then
write_gguf. The in-place output must be byte-identical to it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from gguf.quants import dequantize

from conversion.dsv41_engram_from_hf import (
    build_engram_gguf,
    convert_table,
    engram_extras,
    local_shard_tensors,
    make_ranged,
    make_session,
    write_gguf,
)
from conversion.wp_forge.arch import ARCHS
from conversion.wp_forge.quant import QuantError
from conversion.wp_forge.sink import LocalSink
from conversion.wp_forge.stages import SidecarStage

LAYERS = [1, 14]
ROWS, COLS = 1000, 256        # embed: FP8 rows x 256, 8 E8M0 scales per row (one per 32 columns)
KQ_SHAPE = (16, 64)           # k/q BF16
WKV_SHAPE = (64, 96)          # wkv FP8, E8M0 scale per 32x32 block -> (2, 3)
HPARAMS = {
    "engram_n_heads": 4,
    "engram_head_dim": COLS,
    "engram_max_ngram_size": 3,
    "engram_layer_ids": LAYERS,
    "engram_vocab_size": ROWS,
    "engram_pad_token_id": 0,
}


def _fp8(rng: np.random.Generator, shape: tuple[int, ...]) -> bytes:
    x = torch.from_numpy(rng.standard_normal(shape, dtype=np.float32) * 2.0)
    return x.to(torch.float8_e4m3fn).contiguous().view(torch.uint8).numpy().tobytes()


def _bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> bytes:
    x = torch.from_numpy(rng.standard_normal(shape, dtype=np.float32))
    return x.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


def _e8m0(rng: np.random.Generator, shape: tuple[int, ...]) -> bytes:
    return rng.integers(120, 130, size=shape, dtype=np.uint8).tobytes()


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header: dict = {}
    body = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(body), len(body) + len(raw)]}
        body += raw
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hj += b" " * (-len(hj) % 8)
    path.write_bytes(len(hj).to_bytes(8, "little") + hj + bytes(body))


def _layer_tensors(rng: np.random.Generator, bid: int) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    p = f"layers.{bid}.engram."
    return {
        p + "embed.weight": ("F8_E4M3", (ROWS, COLS), _fp8(rng, (ROWS, COLS))),
        p + "embed.scale": ("F8_E8M0", (ROWS, COLS // 32), _e8m0(rng, (ROWS, COLS // 32))),
        p + "k_weight": ("BF16", KQ_SHAPE, _bf16(rng, KQ_SHAPE)),
        p + "q_weight": ("BF16", KQ_SHAPE, _bf16(rng, KQ_SHAPE)),
        p + "wkv.weight": ("F8_E4M3", WKV_SHAPE, _fp8(rng, WKV_SHAPE)),
        p + "wkv.scale": ("F8_E8M0", (2, 3), _e8m0(rng, (2, 3))),
    }


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """Two shard files (one per layer, like the real checkpoint), plus the staged reference GGUF."""
    root = tmp_path_factory.mktemp("engram-inplace")
    rng = np.random.default_rng(1234)
    shards: dict[int, Path] = {}
    for bid in LAYERS:
        shard = root / f"model-{bid:05d}.safetensors"
        _write_safetensors(shard, _layer_tensors(rng, bid))
        shards[bid] = shard
    remote: dict = {}
    for shard in shards.values():
        remote.update(local_shard_tensors(shard))

    # OLD path: staging bins + write_gguf
    ref = root / "reference-staged.gguf"
    tables: dict[int, Path] = {}
    extras: dict[int, dict] = {}
    for bid in LAYERS:
        url = str(shards[bid])
        ranged = make_ranged(None, url, shards[bid])
        n_rows = ROWS
        row_bytes = int(gguf.quantize(np.zeros((1, COLS), dtype=np.float32), gguf.GGMLQuantizationType.MXFP4).nbytes)
        staging = root / f"engram_{bid}_MXFP4.bin"
        staging.write_bytes(b"\0" * (n_rows * row_bytes))
        convert_table(ranged, remote, bid, staging, 0, workers=1)
        tables[bid] = staging
        extras[bid] = engram_extras(ranged, remote, bid)
    write_gguf(ref, HPARAMS, tables, extras)

    # NEW path: in place, through the local-shard branch
    inplace = root / "inplace" / "engram.gguf"
    inplace.parent.mkdir()
    ranges = {bid: make_ranged(None, str(shards[bid]), shards[bid]) for bid in LAYERS}
    done = build_engram_gguf(inplace, HPARAMS, remote, ranges, workers=1)
    assert done is True
    return SimpleNamespace(root=root, shards=shards, remote=remote, ref=ref, inplace=inplace)


def test_inplace_is_byte_identical_to_staged_reference(synth):
    assert synth.inplace.read_bytes() == synth.ref.read_bytes()


def test_resume_after_partial_layers_is_identical(synth, tmp_path):
    remote, shards = synth.remote, synth.shards
    ranges = {bid: make_ranged(None, str(shards[bid]), shards[bid]) for bid in LAYERS}
    out = tmp_path / "resumed.gguf"
    assert build_engram_gguf(out, HPARAMS, remote, ranges, workers=1, only={LAYERS[0]}) is False
    prog = json.loads((tmp_path / "resumed.gguf.progress.json").read_text())
    assert prog["done"] is False and prog["layers"][str(LAYERS[0])]["done"] is True
    assert build_engram_gguf(out, HPARAMS, remote, ranges, workers=1) is True
    assert out.read_bytes() == synth.ref.read_bytes()


def test_inplace_progress_marks_done_and_no_staging_left(synth):
    prog = json.loads((synth.inplace.parent / "engram.gguf.progress.json").read_text())
    assert prog["done"] is True
    assert sorted(prog["layers"]) == [str(b) for b in LAYERS]
    assert not list(synth.inplace.parent.glob("*.bin"))


def test_mxfp4_rows_decode_close_to_fp8(synth):
    """MXFP4 is lossy; the decode must stay close to the dequantized FP8 values it was built from."""
    reader = gguf.GGUFReader(str(synth.inplace))
    by_name = {t.name: t for t in reader.tensors}
    for bid in LAYERS:
        p = f"layers.{bid}.engram."
        w = np.frombuffer(
            synth.shards[bid].read_bytes()[synth.remote[p + "embed.weight"].offset_start:][:ROWS * COLS],
            dtype=np.uint8,
        ).reshape(ROWS, COLS)
        fp8 = torch.from_numpy(w.copy()).view(torch.float8_e4m3fn).float().numpy()
        s_raw = np.frombuffer(
            synth.shards[bid].read_bytes()[synth.remote[p + "embed.scale"].offset_start:][:ROWS * 8],
            dtype=np.uint8,
        ).reshape(ROWS, 8)
        scale = np.exp2(s_raw.astype(np.float32) - 127.0).repeat(32, axis=1)
        ref = fp8 * scale

        t = by_name[f"blk.{bid}.engram_embd.weight"]
        assert t.tensor_type is gguf.GGMLQuantizationType.MXFP4
        dec = dequantize(np.ascontiguousarray(t.data), gguf.GGMLQuantizationType.MXFP4).reshape(ROWS, COLS)
        rel = np.linalg.norm(dec - ref) / np.linalg.norm(ref)
        assert rel < 0.2, rel


def test_extras_are_f32_at_their_offsets(synth):
    reader = gguf.GGUFReader(str(synth.inplace))
    by_name = {t.name: t for t in reader.tensors}
    for bid in LAYERS:
        ex = engram_extras(make_ranged(None, str(synth.shards[bid]), synth.shards[bid]), synth.remote, bid)
        for key in ("k", "q", "wkv"):
            t = by_name[gguf.TENSOR_NAMES[{"k": gguf.MODEL_TENSOR.ENGRAM_K, "q": gguf.MODEL_TENSOR.ENGRAM_Q,
                                           "wkv": gguf.MODEL_TENSOR.ENGRAM_WKV}[key]].format(bid=bid) + ".weight"]
            assert t.tensor_type is gguf.GGMLQuantizationType.F32
            np.testing.assert_array_equal(np.asarray(t.data, dtype=np.float32).reshape(ex[key].shape), ex[key])


def _source(tmp_path: Path, synth) -> SimpleNamespace:
    cache = tmp_path / "cache"
    cache.mkdir()
    index: dict[str, str] = {}
    for bid, shard in synth.shards.items():
        shutil.copy(shard, cache / shard.name)
        (cache / (shard.name + ".complete")).write_text("")
        for name in synth.remote:
            if name.startswith(f"layers.{bid}."):
                index[name] = shard.name

    def open_shard(_):
        raise AssertionError("in-place engram must not open shards through the source")

    return SimpleNamespace(
        is_gguf=False, repo="fake/repo", revision="abc123", cache_dir=cache,
        tensor_index=lambda: index, hparams=lambda: HPARAMS, open_shard=open_shard,
        release_shard=lambda _s: None,
    )


def _rplan(quant: str) -> SimpleNamespace:
    return SimpleNamespace(
        arch=ARCHS["deepseek41"], hparams=HPARAMS,
        sidecar_paths={"engram": "/models/fake/synthetic-engram.gguf"},
        plan=SimpleNamespace(name="synthetic", quant={"engram": quant}),
    )


def test_sidecar_stage_engram_matches_reference(tmp_path, synth):
    source = _source(tmp_path, synth)
    sink = LocalSink(str(tmp_path / "sidecars"))
    sink.mkdir()
    events: list[dict] = []
    stage = SidecarStage("engram", _rplan("mxfp4"), source, None, sink, events.append, tmp_path / "work")
    size, sha = stage.run()

    ref = synth.ref.read_bytes()
    out = (tmp_path / "sidecars" / "synthetic-engram.gguf").read_bytes()
    assert out == ref
    assert size == len(ref)
    assert sha == hashlib.sha256(ref).hexdigest()
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "stage_start" and kinds[-1] == "stage_done"
    layers = [e for e in events if e["kind"] == "layer_done"]
    assert [e["layer"] for e in layers] == LAYERS and all("secs" in e for e in layers)
    done = events[-1]
    assert done["bytes"] == size and done["sha"] == sha

    # second run: final file and complete progress marker -> resumed, no rebuild
    events.clear()
    again = SidecarStage("engram", _rplan("mxfp4"), source, None, sink, events.append, tmp_path / "work").run()
    assert again == (size, sha)
    assert events == [{"kind": "stage_done", "stage": "engram", "resumed": True}]


def test_sidecar_stage_engram_non_local_sink_puts_and_cleans_workdir(tmp_path, synth):
    source = _source(tmp_path, synth)
    sink = LocalSink(str(tmp_path / "remote-like"))
    sink.mkdir()

    class PutOnlySink:
        """Not a LocalSink, so the stage must build in the workdir and put."""
        def __init__(self, inner):
            self.inner = inner
            self.root = inner.root

        def __getattr__(self, name):
            return getattr(self.inner, name)

    wrapped = PutOnlySink(sink)
    work = tmp_path / "work2"
    size, sha = SidecarStage("engram", _rplan("mxfp4"), source, None, wrapped, lambda e: None, work).run()
    ref = synth.ref.read_bytes()
    assert (tmp_path / "remote-like" / "synthetic-engram.gguf").read_bytes() == ref
    assert size == len(ref) and sha == hashlib.sha256(ref).hexdigest()
    assert not (work / "engram.gguf").exists()
    assert not (work / "synthetic-engram.gguf").exists()


@pytest.mark.parametrize("bad", ["q8_0", "q4_0", "ml8_4"])
def test_sidecar_stage_engram_rejects_non_mxfp4(tmp_path, synth, bad):
    source = _source(tmp_path, synth)
    sink = LocalSink(str(tmp_path / "out"))
    sink.mkdir()
    stage = SidecarStage("engram", _rplan(bad), source, None, sink, lambda e: None, tmp_path / "w")
    with pytest.raises(QuantError, match=f"quant must be 'mxfp4', got '{bad}'"):
        stage.run()
    assert not (tmp_path / "out" / "synthetic-engram.gguf").exists()
