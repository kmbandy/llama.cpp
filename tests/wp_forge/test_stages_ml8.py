"""ExpertStage ml8 (rotated ML8_FP8/ML8_4) checks: _gather + _write_gguf
directly, no C++ tools needed (unlike test_stages.py's end-to-end tests,
which require llama-wp-repack). Reads the one-layer GGUF back with
gguf.GGUFReader to check companion-tensor names/shapes/types.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest

from conversion.wp_forge.arch import ARCHS
from conversion.wp_forge.plan import LayerRange, Ml8Settings, SetSpec
from conversion.wp_forge.sink import LocalSink
from conversion.wp_forge.source import HFSource
from conversion.wp_forge.stages import ExpertStage
from synth import make_synthetic_hf_repo

# ml8_4's QK_ML8=64 needs n_embd (gate/up K) and n_ff (down K) both %64==0;
# n_expert kept small (3) so the per-expert-LUT fused-path check stays cheap.
N_LAYER, N_EXPERT, N_FF, N_EMBD = 2, 3, 128, 64


@pytest.fixture
def env(tmp_path: Path):
    hub = make_synthetic_hf_repo(tmp_path / "hub", n_layer=N_LAYER, n_expert=N_EXPERT,
                                  n_ff=N_FF, n_embd=N_EMBD, nextn=1, engram=False)

    def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    source = HFSource("fake/repo", tmp_path / "cache", fetch=fetch)
    rplan = SimpleNamespace(
        arch=ARCHS["deepseek41"],
        hparams=source.hparams(),
        spine_path=str(tmp_path / "spine.gguf"),
        sidecar_paths={},
        plan=SimpleNamespace(source="fake/repo", name="synthetic", quant={}),
    )
    return SimpleNamespace(source=source, rplan=rplan, tmp_path=tmp_path)


def _stage(env, quant: str, ml8: Ml8Settings, workdir: Path) -> ExpertStage:
    set_spec = SetSpec(
        id="L0-1", role="experts", layers=LayerRange(0, N_LAYER - 1), slice_index=None,
        widths=None, machine="box-a", dir="/models/L0-1", output_base="L0-1",
        quant=quant, est_bytes=0, ml8=ml8,
    )
    sink = LocalSink(str(env.tmp_path / "sink"))
    return ExpertStage([set_spec], env.rplan, env.source, None, {"L0-1": sink}, list().append, workdir)


def _read(path: Path) -> dict[str, "gguf.ReaderTensor"]:
    r = gguf.GGUFReader(str(path))
    return {t.name: t for t in r.tensors}


def test_ml8_4_write_gguf_companion_tensors(env) -> None:
    settings = Ml8Settings(rotation="kronecker", rotation_seed=3, max_b=1024, fit_rows=N_EXPERT * 4)
    stage = _stage(env, "ml8_4", settings, env.tmp_path / "work4")
    stacks, extra = stage._gather(0)
    out = env.tmp_path / "L0.gguf"
    stage._write_gguf(0, stacks, out, extra)
    tensors = _read(out)

    for role, k in (("gate", N_EMBD), ("up", N_EMBD), ("down", N_FF)):
        base = f"blk.0.ffn_{role}_exps"
        w = tensors[f"{base}.weight"]
        assert w.tensor_type == gguf.GGMLQuantizationType.ML8_4
        assert w.shape[-1] == N_EXPERT  # numpy [n_expert, rows, bytes] -> ggml ne [..., n_expert]

        cent = tensors[f"{base}.centroids"]
        assert cent.tensor_type == gguf.GGMLQuantizationType.F8_E4M3
        # numpy shape (n_expert, K/64, 16) -> ggml ne [16, K/64, n_expert]
        assert tuple(cent.shape) == (16, k // 64, N_EXPERT)

        meta = tensors[f"{base}.rotation_meta"]
        assert meta.tensor_type == gguf.GGMLQuantizationType.I32
        a, b, kk, kind = [int(x) for x in meta.data]
        assert kk == k and kind == 1 and a * b == k

        h_a = tensors[f"{base}.rotation_h_a"]
        assert h_a.tensor_type == gguf.GGMLQuantizationType.F32
        assert tuple(h_a.shape) == (a, a)

    # gate and up share the identical rotation (h_a bytes equal); down differs.
    assert np.array_equal(tensors["blk.0.ffn_gate_exps.rotation_h_a"].data,
                           tensors["blk.0.ffn_up_exps.rotation_h_a"].data)


def test_ml8_fp8_no_centroids_sidecar(env) -> None:
    settings = Ml8Settings(rotation="kronecker", rotation_seed=0, max_b=1024, fit_rows=1024)
    stage = _stage(env, "ml8_fp8", settings, env.tmp_path / "work_fp8")
    stacks, extra = stage._gather(0)
    out = env.tmp_path / "Lfp8.gguf"
    stage._write_gguf(0, stacks, out, extra)
    tensors = _read(out)
    for role in ("gate", "up", "down"):
        base = f"blk.0.ffn_{role}_exps"
        assert tensors[f"{base}.weight"].tensor_type == gguf.GGMLQuantizationType.ML8_FP8
        assert f"{base}.centroids" not in tensors  # ML8_FP8 has no centroid LUT
        assert f"{base}.rotation_meta" in tensors
        assert f"{base}.rotation_h_a" in tensors


def test_ml8_rotation_none_no_sidecars(env) -> None:
    settings = Ml8Settings(rotation="none", rotation_seed=0, max_b=1024, fit_rows=1024)
    stage = _stage(env, "ml8_fp8", settings, env.tmp_path / "work_none")
    stacks, extra = stage._gather(0)
    out = env.tmp_path / "Lnone.gguf"
    stage._write_gguf(0, stacks, out, extra)
    tensors = _read(out)
    for role in ("gate", "up", "down"):
        base = f"blk.0.ffn_{role}_exps"
        assert f"{base}.rotation_meta" not in tensors
        assert f"{base}.rotation_h_a" not in tensors


def test_ml8_4_fused_style_per_expert_lut_distinctness(env) -> None:
    """Not a fused-arch synthetic repo (none exists in synth.py yet), but the
    per-expert LUT contract is exercised end-to-end through ExpertStage: this
    synthetic repo's experts are drawn i.i.d. from the same distribution, so
    as a cross-check the LUTs should still legitimately differ between
    experts (different sample draws) -- proving _gather's per-eid ml8_4 path
    calls quantize_role_ml8_4 (fits its own LUT) rather than reusing one fit.
    The stronger "very different scales -> very different LUTs" property is
    covered directly at the math layer in test_ml8.py's
    test_quantize_experts_ml8_4_fits_per_expert_lut (which also proves the
    *fused* np.stack path used by ExpertStage._gather_fused matches this
    per-expert one element for element)."""
    settings = Ml8Settings(rotation="none", rotation_seed=0, max_b=1024, fit_rows=N_EXPERT * 4)
    stage = _stage(env, "ml8_4", settings, env.tmp_path / "work_lut")
    stacks, extra = stage._gather(0)
    cent = extra["down"]["centroids"]  # [n_expert, K/64, 16] uint8
    assert cent.shape[0] == N_EXPERT
    # at least one pair of experts must have a different LUT
    assert not all(np.array_equal(cent[0], cent[e]) for e in range(1, N_EXPERT))
