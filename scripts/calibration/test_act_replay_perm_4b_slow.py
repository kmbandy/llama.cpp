"""Slow CPU integration test: the GGUF->HF V-head reorder inversion recovers the
real Qwen3.5-4B linear-attn weights to within the quant floor.

PROVES the act-replay 4B KL fix end to end: dequant the real ml8 GGUF targets
(attn_qkv, attn_gate) and the frozen fp8 weights (ssm_out, ssm_alpha, ssm_beta),
apply the inverse reorder via act_replay.gguf_to_hf_perm, and assert the result
matches the HF safetensors. Without the reorder these are rel ~1.2-1.4 (orthogonal
/ garbage); with it they hit the 4-bit (rel <= 0.15) and fp8 (rel <= 0.05) floors.

Marked slow and skipped automatically when the model files are absent. Run from
scripts/calibration with PYTHONPATH=../../gguf-py:
    pytest -q test_act_replay_perm_4b_slow.py -m slow
"""
import json
import os

import pytest
import torch

GGUF = "/home/kmbandy/models/mi300x-ggufs/cell_A0_anchor_A3.gguf"
HF_DIR = "/home/kmbandy/models/Qwen3.5-4B-hf"
HF_INDEX = os.path.join(HF_DIR, "model.safetensors.index.json")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.path.exists(GGUF) and os.path.exists(HF_INDEX)),
        reason="real 4B GGUF / HF safetensors not present",
    ),
]


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _hf_getter():
    from safetensors import safe_open

    wmap = json.load(open(HF_INDEX))["weight_map"]
    opened = {}

    def get(name):
        shard = wmap[name]
        f = opened.get(shard)
        if f is None:
            f = opened[shard] = safe_open(os.path.join(HF_DIR, shard), "pt")
        return f.get_tensor(name).float()

    return get


@pytest.fixture(scope="module")
def state():
    from gguf_state import load_ml8_gguf

    return load_ml8_gguf(GGUF, frozen_mode="fp8")


@pytest.fixture(scope="module")
def config():
    return json.load(open(os.path.join(HF_DIR, "config.json")))["text_config"]


def test_attn_qkv_and_gate_recover_under_quant_floor(state, config):
    """ml8 targets: de-rotate (input space, via AttachedTarget rotation) + inverse
    V-reorder (output rows, via gguf_to_hf_perm) -> rel <= 0.15 vs HF."""
    from act_replay import gguf_to_hf_perm, _apply_perm_to_ml8_entry
    from act_replay_student import AttachedTarget
    from kronecker_rotation import KroneckerRotation

    get = _hf_getter()
    cases = {
        "blk.0.attn_qkv.weight": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "blk.0.attn_gate.weight": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    }
    for gname, hfname in cases.items():
        entry = state.ml8[gname]
        shape = tuple(entry["indices"].shape)
        perm = gguf_to_hf_perm(gname, shape, config)
        assert perm is not None, f"{gname}: expected a non-identity 4B reorder"

        permed = _apply_perm_to_ml8_entry(entry, perm)
        at = AttachedTarget(permed)
        w_rot = at.weight().detach().float()      # rotated basis, HF row order

        rot = entry["rotation"]
        kr = KroneckerRotation(rot["h_a"].float(), int(rot["b_dim"]))
        w_hf_basis = kr.inverse(w_rot)            # de-rotate the input space

        hf = get(hfname)
        rel = _rel(w_hf_basis, hf)
        assert rel <= 0.15, f"{gname}: rel {rel:.4f} > 0.15 (4-bit floor)"


def test_frozen_fp8_recover_under_quant_floor(state, config):
    """fp8 frozen weights: out_proj columns / in_proj_a,b rows reorder -> rel<=0.05."""
    from act_replay import gguf_to_hf_perm

    get = _hf_getter()
    cases = {
        "blk.0.ssm_out.weight": "model.language_model.layers.0.linear_attn.out_proj.weight",
        "blk.0.ssm_alpha.weight": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "blk.0.ssm_beta.weight": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
    }
    for gname, hfname in cases.items():
        w = state.frozen[gname].float()
        perm = gguf_to_hf_perm(gname, tuple(w.shape), config)
        assert perm is not None, f"{gname}: expected a non-identity 4B reorder"
        axis, idx = perm
        w = w.index_select(axis, idx)
        hf = get(hfname)
        rel = _rel(w, hf)
        assert rel <= 0.05, f"{gname}: rel {rel:.4f} > 0.05 (fp8 floor)"


def test_without_reorder_is_garbage(state, config):
    """Control: skipping the reorder leaves the V-tensors orthogonal to HF."""
    get = _hf_getter()
    w = state.frozen["blk.0.ssm_out.weight"].float()
    hf = get("model.language_model.layers.0.linear_attn.out_proj.weight")
    assert _rel(w, hf) > 0.5, "expected garbage without the column reorder"
