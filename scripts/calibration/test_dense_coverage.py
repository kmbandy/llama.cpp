"""Enumeration tests for find_dense_full_targets (--dense-coverage ffn|full).

Covers:
  - tier routing is driven by role_targets.classify_role (FFN+attn+ssm_out+gate ->
    ML8; ssm alpha/beta + embed -> FP8; norms native)
  - FFN linears always come FIRST and in find_target_linears order
  - coverage="ffn" yields EXACTLY find_target_linears (today's behavior preserved)
  - ML8_TIER_OVERRIDE re-buckets a whole role uniformly, without dropping tensors

These exercise the post-refactor enumeration where classify_role is the SINGLE
routing truth. configure_roles() MUST be called first (it builds the qwen35
TensorNameMap + applies any tier override) — the prior version of this test omitted
that and failed standalone.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch import nn

from calibrate_ml8_paged import find_dense_full_targets
from calibrate_ml8 import find_target_linears
from role_targets import Tier, configure as configure_roles

_N_LAYERS = 3


def _build_stub(n_layers=_N_LAYERS):
    """Tiny module tree using the REAL Qwen3.5 HF names the qwen35 TensorNameMap
    resolves (verified via classify_role):

      self_attn.{q,k,v,o}_proj                       -> attn_{q,k,v,out}   ML8
      linear_attn.in_proj_qkv / in_proj_z / out_proj -> attn_qkv/gate/ssm_out ML8
      linear_attn.in_proj_a / in_proj_b              -> ssm_alpha/beta      FP8
      mlp.{gate,up,down}_proj                         -> ffn_{gate,up,down}  ML8 (FFN)
      input_layernorm                                -> native (not a Linear)
    Plus top-level model.embed_tokens (nn.Embedding -> FP8) and lm_head (ML8).
    """
    H = 32  # hidden; divisible by 32 so FP8 grouping is valid

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, H, bias=False)
            self.k_proj = nn.Linear(H, H, bias=False)
            self.v_proj = nn.Linear(H, H, bias=False)
            self.o_proj = nn.Linear(H, H, bias=False)

    class LinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(H, H, bias=False)
            self.in_proj_z = nn.Linear(H, H, bias=False)
            self.out_proj = nn.Linear(H, H, bias=False)
            self.in_proj_a = nn.Linear(H, H, bias=False)
            self.in_proj_b = nn.Linear(H, H, bias=False)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(H, H, bias=False)
            self.up_proj = nn.Linear(H, H, bias=False)
            self.down_proj = nn.Linear(H, H, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.linear_attn = LinearAttn()
            self.mlp = MLP()
            self.input_layernorm = nn.LayerNorm(H)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, H)
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(H, 64, bias=False)

    return Model()


def test_ffn_mode_equals_today():
    """coverage='ffn' must yield EXACTLY find_target_linears, same order, all ML8."""
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    today = [(n, m) for (n, m) in find_target_linears(model)]
    ffn = list(find_dense_full_targets(model, coverage="ffn"))
    ffn_pairs = [(n, m) for (n, m, t) in ffn]
    assert [n for n, _ in ffn_pairs] == [n for n, _ in today], (
        f"ffn coverage names diverged:\n  today={[n for n,_ in today]}\n  ffn={[n for n,_ in ffn_pairs]}")
    assert [id(m) for _, m in ffn_pairs] == [id(m) for _, m in today]
    assert all(t is Tier.ML8 for _, _, t in ffn)


def test_full_ffn_prefix_preserved():
    """In full coverage (default tiers), the FFN linears are the leading prefix."""
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    today_names = [n for n, _ in find_target_linears(model)]
    full = list(find_dense_full_targets(model, coverage="full"))
    full_names = [n for n, _, _ in full]
    assert full_names[:len(today_names)] == today_names, (
        f"FFN prefix not preserved:\n  expect={today_names}\n  got={full_names[:len(today_names)]}")


def test_full_tier_routing():
    """Every emitted target carries the classify_role tier; native modules excluded."""
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    full = list(find_dense_full_targets(model, coverage="full"))
    by_name = {n: t for n, _, t in full}

    # FFN + attention + ssm out + ssm gate (in_proj_z) + qkv + lm_head -> ML8
    for suffix in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "self_attn.q_proj", "self_attn.k_proj",
                   "self_attn.v_proj", "self_attn.o_proj",
                   "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                   "linear_attn.out_proj"):
        name = f"model.layers.0.{suffix}"
        assert by_name.get(name) is Tier.ML8, f"{name} -> {by_name.get(name)}"
    assert by_name.get("lm_head") is Tier.ML8

    # ssm alpha/beta (in_proj_a/b) + embed_tokens -> FP8
    assert by_name.get("model.layers.0.linear_attn.in_proj_a") is Tier.FP8
    assert by_name.get("model.layers.0.linear_attn.in_proj_b") is Tier.FP8
    assert by_name.get("model.embed_tokens") is Tier.FP8

    # norms (non-Linear) must NOT appear
    assert "model.layers.0.input_layernorm" not in by_name


def test_full_fp8_after_ml8():
    """Default tiers: all ML8 targets precede all FP8 targets."""
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    full = list(find_dense_full_targets(model, coverage="full"))
    tiers = [t for _, _, t in full]
    last_ml8 = max(i for i, t in enumerate(tiers) if t is Tier.ML8)
    first_fp8 = min(i for i, t in enumerate(tiers) if t is Tier.FP8)
    assert last_ml8 < first_fp8, f"FP8 interleaved with ML8: {tiers}"


def test_full_deterministic_order():
    """Enumeration order is stable across calls; embedding emitted last."""
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    a = [n for n, _, _ in find_dense_full_targets(model, coverage="full")]
    b = [n for n, _, _ in find_dense_full_targets(model, coverage="full")]
    assert a == b
    assert a[-1] == "model.embed_tokens"


def test_unknown_coverage_raises():
    configure_roles("qwen35", _N_LAYERS)
    model = _build_stub()
    try:
        list(find_dense_full_targets(model, coverage="bogus"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown coverage")


def test_tier_override_reroutes_uniformly():
    """ML8_TIER_OVERRIDE moves a whole role between tiers without dropping it.

    Reproduces the Half-1 reallocation: ssm_out/ffn_down/attn_v -> fp8. Every tensor
    of each role must move (uniformly), land in the FP8 bucket, and NONE may vanish
    (the old hardcoded-list enumeration would have silently dropped them)."""
    configure_roles("qwen35", _N_LAYERS,
                    tier_override="ssm_out=fp8,ffn_down=fp8,attn_v=fp8")
    try:
        model = _build_stub()
        full = list(find_dense_full_targets(model, coverage="full"))
        by_name = {n: t for n, _, t in full}
        for L in range(_N_LAYERS):
            assert by_name.get(f"model.layers.{L}.linear_attn.out_proj") is Tier.FP8
            assert by_name.get(f"model.layers.{L}.mlp.down_proj") is Tier.FP8
            assert by_name.get(f"model.layers.{L}.self_attn.v_proj") is Tier.FP8
            # untouched roles stay ML8
            assert by_name.get(f"model.layers.{L}.self_attn.q_proj") is Tier.ML8
            assert by_name.get(f"model.layers.{L}.mlp.up_proj") is Tier.ML8
        # nothing dropped: same tensor COUNT as default (only tiers changed)
        configure_roles("qwen35", _N_LAYERS)
        default = list(find_dense_full_targets(model, coverage="full"))
        assert len(full) == len(default), (
            f"override changed target count {len(default)}->{len(full)} (dropped tensors!)")
    finally:
        configure_roles("qwen35", _N_LAYERS)  # reset global map for other tests


def test_tier_override_unknown_raises():
    for bad in ("ssm_out=q4", "bogus_role=fp8", "ssm_out"):
        try:
            configure_roles("qwen35", _N_LAYERS, tier_override=bad)
        except ValueError:
            continue
        finally:
            configure_roles("qwen35", _N_LAYERS)
        raise AssertionError(f"expected ValueError for bad override {bad!r}")


if __name__ == "__main__":
    test_ffn_mode_equals_today()
    test_full_ffn_prefix_preserved()
    test_full_tier_routing()
    test_full_fp8_after_ml8()
    test_full_deterministic_order()
    test_unknown_coverage_raises()
    test_tier_override_reroutes_uniformly()
    test_tier_override_unknown_raises()
    print("ALL DENSE-COVERAGE TESTS PASSED")
