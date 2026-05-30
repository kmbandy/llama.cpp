"""Enumeration tests for find_dense_full_targets (--dense-coverage ffn|full).

Covers:
  - tier routing (FFN+attn+ssm_out -> ML8; alpha/beta/embed -> FP8; norms native)
  - FFN linears always come FIRST and in find_target_linears order
  - coverage="ffn" yields EXACTLY find_target_linears (today's behavior preserved)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch import nn

# Import without triggering the wp_native pager (paged path) — these enumeration
# helpers are pure-torch. calibrate_ml8_paged guards the pager import, so a plain
# import works even where wp_native is absent.
from calibrate_ml8_paged import find_dense_full_targets
from calibrate_ml8 import find_target_linears
from role_targets import Tier


def _build_stub(n_layers=3):
    """Tiny module tree mirroring the Qwen3.5 HF naming we enumerate over.

    Layout per layer:
      model.layers.{L}.self_attn.{q,k,v,o}_proj   -> ML8
      model.layers.{L}.linear_attn.out_proj       -> ML8 (ssm_out)
      model.layers.{L}.linear_attn.{alpha,beta}_proj -> FP8
      model.layers.{L}.mlp.{gate,up,down}_proj     -> ML8 (FFN)
      model.layers.{L}.input_layernorm             -> native (not a Linear)
    Plus a top-level model.embed_tokens (nn.Embedding -> FP8) and lm_head (ML8).
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
            self.out_proj = nn.Linear(H, H, bias=False)
            self.alpha_proj = nn.Linear(H, H, bias=False)
            self.beta_proj = nn.Linear(H, H, bias=False)

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
    """coverage='ffn' must yield EXACTLY find_target_linears, same order."""
    model = _build_stub()
    today = [(n, m) for (n, m) in find_target_linears(model)]
    ffn = list(find_dense_full_targets(model, coverage="ffn"))
    # tier-tagged -> strip tier
    ffn_pairs = [(n, m) for (n, m, t) in ffn]
    assert [n for n, _ in ffn_pairs] == [n for n, _ in today], (
        f"ffn coverage names diverged:\n  today={[n for n,_ in today]}\n  ffn={[n for n,_ in ffn_pairs]}")
    # identity of module objects, not just names
    assert [id(m) for _, m in ffn_pairs] == [id(m) for _, m in today]
    # all ML8
    assert all(t is Tier.ML8 for _, _, t in ffn)


def test_full_ffn_prefix_preserved():
    """In full coverage, the FFN linears must be the leading prefix, in order."""
    model = _build_stub()
    today_names = [n for n, _ in find_target_linears(model)]
    full = list(find_dense_full_targets(model, coverage="full"))
    full_names = [n for n, _, _ in full]
    assert full_names[:len(today_names)] == today_names, (
        f"FFN prefix not preserved:\n  expect={today_names}\n  got={full_names[:len(today_names)]}")


def test_full_tier_routing():
    """Every emitted target carries the right tier; native modules excluded."""
    model = _build_stub()
    full = list(find_dense_full_targets(model, coverage="full"))
    by_name = {n: t for n, _, t in full}

    # FFN + attention + ssm out + lm_head -> ML8
    for suffix in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                   "self_attn.q_proj", "self_attn.k_proj",
                   "self_attn.v_proj", "self_attn.o_proj",
                   "linear_attn.out_proj"):
        name = f"model.layers.0.{suffix}"
        assert by_name.get(name) is Tier.ML8, f"{name} -> {by_name.get(name)}"
    assert by_name.get("lm_head") is Tier.ML8

    # alpha/beta proj + embed_tokens -> FP8
    assert by_name.get("model.layers.0.linear_attn.alpha_proj") is Tier.FP8
    assert by_name.get("model.layers.0.linear_attn.beta_proj") is Tier.FP8
    assert by_name.get("model.embed_tokens") is Tier.FP8

    # norms (non-Linear) must NOT appear
    assert "model.layers.0.input_layernorm" not in by_name


def test_full_fp8_after_ml8():
    """All ML8 targets must precede all FP8 targets in full coverage."""
    model = _build_stub()
    full = list(find_dense_full_targets(model, coverage="full"))
    tiers = [t for _, _, t in full]
    last_ml8 = max(i for i, t in enumerate(tiers) if t is Tier.ML8)
    first_fp8 = min(i for i, t in enumerate(tiers) if t is Tier.FP8)
    assert last_ml8 < first_fp8, f"FP8 interleaved with ML8: {tiers}"


def test_full_deterministic_order():
    """Enumeration order is stable across calls (deterministic)."""
    model = _build_stub()
    a = [n for n, _, _ in find_dense_full_targets(model, coverage="full")]
    b = [n for n, _, _ in find_dense_full_targets(model, coverage="full")]
    assert a == b
    # embed_tokens (FP8) is emitted last
    assert a[-1] == "model.embed_tokens"


def test_unknown_coverage_raises():
    model = _build_stub()
    try:
        list(find_dense_full_targets(model, coverage="bogus"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown coverage")


if __name__ == "__main__":
    test_ffn_mode_equals_today()
    test_full_ffn_prefix_preserved()
    test_full_tier_routing()
    test_full_fp8_after_ml8()
    test_full_deterministic_order()
    test_unknown_coverage_raises()
    print("ALL DENSE-COVERAGE TESTS PASSED")
