"""test_gptq_cholesky_escalation.py — unit test for the robust GPTQ Cholesky helper.

Root cause (MAD-256, faithful-acts path): the e4m3-activation Hessians X^T X are
severely ill-conditioned. The shipped GPTQ inverse-Cholesky path is

    L     = cholesky(H + damp)
    H_inv = cholesky_inverse(L)
    Hinv_chol = cholesky(H_inv, upper=True)   # <-- fragile second factorization

The first factorization succeeds (H+damp is PD) but the SECOND one fails with
"leading minor of order N not positive-definite" — cholesky_inverse returns an
H_inv that is only approximately symmetric / marginally indefinite in fp32. In
the paged driver that RuntimeError propagates up and the whole tensor is
bf16-backfilled (calibrate_ml8_paged.py:1934), blowing up size and coverage.

The fix is `_cholesky_inv_upper`: symmetrize H_inv (free; bit-identical when the
matrix is already symmetric) and, on a genuine LinAlgError, escalate the damping
geometrically until the factorization is PD. Two properties this test pins down:

  A. RECOVERY   — on a marginal H where the plain double-Cholesky RAISES, the
                  helper returns a finite, upper-triangular factor U with
                  U^T U ≈ H^-1.
  B. EQUIVALENCE — on a well-conditioned H the helper is BIT-IDENTICAL to the
                  plain double-Cholesky and reports n_escalations == 0, so it
                  cannot perturb any tensor that already succeeds (the q1 anchor
                  must not move).

CPU-only, deterministic, fast — runs on /usr/bin/python3 with no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from batched_gptq import _cholesky_inv_upper  # noqa: E402


def _plain_double_cholesky(Hd: torch.Tensor) -> torch.Tensor:
    """The shipped path, verbatim — used as both the failure oracle (A) and the
    bit-equivalence reference (B). `Hd` is the ALREADY-DAMPED Hessian."""
    L = torch.linalg.cholesky(Hd)
    H_inv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(H_inv, upper=True)


def _marginal_indefinite_H(K: int, seed: int, neg_frac: float = 0.5,
                           n_neg: int = 3) -> torch.Tensor:
    """SPD matrix with a few eigenvalues driven NEGATIVE by `neg_frac·mean_diag`.

    Mirrors the production "finite-indefinite" failure: rotation + faithful-e4m3
    Hessians accumulate fp32 roundoff that pushes marginal directions slightly
    below zero. The negative margin is set LARGER than the base GPTQ damping
    (0.01·mean_diag) so the plain double-Cholesky still fails WITH damping on,
    exercising the multi-step escalation path rather than a lucky one-shot."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, K, generator=g)
    H = (A @ A.t() / K).float()
    H = 0.5 * (H + H.t())
    dm = H.diagonal().mean()
    ev, V = torch.linalg.eigh(H)
    ev[:n_neg] = -neg_frac * dm
    H = (V @ torch.diag(ev) @ V.t()).float()
    return 0.5 * (H + H.t())


def _effective_damp(base_damp: torch.Tensor, k: int, escalation: float = 10.0):
    """Damp the helper actually used at recovery attempt k (mirrors its schedule:
    k<=1 → base; k>=2 → base·escalation**(k-1))."""
    return base_damp if k <= 1 else base_damp * (escalation ** (k - 1))


def test_equivalence_on_well_conditioned():
    """B: helper == plain double-Cholesky, bit-for-bit, with zero escalations."""
    K = 256
    eye = torch.eye(K, dtype=torch.float32)
    g = torch.Generator().manual_seed(7)
    A = torch.randn(K, K, generator=g)
    H = (A @ A.t() / K + 1e-2 * torch.eye(K)).float()
    damp = 0.01 * H.diagonal().mean()
    Hd = H + damp * eye

    ref = _plain_double_cholesky(Hd)
    got, n_esc = _cholesky_inv_upper(H, damp, eye)

    assert n_esc == 0, f"well-conditioned H should need no escalation, got {n_esc}"
    max_diff = (ref - got).abs().max().item()
    assert max_diff == 0.0, f"helper must be bit-identical on succeeders, max|diff|={max_diff:.3e}"
    print(f"[B] equivalence OK: bit-identical (max|diff|={max_diff:.1e}), n_escalations=0")


def test_recovery_on_marginal_hessians():
    """A: where the plain path RAISES (even WITH base damping), the helper escalates
    and returns a finite, upper-triangular factor U with U^T U == (H+damp_eff·I)^-1."""
    K = 384
    eye = torch.eye(K, dtype=torch.float32)
    n_total = 0
    n_plain_failed = 0
    n_helper_recovered = 0
    worst_reconstruction = 0.0
    escalations_seen = set()

    for seed in range(12):
        H = _marginal_indefinite_H(K, seed)
        damp = 0.01 * H.diagonal().mean()
        Hd = H + damp * eye
        n_total += 1

        plain_raised = False
        try:
            _plain_double_cholesky(Hd)
        except torch._C._LinAlgError:
            plain_raised = True
            n_plain_failed += 1

        U, n_esc = _cholesky_inv_upper(H, damp, eye)
        assert torch.isfinite(U).all(), f"seed {seed}: helper produced non-finite factor"
        assert U.tril(-1).abs().max().item() == 0.0, f"seed {seed}: factor not upper-triangular"
        escalations_seen.add(n_esc)
        if plain_raised:
            n_helper_recovered += 1
        # U^T U must reconstruct the inverse of the damping the helper ACTUALLY used.
        damp_eff = _effective_damp(damp, n_esc)
        Hinv_true = torch.linalg.inv((H + damp_eff * eye).double())
        rel = ((U.double().t() @ U.double()) - Hinv_true).abs().max().item() / Hinv_true.abs().max().item()
        worst_reconstruction = max(worst_reconstruction, rel)

    print(f"[A] indefinite matrices: total={n_total} plain_failed={n_plain_failed} "
          f"helper_recovered={n_helper_recovered} escalations={sorted(escalations_seen)} "
          f"worst_relerr(U^TU vs (H+damp_eff)^-1)={worst_reconstruction:.2e}")
    assert n_plain_failed == n_total, (
        f"test precondition: every indefinite matrix should break the plain path "
        f"({n_plain_failed}/{n_total})")
    assert n_helper_recovered == n_plain_failed, (
        f"helper must recover ALL plain failures: {n_helper_recovered}/{n_plain_failed}")
    assert max(escalations_seen) >= 2, (
        f"expected multi-step escalation, saw {sorted(escalations_seen)}")
    assert worst_reconstruction < 1e-3, (
        f"recovered factor reconstructs the inverse poorly: rel_err={worst_reconstruction:.2e}")


def main() -> int:
    test_equivalence_on_well_conditioned()
    test_recovery_on_marginal_hessians()
    print("=== PASS ===  Cholesky escalation helper: equivalence + recovery")
    return 0


if __name__ == "__main__":
    sys.exit(main())
