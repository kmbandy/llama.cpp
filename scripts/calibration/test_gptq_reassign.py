# test_gptq_reassign.py
import torch
from batched_gptq import batched_gptq_quantize, batched_gptq_reassign

def test_reassign_matches_quantize_indices_with_fitted_centroids():
    # GPTQ with act_order produces indices via its internal fixed-centroid reassign.
    # batched_gptq_reassign fed those SAME fitted centroids/scales must reproduce them bit-for-bit.
    torch.manual_seed(0)
    E, N, K, GS = 1, 16, 64, 32
    W = torch.randn(E, N, K)
    X = torch.randn(256, K)
    H = (X.t() @ X).unsqueeze(0)                       # [E,K,K] SPD
    out = batched_gptq_quantize(W, H, n_centroids=16, group_size=GS,
                                snap_centroids="none", act_order=True)
    idx = batched_gptq_reassign(W, H, out["centroids_per_group"], out["scale_per_group"],
                                group_size=GS, act_order=True)
    assert torch.equal(idx, out["indices"]), "reassign must reproduce GPTQ act_order indices"


def test_reassign_lowers_H_reconstruction_after_centroid_shift():
    # Indices optimal for OLD centroids become stale when centroids move; a GPTQ
    # re-solve against the NEW centroids must lower H-weighted reconstruction error.
    torch.manual_seed(1)
    E, N, K, GS, NC = 1, 16, 64, 32, 16
    W = torch.randn(E, N, K)
    X = torch.randn(256, K); H = (X.t() @ X).unsqueeze(0)
    base = batched_gptq_quantize(W, H, n_centroids=NC, group_size=GS, act_order=True)
    cents, scales, stale_idx = base["centroids_per_group"], base["scale_per_group"], base["indices"]
    new_cents = (cents * 1.15).sort(dim=-1).values        # simulate Axis-A tuning; keep sorted
    gmap = torch.arange(K) // GS
    def recon_err(idx):
        cg = new_cents[:, gmap, :].unsqueeze(1).expand(E, N, K, NC)   # [E,N,K,NC]
        sc = scales[:, :, gmap]                                       # [E,N,K]
        Wq = cg.gather(3, idx.long().unsqueeze(-1)).squeeze(-1) * sc  # [E,N,K]
        d = (W - Wq).float()
        return torch.einsum("eij,ejk,eik->e", d, H, d).sum().item()
    new_idx = batched_gptq_reassign(W, H, new_cents, scales, group_size=GS, act_order=True)
    assert recon_err(new_idx) < recon_err(stale_idx), "re-solve must lower H-reconstruction"


def test_heavy_rounds_path_runs_through_delegated_reassign():
    # The heavy_rounds loop calls the delegated _reassign (which now passes a
    # precomputed Hinv/perm + the compositor-yield params). Exercise it so the
    # delegation regression-fix (no redundant Cholesky, yield restored) has CPU
    # coverage and produces valid in-range indices.
    torch.manual_seed(2)
    E, N, K, GS, NC = 1, 16, 64, 32, 16
    W = torch.randn(E, N, K)
    X = torch.randn(256, K); H = (X.t() @ X).unsqueeze(0)
    out = batched_gptq_quantize(W, H, n_centroids=NC, group_size=GS,
                                snap_centroids="none", act_order=True, heavy_rounds=1)
    idx = out["indices"]
    assert idx.shape == (E, N, K)
    assert int(idx.min()) >= 0 and int(idx.max()) < NC
