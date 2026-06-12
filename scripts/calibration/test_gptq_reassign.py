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
