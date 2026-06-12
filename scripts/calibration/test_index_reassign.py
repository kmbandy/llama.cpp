# test_index_reassign.py
import torch
from index_reassign import mse_estep

def test_mse_estep_assigns_nearest_centroid():
    # 2 groups, 16 centroids; W_orig exactly equals centroid[g, j]*scale for a known j
    G, NC, N, K = 2, 16, 4, 8
    centroids = torch.randn(G, NC)
    scales = torch.rand(N, G) + 0.5
    gidx = torch.tensor([0,0,0,0,1,1,1,1])
    true_idx = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    # build W_orig from the true assignment
    cent_per_col = centroids[gidx]                                  # [K,NC]
    W = cent_per_col.unsqueeze(0).expand(N,-1,-1).gather(
        2, true_idx.long().unsqueeze(-1)).squeeze(-1) * scales[:, gidx]
    new_idx = mse_estep(W, centroids, scales, gidx)
    assert torch.equal(new_idx, true_idx)                          # recovers exact assignment
