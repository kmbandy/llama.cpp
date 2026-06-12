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

from index_reassign import pv_vstep

def test_pv_vstep_applies_loss_reducing_flips():
    G, NC, N, K = 1, 4, 2, 4
    centroids = torch.tensor([[-1.0, -0.3, 0.3, 1.0]])              # [G,NC]
    scales = torch.ones(N, G)
    gidx = torch.zeros(K, dtype=torch.long)
    idx = torch.zeros(N, K, dtype=torch.uint8)                     # all point at centroid 0 (-1.0)
    # dL/dW positive everywhere → loss decreases if W decreases → want most-negative centroid (already 0)
    dLdW = torch.ones(N, K)
    new_idx, n_flips = pv_vstep(idx, dLdW, centroids, scales, gidx, frac=1.0)
    assert n_flips == 0                                            # already optimal direction
    # now dL/dW negative → loss decreases if W increases → want centroid 3 (+1.0)
    new_idx2, n2 = pv_vstep(idx, -torch.ones(N, K), centroids, scales, gidx, frac=1.0)
    assert (new_idx2 == 3).all() and n2 == N * K
