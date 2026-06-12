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
    idx = torch.zeros(N, K, dtype=torch.uint8)                     # all at centroid 0 (-1.0, the floor)
    h = torch.full((K,), 0.1)                                      # small curvature → large Newton step
    # dL/dW positive: loss falls if W decreases, but cur is already the min centroid → no flip
    new_idx, n_flips = pv_vstep(idx, torch.ones(N, K), h, centroids, scales, gidx, frac=1.0)
    assert n_flips == 0                                            # already at the floor in that direction
    # dL/dW negative + small h: Newton point lands past the top centroid → flip to centroid 3 (+1.0)
    new_idx2, n2 = pv_vstep(idx, -torch.ones(N, K), h, centroids, scales, gidx, frac=1.0)
    assert (new_idx2 == 3).all() and n2 == N * K

def test_pv_vstep_quadratic_picks_near_newton_not_extreme():
    # Constructed degeneracy: codebook [-3,-1,0,1,3], cur=idx0 (W=-3), g=-2, h=1.
    #   linear    g*dW           = [0,-4,-6,-8,-12]   -> argmin j=4  (the EXTREME, dW=6)
    #   quadratic g*dW+.5h*dW^2  = [0,-2,-1.5, 0,  6] -> argmin j=1  (Newton W-g/h = -1)
    G, NC, N, K = 1, 5, 1, 1
    centroids = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]])        # [G,NC]
    scales = torch.ones(N, G)
    gidx = torch.zeros(K, dtype=torch.long)
    idx = torch.zeros(N, K, dtype=torch.uint8)                     # cur = centroid 0 (-3.0)
    dLdW = torch.full((N, K), -2.0)
    h = torch.ones(K)
    new_idx, n_flips = pv_vstep(idx, dLdW, h, centroids, scales, gidx, frac=1.0)
    assert new_idx.item() == 1, f"quadratic should pick near-Newton j=1, got {new_idx.item()}"
    assert new_idx.item() != NC - 1                               # NOT the codebook extreme
    assert n_flips == 1

from index_reassign import index_reassign

def test_index_reassign_dispatch_none_is_noop():
    idx = torch.randint(0,16,(4,8),dtype=torch.uint8)
    out, n = index_reassign(idx, "none", None, None, None, None, None, None)
    assert torch.equal(out, idx) and n == 0
