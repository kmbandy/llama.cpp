import torch
import torch.nn.functional as F
from kl_loss import topk_teacher, kl_topk

def test_topk_shapes():
    t = torch.randn(7, 100)
    idx, vals, tail = topk_teacher(t, 16)
    assert idx.shape == (7, 16) and vals.shape == (7, 16) and tail.shape == (7,)
    full_lse = torch.logsumexp(t, -1)
    rest_lse = torch.logsumexp(t.scatter(1, idx, float("-inf")), -1)
    assert torch.allclose(tail, rest_lse, atol=1e-5)

def test_kl_full_equals_topk():
    g = torch.Generator().manual_seed(0)
    t, s = torch.randn(5, 32, generator=g), torch.randn(5, 32, generator=g)
    idx, vals, tail = topk_teacher(t, 31)
    full = F.kl_div(torch.log_softmax(s, -1), torch.log_softmax(t, -1),
                    log_target=True, reduction="batchmean")
    assert abs(kl_topk(s, idx, vals, tail) - full) < 1e-4

def test_kl_zero_when_equal():
    g = torch.Generator().manual_seed(1)
    t = torch.randn(6, 64, generator=g)
    idx, vals, tail = topk_teacher(t, 8)
    assert kl_topk(t, idx, vals, tail).abs() < 1e-6

def test_kl_mask():
    g = torch.Generator().manual_seed(2)
    t, s = torch.randn(4, 16, generator=g), torch.randn(4, 16, generator=g)
    idx, vals, tail = topk_teacher(t, 4)
    m = torch.tensor([1., 1., 0., 0.])
    masked = kl_topk(s, idx, vals, tail, mask=m)
    first2 = kl_topk(s[:2], idx[:2], vals[:2], tail[:2])
    assert torch.allclose(masked, first2, atol=1e-6)

def test_grad_flows():
    g = torch.Generator().manual_seed(3)
    t = torch.randn(3, 20, generator=g)
    s = torch.randn(3, 20, generator=g, requires_grad=True)
    idx, vals, tail = topk_teacher(t, 5)
    kl_topk(s, idx, vals, tail).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()

def test_k_equals_v_finite():
    g = torch.Generator().manual_seed(4)
    t = torch.randn(3, 8, generator=g)
    s = torch.randn(3, 8, generator=g, requires_grad=True)
    idx, vals, tail = topk_teacher(t, 8)
    loss = kl_topk(s, idx, vals, tail)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(s.grad).all()
