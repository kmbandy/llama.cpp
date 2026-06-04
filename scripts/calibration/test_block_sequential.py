import torch
from faithful_forward import collect_block_hessians

class FakeHook:
    """Mimics FaithfulActHook's H-accumulation contract."""
    def __init__(self, linear): self.linear = linear; self.H = None; self.n_tokens = 0; self._on = False
    def set_hessian_target(self, on): self._on = on
    def reset_hessian(self): self.H = None; self.n_tokens = 0
    def observe(self, x):
        if not self._on: return
        XtX = x.t() @ x
        self.H = XtX if self.H is None else self.H + XtX
        self.n_tokens += x.shape[0]

class FakeBlock(torch.nn.Module):
    def __init__(self, k): super().__init__(); self.lin = torch.nn.Linear(k, k, bias=False)
    def forward(self, x, **kw): return x + self.lin(x)

def test_collect_block_hessians_accumulates_per_target():
    torch.manual_seed(0); k = 8
    block = FakeBlock(k)
    hook = FakeHook(block.lin)
    # adapter.run_block calls the block; the hook observes the linear's input
    def run_block(b, args, kwargs):
        x = args[0]; hook.observe(x); out = b(x, **kwargs); return out, kwargs
    inps = [((torch.randn(4, k),), {}) for _ in range(3)]
    Hs = collect_block_hessians(block, {"lin": hook}, inps, run_block)
    H, n = Hs["lin"]
    assert n == 12 and H.shape == (k, k)
    assert torch.allclose(H, H.t())  # XtX symmetric
