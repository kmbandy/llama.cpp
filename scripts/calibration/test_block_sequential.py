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

from block_arch_adapter import DefaultBlockAdapter, get_adapter

def test_default_adapter_run_block_tuple_and_dict():
    import torch
    blk = torch.nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
    ad = DefaultBlockAdapter()
    x = torch.randn(2, 4, 16)
    out, nkw = ad.run_block(blk, (x,), {})
    assert isinstance(out, torch.Tensor) and out.shape == x.shape
    assert isinstance(nkw, dict)

def test_get_adapter_returns_qwen35_for_qwen35():
    from block_arch_adapter import Qwen35BlockAdapter
    assert isinstance(get_adapter("qwen35"), Qwen35BlockAdapter)
    assert isinstance(get_adapter("some-unknown-arch"), DefaultBlockAdapter)

def test_capture_block_inputs_grabs_args_and_aborts():
    import torch
    from block_sequential import capture_block_inputs
    captured = {"n_downstream": 0}
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = torch.nn.Linear(4, 4)
            self.b1 = torch.nn.Linear(4, 4)
        def forward(self, x):
            h = self.b0(x)
            captured["n_downstream"] += 1   # must NOT run after capture
            return self.b1(h)
    m = Tiny()
    calib = [torch.randn(1, 4) for _ in range(3)]
    inps = capture_block_inputs(m, m.b0, calib, device="cpu")
    assert len(inps) == 3
    args, kwargs = inps[0]
    assert args[0].shape == (1, 4)
    assert captured["n_downstream"] == 0   # sentinel aborted before downstream
