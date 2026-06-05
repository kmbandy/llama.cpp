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

def _make_hook_factory():
    """Returns a hook CLASS; run_walk calls hook_factory() to get a fresh instance.
    Mimics the real FaithfulActHook contract used by collect_block_hessians +
    run_walk (set_hessian_target/reset_hessian/.H/.n_tokens/install/remove/.rotation)."""
    import torch
    class _H:
        def __init__(self):
            self.H = None; self.n_tokens = 0; self._on = False
            self._handle = None; self.rotation = None
        def set_hessian_target(self, on): self._on = on
        def reset_hessian(self): self.H = None; self.n_tokens = 0
        def install(self, block, leaf):
            mod = dict(block.named_modules())[leaf]
            def pre(m, inp):
                if self._on:
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    XtX = x.t() @ x
                    self.H = XtX if self.H is None else self.H + XtX
                    self.n_tokens += x.shape[0]
            self._handle = mod.register_forward_pre_hook(pre)
        def remove(self):
            if self._handle is not None:
                self._handle.remove(); self._handle = None
    return _H

def test_walk_propagates_quantized_output_to_next_block():
    import torch
    from block_sequential import run_walk
    from block_arch_adapter import SubGroup
    k = 4
    class Blk(torch.nn.Module):
        def __init__(s): super().__init__(); s.lin = torch.nn.Linear(k, k, bias=False)
        def forward(s, x, **kw): return s.lin(x)   # no residual: zeroing is observable
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            class Inner(torch.nn.Module):
                def __init__(ss): super().__init__(); ss.layers = torch.nn.ModuleList([Blk(), Blk()])
            s.model = Inner()
        def forward(s, x):
            h = x
            for b in s.model.layers: h = b(h)
            return h
    torch.manual_seed(0)
    m = M()
    seen = {}
    def fake_quantize(name, layer, idx, H, n_tok, sum_abs, rotation_hook):
        seen[name] = H.clone()
        with torch.no_grad(): layer.weight.zero_()   # "quantized" => zero weight
    class Adapter:
        def iter_blocks(s, model): return list(model.model.layers)
        def run_block(s, b, args, kwargs): return b(*args, **kwargs), kwargs
        def ml8_targets(s, b, i, is_ml8): return [SubGroup(names=["lin"])]
    calib = [torch.randn(2, k) for _ in range(2)]
    run_walk(m, Adapter(), calib, "cpu",
             is_ml8=lambda n: True, quantize_fn=fake_quantize,
             hook_factory=_make_hook_factory())
    names = sorted(seen)   # ["model.layers.0.lin", "model.layers.1.lin"]
    assert len(names) == 2
    # Block 1's weight was zeroed during quantization, so block 2 received an
    # all-zero input; its target Hessian must be all-zero. If propagation were
    # missing (block 2 saw the ORIGINAL block-1 output) H would be nonzero.
    assert torch.count_nonzero(seen[names[1]]) == 0, "block 2 did NOT see quantized upstream"
