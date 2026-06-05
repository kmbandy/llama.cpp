"""Tier 2: adapter.run_block must reproduce the reference full-forward block output,
per block kind. Fails loudly if a delta-net block carries state we didn't capture.
The 0.8B is all delta-net, so this validates the SSM block kind (the known risk)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch, transformers, pytest
from block_arch_adapter import get_adapter
from fla_compat import apply_fla_cpu_fallback

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"

@pytest.mark.slow
def test_run_block_reproduces_reference_per_kind():
    m = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, trust_remote_code=True).eval()
    apply_fla_cpu_fallback(m, "cpu")   # REQUIRED for CPU forward (fla Triton -> torch ref)
    adapter = get_adapter("qwen35")
    blocks = adapter.iter_blocks(m)

    cap = {}   # block_idx -> (in_args, in_kwargs, ref_out)
    handles = []
    def mk(i):
        def pre(mod, args, kwargs): cap[i] = [args, kwargs, None]
        def post(mod, args, output):
            out = output[0] if isinstance(output, tuple) else output
            cap[i][2] = out.detach()
        return pre, post
    for i, b in enumerate(blocks):
        pre, post = mk(i)
        handles += [b.register_forward_pre_hook(pre, with_kwargs=True),
                    b.register_forward_hook(post)]
    ids = torch.randint(0, 1000, (1, 32))
    with torch.no_grad():
        m(ids)
    for h in handles: h.remove()

    def kind(b):
        return "ssm" if any("linear_attn" in n for n, _ in b.named_modules()) else "attn"

    seen = {}
    for i, b in enumerate(blocks):
        in_args, in_kwargs, ref_out = cap[i]
        with torch.no_grad():
            out, _ = adapter.run_block(b, in_args, in_kwargs)
        rel = (out - ref_out).norm() / ref_out.norm().clamp_min(1e-9)
        seen.setdefault(kind(b), rel.item())
        assert rel < 1e-4, f"block {i} ({kind(b)}) run_block diverged: rel={rel:.2e}"
    # 0.8B is all delta-net; require the SSM kind was exercised (the known risk).
    assert "ssm" in seen, f"expected at least one delta-net block, saw {list(seen)}"
    print(f"run_block equivalence OK; max rel by kind: {seen}")
