"""DIAGNOSTIC (R9700/gfx1201): is the fla gated-delta-rule Triton kernel's
BACKWARD correct on RDNA?

The CPU investigation cleared every other component; the only thing the GPU run
does that CPU can't is run the real fla `chunk_gated_delta_rule` Triton kernel.
The fla_compat shim only ever validated its FORWARD on RDNA (inference/PPL). The
act-replay trainer is the first to run its BACKWARD on gfx1201.

This runs the fla Triton kernel and the CPU-PROVEN torch reference
(`torch_chunk_gated_delta_rule`) on the SAME tiny fp32 inputs on the R9700, and
compares forward AND backward. A large backward mismatch (with the torch ref
matching finite differences) pins the divergence on the fla backward.

Tiny tensors (<1MB), single small kernel — targets gfx1201 by gcnArchName, sets
oom_score_adj=600. No 4B load.
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")
except OSError:
    pass

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk
except Exception as e:
    print("fla import failed:", e); raise SystemExit(1)


def pick_gfx1201():
    if not torch.cuda.is_available():
        raise SystemExit("no ROCm/CUDA device visible")
    for i in range(torch.cuda.device_count()):
        arch = getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")
        if "gfx1201" in arch:
            return i, arch
    raise SystemExit("gfx1201 (R9700) not found among visible devices")


def mk_inputs(dev, *, B=1, H=4, T=64, Dk=64, Dv=64, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float32).to(dev)
    q = r(B, T, H, Dk); k = r(B, T, H, Dk); v = r(B, T, H, Dv)
    beta = torch.rand(B, T, H, generator=g).to(dev)                # in (0,1)
    gate = (-torch.rand(B, T, H, generator=g) * 0.5).to(dev)        # log-decay <0
    return q, k, v, beta, gate


def run_impl(fn, q, k, v, beta, gate, *, is_fla):
    q = q.clone().requires_grad_(True); k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True); beta = beta.clone().requires_grad_(True)
    gate = gate.clone().requires_grad_(True)
    if is_fla:
        out, _ = fn(q, k, v, g=gate, beta=beta, initial_state=None,
                    output_final_state=False, use_qk_l2norm_in_kernel=True)
    else:
        out, _ = fn(q, k, v, gate, beta, chunk_size=64, initial_state=None,
                    output_final_state=False, use_qk_l2norm_in_kernel=True)
    loss = (out.float() ** 2).sum()
    loss.backward()
    grads = {"q": q.grad, "k": k.grad, "v": v.grad, "beta": beta.grad, "g": gate.grad}
    return out.detach().float(), {n: (gr.detach().float() if gr is not None else None)
                                  for n, gr in grads.items()}, float(loss)


def main():
    idx, arch = pick_gfx1201()
    dev = torch.device(f"cuda:{idx}")
    print(f"device cuda:{idx} = {arch}  free/total(GB):",
          [round(x / 1e9, 2) for x in torch.cuda.mem_get_info(idx)])
    q, k, v, beta, gate = mk_inputs(dev)

    print("\n-- running torch reference (CPU-proven correct) on GPU --")
    out_ref, g_ref, l_ref = run_impl(torch_chunk_gated_delta_rule, q, k, v, beta, gate, is_fla=False)
    print(f"   loss_ref = {l_ref:.6f}")

    print("-- running fla Triton kernel (the suspect) on GPU --")
    out_fla, g_fla, l_fla = run_impl(fla_chunk, q, k, v, beta, gate, is_fla=True)
    print(f"   loss_fla = {l_fla:.6f}")

    fdiff = (out_fla - out_ref).abs().max().item()
    odiff = (out_fla - out_ref).abs().mean().item()
    print(f"\nFORWARD  max|Δ|={fdiff:.3e}  mean|Δ|={odiff:.3e}")
    print("BACKWARD grad agreement (fla vs torch-ref):")
    worst = 0.0
    for n in ("q", "k", "v", "beta", "g"):
        a, b = g_fla[n], g_ref[n]
        if a is None or b is None:
            print(f"   {n:>4}: MISSING grad (fla={a is not None}, ref={b is not None})")
            worst = float("inf"); continue
        denom = b.abs().max().item() or 1.0
        rel = (a - b).abs().max().item() / denom
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)).item()
        print(f"   {n:>4}: max|Δ|={ (a-b).abs().max().item():.3e}  rel={rel:.3e}  cos={cos:+.5f}")
        worst = max(worst, rel)

    print("\nVERDICT:",
          "fla BACKWARD matches the proven reference — NOT the bug"
          if worst < 1e-2 else
          f"fla BACKWARD DISAGREES with the proven reference (worst rel {worst:.2e}) "
          f"-> fla gated-delta Triton backward is wrong on gfx1201")


if __name__ == "__main__":
    main()
