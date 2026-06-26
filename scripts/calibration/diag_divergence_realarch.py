"""DIAGNOSTIC (CPU-only, no GPU): does the REAL qwen3_5 arch (linear-attention
gated-delta scan + periodic full-attention) train correctly with ml8-quantized
projections, or does it reproduce the act-replay DIVERGENCE?

The CPU bisection (diag_divergence_bisect.py) cleared the entire Python training
PATH — loss/STE/optimizer, rotation+faithful-acts, stacking+residual, gradient
checkpointing (bit-identical), windowed CachedTeacher (byte-exact). All descended.
So the divergence must be real-model-specific. The single biggest untested piece
is the linear-attention scan itself (the fla gated-delta-net), absent from the
plain-linear stub. This builds a TINY real qwen3_5 (4 layers: 3 linear-attn + 1
full-attn, real per-head dims, RANDOM weights, ~2.8MB) and runs the REAL train().

Setup faithful to act-replay: each targeted Linear gets a synthetic ml8 target;
the TEACHER's matching weight is the EXACT dequant (W_rot @ Q.T for rotated
targets, so teacher==student at zero quant error). The only gap is the e4m3
quantization the trainer is meant to close -> correct behavior is DESCENT.

CPU-ONLY (GPUs hidden) — never touches the fleet 6900xt or the R9700.
"""
import os
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
try:
    open("/proc/self/oom_score_adj", "w").write("600")   # sacrifice THIS, never the desktop
except OSError:
    pass

import sys
import copy
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
import fla_compat
from act_replay import _LMWrap, train
from act_replay_student import attach_to_linear
from kronecker_rotation import KroneckerRotation, factor_for_dim
from teacher_source import LiveTeacher
from kl_loss import kl_topk

VOCAB = 256


DTYPE = torch.float32   # overridden per-run


def tiny_model():
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_key_head_dim=32,
        linear_num_value_heads=4, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=4, max_position_embeddings=2048, tie_word_embeddings=True)
    m = Qwen3_5ForCausalLM(cfg).to(DTYPE).eval()
    fla_compat.apply_fla_cpu_fallback(m, torch.device("cpu"))
    return m


def mk_target(out_f, in_f, *, rotation, seed):
    g = torch.Generator().manual_seed(seed)
    cent = torch.randn(1, 16, generator=g) * 0.1          # G=1 group (whole row)
    t = {"indices": torch.randint(0, 16, (out_f, in_f), generator=g),
         "scales": torch.rand(out_f, 1, generator=g) * 0.05 + 0.02,
         "centroids": cent, "rotation": None}
    if rotation:
        a, b = factor_for_dim(in_f)
        from kronecker_rotation import random_orthogonal
        t["rotation"] = {"h_a": random_orthogonal(a, seed=seed), "a_dim": a,
                         "b_dim": b, "in_features": in_f}
    return t


def dequant_exact(t):
    """Exact ml8 dequant W[r,c] = centroids[0, idx[r,c]] * scales[r,0] (G=1, no snap)."""
    cent = t["centroids"][0]                                # [16]
    return cent[t["indices"].long()] * t["scales"]         # [out,in]


def build(rotation, perturb=0.15):
    teacher = tiny_model()
    student = copy.deepcopy(teacher)
    t_mods = dict(teacher.named_modules())
    s_mods = dict(student.named_modules())
    names = [n for n, m in student.named_modules()
             if isinstance(m, nn.Linear) and "lm_head" not in n]
    ats = []
    for i, name in enumerate(names):
        lin_s = s_mods[name]
        out_f, in_f = lin_s.weight.shape
        t = mk_target(out_f, in_f, rotation=rotation, seed=1000 + i)
        W_rot = dequant_exact(t)                            # [out,in], rotated basis
        if rotation:
            rot = KroneckerRotation(t["rotation"]["h_a"], t["rotation"]["b_dim"])
            Q = rot.forward(torch.eye(in_f))                # [in,in]; x@Q
            W_teacher = W_rot @ Q.T                          # so x@W_teacher.T == (x@Q)@W_rot.T
        else:
            W_teacher = W_rot
        with torch.no_grad():
            t_mods[name].weight.copy_(W_teacher)            # teacher = exact (no quant)
        pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in t.items()}
        if t["rotation"] is not None:
            pert["rotation"] = dict(t["rotation"])
        pert["centroids"] = pert["centroids"] + perturb     # gap to close
        ats.append(attach_to_linear(lin_s, pert,
                                    faithful_acts=(t["rotation"] is not None)))
    teacher.requires_grad_(False)
    return _LMWrap(student), _LMWrap(teacher), ats


def run(name, *, rotation, steps=30):
    torch.manual_seed(0)
    student, teacher, ats = build(rotation)
    src = LiveTeacher(teacher, 8)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, VOCAB, (1, 16), generator=g) for _ in range(4)]
    train_idx, hold_idx = torch.arange(3), torch.tensor([3])

    def kl_now():
        tot = 0.0
        with torch.no_grad():
            for i, ids in enumerate(batches):
                idx, vals, tail = src.get(i, ids)
                lg = student(ids)
                tot += kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail).item()
        return tot / len(batches)

    kl0 = kl_now()
    params = [p for at in ats for p in (at.centroids, at.scales)]
    opt = torch.optim.Adam(params, lr=1e-3)
    # print a short trajectory so a monotonic blow-up is visible, not just endpoints
    traj = [kl0]
    for _ in range(6):
        train(student, src, batches, train_idx, hold_idx, opt,
              grad_accum=1, epochs=1, eval_interval=0)
        traj.append(kl_now())
    train(student, src, batches, train_idx, hold_idx, opt,
          grad_accum=1, epochs=steps, eval_interval=0)
    kl1 = kl_now()
    verdict = "DESCEND" if kl1 < kl0 else f"DIVERGE (x{kl1/max(kl0,1e-9):.1f})"
    print(f"  {name:<34} KL {kl0:.4f} -> {kl1:.4f}   {verdict}")
    print(f"     first-6-step trajectory: " + " -> ".join(f"{v:.4f}" for v in traj))
    return kl1 < kl0


if __name__ == "__main__":
    print("Real qwen3_5 arch (linear-attn scan + full-attn), ml8-quantized "
          "projections, real train().\n")
    print("--- fp32 (matches the CPU torch-reference scan) ---")
    a = run("1) ml8 weights, NO rotation", rotation=False)
    b = run("2) ml8 + rotation + faithful-acts", rotation=True)
    print("\n--- bf16 (matches the GPU run's compute dtype) ---")
    DTYPE = torch.bfloat16
    c = run("3) bf16, NO rotation", rotation=False)
    d = run("4) bf16, rotation + faithful-acts", rotation=True)
    print("\nVerdict:",
          "ALL DESCEND — neither arch nor bf16 reproduces it; cause is GPU-kernel-specific "
          "(fla Triton scan backward on gfx1201)" if all([a, b, c, d])
          else "DIVERGENCE REPRODUCED on CPU — bisect from here")
