"""DIAGNOSTIC (R9700/gfx1201): does the REAL fla Triton scan + gradient
checkpointing + bf16 reproduce the act-replay divergence at TINY scale?

CPU cleared everything (incl. the fla backward in isolation). The remaining
candidates are GPU/real-only: (A) gradient-checkpoint RECOMPUTE x the fla Triton
autograd.Function, (C) the shim bf16 output-downcast on the backward. This runs
the tiny real qwen3_5 (3 linear-attn gated-delta + 1 full-attn, ~2.8MB) ON
gfx1201 with the REAL fla kernel (apply_fla_arch_shim -> fp32 scan on RDNA), bf16
compute, ml8-quantized projections, real train() — WITH checkpointing and
WITHOUT. Teacher = exact dequant (W_rot @ Q.T) so descent is correct behavior.

- both DESCEND      -> (A)+(C) cleared; cause is real-4B-SCALE (next: short
                       instrumented 4B run).
- with-ckpt DIVERGES, without descends -> checkpoint x fla is the bug (tiny repro!).
- both DIVERGE      -> fla-in-full-model / bf16+fla path.

Tiny VRAM (<1GB), gfx1201-targeted, oom_score_adj=600.
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")
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
from kronecker_rotation import KroneckerRotation, factor_for_dim, random_orthogonal
from teacher_source import LiveTeacher
from kl_loss import kl_topk

VOCAB = 256
DTYPE = torch.bfloat16


def pick_gfx1201():
    if not torch.cuda.is_available():
        raise SystemExit("no ROCm device visible")
    for i in range(torch.cuda.device_count()):
        if "gfx1201" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""):
            return torch.device(f"cuda:{i}")
    raise SystemExit("gfx1201 (R9700) not found")


def tiny_model(dev):
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_key_head_dim=32,
        linear_num_value_heads=4, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        full_attention_interval=4, max_position_embeddings=2048, tie_word_embeddings=True)
    return Qwen3_5ForCausalLM(cfg).to(dev).to(DTYPE).eval()


def mk_target(out_f, in_f, *, seed):
    g = torch.Generator().manual_seed(seed)
    a, b = factor_for_dim(in_f)
    return {"indices": torch.randint(0, 16, (out_f, in_f), generator=g),
            "scales": torch.rand(out_f, 1, generator=g) * 0.05 + 0.02,
            "centroids": torch.randn(1, 16, generator=g) * 0.1,
            "rotation": {"h_a": random_orthogonal(a, seed=seed), "a_dim": a,
                         "b_dim": b, "in_features": in_f}}


def dequant_exact(t):
    return t["centroids"][0][t["indices"].long()] * t["scales"]   # [out,in], rotated basis


def build(dev, ckpt):
    teacher = tiny_model(dev)
    student = copy.deepcopy(teacher)
    t_mods, s_mods = dict(teacher.named_modules()), dict(student.named_modules())
    names = [n for n, m in student.named_modules()
             if isinstance(m, nn.Linear) and "lm_head" not in n]
    ats = []
    for i, name in enumerate(names):
        lin_s = s_mods[name]
        out_f, in_f = lin_s.weight.shape
        t = mk_target(out_f, in_f, seed=2000 + i)
        W_rot = dequant_exact(t)                                  # fp32
        rot = KroneckerRotation(t["rotation"]["h_a"], t["rotation"]["b_dim"])
        Q = rot.forward(torch.eye(in_f))
        W_teacher = (W_rot @ Q.T).to(dev).to(DTYPE)
        with torch.no_grad():
            t_mods[name].weight.copy_(W_teacher)
        pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in t.items()}
        pert["rotation"] = dict(t["rotation"])
        pert["centroids"] = pert["centroids"] + 0.15
        at = attach_to_linear(lin_s, pert, faithful_acts=True)
        at.to(dev)                                                # codebook leaves -> GPU
        ats.append(at)
    # REAL fla kernel, fp32 scan on RDNA (NOT the cpu fallback)
    fla_compat.apply_fla_arch_shim(teacher, dev)
    fla_compat.apply_fla_arch_shim(student, dev)
    if ckpt:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher.requires_grad_(False)
    return _LMWrap(student, dev), _LMWrap(teacher, dev), ats


def run(name, dev, ckpt, steps=12):
    torch.manual_seed(0)
    student, teacher, ats = build(dev, ckpt)
    src = LiveTeacher(teacher, 8)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, VOCAB, (1, 16), generator=g).to(dev) for _ in range(4)]
    train_idx, hold_idx = torch.arange(3), torch.tensor([3])

    def kl_now():
        tot = 0.0
        with torch.no_grad():
            for i, ids in enumerate(batches):
                idx, vals, tail = src.get(i, ids)
                lg = student(ids)
                tot += kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail).item()
        return tot / len(batches)

    traj = [kl_now()]
    params = [p for at in ats for p in (at.centroids, at.scales)]
    opt = torch.optim.Adam(params, lr=1e-3)
    print(f"[{name}] step0 KL={traj[0]:.4f}  (compiling fla kernels on first step...)",
          flush=True)
    for s in range(steps):
        train(student, src, batches, train_idx, hold_idx, opt,
              grad_accum=1, epochs=1, eval_interval=0)
        traj.append(kl_now())
        print(f"[{name}] step{s+1} KL={traj[-1]:.4f}", flush=True)
    verdict = "DESCEND" if traj[-1] < traj[0] else f"DIVERGE (x{traj[-1]/max(traj[0],1e-9):.1f})"
    print(f"[{name}] {traj[0]:.4f} -> {traj[-1]:.4f}  {verdict}", flush=True)
    return traj[-1] < traj[0]


if __name__ == "__main__":
    dev = pick_gfx1201()
    print(f"device {dev} = {torch.cuda.get_device_properties(dev.index).gcnArchName}  "
          f"free/total(GB): {[round(x/1e9,2) for x in torch.cuda.mem_get_info(dev.index)]}",
          flush=True)
    a = run("ckpt=OFF", dev, ckpt=False)
    b = run("ckpt=ON ", dev, ckpt=True)
    print("\nVERDICT:",
          "both DESCEND — GPU path (fla+ckpt+bf16) is fine; cause is real-4B-SCALE" if (a and b)
          else "checkpoint x fla is the bug (tiny GPU repro)" if (a and not b)
          else "fla-in-full-model / bf16+fla path diverges (both configs)", flush=True)
