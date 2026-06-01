#!/usr/bin/env python3
"""down_proj offline rig — reconstruct the TRUE SwiGLU intermediate and test
bpv-neutral levers (AWQ first) on the one expert matrix where they should bite.

WHY THIS EXISTS:
  gate_proj and up_proj both consume x (the post-norm MoE hidden state), which
  awq_sweep.py showed is already smooth/incoherent — AWQ moves it +0.05..0.27 dB,
  a dud. down_proj is different: its input is the SwiGLU intermediate
      interm = silu(x·W_gate^T) * (x·W_up^T)        # ⊙ elementwise
  a PRODUCT of two projections → fat-tailed, outlier-heavy. That's the spikiest
  activation in the transformer and exactly what AWQ/scaling is designed for.

  The router capture (route-capture-*.pt) only saved x, so the rig has been
  scoring down_proj under the WRONG (smooth-x) Hessian. Here we reconstruct the
  real intermediate offline from captured x + bf16 gate/up weights, build the
  pooled shared H_down = E[interm interm^T] (matching the pipeline's per-layer
  shared down Hessian, calibrate_ml8_paged.py:875/1033), and quantize the down
  weights under it — faithful to production.

AWQ measurement is free here for the same reason as awq_sweep.py: scaling the
down-weight columns by s and the intermediate by 1/s leaves the quantizer's
reported y_snr_db equal to the original-space output SNR.

Assumption: act_fn = SiLU (z·sigmoid(z)). The pipeline comment and Qwen3 MoE
both use silu(gate)*up; flagged here so it's auditable.

Usage:
  python3 down_proj_rig.py --capture /home/kmbandy/models/route-capture-L0-L20.pt \
      --gguf /home/kmbandy/models/Qwen3.6-35B-A3B-bf16.gguf \
      --n-experts 64 --device cuda:0
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402
from batched_gptq import batched_gptq_quantize  # noqa: E402


def load_experts(gguf_path, layer, suffix, device):
    r = gguf.GGUFReader(gguf_path)
    name = f"blk.{layer}.{suffix}.weight"
    t = next((t for t in r.tensors if t.name == name), None)
    if t is None:
        raise RuntimeError(f"{name} not found")
    arr = np.asarray(t.data, dtype=np.uint8).copy()
    shp = [int(s) for s in reversed(list(t.shape))]
    return torch.from_numpy(arr).view(torch.bfloat16).reshape(*shp).to(torch.float32).to(device)


def quantize_get_ysnr(W_stack, H, gs, nc, snap):
    E = W_stack.shape[0]
    H_stack = H.unsqueeze(0).repeat(E, 1, 1)
    out = batched_gptq_quantize(W_stack, H_stack, n_centroids=nc,
                                group_size=gs, snap_centroids=snap, fit_loss="mse")
    y = out["y_snr_db"]; y = y[~torch.isnan(y)]
    del H_stack, out
    if W_stack.is_cuda:
        torch.cuda.empty_cache()
    return float(y.median()) if y.numel() else float("nan")


def awq_transform(W_stack, H, s):
    inv = 1.0 / s
    Wp = W_stack * s.view(1, 1, -1)
    Hp = H * inv.view(-1, 1) * inv.view(1, -1)
    return Wp.contiguous(), Hp.contiguous()


def build_intermediate_and_H(x, logits, Wg, Wu, top_k, device):
    """Reconstruct SwiGLU intermediates for ALL routed (token,expert) pairs and
    return (pooled H_down [I,I], per-channel outlier diagnostic, total tokens)."""
    N, K = x.shape
    Eall, I, _ = Wg.shape
    sel = logits.topk(top_k, dim=1).indices                 # [N, top_k]
    oh = torch.zeros(N, Eall, device=device, dtype=torch.bool)
    oh.scatter_(1, sel, True)
    H = torch.zeros(I, I, device=device, dtype=torch.float32)
    n_tot = 0
    kurt_acc = torch.zeros(I, device=device)                # 4th-moment per channel (outlier gauge)
    sq_acc = torch.zeros(I, device=device)
    for e in range(Eall):
        idx = oh[:, e].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        Xe = x[idx]                                         # [n_e, K]
        g = F.silu(Xe @ Wg[e].t())                          # [n_e, I]
        u = Xe @ Wu[e].t()                                  # [n_e, I]
        interm = g * u                                      # [n_e, I]  fat-tailed
        H += interm.t() @ interm
        sq_acc += interm.pow(2).sum(0)
        kurt_acc += interm.pow(4).sum(0)
        n_tot += idx.numel()
    H /= max(n_tot, 1)
    var = (sq_acc / max(n_tot, 1)).clamp_min(1e-12)
    kurt = (kurt_acc / max(n_tot, 1)) / var.pow(2)          # per-channel kurtosis (3=gaussian)
    return H, kurt, n_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--n-experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--n-centroids", type=int, default=16)
    p.add_argument("--snap", default="e4m3")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    alphas = [float(a) for a in args.alphas.split(",")]
    blob = torch.load(args.capture, map_location="cpu", weights_only=False)
    print(f"loaded {args.capture}: layers {sorted(blob.keys())}  device={dev}")
    print(f"config: gs{args.group_size} nc{args.n_centroids} snap={args.snap}  "
          f"DOWN_PROJ via reconstructed SwiGLU intermediate  E_sample={args.n_experts}")

    for L in sorted(blob.keys()):
        x = blob[L]["x"].to(device=dev, dtype=torch.float32)
        logits = blob[L]["logits"].to(device=dev, dtype=torch.float32)
        Wg = load_experts(args.gguf, int(L), "ffn_gate_exps", dev)   # [E, I, K]
        Wu = load_experts(args.gguf, int(L), "ffn_up_exps", dev)     # [E, I, K]
        Wd = load_experts(args.gguf, int(L), "ffn_down_exps", dev)   # [E, K_out, I]
        H, kurt, n_tot = build_intermediate_and_H(x, logits, Wg, Wu, args.top_k, dev)
        I = H.shape[0]
        # busiest experts for the down-weight sample
        sel = logits.topk(args.top_k, dim=1).indices
        oh = torch.zeros(x.shape[0], Wg.shape[0], device=dev, dtype=torch.bool); oh.scatter_(1, sel, True)
        pick = oh.sum(0).argsort(descending=True)[:min(args.n_experts, Wd.shape[0])]
        W_stack = Wd[pick].contiguous()                              # [E, K_out, I]
        del Wg, Wu, Wd
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

        # Outlier diagnostic: SwiGLU intermediate kurtosis vs gaussian (3).
        kmed = float(kurt.median()); kmax = float(kurt.max())
        print(f"\n{'='*72}\nLAYER {L} ffn_down_exps:  E={W_stack.shape[0]}  W_e={tuple(W_stack.shape[1:])}  I={I}  n_tok={n_tot}")
        print(f"   [intermediate outlier gauge] per-channel kurtosis: median={kmed:.1f} max={kmax:.0f}  (gaussian=3.0; high ⇒ fat tails ⇒ AWQ headroom)")
        print(f"   {'config':<26}{'med Y_SNR':>10}   note")
        act_rms = torch.diagonal(H).clamp_min(1e-12).sqrt()
        base = None
        for a in alphas:
            s = act_rms.pow(a); s = s / s.mean()
            Wp, Hp = awq_transform(W_stack, H, s)
            med = quantize_get_ysnr(Wp, Hp, args.group_size, args.n_centroids, args.snap)
            if base is None:
                base = med
            tag = "BASELINE (s=1)" if a == 0.0 else f"AWQ act^{a}"
            print(f"   {tag:<26}{med:>9.2f}   (Δ {med-base:+.2f})")


if __name__ == "__main__":
    main()
