"""DIAGNOSTIC (R9700/gfx1201, single-model, RAM-bounded): localize the act-replay
REAL-4B-SCALE divergence by instrumenting a SHORT real run.

Everything below 4B scale was cleared (algorithm, arch, fla fwd+bwd, ckpt, bf16,
windowed teacher). The cause is real-scale: real activation/gradient magnitudes,
the actual A3 centroid/scale/rotation values, 32 layers, 136 tensors. This loads
the 4B bf16 ONCE, computes the bf16 TEACHER top-K in-memory (no second model, no
cache rebuild), rehydrates the REAL A3 ml8 targets in place -> student, then runs
~16 instrumented optimizer steps logging:
  - train KL (does it RISE = divergence?)
  - centroids vs scales grad-norm; top tensors by grad-norm (WHICH blows up)
  - scales global min/max drift (toward 0/neg = weight collapse)
  - centroid e4m3 saturation count
Matches the real run: grad_accum=8, lr_cent=1e-3, lr_scale=1e-4, K=256, response
mask, grad checkpointing, fp8 install. Single model resident (host peak ~6GB).
oom_score_adj=600 in-process; launch under systemd-run MemoryHigh=9G/Max=11G.
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")   # sacrifice THIS, never desktop
except OSError:
    pass

import sys
import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from act_replay import (load_hf_model, _LMWrap, _attach_one, _install_one,
                        map_gguf_to_hf, assistant_delimiters, batch_response_mask)
from act_replay_student import select_targets
from gguf_state import open_ml8_gguf, list_ml8_names
from kl_loss import topk_teacher, kl_topk
from centroid_quantizer import snap_to_e4m3

MODEL = "/home/kmbandy/models/Qwen3.5-4B-hf"
GGUF = "/home/kmbandy/models/mi300x-ggufs/cell_A0_anchor_A3.gguf"
TEXT = "/home/kmbandy/models/phase2/sizesweep/heldouts/wiki_chat_eval.txt"
K = 256
SEQ = 1024
N_WIN = 24
GRAD_ACCUM = 8
LR_CENT, LR_SCALE = 1e-3, 1e-4
MAX_STEPS = 16


def pick_gfx1201():
    for i in range(torch.cuda.device_count()):
        if "gfx1201" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""):
            return torch.device(f"cuda:{i}")
    raise SystemExit("gfx1201 not found")


def make_windows(tok, device):
    text = open(TEXT).read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    wins = []
    for s in range(0, len(ids) - SEQ, SEQ):
        wins.append(ids[s:s + SEQ].unsqueeze(0).to(device))
        if len(wins) >= N_WIN:
            break
    return wins


def sat_frac(c):
    """fraction of centroids whose e4m3 snap moved them (near lattice edges/saturation)."""
    snapped = snap_to_e4m3(c)
    return float((snapped.abs() >= 448.0).float().mean())


def main():
    dev = pick_gfx1201()
    print(f"device {dev} = {torch.cuda.get_device_properties(dev.index).gcnArchName}; "
          f"VRAM free/total(GB) {[round(x/1e9,2) for x in torch.cuda.mem_get_info(dev.index)]}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    resp = assistant_delimiters(tok)

    print("[load] bf16 4B (streaming)...", flush=True)
    model = load_hf_model(MODEL, dev, grad_ckpt=True)
    wrapped = _LMWrap(model, dev)
    wins = make_windows(tok, dev)
    print(f"[data] {len(wins)} windows x {SEQ} tok", flush=True)

    # ── bf16 TEACHER top-K in-memory (single model, BEFORE quantization) ──
    print("[teacher] computing bf16 top-K (no second model)...", flush=True)
    teacher = []
    with torch.no_grad():
        for ids in wins:
            lg = wrapped(ids)
            teacher.append(topk_teacher(lg.reshape(-1, lg.shape[-1]), K))
            del lg
    torch.cuda.empty_cache()

    # ── rehydrate REAL A3 ml8 targets in place -> student ──
    print("[rehydrate] attaching real A3 ml8 targets + fp8 install...", flush=True)
    modules = dict(model.named_modules())
    mcfg = getattr(model, "config", None)
    selected = set(select_targets(list_ml8_names(GGUF), train="ml8", skip=""))
    targets, warn = {}, {"warned": False}
    n_fp8 = 0
    _, stream = open_ml8_gguf(GGUF, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, mcfg)
        else:
            n_fp8 += _install_one(modules, name, payload, map_gguf_to_hf, dev,
                                  torch.bfloat16, mcfg, warn)
        del payload
    print(f"[rehydrate] {len(targets)} ml8 targets, {n_fp8} fp8 installed", flush=True)

    cent = [at.centroids for at in targets.values()]
    scl = [at.scales for at in targets.values()]
    opt = torch.optim.Adam([{"params": cent, "lr": LR_CENT},
                            {"params": scl, "lr": LR_SCALE}])
    names = list(targets.keys())

    def gn(p):
        return 0.0 if p.grad is None else float(p.grad.detach().norm())

    print(f"\n{'step':>4} {'trainKL':>9} {'gC':>9} {'gS':>9} "
          f"{'sMin':>8} {'sMax':>8} {'cMax':>7} {'sat%':>6}  topgrad-tensors", flush=True)
    micro = 0
    step = 0
    opt.zero_grad()
    accum_kl = 0.0
    model.train()
    epoch = 0
    while step < MAX_STEPS:
        for i, ids in enumerate(wins):
            idx, vals, tail = teacher[i]
            logits = wrapped(ids)
            V = logits.shape[-1]
            mask = batch_response_mask(ids, *resp).reshape(-1).to(dev)
            loss = kl_topk(logits.reshape(-1, V), idx, vals, tail, mask=mask) / GRAD_ACCUM
            loss.backward()
            accum_kl += float(loss.item()) * GRAD_ACCUM
            del loss, logits, mask
            micro += 1
            if micro % GRAD_ACCUM == 0:
                # ── instrument BEFORE the step ──
                gC = sum(gn(p) for p in cent)
                gS = sum(gn(p) for p in scl)
                per = sorted(((gn(targets[n].centroids) + gn(targets[n].scales), n)
                              for n in names), reverse=True)[:4]
                smin = min(float(p.detach().min()) for p in scl)
                smax = max(float(p.detach().max()) for p in scl)
                cmax = max(float(p.detach().abs().max()) for p in cent)
                sat = max(sat_frac(p.detach()) for p in cent)
                top = ", ".join(f"{n.split('.',2)[-1]}:{g:.1e}" for g, n in per)
                print(f"{step:>4} {accum_kl/GRAD_ACCUM:>9.4f} {gC:>9.2e} {gS:>9.2e} "
                      f"{smin:>8.3f} {smax:>8.3f} {cmax:>7.2f} {sat*100:>5.1f}  {top}",
                      flush=True)
                opt.step()
                opt.zero_grad()
                step += 1
                accum_kl = 0.0
                if step >= MAX_STEPS:
                    break
        epoch += 1
    print("\n[done] if trainKL RISES + a param/tensor's grad explodes or scales "
          "drift to 0/neg, that's the real-scale failure mode.", flush=True)


if __name__ == "__main__":
    main()
