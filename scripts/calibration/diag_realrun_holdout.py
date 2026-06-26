"""DIAGNOSTIC v2 (R9700, single-model): read the act-replay divergence on a FIXED
holdout (apples-to-apples, like the trainer's eval_kl) and test the gradient-clip
fix — two arms in ONE model load.

v1 (diag_realrun_instrument.py) measured the per-step TRAIN batch KL, which uses
different windows each step -> confounded (bounced 0.08..4 from data variance).
This holds out 4 FIXED windows, trains on the other 20, and logs holdout_kl every
optimizer step (the true divergence signal). Then it RESTORES the initial
centroids/scales and re-runs WITH gradient clipping. If no-clip diverges and clip
descends, clipping is the fix.

Matches the real run: grad_accum=8, lr_cent=1e-3, lr_scale=1e-4, K=256, real A3
targets, fp8 install, grad checkpointing, fla fp32 shim. Single model resident.
oom_score_adj=600; launch under systemd-run MemoryHigh=9G/Max=11G.
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")
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

MODEL = "/home/kmbandy/models/Qwen3.5-4B-hf"
GGUF = "/home/kmbandy/models/mi300x-ggufs/cell_A0_anchor_A3.gguf"
CORPUS = "mix"            # real overnight regime: chat mix + response masking
K, SEQ, N_WIN = 256, 1024, 48
N_HOLD = 8
GRAD_ACCUM, LR_CENT, LR_SCALE = 8, 1e-3, 1e-4
MAX_STEPS = 30


def pick_gfx1201():
    for i in range(torch.cuda.device_count()):
        if "gfx1201" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""):
            return torch.device(f"cuda:{i}")
    raise SystemExit("gfx1201 not found")


def main():
    dev = pick_gfx1201()
    print(f"device {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    resp = assistant_delimiters(tok)
    from calib_corpus import collect_calibration
    print(f"[data] drawing {N_WIN}x{SEQ} from '{CORPUS}' corpus (chat-templated)...", flush=True)
    raw = collect_calibration(tok, n_samples=N_WIN, seq_len=SEQ, composition=CORPUS,
                              seed=0, token_budget=N_WIN * SEQ)
    wins = [(b if b.dim() == 2 else b.unsqueeze(0)).to(dev) for b in raw][:N_WIN]
    n_masked = sum(int(batch_response_mask(w, *resp).sum() > 0) for w in wins)
    train_w, hold_w = wins[:-N_HOLD], wins[-N_HOLD:]
    print(f"[data] {len(train_w)} train + {len(hold_w)} holdout windows; "
          f"{n_masked}/{len(wins)} have assistant spans (response-masked)", flush=True)

    print("[load] bf16 4B...", flush=True)
    model = load_hf_model(MODEL, dev, grad_ckpt=True)
    wrapped = _LMWrap(model, dev)

    print("[teacher] bf16 top-K...", flush=True)
    teach_tr, teach_ho = [], []
    with torch.no_grad():
        for ids in train_w:
            lg = wrapped(ids); teach_tr.append(topk_teacher(lg.reshape(-1, lg.shape[-1]), K)); del lg
        for ids in hold_w:
            lg = wrapped(ids); teach_ho.append(topk_teacher(lg.reshape(-1, lg.shape[-1]), K)); del lg
    torch.cuda.empty_cache()

    print("[rehydrate] real A3 ml8 + fp8...", flush=True)
    modules = dict(model.named_modules()); mcfg = getattr(model, "config", None)
    selected = set(select_targets(list_ml8_names(GGUF), train="ml8", skip=""))
    targets, warn, n_fp8 = {}, {"warned": False}, 0
    _, stream = open_ml8_gguf(GGUF, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, mcfg)
        else:
            n_fp8 += _install_one(modules, name, payload, map_gguf_to_hf, dev,
                                  torch.bfloat16, mcfg, warn)
        del payload
    print(f"[rehydrate] {len(targets)} ml8, {n_fp8} fp8", flush=True)

    cent = [at.centroids for at in targets.values()]
    scl = [at.scales for at in targets.values()]
    init = [(p.detach().clone()) for p in cent + scl]   # snapshot for arm reset

    def restore():
        with torch.no_grad():
            for p, s in zip(cent + scl, init):
                p.copy_(s)

    def holdout_kl():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for ids, (idx, vals, tail) in zip(hold_w, teach_ho):
                lg = wrapped(ids)
                m = batch_response_mask(ids, *resp).reshape(-1).to(dev)
                tot += float(kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail, mask=m).item())
                del lg
        model.train()
        return tot / len(hold_w)

    def gn(p):
        return 0.0 if p.grad is None else float(p.grad.detach().norm())

    def run_arm(label, lr_cent, lr_scale, warmup):
        restore()
        opt = torch.optim.Adam([{"params": cent, "lr": lr_cent},
                                {"params": scl, "lr": lr_scale}])
        model.train()
        print(f"\n=== ARM {label} (lr_c={lr_cent:.0e} lr_s={lr_scale:.0e} "
              f"warmup={warmup}) ===", flush=True)
        print(f"{'step':>4} {'holdoutKL':>10} {'lr_c':>9} {'gC':>9} {'gS':>9}", flush=True)
        print(f"{0:>4} {holdout_kl():>10.4f} {'-':>9} {'-':>9} {'-':>9}", flush=True)
        micro = step = 0; opt.zero_grad()
        while step < MAX_STEPS:
            for ids, (idx, vals, tail) in zip(train_w, teach_tr):
                lg = wrapped(ids)
                m = batch_response_mask(ids, *resp).reshape(-1).to(dev)
                loss = kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail, mask=m) / GRAD_ACCUM
                loss.backward(); del loss, lg, m
                micro += 1
                if micro % GRAD_ACCUM == 0:
                    sc = min(1.0, (step + 1) / warmup) if warmup else 1.0
                    opt.param_groups[0]["lr"] = lr_cent * sc
                    opt.param_groups[1]["lr"] = lr_scale * sc
                    gC, gS = sum(gn(p) for p in cent), sum(gn(p) for p in scl)
                    opt.step(); opt.zero_grad(); step += 1
                    print(f"{step:>4} {holdout_kl():>10.4f} {lr_cent*sc:>9.2e} "
                          f"{gC:>9.2e} {gS:>9.2e}", flush=True)
                    if step >= MAX_STEPS:
                        break

    # HINT run: extend today's most-promising config (2e-4) with warmup (kills the
    # Adam first-step bump) + 30 steps -> does holdoutKL break BELOW the 0.16 PTQ
    # start (real learning / headroom exists), or plateau at break-even (little
    # headroom -> reconsider the act-replay codebook-FT premise)?
    run_arm("lr2e-4-warmup5", 2e-4, 2e-5, warmup=5)
    print("\n[done] if holdoutKL ends well below 0.16 => act-replay has headroom + "
          "learns once tuned; if it plateaus ~0.16 => codebook-FT near headroom-less.",
          flush=True)


if __name__ == "__main__":
    main()
