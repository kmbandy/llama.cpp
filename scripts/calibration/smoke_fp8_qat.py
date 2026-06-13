"""MAD-281 Task C.3 — 0.8B three-rung fp8 QAT smoke (R9700, single model load).

The composed gate for the whole fp8 trainer: load Qwen3.5-0.8B bf16, capture its
own logits as the (lossless-compression) bf16 teacher top-K, rehydrate the A-cell
ml8+fp8 student INTO the same model, attach the fp8 engine (Ml8Fp8Fn fwd+bwd), and
run THREE arms on the SAME fixed data/seed/holdout:

    frozen  = --reassign none   (Axis A only: continuous centroids/scales)
    mse     = --reassign mse    (+ re-solve indices vs the W_orig anchor)
    pv      = --reassign pv     (+ PV-tuning linearized flip using dL/dW_raw)

Gate (each rung earns its keep): pv <= mse <= frozen final holdout KL. Not a code
failure if violated — it's a finding about where the discrete axis pays off.

Single model resident. oom_score_adj=600; launch under systemd-run MemoryMax=11G.
Mirrors diag_realrun_holdout.py (the proven holdout harness).
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")
except OSError:
    pass

import sys
import time
import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from act_replay import (load_hf_model, _LMWrap, _attach_one, _install_one,
                        map_gguf_to_hf, assistant_delimiters, batch_response_mask,
                        reassign_targets, lr_warmup_cosine, split_batches_seq,
                        collect_target_hessians, gptq_reassign_targets)
from act_replay_student import select_targets
from gguf_state import open_ml8_gguf, list_ml8_names
from kl_loss import topk_teacher, kl_topk
from fp8_qat import Ml8Fp8Fn

def _mem(tag):
    """VRAM breakdown probe: current allocated, reserved, and running peak."""
    import torch
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    print(f"[mem] {tag:30s} alloc={a:6.2f}  reserved={r:6.2f}  peak={p:6.2f} GB", flush=True)


MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"
GGUF = "/home/kmbandy/models/act_replay/Qwen3.5-0.8B-ml8.gguf"   # A-cell ml8+fp8
CORPUS = "mix"
K, SEQ, N_WIN = 256, 1024, 48
N_HOLD = 8
GRAD_ACCUM, LR_CENT, LR_SCALE = 8, 2e-4, 2e-5
MAX_STEPS, WARMUP = 60, 5
REASSIGN_INTERVAL, REASSIGN_FRAC = 10, 0.1
LOSS_SCALE = 1.0
RESULTS = "/home/kmbandy/models/act_replay/MAD281_RUNG_RESULTS.md"


def pick_gfx1201():
    for i in range(torch.cuda.device_count()):
        if "gfx1201" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""):
            return torch.device(f"cuda:{i}")
    raise SystemExit("gfx1201 (R9700) not found")


def main():
    global MODEL, GGUF   # allow --model/--gguf to override the module defaults
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="frozen,mse,pv",
                    help="comma list of arms: frozen,mse,pv,gptq,gptq-interleave "
                         "(pv expands over --pv-fracs; gptq = rung-A single re-solve; "
                         "gptq-interleave = rung-B re-solve every REASSIGN_INTERVAL steps)")
    ap.add_argument("--pv-fracs", default="0.1",
                    help="comma list of pv trust-region fractions (one pv arm each)")
    ap.add_argument("--eval-interval", type=int, default=1,
                    help="holdout-eval cadence in optimizer steps (>1 = much faster)")
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--model", default=MODEL, help="bf16 HF model dir (teacher + host)")
    ap.add_argument("--gguf", default=GGUF, help="rotated/faithful ml8 GGUF to rehydrate")
    ap.add_argument("--train-seq-len", type=int, default=None,
                    help="chunk TRAIN windows to this length (caps per-step activation "
                         "memory; the MAD-264 peak-VRAM-prop-tokens lever). Holdout eval "
                         "keeps the full --seq-len, so the verdict metric is unaffected. "
                         "None = no chunking (fits small models at full length).")
    args = ap.parse_args()
    ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
    # W_orig (the [N,K] fp32 mse/pv reassign anchor) summed over all ml8 targets is
    # ~the whole model in fp32 — it OOM'd the 4B on the 32GB R9700. Only the mse/pv
    # arms use it; frozen/gptq/gptq-interleave don't, so skip it unless asked for.
    NEEDS_W_ORIG = any(a in ("mse", "pv") for a in ARMS)
    PV_FRACS = [float(x) for x in args.pv_fracs.split(",") if x.strip()]
    EVAL_INTERVAL = max(1, args.eval_interval)
    STEPS = args.steps
    MODEL = args.model      # shadow module defaults so the harness runs on any size (e.g. 4B)
    GGUF = args.gguf

    dev = pick_gfx1201()
    print(f"[dev] {dev}  loss_scale={LOSS_SCALE}  arms={ARMS} pv_fracs={PV_FRACS} "
          f"steps={STEPS} eval_interval={EVAL_INTERVAL}", flush=True)
    Ml8Fp8Fn.loss_scale = LOSS_SCALE
    tok = AutoTokenizer.from_pretrained(MODEL)
    resp = assistant_delimiters(tok)

    from calib_corpus import collect_calibration
    print(f"[data] {N_WIN}x{SEQ} from '{CORPUS}' (chat-templated, seed 0)...", flush=True)
    raw = collect_calibration(tok, n_samples=N_WIN, seq_len=SEQ, composition=CORPUS,
                              seed=0, token_budget=N_WIN * SEQ)
    wins = [(b if b.dim() == 2 else b.unsqueeze(0)).to(dev) for b in raw][:N_WIN]
    n_masked = sum(int(batch_response_mask(w, *resp).sum() > 0) for w in wins)
    train_w, hold_w = wins[:-N_HOLD], wins[-N_HOLD:]
    # Token-bounding lever (the 4B-on-32GB knob, mirroring act_replay --train-seq-len /
    # the MAD-264 peak-VRAM-prop-tokens result): tile each TRAIN window into shorter
    # windows BEFORE the teacher pass so the per-step forward+backward activation graph
    # is bounded. Done pre-teacher so each window's top-K teacher matches it. Holdout
    # stays full --seq-len (eval/verdict unaffected).
    if args.train_seq_len:
        n_before = len(train_w)
        train_w, _ = split_batches_seq(
            train_w, torch.arange(len(train_w)), args.train_seq_len)
        print(f"[data] train windows tiled {n_before} -> {len(train_w)} "
              f"@ seq_len {args.train_seq_len} (holdout stays {SEQ})", flush=True)
    print(f"[data] {len(train_w)} train + {len(hold_w)} holdout; "
          f"{n_masked}/{len(wins)} have assistant spans", flush=True)

    print("[load] bf16 0.8B...", flush=True)
    model = load_hf_model(MODEL, dev, grad_ckpt=True)
    wrapped = _LMWrap(model, dev)
    _mem("after model load (bf16 student)")

    print("[teacher] bf16 top-K (captured pre-quant = lossless teacher)...", flush=True)
    teach_tr, teach_ho = [], []
    with torch.no_grad():
        for ids in train_w:
            lg = wrapped(ids); teach_tr.append(topk_teacher(lg.reshape(-1, lg.shape[-1]), K)); del lg
        for ids in hold_w:
            lg = wrapped(ids); teach_ho.append(topk_teacher(lg.reshape(-1, lg.shape[-1]), K)); del lg
    torch.cuda.empty_cache()
    _mem("after teacher pass (+top-K cache)")

    print("[rehydrate] A-cell ml8 + fp8 (fp8 engine attached)...", flush=True)
    modules = dict(model.named_modules()); mcfg = getattr(model, "config", None)
    selected = set(select_targets(list_ml8_names(GGUF), train="ml8", skip=""))
    targets, warn, n_fp8 = {}, {"warned": False}, 0
    _, stream = open_ml8_gguf(GGUF, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, mcfg, fp8=True,
                                            keep_w_orig=NEEDS_W_ORIG,
                                            free_host_weight=True)
        else:
            n_fp8 += _install_one(modules, name, payload, map_gguf_to_hf, dev,
                                  torch.bfloat16, mcfg, warn)
        del payload
    print(f"[rehydrate] {len(targets)} ml8 (fp8 fwd+bwd), {n_fp8} fp8-tier", flush=True)
    torch.cuda.empty_cache()   # release the freed dead bf16 ml8-layer weights
    _mem("after rehydrate (+ml8 buffers)")

    tlist = list(targets.values())
    cent = [at.centroids for at in tlist]
    scl = [at.scales for at in tlist]
    # snapshot for arm reset — INCLUDING indices (mse/pv mutate them in place)
    init_cs = [p.detach().clone() for p in cent + scl]
    # Arm-reset snapshot of all 200 targets' indices — keep it on the HOST, not VRAM
    # (it's ~3.5GB of uint8 on the 4B, only touched between arms; restore() copies it
    # back H2D). Saves that VRAM from the per-step training budget.
    init_idx = [at.indices.detach().cpu() for at in tlist]

    def restore():
        with torch.no_grad():
            for p, s in zip(cent + scl, init_cs):
                p.copy_(s)
            for at, s in zip(tlist, init_idx):
                at.indices.copy_(s)

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

    def run_arm(label, reassign_mode, reassign_frac, eval_interval, max_steps):
        restore()
        opt = torch.optim.Adam([{"params": cent, "lr": LR_CENT},
                                {"params": scl, "lr": LR_SCALE}])
        base = [LR_CENT, LR_SCALE]
        model.train()
        k0 = holdout_kl()
        print(f"\n=== ARM {label}  (reassign={reassign_mode} frac={reassign_frac:g}) ===", flush=True)
        # Only the pv reassign path reads the dense [N,K] dL/dW side channel; capturing
        # it otherwise hoards ~the whole model in fp32 (OOM'd the 4B). Off for all else.
        Ml8Fp8Fn.capture_dLdW = (reassign_mode == "pv")
        print(f"{'step':>4} {'holdoutKL':>10} {'lr_c':>9} {'flips':>10}", flush=True)
        print(f"{0:>4} {k0:>10.4f} {'-':>9} {'-':>10}", flush=True)
        micro = step = 0; opt.zero_grad(); t0 = time.time(); kf = k0
        torch.cuda.reset_peak_memory_stats(); _mem("arm resident (pre-train)")
        _probed = [False]
        gptq_H = {"H": None}   # rung B: rotated Hessian, collected once, reused per re-solve
        while step < max_steps:
            for ids, (idx, vals, tail) in zip(train_w, teach_tr):
                lg = wrapped(ids)
                m = batch_response_mask(ids, *resp).reshape(-1).to(dev)
                loss = kl_topk(lg.reshape(-1, lg.shape[-1]), idx, vals, tail, mask=m) / GRAD_ACCUM
                loss.backward(); del loss, lg, m
                if not _probed[0]:
                    _mem("after 1st fwd+bwd (PEAK)"); _probed[0] = True
                micro += 1
                if micro % GRAD_ACCUM == 0:
                    mult = lr_warmup_cosine(step + 1, WARMUP, max_steps)
                    opt.param_groups[0]["lr"] = base[0] * mult
                    opt.param_groups[1]["lr"] = base[1] * mult
                    opt.step(); opt.zero_grad(); step += 1
                    flips = 0
                    # mse/pv reassign in-loop; gptq (Axis B v2) is a single post-Axis-A
                    # re-solve done after the loop, so it trains frozen here.
                    if reassign_mode in ("mse", "pv") and step % REASSIGN_INTERVAL == 0:
                        flips = reassign_targets(tlist, reassign_mode, frac=reassign_frac)
                    # rung B: interleaved full-H GPTQ re-solve every REASSIGN_INTERVAL steps.
                    # H collected once (lazily) and reused — centroid drift, not H drift, is
                    # what stales the indices. Each re-solve re-optimizes indices for the
                    # centroids Axis A has moved since the last one.
                    if reassign_mode == "gptq-interleave" and step % REASSIGN_INTERVAL == 0:
                        if gptq_H["H"] is None:
                            print(f"[arm {label}] collecting rotated Hessians "
                                  f"({len(targets)} targets)...", flush=True)
                            gptq_H["H"] = collect_target_hessians(targets, train_w, model, dev)
                            model.train()
                        flips = gptq_reassign_targets(targets, gptq_H["H"],
                                                      percdamp=0.05, act_order=True)
                    # Eval cadence: reassign_interval is a multiple of eval_interval in
                    # practice, so every reassign step also reports its post-flip KL.
                    if step % eval_interval == 0 or step >= max_steps:
                        kf = holdout_kl()
                        print(f"{step:>4} {kf:>10.4f} {base[0]*mult:>9.2e} "
                              f"{flips:>10}", flush=True)
                    if step >= max_steps:
                        break
        sps = max_steps / (time.time() - t0)
        print(f"[arm {label}] start {k0:.4f} -> final {kf:.4f}  ({sps:.3f} steps/s)", flush=True)
        # Axis B rung A: ONE full-H GPTQ index re-solve against the now-tuned centroids.
        # Collect the rotated activation Hessian fresh for THIS arm's tuned state, then
        # re-solve indices (sequential, H^-1-compensated — cannot diverge like pv).
        if reassign_mode == "gptq":
            print(f"[arm {label}] collecting rotated Hessians ({len(targets)} targets)...", flush=True)
            H_by_name = collect_target_hessians(targets, train_w, model, dev)
            nflip = gptq_reassign_targets(targets, H_by_name, percdamp=0.05, act_order=True)
            kf = holdout_kl()
            print(f"[arm {label}] post-GPTQ-reassign KL {kf:.4f}  ({nflip} indices changed)", flush=True)
        return k0, kf

    # Build arm specs from CLI: frozen/mse once, pv once per requested frac.
    arm_specs = []
    for a in ARMS:
        if a == "frozen":
            arm_specs.append(("frozen", "none", 0.0))
        elif a == "mse":
            arm_specs.append(("mse", "mse", REASSIGN_FRAC))
        elif a == "pv":
            for fr in PV_FRACS:
                arm_specs.append((f"pv_f{fr:g}", "pv", fr))
        elif a == "gptq":
            arm_specs.append(("gptq", "gptq", 0.0))
        elif a == "gptq-interleave":
            arm_specs.append(("gptqi", "gptq-interleave", 0.0))
    results = {}
    for label, mode, fr in arm_specs:
        results[label] = run_arm(label, mode, fr, EVAL_INTERVAL, STEPS)

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"{'arm':>14} {'start_KL':>10} {'final_KL':>10}", flush=True)
    for label, (k0, kf) in results.items():
        print(f"{label:>14} {k0:>10.4f} {kf:>10.4f}", flush=True)

    with open(RESULTS, "a") as fh:
        fh.write(f"\n## MAD-281 C.3 fp8 QAT smoke — {time.strftime('%Y-%m-%d %H:%M')}\n")
        fh.write(f"- model: {os.path.basename(MODEL)}  gguf: {os.path.basename(GGUF)} | "
                 f"{len(targets)} ml8 (fp8 fwd+bwd) + {n_fp8} fp8-tier\n")
        fh.write(f"- K={K} seq={SEQ} train={len(train_w)} hold={len(hold_w)} | "
                 f"lr_cent={LR_CENT} lr_scale={LR_SCALE} warmup={WARMUP} steps={STEPS} "
                 f"reassign_interval={REASSIGN_INTERVAL} eval_interval={EVAL_INTERVAL} "
                 f"loss_scale={LOSS_SCALE}\n")
        for label, (k0, kf) in results.items():
            fh.write(f"- {label:>14}: start {k0:.4f} -> final {kf:.4f}\n")
    print(f"[written] {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
