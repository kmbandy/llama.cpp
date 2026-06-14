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
from centroid_quantizer import snap_to_e4m3
from gguf_state import open_ml8_gguf, list_ml8_names
from kl_loss import topk_teacher, kl_topk
from fp8_qat import Ml8Fp8Fn

def _host_rss():
    """Host memory: (RssAnon, RssFile) GB from /proc/self/status. RssAnon is the
    real pressure (private dirty); RssFile is mostly reclaimable mmap (e.g. the
    device_map checkpoint), so it inflates RSS without being a true cost."""
    anon = file = 0.0
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("RssAnon:"):
                    anon = int(ln.split()[1]) / 1e6      # kB -> GB
                elif ln.startswith("RssFile:"):
                    file = int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return anon, file

def _mem(tag):
    """VRAM + host-RAM breakdown probe."""
    import torch
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    p = torch.cuda.max_memory_allocated() / 1e9
    ha, hf = _host_rss()
    print(f"[mem] {tag:30s} vram(alloc={a:6.2f} reserved={r:6.2f} peak={p:6.2f}) "
          f"host(anon={ha:5.2f} file={hf:5.2f}) GB", flush=True)


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
    global MODEL, GGUF, LR_CENT, LR_SCALE, WARMUP, N_WIN, LOSS_SCALE   # CLI overrides
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
    ap.add_argument("--lr-cent", type=float, default=LR_CENT,
                    help="Axis-A centroid LR. Default (2e-4) was tuned on the 0.8B; a "
                         "bigger model with a larger PTQ gap overshoots at the warmup "
                         "peak — lower it (e.g. 5e-5) for the 4B (MAD-283).")
    ap.add_argument("--lr-scale", type=float, default=LR_SCALE, help="Axis-A scale LR")
    ap.add_argument("--warmup-steps", type=int, default=WARMUP,
                    help="linear LR warmup steps before cosine decay")
    ap.add_argument("--n-win", type=int, default=N_WIN,
                    help="total calib windows (train = n_win - n_hold). More windows = "
                         "more data for the trainer to actually descend below the PTQ "
                         "floor (MAD-283: small regimes only break-even).")
    ap.add_argument("--loss-scale", type=float, default=LOSS_SCALE,
                    help="Ml8Fp8Fn.loss_scale (gradient scaling)")
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
    LR_CENT = args.lr_cent; LR_SCALE = args.lr_scale; WARMUP = args.warmup_steps
    N_WIN = args.n_win; LOSS_SCALE = args.loss_scale

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
    fp8_tier_gguf = []   # gguf names that took the bf16 fp8-tier install branch (diag)
    _, stream = open_ml8_gguf(GGUF, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, mcfg, fp8=True,
                                            keep_w_orig=NEEDS_W_ORIG,
                                            free_host_weight=True)
        else:
            inst = _install_one(modules, name, payload, map_gguf_to_hf, dev,
                                torch.bfloat16, mcfg, warn)
            n_fp8 += inst
            if inst:
                fp8_tier_gguf.append(name)
        del payload
    print(f"[rehydrate] {len(targets)} ml8 (fp8 fwd+bwd), {n_fp8} fp8-tier", flush=True)
    torch.cuda.empty_cache()   # release the freed dead bf16 ml8-layer weights
    _mem("after rehydrate (+ml8 buffers)")

    # ── env-gated module enumeration (ENUMERATE_MODULES=1): classify every
    # residual nn.Linear that still runs a bf16 aten::mm into fp8-tier-as-bf16
    # vs untiered, size each by N*K (a FLOP proxy at fixed token count M), then
    # exit. This sizes MAD-290 lever #1 (route frozen bf16 linears -> fp8 a8w8):
    # how much of the profiled ~4.5s aten::mm is fp8-tier vs untiered residual.
    if os.environ.get("ENUMERATE_MODULES"):
        import torch.nn as _nn, sys as _sys
        ml8_hf, fp8_hf = set(), set()
        for _g in targets:
            try: ml8_hf.add(map_gguf_to_hf(_g))
            except KeyError: pass
        for _g in fp8_tier_gguf:
            try: fp8_hf.add(map_gguf_to_hf(_g))
            except KeyError: pass
        cats = {"ml8": [], "fp8_tier": [], "untiered": []}
        nonstd = 0
        for nm, mod in model.named_modules():
            if isinstance(mod, _nn.Linear):
                c = "ml8" if nm in ml8_hf else ("fp8_tier" if nm in fp8_hf else "untiered")
                w = getattr(mod, "weight", None)
                if w is None or w.dim() != 2:
                    # ml8-attached shell / freed weight: count membership, skip sizing
                    nonstd += 1
                    cats[c].append((nm, 0, 0, "non-2d"))
                    continue
                N, Kk = int(w.shape[0]), int(w.shape[1])
                cats[c].append((nm, N, Kk, str(w.dtype)))
        print(f"(non-2d Linear shells skipped for sizing: {nonstd})", flush=True)
        print("\n===== MODULE ENUMERATION (residual nn.Linear) =====", flush=True)
        tot_nk = sum(N * Kk for c in cats for _, N, Kk, _ in cats[c]) or 1
        for c in ("ml8", "fp8_tier", "untiered"):
            n = len(cats[c]); nk = sum(N * Kk for _, N, Kk, _ in cats[c])
            params = nk / 1e6
            print(f"[{c:9}] count={n:4d}  sum(N*K)={params:9.1f}M  "
                  f"FLOP-share={100*nk/tot_nk:5.1f}%", flush=True)
        # the bf16 aten::mm cost is fp8_tier + untiered; report their internal split
        bf16_nk = sum(N * Kk for c in ("fp8_tier", "untiered") for _, N, Kk, _ in cats[c]) or 1
        for c in ("fp8_tier", "untiered"):
            nk = sum(N * Kk for _, N, Kk, _ in cats[c])
            print(f"   of-bf16-mm: {c:9} = {100*nk/bf16_nk:5.1f}%", flush=True)
        # break down untiered + fp8_tier by unique (name-suffix, shape) so we can
        # see WHICH projections dominate (gate/up/down/qkv/head/ssm)
        from collections import Counter as _Counter
        for c in ("fp8_tier", "untiered"):
            agg = _Counter()
            aggnk = {}
            for nm, N, Kk, _ in cats[c]:
                suf = nm.split(".")[-1]
                key = f"{suf}[{N}x{Kk}]"
                agg[key] += 1; aggnk[key] = aggnk.get(key, 0) + N * Kk
            print(f"-- {c} breakdown by (proj, shape) --", flush=True)
            for key, cnt in sorted(agg.items(), key=lambda kv: -aggnk[kv[0]]):
                print(f"   {cnt:4d}x {key:28} sum(N*K)={aggnk[key]/1e6:8.1f}M", flush=True)
        _sys.stdout.flush()
        _sys.exit(0)

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
        # RAM-safe streaming anchor for the gptq / gptq-interleave arms. Those skip
        # capturing W_orig (keep_w_orig=False) — holding all ~200 fp32 anchors ≈ the
        # whole 4B model (~16GB) and OOM'd the GPU. Instead reconstruct the INITIAL
        # ml8-dequant PTQ anchor on demand (peak ~one [N,K] ~100MB, freed after each
        # target). Uses the pre-train snapshot (init_cs / init_idx, in tlist order),
        # bit-exact to AttachedTarget.weight() at attach time:
        #   snap_to_e4m3(init_centroids)[gidx] gathered by init_indices.long()
        #   times init_scales[:, gidx].
        # The anchor is the FIXED initial PTQ target (init indices, NOT the mutated
        # current ones), so it stays constant across interleaved re-solves.
        name_to_i = {nm: i for i, nm in enumerate(targets.keys())}
        n = len(tlist)

        def anchor_provider(name, at):
            i = name_to_i[name]
            init_cent = init_cs[i].to(dev)              # [G, NC]
            init_scl_i = init_cs[n + i].to(dev)         # [N, G]
            init_idx_i = init_idx[i].to(dev)            # [N, K] uint8
            gidx = at.gidx                              # [K] long
            cent_snap = snap_to_e4m3(init_cent)         # [G, NC]
            cent_per_col = cent_snap[gidx]              # [K, NC]
            idx_long = init_idx_i.long()                # [N, K]
            gathered = cent_per_col.unsqueeze(0).expand(
                init_idx_i.shape[0], -1, -1).gather(
                2, idx_long.unsqueeze(-1)).squeeze(-1)  # [N, K]
            return gathered * init_scl_i[:, gidx]       # [N, K]
        # ── env-gated matmul tracer (TRACE_MM=1): attribute the big aten::mm calls
        # to their python call-site (fla scan is Triton, so these are HF-modeling `@`
        # ops, not fla). Patches torch matmul entry points, runs ONE fwd+bwd, prints
        # big matmuls grouped by (shapes, source line), exits. MAD-290 diagnostic.
        if os.environ.get("TRACE_MM"):
            import sys as _sys, traceback as _tb, collections as _c
            _sw, _st = (train_w, teach_tr) if train_w else (hold_w, teach_ho)
            ids0, (i0, v0, t0_) = _sw[0], _st[0]
            m0 = batch_response_mask(ids0, *resp).reshape(-1).to(dev)
            agg = _c.defaultdict(int)
            THRESH = 500_000   # output-numel threshold for "big"
            _real_tmm = torch.Tensor.matmul
            _real_fmm = torch.matmul
            _real_mm = torch.mm

            def _site():
                for fr in reversed(_tb.extract_stack()[:-2]):
                    fn = fr.filename
                    if "smoke_fp8_qat" in fn or "fp8_qat.py" in fn:
                        continue
                    if "/torch/" in fn and "modeling_" not in fn:
                        continue
                    return f"{fn.split('/')[-1]}:{fr.lineno} {fr.name}"
                return "?"

            def _rec(out, a, b):
                try:
                    if out.dim() >= 2 and out.numel() >= THRESH:
                        agg[(tuple(a.shape), tuple(b.shape), _site())] += 1
                except Exception:
                    pass
                return out

            _real_dmm = torch.Tensor.__matmul__
            torch.Tensor.matmul = lambda self, other, *a, **k: _rec(_real_tmm(self, other, *a, **k), self, other)
            torch.Tensor.__matmul__ = lambda self, other: _rec(_real_dmm(self, other), self, other)
            torch.matmul = lambda a, b, *ar, **k: _rec(_real_fmm(a, b, *ar, **k), a, b)
            torch.mm = lambda a, b, *ar, **k: _rec(_real_mm(a, b, *ar, **k), a, b)
            opt.zero_grad(set_to_none=True)
            lg = wrapped(ids0)
            loss = kl_topk(lg.reshape(-1, lg.shape[-1]), i0, v0, t0_, mask=m0)
            loss.backward()
            torch.cuda.synchronize()
            torch.Tensor.matmul = _real_tmm; torch.Tensor.__matmul__ = _real_dmm
            torch.matmul = _real_fmm; torch.mm = _real_mm
            print("\n===== BIG MATMUL CALL-SITES (TRACE_MM) =====", flush=True)
            print(f"(captured {sum(agg.values())} big-matmul calls)", flush=True)
            for (sa, sb, site), cnt in sorted(agg.items(), key=lambda kv: -kv[1])[:24]:
                print(f"  {cnt:4d}x  {list(sa)} @ {list(sb)}   <- {site}", flush=True)
            _sys.exit(0)

        # ── env-gated dispatch profiler (PROFILE_STEPS=N): profile the REAL ml8
        # micro-step (fwd through fp8 LUT + SSM, kl_topk loss, backward), report
        # GPU-bound vs host-bound via CPU-blocked-on-GPU time, then exit. Same code
        # path as the train loop below, so it is faithful, not a proxy.
        if os.environ.get("PROFILE_STEPS"):
            import time as _t, sys as _sys
            from torch.profiler import profile as _profile, ProfilerActivity as _PA
            NPF = int(os.environ["PROFILE_STEPS"])
            _sw, _st = (train_w, teach_tr) if train_w else (hold_w, teach_ho)
            ids0, (i0, v0, t0_) = _sw[0], _st[0]
            m0 = batch_response_mask(ids0, *resp).reshape(-1).to(dev)

            def _micro():
                opt.zero_grad(set_to_none=True)
                lg = wrapped(ids0)
                loss = kl_topk(lg.reshape(-1, lg.shape[-1]), i0, v0, t0_, mask=m0)
                loss.backward()

            for _ in range(3):
                _micro()
            torch.cuda.synchronize()
            _s = _t.perf_counter()
            for _ in range(8):
                _micro()
            torch.cuda.synchronize()
            wall = (_t.perf_counter() - _s) / 8 * 1e3
            torch.cuda.synchronize()
            # Print wall early + allow a profiler-free exit: the torch profiler with
            # record_shapes chokes on the FWHT rotation's butterfly op explosion
            # (thousands of slice/cat/add); WALL_ONLY gets a clean step time.
            print(f"\n[wall-only] {wall:8.1f} ms/micro", flush=True)
            if os.environ.get("WALL_ONLY"):
                import sys as _sys2
                _sys2.exit(0)
            with _profile(activities=[_PA.CPU, _PA.CUDA], record_shapes=True) as prof:
                for _ in range(NPF):
                    _micro()
                torch.cuda.synchronize()
            ka = prof.key_averages()

            def _dev(e):
                for a in ("self_device_time_total", "self_cuda_time_total"):
                    if hasattr(e, a):
                        return getattr(e, a)
                return 0

            def _cpu(e):
                return getattr(e, "self_cpu_time_total", 0)

            def _ck(subs):
                return sum(e.count for e in ka if any(s in e.key for s in subs))

            sync_ms = sum(_cpu(e) for e in ka if "Synchronize" in e.key) / NPF / 1e3
            launch_ms = sum(_cpu(e) for e in ka if "LaunchKernel" in e.key) / NPF / 1e3
            launches = _ck(["LaunchKernel"]) / NPF
            print("\n===== REAL ML8 MICRO-STEP PROFILE =====", flush=True)
            print(f"[wall]                              {wall:8.1f} ms/micro", flush=True)
            print(f"[CPU blocked-on-GPU (Synchronize)]  {sync_ms:8.1f} ms/micro "
                  f"=> {sync_ms / wall * 100:4.0f}% of wall  (HIGH => GPU-bound)", flush=True)
            print(f"[CPU dispatch (LaunchKernel self)]  {launch_ms:8.1f} ms/micro "
                  f"({launches:.0f} launches)  (HIGH + low-sync => HOST-bound)", flush=True)
            print("-- top DEVICE (GPU) kernels --", flush=True)
            for e in sorted(ka, key=_dev, reverse=True)[:14]:
                print(f"  {_dev(e) / NPF / 1e3:8.2f} ms  x{e.count // NPF:>5}  {e.key[:46]}", flush=True)
            print("-- top HOST self --", flush=True)
            for e in sorted(ka, key=_cpu, reverse=True)[:14]:
                print(f"  {_cpu(e) / NPF / 1e3:8.2f} ms  x{e.count // NPF:>5}  {e.key[:46]}", flush=True)
            # ── GEMM attribution by operand shape: which matmuls own the bf16 mm
            # cost? lm_head is 2560x248320, in_proj 2560x32; small head-dim shapes
            # => SSM functional matmuls (fla f32 scan), NOT routable via Linear attach.
            kas = prof.key_averages(group_by_input_shape=True)
            gemm = [e for e in kas if e.key in ("aten::mm", "aten::addmm",
                    "aten::matmul", "aten::bmm")]
            print("-- GEMM (aten::mm/addmm/bmm) by operand shape (device time) --",
                  flush=True)
            for e in sorted(gemm, key=_dev, reverse=True)[:20]:
                shp = getattr(e, "input_shapes", "")
                print(f"  {_dev(e) / NPF / 1e3:8.2f} ms  x{e.count // NPF:>5}  "
                      f"{e.key:11} {str(shp)[:60]}", flush=True)
            _sys.exit(0)

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
                                                      percdamp=0.05, act_order=True,
                                                      anchor_provider=anchor_provider)
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
            nflip = gptq_reassign_targets(targets, H_by_name, percdamp=0.05,
                                          act_order=True,
                                          anchor_provider=anchor_provider)
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
