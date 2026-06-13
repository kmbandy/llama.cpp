"""MAD-281 D.3 debug — gather REAL evidence on why curvature-corrected pv still
diverged at frac=0.1. Loads the 0.8B ml8+fp8 student, does ONE real backward to
populate Ml8Fp8Fn.last_dLdW (g) and last_h (h), then on representative targets
dumps: g/h magnitudes, Newton step |Δw*|=|g/h| vs codebook spacing, and the
best_j histogram (interior vs codebook-extreme) for BOTH the linear and the
quadratic criterion. No fix here — only measurement.
"""
import os
try:
    open("/proc/self/oom_score_adj", "w").write("600")
except OSError:
    pass
import sys, torch
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from act_replay import (load_hf_model, _LMWrap, _attach_one, _install_one,
                        map_gguf_to_hf, assistant_delimiters, batch_response_mask)
from act_replay_student import select_targets
from gguf_state import open_ml8_gguf, list_ml8_names
from kl_loss import topk_teacher, kl_topk
from fp8_qat import Ml8Fp8Fn
from centroid_quantizer import snap_to_e4m3

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"
GGUF = "/home/kmbandy/models/act_replay/Qwen3.5-0.8B-ml8.gguf"
K, SEQ, N_WIN = 256, 1024, 8


def pick():
    for i in range(torch.cuda.device_count()):
        if "gfx1201" in getattr(torch.cuda.get_device_properties(i), "gcnArchName", ""):
            return torch.device(f"cuda:{i}")
    raise SystemExit("no gfx1201")


def main():
    dev = pick()
    tok = AutoTokenizer.from_pretrained(MODEL)
    resp = assistant_delimiters(tok)
    from calib_corpus import collect_calibration
    raw = collect_calibration(tok, n_samples=N_WIN, seq_len=SEQ, composition="mix",
                              seed=0, token_budget=N_WIN * SEQ)
    wins = [(b if b.dim() == 2 else b.unsqueeze(0)).to(dev) for b in raw][:N_WIN]
    print("[load] bf16 0.8B...", flush=True)
    model = load_hf_model(MODEL, dev, grad_ckpt=True)
    wrapped = _LMWrap(model, dev)

    # Teacher = bf16 model logits captured PRE-QUANT (mirrors smoke_fp8_qat order).
    ids = wins[0]
    print("[teacher] bf16 top-K (pre-quant)...", flush=True)
    with torch.no_grad():
        lg0 = wrapped(ids)
        teach = topk_teacher(lg0.reshape(-1, lg0.shape[-1]), K); del lg0
    torch.cuda.empty_cache()

    modules = dict(model.named_modules()); mcfg = getattr(model, "config", None)
    selected = set(select_targets(list_ml8_names(GGUF), train="ml8", skip=""))
    targets = {}
    _, stream = open_ml8_gguf(GGUF, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, mcfg, fp8=True)
        else:
            _install_one(modules, name, payload, map_gguf_to_hf, dev, torch.bfloat16, mcfg, {"warned": False})
        del payload
    print(f"[attach] {len(targets)} ml8 fp8 targets", flush=True)

    # one real backward (bf16 teacher vs quantized student) to populate g and h
    model.train()
    Ml8Fp8Fn.capture_dLdW = True       # this diagnostic reads the pv dL/dW side channel
    lg = wrapped(ids)
    m = batch_response_mask(ids, *resp).reshape(-1).to(dev)
    loss = kl_topk(lg.reshape(-1, lg.shape[-1]), *teach, mask=m)
    loss.backward()
    print(f"[bwd] loss={float(loss):.4f}  stashed g={len(Ml8Fp8Fn.last_dLdW)} h={len(Ml8Fp8Fn.last_h)}", flush=True)

    def q(t):  # quantiles
        t = t.flatten().float().abs()
        return [float(t.quantile(x)) for x in (0.5, 0.9, 0.99, 1.0)]

    names = list(targets.keys())
    sample = names[:2] + names[len(names)//2:len(names)//2+1] + names[-1:]
    print("\n" + "=" * 100)
    for nm in sample:
        at = targets[nm]
        key = id(at.indices)
        g = Ml8Fp8Fn.last_dLdW.get(key)
        h = Ml8Fp8Fn.last_h.get(key)
        if g is None or h is None:
            print(f"[skip] {nm}: no stash"); continue
        cent = snap_to_e4m3(at.centroids).detach()      # [G,NC]
        scl = at.scales.detach()                        # [N,G]
        gidx = at.gidx
        N, Kk = at.indices.shape
        NC = cent.shape[1]
        scale_col = scl[:, gidx]                         # [N,K]
        cent_cols = cent[gidx]                           # [K,NC]
        cur = cent_cols.unsqueeze(0).expand(N, -1, -1).gather(
            2, at.indices.long().unsqueeze(-1)).squeeze(-1)   # [N,K]
        dW = (cent_cols.unsqueeze(0) - cur.unsqueeze(-1)) * scale_col.unsqueeze(-1)  # [N,K,NC]
        hb = h.reshape(1, Kk, 1)
        dL_lin = g.unsqueeze(-1) * dW
        dL_quad = g.unsqueeze(-1) * dW + 0.5 * hb * dW * dW
        # bounded: max_step=1 value-rank trust region
        rank = cent.argsort(1).argsort(1)               # [G,NC]
        rank_cols = rank[gidx]                           # [K,NC]
        cur_rank = rank_cols.unsqueeze(0).expand(N, -1, -1).gather(
            2, at.indices.long().unsqueeze(-1)).squeeze(-1)
        allowed = (rank_cols.unsqueeze(0) - cur_rank.unsqueeze(-1)).abs() <= 1
        dL_bnd = torch.where(allowed, dL_quad, torch.full_like(dL_quad, float("inf")))
        best_bnd_dL, bj_bnd = dL_bnd.min(-1)
        # ΔW of would-be flips (improve>0) under each scheme
        dW_quad = dW.gather(2, dL_quad.min(-1).indices.unsqueeze(-1)).squeeze(-1)
        dW_bnd = dW.gather(2, bj_bnd.unsqueeze(-1)).squeeze(-1)
        imp_quad = (-dL_quad.min(-1).values).clamp_min(0)
        imp_bnd = (-best_bnd_dL).clamp_min(0)
        fl_quad = 100.0 * float((imp_quad > 0).sum()) / imp_quad.numel()
        fl_bnd = 100.0 * float((imp_bnd > 0).sum()) / imp_bnd.numel()
        maxdw_quad = float(dW_quad.abs()[imp_quad > 0].max()) if (imp_quad > 0).any() else 0.0
        maxdw_bnd = float(dW_bnd.abs()[imp_bnd > 0].max()) if (imp_bnd > 0).any() else 0.0
        bj_lin = dL_lin.min(-1).indices.reshape(-1)
        bj_quad = dL_quad.min(-1).indices.reshape(-1)
        def ext_pct(bj):
            return 100.0 * float((bj == 0).sum() + (bj == NC - 1).sum()) / bj.numel()
        # codebook spacing in W units (per-group median adjacent gap * median scale)
        gaps = (cent.sort(1).values.diff(dim=1)).abs()
        med_gap_w = float(gaps.median()) * float(scl.median())
        # Newton step magnitude |g/h| (h broadcast over rows), in W units
        newton = (g.abs() / h.reshape(1, Kk).clamp_min(1e-12))   # [N,K]
        print(f"{nm}  [N={N} K={Kk} NC={NC}]")
        print(f"  |g|=dW_raw   q50/q90/q99/max = {q(g)}")
        print(f"   h=E[x^2]    q50/q90/q99/max = {q(h)}   (min={float(h.min()):.3e})")
        print(f"  |Newton g/h| q50/q90/q99/max = {q(newton)}   vs codebook gap(W)≈{med_gap_w:.3e}")
        print(f"  best_j extremes:  linear={ext_pct(bj_lin):5.1f}%   quadratic={ext_pct(bj_quad):5.1f}%   bounded={ext_pct(bj_bnd.reshape(-1)):5.1f}%")
        print(f"  flip%  (improve>0):  quad-unbounded={fl_quad:5.1f}%   bounded(max_step=1)={fl_bnd:5.1f}%")
        print(f"  max |ΔW| of a flip:  quad-unbounded={maxdw_quad:.3e}   bounded={maxdw_bnd:.3e}   (codebook gap≈{med_gap_w:.3e})")
        print("-" * 100, flush=True)


if __name__ == "__main__":
    main()
