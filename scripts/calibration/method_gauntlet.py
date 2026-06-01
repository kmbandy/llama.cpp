#!/usr/bin/env python3
"""ml8 method gauntlet — local model-level PPL sweep of corpus + heavy-fine-tune levers.

Calibrate -> convert (full coverage, --mtp-fp8) -> llama-perplexity, per recipe cell, and
log model-level PPL + GGUF size vs the UD-Q4_K_XL / bf16 references. Designed for the cheap
0.8B dense test bed (see docs/superpowers/2026-05-31-ml8-method-gauntlet.md) before spending
the MI300X pod; the same harness scales to 2B/4B by passing --model/--base/--ud.

Staged main-effects design (non-adaptive so it runs unattended):
  stage1  token count   : n_samples {32,128,512} @ seq_len 1024
  stage2  seq_len        : token-matched ~131k {128x1024, 64x2048, 32x4096}
  stage3  heavy fine-tune: heavy_rounds {0,4,8} (+ heavy_steps 120) @ n128/s2048
  stage4  secondary      : snap e4m3 / mag_weighted / group_size_down=32 @ n128/s2048 heavy=4

Usage:
  python3 method_gauntlet.py --stage 1            # run one stage
  python3 method_gauntlet.py --all                # run all stages
  python3 method_gauntlet.py --list               # preview cells, run nothing
Results append to <workdir>/results.json + results.md; completed cells are skipped on re-run.
GGUFs are deleted after measuring; blob dirs are kept (resume / inspection).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/kmbandy/GitHub/llama.cpp")
PPL_BIN = ROOT / "build-hip/bin/llama-perplexity"
WIKI = ROOT / "wikitext-2-raw/wiki.test.raw"
GGUF_PY = ROOT / "gguf-py"

# ── per-model config (override via CLI for 2B/4B) ───────────────────────────
DEFAULTS = dict(
    model="/home/kmbandy/models/Qwen3.5-0.8B-hf",
    base="/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf",
    ud="/home/kmbandy/Downloads/Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    arch="qwen35",
    workdir="/home/kmbandy/models/gauntlet-0p8b",
    ref_bf16=18.37, ref_ud=18.50,   # measured 2026-05-31, 8 chunks c512
)

# Fixed base recipe = the A0 command that produced PPL 19.40 (rotation kronecker, gs64, nc16,
# percdamp 0.01, fit mse, snap none[convert snaps to e4m3], heavy 0, 32x1024).
# NOTE: --device is injected per-run from cfg["cal_device"] (default cuda:0 = R9700), so the
# calibrate step can be pinned to cuda:1 (6900 XT) for parallel sample-cranking.
BASE = {"--rotation": "kronecker", "--group-size": "64", "--n-centroids": "16",
        "--percdamp": "0.01", "--fit-loss": "mse", "--dense-coverage": "full",
        "--n-samples": "32", "--seq-len": "1024"}

# ── the staged cells (name -> overrides on BASE) ────────────────────────────
STAGES = {
    1: [  # token count
        ("s1_n32_sl1024",  {}),                                   # == A0 baseline
        ("s1_n128_sl1024", {"--n-samples": "128"}),
        ("s1_n512_sl1024", {"--n-samples": "512"}),
    ],
    2: [  # seq_len, token-matched ~131k
        ("s2_n128_sl1024", {"--n-samples": "128", "--seq-len": "1024"}),
        ("s2_n64_sl2048",  {"--n-samples": "64",  "--seq-len": "2048"}),
        ("s2_n32_sl4096",  {"--n-samples": "32",  "--seq-len": "4096"}),
    ],
    3: [  # heavy fine-tune @ a solid base (n128/s2048 ~262k tokens). The decisive
          # question: does our heavy FT produce a MODEL-LEVEL PPL gain (never measured;
          # we only have matrix-level +0.3dB Y_SNR)? The heavy-on minus heavy-off delta
          # is leakage-robust (both calibrate+eval on wikitext equally) so this answers
          # the FT question even before the corpus/eval-leakage fix.
        ("s3_heavy0",      {"--n-samples": "128", "--seq-len": "2048"}),                                                 # off baseline
        ("s3_heavy4",      {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True}),      # rounds
        ("s3_heavy8",      {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "8", "--act-order": True}),
        ("s3_heavy4_steps120", {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--heavy-steps": "120", "--act-order": True}),  # steps
        # LR sweep — the dominant un-tuned knob (default 1e-2 was a never-swept guess).
        # Bracket the centroid LR half an order each way on the rounds-4 base.
        ("s3_heavy4_lrc3e2", {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True, "--heavy-lr-cent": "3e-2"}),
        ("s3_heavy4_lrc3e3", {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True, "--heavy-lr-cent": "3e-3"}),
    ],
    4: [  # secondary levers on the heavy base
        ("s4_snap_e4m3",   {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True, "--snap-centroids": "e4m3"}),
        ("s4_magweighted", {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True, "--fit-loss": "mag_weighted"}),
        ("s4_gsdown32",    {"--n-samples": "128", "--seq-len": "2048", "--heavy-rounds": "4", "--act-order": True, "--group-size-down": "32"}),
    ],
    5: [  # corpus CONTENT @ fixed 262k (n128/sl2048), heavy off. The composition lever —
          # which is more transferable across token budgets than size, so settle it at a
          # fixed budget. Read = best AVERAGE across wikitext + the never-train held-out
          # eval (--holdout-eval), which also de-biases vs UD's leakage-free measurement.
        ("c_wiki", {"--n-samples": "128", "--seq-len": "2048", "--corpus": "wiki"}),   # control
        ("c_mix",  {"--n-samples": "128", "--seq-len": "2048", "--corpus": "mix"}),    # Unsloth analog
        ("c_code", {"--n-samples": "128", "--seq-len": "2048", "--corpus": "code"}),
        ("c_math", {"--n-samples": "128", "--seq-len": "2048", "--corpus": "math"}),
        ("c_chat", {"--n-samples": "128", "--seq-len": "2048", "--corpus": "chat"}),
    ],
    6: [  # W4A8 deployment-faithful calibration — paired toggle vs the fla zero-point
          # 19.2678. q1_off is recipe-identical to stage-5 c_wiki, so it MUST reproduce
          # 19.2678 (Gate C, refactor-neutrality). q2/q3/q4 add the faithful tiers + heavy
          # on top with the SAME corpus seed, so the paired Δ cancels run-to-run noise.
        ("q1_off",    {"--n-samples": "128", "--seq-len": "2048", "--corpus": "wiki"}),
        ("q2_acts",   {"--n-samples": "128", "--seq-len": "2048", "--corpus": "wiki",
                       "--faithful-acts": True}),
        ("q3_actswt", {"--n-samples": "128", "--seq-len": "2048", "--corpus": "wiki",
                       "--faithful-acts": True, "--faithful-weights": True}),
        ("q4_heavy",  {"--n-samples": "128", "--seq-len": "2048", "--corpus": "wiki",
                       "--faithful-acts": True, "--faithful-weights": True,
                       "--heavy-rounds": "4", "--act-order": True}),
    ],
}


def recipe_args(overrides):
    r = dict(BASE); r.update(overrides)
    args = []
    for k, v in r.items():
        if v is True:
            args.append(k)            # store_true flag
        elif v is False or v is None:
            continue                  # store_true flag left off
        else:
            args += [k, str(v)]
    return args


def parse_ppl(text):
    for line in text.splitlines():
        if "Final estimate: PPL =" in line:
            try:
                return float(line.split("PPL =")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def run_cell(name, overrides, cfg):
    work = Path(cfg["workdir"]); work.mkdir(parents=True, exist_ok=True)
    out_dir = work / name; gguf = work / f"{name}.gguf"
    marker = out_dir / ".gauntlet_calib_ok"     # written only on a clean calibrate
    env = {"PYTHONPATH": str(GGUF_PY)}
    import os; env = {**os.environ, **env}

    t0 = time.time()
    # 1. calibrate (on cfg["cal_device"]). Skip if a completed calibration's blobs are
    #    already present — lets the 6900 XT (cuda:1) crank blobs that the R9700 later
    #    converts+ppls without redoing the heavy forward pass.
    if marker.exists():
        print(f"[calib-cached] {name} (reusing blobs in {out_dir.name})")
    else:
        cal = subprocess.run(
            [sys.executable, str(ROOT / "scripts/calibration/calibrate_ml8_paged.py"),
             "--model", cfg["model"], "--gguf", cfg["base"], "--arch", cfg["arch"],
             "--device", cfg["cal_device"],
             "--output-dir", str(out_dir)] + recipe_args(overrides),
            cwd=ROOT, env=env, capture_output=True, text=True)
        if cal.returncode != 0:
            return {"name": name, "ppl": None, "size_mb": None, "status": "CALIB_FAIL",
                    "tail": cal.stderr.strip().splitlines()[-3:]}
        marker.write_text(json.dumps({"recipe": overrides, "cal_device": cfg["cal_device"]}))

    # calib-only: stop here, keep blobs, leave convert+ppl for the R9700 pass.
    if cfg.get("calib_only"):
        return {"name": name, "ppl": None, "size_mb": None, "status": "CALIB_OK",
                "secs": round(time.time() - t0), "out_dir": out_dir.name}

    # 2. convert
    conv = subprocess.run(
        [sys.executable, str(ROOT / "scripts/calibration/ml8_to_gguf.py"),
         "--base-gguf", cfg["base"], "--calib-dir", str(out_dir),
         "--out-gguf", str(gguf), "--allow-partial"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if conv.returncode != 0 or not gguf.exists():
        return {"name": name, "ppl": None, "size_mb": None, "status": "CONVERT_FAIL",
                "tail": conv.stderr.strip().splitlines()[-3:]}
    size_mb = round(gguf.stat().st_size / 1048576)
    # 3. perplexity on cfg["ppl_device"] (stderr merged — llama.cpp logs there)
    def _ppl(eval_path):
        r = subprocess.run(
            [str(PPL_BIN), "--no-mmap", "-m", str(gguf), "-ngl", "99",
             "--device", cfg["ppl_device"], "-f", str(eval_path), "-c", "512", "--chunks", "8"],
            cwd=ROOT, capture_output=True, text=True)
        return parse_ppl(r.stdout + r.stderr)
    ppl = _ppl(WIKI)
    # Optional held-out (never-train) eval — detects calibration over-fit / eval leakage.
    ppl_holdout = _ppl(cfg["holdout_txt"]) if cfg.get("holdout_txt") else None
    gguf.unlink(missing_ok=True)                # keep blobs, drop the big gguf
    return {"name": name, "ppl": ppl, "ppl_holdout": ppl_holdout, "size_mb": size_mb,
            "status": "OK" if ppl else "PPL_FAIL", "secs": round(time.time() - t0)}


def write_results(results, cfg):
    work = Path(cfg["workdir"])
    (work / "results.json").write_text(json.dumps(results, indent=2))
    lines = ["# ml8 method gauntlet results", "",
             f"refs: bf16 {cfg['ref_bf16']} | UD {cfg['ref_ud']} | beat UD = PPL < {cfg['ref_ud']}", "",
             "| cell | PPL | Δ vs UD | held-out PPL | size MB | status | secs |",
             "|---|---|---|---|---|---|---|"]
    for r in results.values():
        d = f"{r['ppl']-cfg['ref_ud']:+.3f}" if r.get("ppl") else "—"
        ho = r.get("ppl_holdout") or "—"
        lines.append(f"| {r['name']} | {r.get('ppl','—')} | {d} | {ho} | "
                     f"{r.get('size_mb','—')} | {r['status']} | {r.get('secs','—')} |")
    (work / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=list(STAGES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cal-device", default="cuda:0",
                    help="torch device for the calibrate step (cuda:0=R9700, cuda:1=6900 XT)")
    ap.add_argument("--ppl-device", default="ROCm0",
                    help="llama.cpp device for perplexity (ROCm0=R9700; gfx1030 can't run ml8)")
    ap.add_argument("--calib-only", action="store_true",
                    help="only calibrate (crank blobs on the 6900 XT); skip convert+ppl")
    ap.add_argument("--holdout-eval", action="store_true",
                    help="also eval each cell on the never-train held-out set (quant_so_eval) "
                         "— the leakage-free read; recommended for the content stage (5)")
    ap.add_argument("--cell", default=None,
                    help="run only the named cell(s) within the selected stage/--all "
                         "(comma-separated). Use with a fresh --workdir to force a clean "
                         "re-calibrate (e.g. fla/QAT attribution runs).")
    for k in ("model", "base", "ud", "arch", "workdir"):
        ap.add_argument(f"--{k}", default=DEFAULTS[k])
    args = ap.parse_args()
    cfg = {**DEFAULTS, **{k: getattr(args, k) for k in ("model", "base", "ud", "arch", "workdir")}}
    cfg["cal_device"] = args.cal_device
    cfg["ppl_device"] = args.ppl_device
    cfg["calib_only"] = args.calib_only
    cfg["holdout_txt"] = None
    if args.holdout_eval and not args.list:
        sys.path.insert(0, str(ROOT / "scripts/calibration"))
        from calib_corpus import write_holdout_eval_txt
        Path(cfg["workdir"]).mkdir(parents=True, exist_ok=True)
        cfg["holdout_txt"] = write_holdout_eval_txt(Path(cfg["workdir"]) / "holdout_quant_so.txt")

    cells = []
    if args.all:
        for s in sorted(STAGES): cells += STAGES[s]
    elif args.stage:
        cells = STAGES[args.stage]
    else:
        cells = STAGES[1]
    if args.cell:
        wanted = set(args.cell.split(","))
        cells = [(n, o) for (n, o) in cells if n in wanted]
        if not cells:
            print(f"[error] no cells match --cell {args.cell!r} in the selected stage"); return
    if args.list:
        for n, o in cells: print(f"  {n}: {o}")
        return

    rpath = Path(cfg["workdir"]) / "results.json"
    results = json.loads(rpath.read_text()) if rpath.exists() else {}
    done_status = "CALIB_OK" if args.calib_only else "OK"
    for name, ov in cells:
        if name in results and results[name].get("status") == done_status:
            print(f"[skip] {name} (done: {results[name].get('status')} "
                  f"PPL={results[name].get('ppl')})"); continue
        print(f"[run ] {name} {ov}")
        results[name] = run_cell(name, ov, cfg)
        print(f"[done] {name}: {results[name]}")
        write_results(results, cfg)


if __name__ == "__main__":
    main()
