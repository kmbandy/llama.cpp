"""Sweep ml8 GEMV kernel configs (G.6.h M1).

Iterates (BN, K_COOP, USE_LDS_A, LAYOUT) over the legal grid, sets
ML8_USE_GEMV=1 and the four ML8_GEMV_* env vars, runs llama-completion
with 32 decode tokens, parses the eval t/s from common_perf_print.

Outputs a sorted table; saves full results to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


LLAMA = "/home/kmbandy/GitHub/llama.cpp/build-hip/bin/llama-completion"
MODEL = "/home/kmbandy/models/Qwen3.5-4B-ml8_4-cellE.gguf"


def run_config(bn: int, kc: int, lds: int, layout: int, prompt: str, n_predict: int = 32, n_runs: int = 2) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/kmbandy/GitHub/triton/python"
    env["ML8_USE_GEMV"]     = "1"
    env["ML8_GEMV_BN"]      = str(bn)
    env["ML8_GEMV_K_COOP"]  = str(kc)
    env["ML8_GEMV_LDS_A"]   = str(lds)
    env["ML8_GEMV_LAYOUT"]  = str(layout)

    cmd = [
        LLAMA, "-m", MODEL, "-ngl", "99", "--device", "ROCm0",
        "-p", prompt, "-n", str(n_predict),
        "--no-warmup", "-no-cnv", "--no-mmap",
    ]
    ts_runs = []
    for _ in range(n_runs):
        try:
            res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
            combined = res.stdout + "\n" + res.stderr
            m = re.search(r"eval time\s*=\s*[\d.]+\s*ms\s*/\s*\d+\s*runs\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)", combined)
            if m:
                ts_runs.append(float(m.group(1)))
            else:
                # GEMV dispatch miss → fail entry
                if "no template matches" in combined:
                    return {"bn": bn, "kc": kc, "lds": lds, "layout": layout,
                            "ts": None, "err": "dispatch_miss"}
                return {"bn": bn, "kc": kc, "lds": lds, "layout": layout,
                        "ts": None, "err": "parse_fail"}
        except subprocess.TimeoutExpired:
            return {"bn": bn, "kc": kc, "lds": lds, "layout": layout,
                    "ts": None, "err": "timeout"}
        except Exception as e:
            return {"bn": bn, "kc": kc, "lds": lds, "layout": layout,
                    "ts": None, "err": str(e)}

    return {
        "bn": bn, "kc": kc, "lds": lds, "layout": layout,
        "ts": max(ts_runs),  # best of N (warm cache + outlier resistance)
        "ts_all": ts_runs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/kmbandy/GitHub/llama.cpp/tests/perf-baseline/ppl-kv/gemv_sweep_m1.json"))
    ap.add_argument("--n-predict", type=int, default=32)
    args = ap.parse_args()

    prompt = "Hello, my name is"

    # Pruned legal grid.
    BNS    = [8, 16, 32, 64, 128]
    KCS    = [1, 2, 4, 8]
    LDS    = [0, 1]
    LAYOUT = [0, 1]   # only meaningful when KC > 1

    configs = []
    for bn in BNS:
        for kc in KCS:
            if bn * kc > 1024:
                continue
            if bn * kc < 16:   # too few threads to fill a wave well
                continue
            for lds in LDS:
                if kc == 1:
                    configs.append((bn, kc, lds, 0))   # LAYOUT irrelevant
                else:
                    for lay in LAYOUT:
                        configs.append((bn, kc, lds, lay))

    print(f"=== ml8 GEMV M1 sweep: {len(configs)} configs ===")
    print(f"  baseline (Triton M=16 path, no GEMV): ~20.30 t/s decode")
    print(f"  f16 reference: ~50.89 t/s decode")
    print()

    t0 = time.time()
    results = []
    for i, (bn, kc, lds, lay) in enumerate(configs):
        r = run_config(bn, kc, lds, lay, prompt, args.n_predict)
        results.append(r)
        ts_str = f"{r['ts']:.2f}" if r.get("ts") is not None else (r.get("err") or "?")
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(configs)}]  BN={bn:3d}  KC={kc}  LDS={lds}  LAY={lay}  ->  {ts_str:>10}  t/s   ({elapsed:.0f}s elapsed)")

    args.out.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("ts") is not None]
    ok.sort(key=lambda r: -r["ts"])
    print()
    print("=== TOP 15 ===")
    print(f"{'rank':>4}  {'BN':>3}  {'KC':>2}  {'LDS':>3}  {'LAY':>3}  {'t/s':>8}")
    for i, r in enumerate(ok[:15]):
        print(f"{i+1:>4}  {r['bn']:>3}  {r['kc']:>2}  {r['lds']:>3}  {r['layout']:>3}  {r['ts']:>8.2f}")

    if ok:
        winner = ok[0]
        print(f"\n  WINNER: BN={winner['bn']}, KC={winner['kc']}, LDS={winner['lds']}, LAYOUT={winner['layout']}  -> {winner['ts']:.2f} t/s")
        print(f"  vs baseline 20.30: {winner['ts']/20.30:.2f}x")
        print(f"  vs f16 50.89:      {winner['ts']/50.89:.2f}x (target 1.0+)")


if __name__ == "__main__":
    main()
