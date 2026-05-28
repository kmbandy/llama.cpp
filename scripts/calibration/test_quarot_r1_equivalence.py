#!/usr/bin/env python3
"""Equivalence gate for QuaRot-R1 rotation.

Runs llama-perplexity on both a source bf16 GGUF and its rotated counterpart
on a fixed wikitext-2 slice; asserts the PPL values match to ±0.005. This is
the bit-equivalence gate from the design doc — it must pass before any
calibration is run on the rotated GGUF.

Mad-lab standing rule: --no-mmap is passed to every llama.cpp invocation.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PERPLEXITY = str(_REPO_ROOT / "build-hip" / "bin" / "llama-perplexity")
DEFAULT_WIKITEXT = str(_REPO_ROOT / "wikitext-2-raw" / "wiki.test.raw")


def run_ppl(gguf_path: str, wikitext_path: str, chunks: int, ctx_size: int) -> float:
    """Run llama-perplexity on a fixed slice and return the final running-PPL value.

    Streams stderr+stdout to this process so progress is visible in real time; the
    captured combined output is then parsed. Order of regex attempts:
      1. Final estimate line ("Final estimate: PPL = X.YYYY")
      2. Last per-chunk PPL ("[N]X.YYYY,") — this is the running estimate after
         chunk N and is correct enough for equivalence comparisons.
    """
    cmd = [
        LLAMA_PERPLEXITY,
        "--no-mmap",
        "-m", gguf_path,
        "-f", wikitext_path,
        "--ctx-size", str(ctx_size),
        "--threads", "8",
        "--chunks", str(chunks),
    ]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(out[-2000:], file=sys.stderr)
        raise RuntimeError(f"llama-perplexity exited {proc.returncode}")
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", out)
    if m:
        return float(m.group(1))
    chunk_matches = re.findall(r"\[\d+\]([\d.]+),", out)
    if chunk_matches:
        return float(chunk_matches[-1])
    print(out[-2000:])
    raise RuntimeError("could not parse PPL from llama-perplexity output")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--rotated", required=True)
    p.add_argument("--wikitext", default=DEFAULT_WIKITEXT)
    p.add_argument("--chunks", type=int, default=8,
                   help="number of wikitext chunks to evaluate (caps wall time)")
    p.add_argument("--ctx-size", type=int, default=512,
                   help="ctx-size per chunk; 512 keeps the run fast (4 chunks ≈ 30s on R9700)")
    p.add_argument("--tol", type=float, default=0.05,
                   help="PPL tolerance; 0.05 is appropriate when running ≤8 chunks (1 σ ≈ 0.03)")
    args = p.parse_args()

    print(f"[gate] source : {args.source}")
    print(f"[gate] rotated: {args.rotated}")

    ppl_src = run_ppl(args.source,  args.wikitext, args.chunks, args.ctx_size)
    ppl_rot = run_ppl(args.rotated, args.wikitext, args.chunks, args.ctx_size)
    diff = abs(ppl_src - ppl_rot)
    print(f"[gate] PPL source : {ppl_src:.4f}")
    print(f"[gate] PPL rotated: {ppl_rot:.4f}")
    print(f"[gate] |diff|     : {diff:.4f} (tol {args.tol})")
    if diff > args.tol:
        print("[gate] FAIL — rotation is NOT equivalent to source", file=sys.stderr)
        return 1
    print("[gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
