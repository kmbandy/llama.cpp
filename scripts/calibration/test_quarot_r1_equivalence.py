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


LLAMA_PERPLEXITY = "/home/kmbandy/GitHub/llama.cpp/build/bin/llama-perplexity"
DEFAULT_WIKITEXT = "/home/kmbandy/wikitext-2-raw/wiki.test.raw"


def run_ppl(gguf_path: str, wikitext_path: str, max_tokens: int) -> float:
    """Run llama-perplexity and return the final PPL value."""
    cmd = [
        LLAMA_PERPLEXITY,
        "--no-mmap",
        "-m", gguf_path,
        "-f", wikitext_path,
        "--ctx-size", "4096",
        "--threads", "8",
        "-n", str(max_tokens),
    ]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"llama-perplexity exited {proc.returncode}")
    # Parse the final "perplexity: X.XXXX ± Y.YYYY" line.
    m = re.search(r"perplexity:\s+([\d.]+)\s+±", proc.stdout)
    if not m:
        m = re.search(r"Final estimate: PPL = ([\d.]+)", proc.stdout)
    if not m:
        print(proc.stdout[-2000:])
        raise RuntimeError("could not parse PPL from llama-perplexity output")
    return float(m.group(1))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--rotated", required=True)
    p.add_argument("--wikitext", default=DEFAULT_WIKITEXT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--tol", type=float, default=0.005)
    args = p.parse_args()

    print(f"[gate] source : {args.source}")
    print(f"[gate] rotated: {args.rotated}")

    ppl_src = run_ppl(args.source,  args.wikitext, args.max_tokens)
    ppl_rot = run_ppl(args.rotated, args.wikitext, args.max_tokens)
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
