#!/usr/bin/env python3
"""CPU-only token-budget measurement for the content sweep.

Loads the tokenizer and runs the REAL collect_calibration code paths (same seed as
calibrate_ml8_paged uses) to report, per corpus: number of samples actually returned,
total tokens, and mean tokens/sample. Grounds the token-matched n_samples for Q1.

No GPU, no model weights — tokenizer + JSONL sampling only.
"""
import sys, time
sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/scripts/calibration")
from transformers import AutoTokenizer
from calib_corpus import collect_calibration

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"
SEQ_LEN = 2048
N = 128
SEED = 0

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print(f"{'corpus':10s} {'n_ret':>6s} {'tot_tok':>9s} {'mean_tok':>9s} {'n_for_77k':>10s} {'secs':>6s}")
print("-" * 60)
TARGET = None
for corpus in ["wiki", "mix", "code", "math", "chat"]:
    t0 = time.time()
    try:
        calib = collect_calibration(tok, n_samples=N, seq_len=SEQ_LEN,
                                    composition=corpus, seed=SEED)
    except Exception as e:
        print(f"{corpus:10s} ERROR {e}")
        continue
    n_ret = len(calib)
    tot = sum(c.numel() for c in calib)
    mean = tot / max(n_ret, 1)
    if corpus == "wiki":
        TARGET = tot  # wiki@128 IS the q3 budget — match everything to this
    n_for_target = round(TARGET / mean) if (TARGET and mean) else 0
    print(f"{corpus:10s} {n_ret:6d} {tot:9d} {mean:9.1f} {n_for_target:10d} {time.time()-t0:6.1f}")

print(f"\nTARGET budget (wiki@{N}x{SEQ_LEN}) = {TARGET} tokens")
