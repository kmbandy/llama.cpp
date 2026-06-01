#!/usr/bin/env python3
"""CPU-only verification of --token-budget draws (warms the budget-mode NVMe cache)."""
import sys, time
sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/scripts/calibration")
from transformers import AutoTokenizer
from calib_corpus import collect_calibration

MODEL = "/home/kmbandy/models/Qwen3.5-0.8B-hf"
BUDGET, SEQ_LEN, SEED = 20966, 2048, 0
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print(f"{'corpus':10s} {'n_samp':>7s} {'tot_tok':>9s} {'Δ%':>7s} {'secs':>6s}")
print("-" * 44)
wiki_ids = None
for corpus in ["wiki", "mix", "code", "math", "chat"]:
    t0 = time.time()
    c = collect_calibration(tok, n_samples=128, seq_len=SEQ_LEN, composition=corpus,
                            seed=SEED, token_budget=BUDGET)
    tot = sum(t.numel() for t in c)
    dpct = 100.0 * (tot - BUDGET) / BUDGET
    print(f"{corpus:10s} {len(c):7d} {tot:9d} {dpct:+6.1f}% {time.time()-t0:6.1f}")
    if corpus == "wiki":
        wiki_ids = c

# Neutrality: budget-mode wiki must equal the plain n=128 wiki draw (same 35 rows, same order).
plain = collect_calibration(tok, n_samples=128, seq_len=SEQ_LEN, composition="wiki", seed=SEED)
same = (len(plain) == len(wiki_ids)
        and all(a.shape == b.shape and (a == b).all() for a, b in zip(plain, wiki_ids)))
print(f"\nwiki budget-mode == plain n=128 draw: {same} "
      f"(budget {len(wiki_ids)} samples, plain {len(plain)})")
