#!/usr/bin/env python3
"""How accurate is EMBEDDING-ONLY routing at each layer depth?

The deep-layer prefetch design (2026-08-02 scoping, revisited 2026-08-06)
rests on one claim: a drafted token ID gives its embedding exactly, so
applying layer L's router to the raw embedding approximates layer L's true
routing, with accuracy decaying in depth. This measures that decay against
ground truth, offline, from a SPEC=0 capture run (strict one-token-per-step
so REF-log decode segments align 1:1 with generated tokens).

Replicates build_moe_ffn for DS4: logits = W_l @ e; probs =
sqrt(softplus(logits)); selection = top-k of (probs + exp_probs_b). The
actual experts come from the workers' WP_REF_LOG streams (union across
workers; each worker only sees its shard).

Usage:
  probe-embed-routing.py --gguf ds4-dense.gguf --gen-json gen.json \
      --ref ref-w-r9700.txt ref-w-1070.txt ref-w-480.txt [--m 8 16 32]

Output: per-layer recall@M (all experts, and the 85-255 R9700 shard alone).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

HASH_LAYERS = 3  # layers 0..2 route by token id (tid2eid); already hinted exactly


def bf16(t):
    """gguf-py returns BF16 tensors as uint8 rows of n_embd*2 bytes."""
    raw = np.asarray(t.data)
    if raw.dtype == np.float32:
        return raw
    u = raw.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def load_ref(path):
    """-> list of (layer, set(experts)) in stream order."""
    out = []
    for line in open(path):
        f = line.split()
        if len(f) < 2:
            continue
        # skip the trailing nt=<n_tokens> phase column (2026-08-08, D5) -- it is
        # not an expert id, and int() on a bare trailing integer would silently
        # inject a phantom expert into every step's set.
        out.append((int(f[0]), set(int(x) for x in f[1:] if not x.startswith("nt="))))
    return out


def segments(refs):
    """Split a ref stream into steps at layer wrap-around."""
    steps, cur, prev = [], {}, None
    for layer, ex in refs:
        if prev is not None and layer < prev and cur:
            steps.append(cur)
            cur = {}
        cur.setdefault(layer, set()).update(ex)
        prev = layer
    if cur:
        steps.append(cur)
    return steps


ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--gen-json")
ap.add_argument("--tokens-npy", help="per-step token ids recovered from the "
                "hash-layer sets via tid2eid (-1 = unrecovered, step skipped); "
                "alignment is exact by construction")
ap.add_argument("--ref", nargs="+", required=True)
ap.add_argument("--m", type=int, nargs="+", default=[8, 16, 32])
a = ap.parse_args()

if a.tokens_npy:
    tokens = np.load(a.tokens_npy).tolist()
else:
    tokens = json.load(open(a.gen_json))["tokens"]
print("step tokens: %d" % len(tokens))

r = GGUFReader(a.gguf)
tensors = {t.name: t for t in r.tensors}
embd = tensors["token_embd.weight"]
n_layers = 1 + max(
    int(n.split(".")[1]) for n in tensors if n.startswith("blk.") and "ffn_gate_inp" in n)
print("layers with routers: %d" % n_layers)

# Merge each worker's stream into per-step per-layer actual sets. Workers see
# disjoint shards of the same request sequence, so their step counts match.
per_worker = [segments(load_ref(p)) for p in a.ref]
n_steps = min(len(s) for s in per_worker)
# First segment is the prefill sweep (one giant union per layer); drop it.
decode = []
for i in range(1, n_steps):
    merged = {}
    for s in per_worker:
        for layer, ex in s[i].items():
            merged.setdefault(layer, set()).update(ex)
    decode.append(merged)
print("decode steps: %d (tokens %d)" % (len(decode), len(tokens)))
n = min(len(decode), len(tokens))
if abs(len(decode) - len(tokens)) > 2:
    print("WARNING: step/token count mismatch > 2 -- alignment suspect")

emb_all = bf16(embd)          # (vocab, 4096)
Ms = sorted(a.m)
stats = {}                     # layer -> M -> [hit, total] ; plus '9700' variant
for L in range(HASH_LAYERS, n_layers):
    W = bf16(tensors["blk.%d.ffn_gate_inp.weight" % L])   # (256, 4096)
    b = np.asarray(tensors["blk.%d.exp_probs_b.bias" % L].data, dtype=np.float32)
    sel_all = {}
    for i in range(n):
        if tokens[i] < 0:
            continue
        e = emb_all[tokens[i]].astype(np.float32)
        logits = W.astype(np.float32) @ e
        probs = np.sqrt(np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0))
        sel = probs + b
        order = np.argsort(-sel)
        actual = decode[i].get(L)
        if not actual:
            continue
        for M in Ms:
            pred = set(order[:M].tolist())
            key = (L, M, "all")
            h, t = stats.get(key, (0, 0))
            stats[key] = (h + len(pred & actual), t + len(actual))
            act97 = {x for x in actual if x >= 85}
            if act97:
                pred97 = {x for x in pred if x >= 85}
                key = (L, M, "r97")
                h, t = stats.get(key, (0, 0))
                stats[key] = (h + len(pred97 & act97), t + len(act97))

print("\nlayer  " + "  ".join("R@%-3d" % M for M in Ms) + "   (all | r9700-shard)")
band = {M: [0, 0] for M in Ms}
band97 = {M: [0, 0] for M in Ms}
for L in range(HASH_LAYERS, n_layers):
    cells = []
    for M in Ms:
        h, t = stats.get((L, M, "all"), (0, 0))
        h9, t9 = stats.get((L, M, "r97"), (0, 0))
        band[M][0] += h; band[M][1] += t
        band97[M][0] += h9; band97[M][1] += t9
        cells.append("%4.1f|%4.1f" % (100.0 * h / t if t else 0,
                                      100.0 * h9 / t9 if t9 else 0))
    print("%5d  %s" % (L, "  ".join(cells)))
print("\nTOTAL  " + "  ".join(
    "%4.1f|%4.1f" % (100.0 * band[M][0] / band[M][1] if band[M][1] else 0,
                     100.0 * band97[M][0] / band97[M][1] if band97[M][1] else 0)
    for M in Ms))
