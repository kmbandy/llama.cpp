#!/usr/bin/env python3
"""The day-one falsifier for the cross-layer prediction build.

Reads a WP_PREDICT_CAPTURE stream (WPC1 records: the router INPUT h_L and the
ACTUAL selected experts, per layer, per decode-shaped dispatch) and measures,
offline, what the live predictor would have scored at every lookahead depth:
apply router_{L+k} to h_L, take top-M per token, compare against the true
selection at L+k in the SAME step.

This is the k>=2 measurement that has never been taken. The 2026-07-19 basis
(rank1 0.973 / rank2 0.914 / rank3 0.814) is k=1 only; if precision cliffs by
k=2-3 the constant-lead design falls back to k=1, and if it cliffs AT k=1 on
this capture the build's premise is dead and we stop before burning arms.

Capture record (little-endian, see graph_dispatcher::capture_routing):
  u32 magic 'WPC1' (0x31435057)
  i32 layer, n_tokens, n_embd, n_expert_used
  f32 h[n_tokens * n_embd]
  i32 sel[n_tokens * n_expert_used]

Steps are delimited by layer wrap-around (layer <= previous layer), matching
the live emission order: one record per dispatched layer, layers ascending
within a step.

Usage:
  analyze-pred-capture.py --gguf ds4-dense.gguf --capture routing-capture.*.bin \
      [--k 1 2 3 4] [--m 1 2 3 4 6]

Output: per-k recall@M of the union-per-token prediction, overall and split by
layer band, plus the byte-budget framing (mean predicted-set size per step).
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/kmbandy/GitHub/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

MAGIC = 0x31435057


def bf16(t):
    raw = np.asarray(t.data)
    if raw.dtype == np.float32:
        return raw
    u = raw.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def read_records(path):
    """-> list of (layer, h[n_tokens, n_embd] f32, sel[n_tokens, n_used] i32)."""
    out = []
    with open(path, "rb") as f:
        while True:
            head = f.read(20)
            if len(head) < 20:
                break
            magic, layer, n_tokens, n_embd, n_used = struct.unpack("<Iiiii", head)
            if magic != MAGIC:
                raise SystemExit(f"{path}: bad magic {magic:#x} at offset {f.tell() - 20} "
                                 "(truncated or foreign file)")
            h = np.frombuffer(f.read(4 * n_tokens * n_embd), dtype=np.float32)
            sel = np.frombuffer(f.read(4 * n_tokens * n_used), dtype=np.int32)
            if h.size != n_tokens * n_embd or sel.size != n_tokens * n_used:
                print(f"WARNING: {path} ends mid-record; dropping the tail", file=sys.stderr)
                break
            out.append((layer, h.reshape(n_tokens, n_embd), sel.reshape(n_tokens, n_used)))
    return out


def to_steps(records):
    """Split the flat record stream into steps at layer wrap-around."""
    steps, cur, prev = [], {}, None
    for layer, h, sel in records:
        if prev is not None and layer <= prev and cur:
            steps.append(cur)
            cur = {}
        cur[layer] = (h, sel)
        prev = layer
    if cur:
        steps.append(cur)
    return steps


ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--capture", nargs="+", required=True)
ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--m", type=int, nargs="+", default=[1, 2, 3, 4, 6])
a = ap.parse_args()

records = []
for path in a.capture:
    records.extend(read_records(path))
steps = to_steps(records)
print(f"records: {len(records)}  steps: {len(steps)}")
if not steps:
    raise SystemExit("no steps -- was the capture run decode-shaped?")

r = GGUFReader(a.gguf)
tensors = {t.name: t for t in r.tensors}
layers = sorted({layer for step in steps for layer in step})
routers = {}
for layer in layers:
    wname = "blk.%d.ffn_gate_inp.weight" % layer
    bname = "blk.%d.exp_probs_b.bias" % layer
    if wname in tensors and bname in tensors:
        routers[layer] = (bf16(tensors[wname]).astype(np.float32),
                          np.asarray(tensors[bname].data, dtype=np.float32))
print(f"layers seen: {layers[0]}..{layers[-1]}  routers loaded: {len(routers)}")

Ks, Ms = sorted(a.k), sorted(a.m)
# stats[(k, M, band)] = [hits, total]; band 0 = layers < 20, 1 = 20..39, 2 = 40+
stats = {}
pred_set_sizes = {k: [] for k in Ks}
for step in steps:
    for layer, (h, _sel_here) in step.items():
        for k in Ks:
            target = layer + k
            if target not in step or target not in routers:
                continue
            W, b = routers[target]
            _h_t, sel_t = step[target]
            # W from gguf-py is (n_expert, n_embd) row-major -> h @ W.T
            logits = h @ W.T
            probs = np.sqrt(np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0.0))
            scores = probs + b
            order = np.argsort(-scores, axis=1)
            band = 0 if target < 20 else (1 if target < 40 else 2)
            for M in Ms:
                pred = set(order[:, :M].ravel().tolist())
                actual = set(sel_t.ravel().tolist())
                key = (k, M, band)
                hit, tot = stats.get(key, (0, 0))
                stats[key] = (hit + len(pred & actual), tot + len(actual))
                if M == max(Ms):
                    pred_set_sizes[k].append(len(pred))

print("\n== recall@M of union-per-token prediction, by lookahead k ==")
hdr = "k    " + "  ".join("R@%-3d" % M for M in Ms) + "   bands(<20 | 20-39 | 40+) at M=%d" % Ms[-1]
print(hdr)
for k in Ks:
    cells = []
    for M in Ms:
        h = sum(stats.get((k, M, b), (0, 0))[0] for b in range(3))
        t = sum(stats.get((k, M, b), (0, 0))[1] for b in range(3))
        cells.append("%5.1f" % (100.0 * h / t if t else 0.0))
    bands = []
    for b in range(3):
        h, t = stats.get((k, Ms[-1], b), (0, 0))
        bands.append("%5.1f" % (100.0 * h / t if t else 0.0))
    sizes = pred_set_sizes[k]
    mean_sz = sum(sizes) / len(sizes) if sizes else 0.0
    print("%-4d %s   %s   mean-union %.1f experts/layer" %
          (k, "  ".join(cells), " | ".join(bands), mean_sz))
print("\nREAD THIS AGAINST: k=1 basis rank1 0.973 / rank2 0.914 / rank3 0.814 "
      "(2026-07-19); random = M*n_tokens-union / 256.")
