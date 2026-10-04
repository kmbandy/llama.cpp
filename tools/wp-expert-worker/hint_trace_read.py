#!/usr/bin/env python3
"""Reader for WP_HINT_TRACE files written by the pipeline spine
(graph_dispatcher::trace_layer / trace_loop in src/pipeline/pipe-expert-dispatch-graph.cpp).

Pure numpy. Usage:  hint_trace_read.py TRACE.bin [--max-m 6] [--max-d 6]

Format (little-endian):
  header (48 B): "WPHT1\\0\\0\\0" u32 version u32 n_expert u32 n_embd u32 trace_k
                 u32 top_n u32 pid u32 reserved[4]
  record: u32 type, u32 payload_bytes, payload
    1 ROUTING: u32 step i32 layer u32 n_rows u32 k u32 phantom_mask u32 flags
               i32 ids[n_rows][k] f32 gate[n_rows][k]
    2 PRED:    u32 step i32 src_layer u32 n_rows u32 n_dist u32 top_n
               i32 dist[n_dist]; then [n_dist][n_rows]: u16 ids[top_n] f32 score[top_n] f32 prob[top_n]
    3 FOOTER:  u64 dropped_pred u64 dropped_all
    4 DRAFT:   u32 step u32 n_slots u32 n_embd_h u32 flags; i32 token[n_slots] f32 conf[n_slots]
               i32 anchor; f32 hidden[n_slots][n_embd_h]  (n_embd_h may be 0)
               flags bit0 hidden present, bit1 hidden = post-norm LM-head input, bits8..15 seq id.
               token -1 = slot not drafted. Slot i's token is verify row i+1 (row 0 = anchor).
               `step` = the draft decode's step; its verify is the next target decode (step+1).
    5 BATCH:   u32 step u32 n_rows i32 pos0 i32 token[n_rows]   (input tokens of each <=32-row decode)
  Row 0 = committed token, rows 1..n = drafted. dist = target_layer - src_layer.
  Only decode/verify ubatches (<= 32 rows) are traced; prefill is skipped.
  A step is one begin_decode() call (draft-model decodes, if they share the
  dispatcher, also advance it). Accepted-count is not known on the spine and is
  not recorded.
"""
import argparse
import struct
import sys

import numpy as np

HDR = 48


class Trace:
    def __init__(self):
        self.n_expert = self.n_embd = self.trace_k = self.top_n = self.pid = 0
        self.routing = {}   # (step, layer) -> dict(ids[n_rows,k] i32, gate[n_rows,k] f32, phantom u32)
        self.pred = {}      # (step, src_layer) -> dict(dist[n_dist], ids[n_dist,n_rows,top_n] u16, score, prob f32)
        self.dropped_pred = self.dropped_all = 0
        self.draft = {}     # step -> dict(tokens[n_slots] i32, conf f32, anchor, hidden[n_slots,n_embd_h] f32|None, flags, seq)
        self.draft_all = []  # every DRAFT record in file order (multi-seq: several per step)
        self.batch = {}     # step -> dict(tokens[n_rows] i32, pos0)

    def verify_step_of_draft(self, step):
        """Step of the target verify batch that consumes the DRAFT record at `step`:
        the first later step whose BATCH starts with the anchor token and has >1 rows.
        Falls back to step+1."""
        d = self.draft.get(step)
        for s in range(step + 1, step + 8):
            b = self.batch.get(s)
            if b is not None and len(b["tokens"]) > 1 and d is not None and b["tokens"][0] == d["anchor"]:
                return s
        return step + 1


def load(path):
    buf = open(path, "rb").read()
    if buf[:5] != b"WPHT1":
        raise ValueError("not a WP_HINT_TRACE file")
    t = Trace()
    (_ver, t.n_expert, t.n_embd, t.trace_k, t.top_n, t.pid) = struct.unpack_from("<6I", buf, 8)
    pos = HDR
    while pos + 8 <= len(buf):
        rtype, nbytes = struct.unpack_from("<II", buf, pos)
        pos += 8
        if pos + nbytes > len(buf):
            break   # truncated tail (run killed mid-write)
        p = pos
        if rtype == 1:
            step, layer, n_rows, k, phantom, _flags = struct.unpack_from("<IiIIII", buf, p)
            p += 24
            n = n_rows * k
            ids = np.frombuffer(buf, "<i4", n, p).reshape(n_rows, k)
            gate = np.frombuffer(buf, "<f4", n, p + 4 * n).reshape(n_rows, k)
            t.routing[(step, layer)] = dict(ids=ids, gate=gate, phantom=phantom)
        elif rtype == 2:
            step, src, n_rows, n_dist, top_n = struct.unpack_from("<IiIII", buf, p)
            p += 20
            dist = np.frombuffer(buf, "<i4", n_dist, p)
            p += 4 * n_dist
            cells = n_dist * n_rows
            ids = np.empty((cells, top_n), "<u2")
            score = np.empty((cells, top_n), "<f4")
            prob = np.empty((cells, top_n), "<f4")
            for c in range(cells):
                ids[c] = np.frombuffer(buf, "<u2", top_n, p)
                p += 2 * top_n
                score[c] = np.frombuffer(buf, "<f4", top_n, p)
                p += 4 * top_n
                prob[c] = np.frombuffer(buf, "<f4", top_n, p)
                p += 4 * top_n
            t.pred[(step, src)] = dict(
                dist=dist,
                ids=ids.reshape(n_dist, n_rows, top_n),
                score=score.reshape(n_dist, n_rows, top_n),
                prob=prob.reshape(n_dist, n_rows, top_n),
            )
        elif rtype == 3:
            t.dropped_pred, t.dropped_all = struct.unpack_from("<QQ", buf, p)
        elif rtype == 4:
            step, n_slots, n_h, flags = struct.unpack_from("<IIII", buf, p)
            p += 16
            tok = np.frombuffer(buf, "<i4", n_slots, p)
            conf = np.frombuffer(buf, "<f4", n_slots, p + 4 * n_slots)
            p += 8 * n_slots
            (anchor,) = struct.unpack_from("<i", buf, p)
            p += 4
            hid = None
            if n_h > 0:
                hid = np.frombuffer(buf, "<f4", n_slots * n_h, p).reshape(n_slots, n_h)
            rec = dict(tokens=tok, conf=conf, anchor=anchor, hidden=hid, flags=flags, seq=(flags >> 8) & 0xFF)
            t.draft_all.append((step, rec))
            if step not in t.draft or rec["seq"] == 0:
                t.draft[step] = rec
        elif rtype == 5:
            step, n_rows, pos0 = struct.unpack_from("<IIi", buf, p)
            p += 12
            t.batch[step] = dict(tokens=np.frombuffer(buf, "<i4", n_rows, p), pos0=pos0)
        pos += nbytes
    return t


def precision_recall(t, d, top_m, max_rows=32):
    """ROUTER2 raw top-`top_m` at distance d vs actual routing of layer src+d.
    Returns (precision[max_rows], recall[max_rows], n[max_rows]) per row index;
    NaN where a row never occurred. Phantom (padding) rows are skipped."""
    hit = np.zeros(max_rows)
    pred_n = np.zeros(max_rows)
    act_n = np.zeros(max_rows)
    cnt = np.zeros(max_rows)
    for (step, src), pr in t.pred.items():
        di = np.nonzero(pr["dist"] == d)[0]
        rt = t.routing.get((step, src + d))
        if len(di) == 0 or rt is None:
            continue
        pids = pr["ids"][di[0]]   # [n_rows, top_n]
        for r in range(min(pids.shape[0], rt["ids"].shape[0], max_rows)):
            if (rt["phantom"] >> r) & 1:
                continue
            p = set(int(x) for x in pids[r, :top_m] if x != 0xFFFF)
            a = set(int(x) for x in rt["ids"][r])
            hit[r] += len(p & a)
            pred_n[r] += len(p)
            act_n[r] += len(a)
            cnt[r] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return hit / pred_n, hit / act_n, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--max-m", type=int, default=6, help="top-M for the example table")
    ap.add_argument("--max-d", type=int, default=0, help="max distance (default: trace_k)")
    args = ap.parse_args()
    t = load(args.trace)
    print(f"n_expert={t.n_expert} trace_k={t.trace_k} routing_records={len(t.routing)} "
          f"pred_records={len(t.pred)} draft_records={len(t.draft_all)} batch_records={len(t.batch)} dropped_pred={t.dropped_pred} dropped_all={t.dropped_all}")
    for d in range(1, (args.max_d or t.trace_k) + 1):
        prec, rec, n = precision_recall(t, d, args.max_m)
        rows = [r for r in range(len(n)) if n[r] > 0][:8]
        if not rows:
            continue
        print(f"d={d} top-{args.max_m}:")
        for r in rows:
            print(f"  row {r}: precision={prec[r]:.3f} recall={rec[r]:.3f} (n={int(n[r])})")


if __name__ == "__main__":
    sys.exit(main())
