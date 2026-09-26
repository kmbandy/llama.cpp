#!/usr/bin/env python3
"""Join the spine's WP_DISPATCH_REQ_LOG(.writer) with one or more workers'
WP_REQ_LOG files and print a per-request/per-chunk wait-time breakdown.

Requires the *_TIMELINE sub-knobs on both sides:
  spine  : WP_DISPATCH_REQ_LOG=<path> WP_DISPATCH_REQ_LOG_TIMELINE=1
  worker : WP_REQ_LOG=<path>          WP_REQ_LOG_TIMELINE=1
(WP_WORKER_PIPELINE defaults to 1 already; leave it alone.)

Correlation key: (seq_id, chunk_index, worker_index). worker_index is the
position of the worker's endpoint in the dispatcher's constructor argument
list (0-based) -- pass each worker's own WP_REQ_LOG file labelled with that
index via --worker INDEX:PATH.

Usage:
    wp-req-timeline.py --spine spine.reqlog --writer spine.reqlog.writer \
        --worker 0:worker-a.reqlog --worker 1:worker-b.reqlog

--writer is optional but strongly recommended -- without it, spine-side send
queueing/wire time cannot be split out (the script falls back to folding it
into "request wire+recv").

Clock alignment: the spine and each worker read CLOCK_REALTIME independently
(NTP-skewed relative to each other; their CLOCK_MONOTONIC readings are not
comparable across hosts at all). This script estimates, per worker, the
offset O = worker_realtime - spine_realtime via the min-latency method: for
every spine->worker message, one-way delay >= 0 bounds O from above
(worker_recv - spine_send); for every worker->spine message, one-way delay
>= 0 bounds O from below (worker_send - spine_recv). The midpoint of
[lower, upper] is reported as the estimate, and (upper - lower) / 2 as its
accuracy -- half the round-trip jitter seen in the sample, which is the best
this method can do without a dedicated clock-sync probe.
"""
import argparse
import sys
from collections import defaultdict

UINT32_MAX = 0xFFFFFFFF

# --- spine WP_DISPATCH_REQ_LOG -----------------------------------------
# Base columns (always present), see write_request_log() in
# src/pipeline/pipe-expert-dispatcher.cpp:
SPINE_BASE_COLS = [
    "layer", "n_tokens", "worker_index", "n_experts", "ns_before_await",
    "ns_blocked", "ns_issue_done", "ns_await_recv", "resp_bytes", "ns_unpack",
    "await_start_ns", "await_end_ns", "seq_id", "chunk_index",
]
# WP_DISPATCH_REQ_LOG_TIMELINE=1 appends these four:
SPINE_TIMELINE_COLS = ["ns_hdr_wait", "ns_body_wait", "epoch_await_start_ns", "epoch_await_end_ns"]

# --- spine WP_DISPATCH_REQ_LOG.writer -----------------------------------
WRITER_BASE_COLS = ["ns_queued", "ns_send", "frame_type", "seq_id", "bytes", "endpoint"]
WRITER_TIMELINE_COLS = ["chunk_index", "epoch_send_ns"]

# --- worker WP_REQ_LOG (aggregate file; timeline block is always the last
# 7 fields regardless of how many base/per-device columns precede it) -----
WORKER_TIMELINE_TAIL = [
    "chunk_index", "seq_id", "ns_recv_body", "ns_req_decode",
    "ns_resp_send", "ns_queue_wait", "epoch_recv_ns",
]
# A handful of named base columns used below, by fixed 1-indexed position
# (see the WP_REQ_LOG column-order comment in wp-expert-worker.cpp).
WORKER_NS_WALL_IDX = 7   # 1-indexed
WORKER_EPOCH_END_IDX = 22


def parse_rows(path, expect_min_cols):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < expect_min_cols:
                continue
            rows.append(fields)
    return rows


def parse_spine_req_log(path):
    rows = parse_rows(path, len(SPINE_BASE_COLS))
    out = []
    for fields in rows:
        timeline = len(fields) >= len(SPINE_BASE_COLS) + len(SPINE_TIMELINE_COLS)
        rec = dict(zip(SPINE_BASE_COLS, fields[: len(SPINE_BASE_COLS)]))
        if timeline:
            rec.update(dict(zip(SPINE_TIMELINE_COLS, fields[len(SPINE_BASE_COLS):len(SPINE_BASE_COLS) + len(SPINE_TIMELINE_COLS)])))
        for k in rec:
            rec[k] = int(rec[k])
        rec["_timeline"] = timeline
        out.append(rec)
    return out


def parse_writer_log(path):
    if path is None:
        return []
    rows = parse_rows(path, len(WRITER_BASE_COLS))
    out = []
    for fields in rows:
        base = fields[: len(WRITER_BASE_COLS)]
        rec = dict(zip(WRITER_BASE_COLS, base))
        rest = fields[len(WRITER_BASE_COLS):]
        if len(rest) >= len(WRITER_TIMELINE_COLS):
            rec.update(dict(zip(WRITER_TIMELINE_COLS, rest[: len(WRITER_TIMELINE_COLS)])))
        for k, v in list(rec.items()):
            if k == "endpoint":
                continue
            rec[k] = int(v)
        if "chunk_index" not in rec:
            rec["chunk_index"] = UINT32_MAX
        out.append(rec)
    return out


def parse_worker_req_log(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < WORKER_EPOCH_END_IDX + len(WORKER_TIMELINE_TAIL):
                # Not a timeline row (or truncated) -- skip; this script has
                # nothing to join it against.
                continue
            tail = fields[-len(WORKER_TIMELINE_TAIL):]
            rec = dict(zip(WORKER_TIMELINE_TAIL, tail))
            for k in rec:
                rec[k] = int(rec[k])
            rec["ns_wall"] = int(fields[WORKER_NS_WALL_IDX - 1])
            # epoch_end (worker's own column, see wp-expert-worker.cpp) is
            # CLOCK_REALTIME as fractional SECONDS ("%.6f"), i.e. microsecond
            # precision -- convert to ns for comparison against the spine's
            # epoch_await_end_ns (already integer ns).
            rec["epoch_end_ns"] = int(round(float(fields[WORKER_EPOCH_END_IDX - 1]) * 1e9))
            out.append(rec)
    return out


def estimate_offset(spine_send, worker_recv, worker_send, spine_recv):
    """min-latency clock offset estimate: O = worker_realtime - spine_realtime.

    spine_send / worker_recv: parallel lists, one entry per spine->worker
    message (epoch_send_ns from the writer log, epoch_recv_ns from the
    worker log). worker_send / spine_recv: parallel lists for the response
    direction (worker's epoch_end, spine's epoch_await_end_ns).
    Returns (offset_ns, accuracy_ns, n_forward, n_backward) or None if there
    is nothing to estimate from in one direction.
    """
    upper_candidates = [wr - ss for ss, wr in zip(spine_send, worker_recv) if ss > 0 and wr > 0]
    lower_candidates = [ws - sr for ws, sr in zip(worker_send, spine_recv) if ws > 0 and sr > 0]
    if not upper_candidates or not lower_candidates:
        return None
    upper = min(upper_candidates)
    lower = max(lower_candidates)
    offset = (upper + lower) / 2.0
    # upper < lower is possible (and NOT a bug) when the true one-way delays
    # are smaller than the clock read jitter itself -- e.g. two processes on
    # the same host, as in this script's own test, where "network delay" is
    # sub-microsecond and CLOCK_REALTIME reads a few hundred ns apart can
    # invert the bound order. Report the crossing honestly as the accuracy
    # bound rather than a misleading negative number: the true offset is
    # known only to within this width either way.
    accuracy = abs(upper - lower) / 2.0
    return offset, accuracy, len(upper_candidates), len(lower_candidates), upper >= lower


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spine", required=True, help="spine WP_DISPATCH_REQ_LOG file")
    ap.add_argument("--writer", default=None, help="spine WP_DISPATCH_REQ_LOG.writer file")
    ap.add_argument("--worker", action="append", default=[], metavar="INDEX:PATH",
                     help="worker_index:WP_REQ_LOG path; repeatable, one per worker")
    args = ap.parse_args()

    spine_rows = parse_spine_req_log(args.spine)
    if not spine_rows:
        print("wp-req-timeline: no rows parsed from spine log " + args.spine, file=sys.stderr)
        return 1
    writer_rows = parse_writer_log(args.writer)

    worker_logs = {}
    for spec in args.worker:
        if ":" not in spec:
            print("wp-req-timeline: --worker must be INDEX:PATH, got " + spec, file=sys.stderr)
            return 1
        idx_str, path = spec.split(":", 1)
        worker_logs[int(idx_str)] = parse_worker_req_log(path)
    if not worker_logs:
        print("wp-req-timeline: at least one --worker INDEX:PATH is required", file=sys.stderr)
        return 1

    # Index worker rows by (seq_id, chunk_index).
    worker_by_key = {}
    for widx, rows in worker_logs.items():
        for r in rows:
            worker_by_key[(widx, r["seq_id"], r["chunk_index"])] = r

    # Index writer rows by (seq_id, chunk_index) too -- endpoint carries the
    # worker identity on that side, worker_index is what we key by here, so
    # collect ALL writer rows matching a seq_id/chunk_index and let the
    # per-spine-row join pick by worker_index-derived endpoint count. With
    # exactly one frame per (seq_id, chunk_index) per worker this is a
    # straight dict lookup; multiple frames of the same seq_id to different
    # workers are disambiguated by chunk_index already carrying no worker
    # identity, so ties are broken by insertion order (fine for the common
    # 1-worker-per-request-leg case this task's live topology uses).
    writer_by_key = defaultdict(list)
    for w in writer_rows:
        writer_by_key[(w["seq_id"], w["chunk_index"])].append(w)

    joined = []
    unmatched_worker = 0
    for s in spine_rows:
        key_generic = (s["seq_id"], s["chunk_index"])
        widx = s["worker_index"]
        wkey = (widx, s["seq_id"], s["chunk_index"])
        w = worker_by_key.get(wkey)
        if w is None:
            unmatched_worker += 1
            continue
        wr = None
        if writer_by_key.get(key_generic):
            wr = writer_by_key[key_generic].pop(0)
        joined.append((s, w, wr))

    if not joined:
        print("wp-req-timeline: no spine row joined to a worker row", file=sys.stderr)
        return 1
    if unmatched_worker:
        print("wp-req-timeline: %d spine row(s) did not join to a worker row (see --worker "
              "INDEX:PATH -- is INDEX the worker's position in the dispatcher's endpoint "
              "list?)" % unmatched_worker, file=sys.stderr)
        return 1

    # --- Sanity: every interval must be non-negative, and the two spine-side
    # sub-intervals must sum to the total spine-side wait -- this is an exact
    # arithmetic identity from write_request_log() (ns_hdr_wait + ns_body_wait
    # == ns_blocked, both computed from the same three timestamps), not a
    # heuristic, so any violation means a parsing/join bug rather than jitter.
    bad = 0
    for s, w, wr in joined:
        for key in ("ns_before_await", "ns_hdr_wait", "ns_body_wait", "ns_unpack"):
            if s.get(key, 0) < 0:
                bad += 1
        for key in ("ns_queue_wait", "ns_req_decode", "ns_resp_send", "ns_wall"):
            if w.get(key, 0) < 0:
                bad += 1
        if s["_timeline"]:
            total = s["ns_hdr_wait"] + s["ns_body_wait"]
            if total != s["ns_blocked"]:
                print("wp-req-timeline: seq_id=%d chunk=%d: ns_hdr_wait+ns_body_wait=%d != "
                      "ns_blocked=%d" % (s["seq_id"], s["chunk_index"], total, s["ns_blocked"]),
                      file=sys.stderr)
                bad += 1
        if wr is not None and wr["ns_queued"] + wr["ns_send"] < 0:
            bad += 1
    if bad:
        print("wp-req-timeline: %d row(s) failed the non-negative/sums-sensibly check" % bad,
              file=sys.stderr)
        return 1

    # --- Per-worker clock offset estimate -----------------------------
    offsets = {}
    for widx in worker_logs:
        spine_send, worker_recv, worker_send, spine_recv = [], [], [], []
        for s, w, wr in joined:
            if s["worker_index"] != widx:
                continue
            if wr is not None and wr.get("epoch_send_ns"):
                spine_send.append(wr["epoch_send_ns"])
                worker_recv.append(w["epoch_recv_ns"])
            if s.get("epoch_await_end_ns"):
                worker_send.append(w["epoch_end_ns"])
                spine_recv.append(s["epoch_await_end_ns"])
        offsets[widx] = estimate_offset(spine_send, worker_recv, worker_send, spine_recv)

    # --- Per-request breakdown -----------------------------------------
    # All intervals below are ns. Terms:
    #   spine_busy_before_await : spine thread doing other work before it
    #                             even started waiting on this response
    #                             (head-of-line on the spine thread).
    #   send_queue               : writer FIFO queueing (enqueue -> send start).
    #   send_wire                : the send() syscall itself.
    #   worker_queue             : worker reader-queue wait before compute.
    #   worker_decode            : this frame's request decode.
    #   worker_dispatch          : ns_wall - decode - resp_send (approx: the
    #                              remainder after decode and response send,
    #                              i.e. compute + response encode).
    #   worker_resp_send         : worker's response encode+send.
    #   response_wire_and_recv   : ns_hdr_wait minus everything already
    #                              counted on the worker side that overlaps
    #                              it is NOT subtracted here (ns_hdr_wait is
    #                              spine-clock await_start -> header landing,
    #                              i.e. wire + worker queue + worker service
    #                              as a single spine-side number; report it
    #                              alongside the worker-side breakdown rather
    #                              than trying to net them against each other
    #                              across two different clocks without an
    #                              offset -- see ns_hdr_wait_spine_side below).
    #   response_body            : ns_body_wait (header landed -> body done).
    #   decode                   : spine-side unpack of the response.
    per_layer = defaultdict(lambda: defaultdict(float))
    per_layer_n = defaultdict(int)
    totals = defaultdict(float)
    n = 0
    for s, w, wr in joined:
        n += 1
        layer = s["layer"]
        per_layer_n[layer] += 1
        per_layer[layer]["spine_busy_before_await"] += s["ns_before_await"]
        per_layer[layer]["response_hdr_wait_spine_side"] += s.get("ns_hdr_wait", 0)
        per_layer[layer]["response_body_wait"] += s.get("ns_body_wait", s["ns_blocked"])
        # ns_unpack (spine's decode_started -> now()) is only stamped from a
        # real clock reading when WP_DS4_LAYER_TRACE=1 is ALSO set on the
        # spine (see receive_partial()'s decode_started in
        # pipe-expert-dispatcher.cpp); otherwise decode_started is a
        # default-constructed (epoch-0) time_point and ns_unpack is really
        # "steady_clock::now() since its own epoch", a multi-second-or-larger
        # number with no timing meaning. Anything over 1 second is treated as
        # that unmeasured case rather than reported as a real decode cost.
        if s["ns_unpack"] < 1_000_000_000:
            per_layer[layer]["spine_decode"] += s["ns_unpack"]
            per_layer[layer]["spine_decode_n"] += 1
        per_layer[layer]["worker_queue_wait"] += w["ns_queue_wait"]
        per_layer[layer]["worker_decode"] += w["ns_req_decode"]
        per_layer[layer]["worker_resp_send"] += w["ns_resp_send"]
        dispatch_approx = w["ns_wall"] - w["ns_req_decode"] - w["ns_resp_send"]
        per_layer[layer]["worker_dispatch_approx"] += max(dispatch_approx, 0)
        if wr is not None:
            per_layer[layer]["send_queue"] += wr["ns_queued"]
            per_layer[layer]["send_wire"] += wr["ns_send"]
    for layer, d in per_layer.items():
        for k, v in d.items():
            totals[k] += v

    print("wp-req-timeline: %d spine rows, %d joined to a worker row, %d unmatched"
          % (len(spine_rows), len(joined), unmatched_worker))
    for widx, est in sorted(offsets.items()):
        if est is None:
            print("  worker %d: clock offset -- not enough samples in one direction" % widx)
            continue
        offset, accuracy, nf, nb, bounds_ok = est
        note = "" if bounds_ok else " (bounds crossed: true delay is within read jitter, e.g. same host)"
        print("  worker %d: clock offset = %+.3f ms (+/- %.3f ms, %d forward / %d backward samples)%s"
              % (widx, offset / 1e6, accuracy / 1e6, nf, nb, note))

    print("\nPer-layer breakdown (mean ns over the layer's requests/chunks):")
    header = ["layer", "n", "spine_busy", "send_q", "send_wire", "hdr_wait(spine)",
              "worker_q", "worker_decode", "worker_dispatch~", "worker_send", "body_wait", "spine_decode"]
    print("  " + " ".join("%-16s" % h for h in header))
    def fmt_spine_decode(total_ns, sample_n):
        if sample_n == 0:
            return "n/a*"
        return "%.0f" % (total_ns / sample_n)

    for layer in sorted(per_layer):
        d = per_layer[layer]
        cnt = per_layer_n[layer]
        row = [layer, cnt,
               d["spine_busy_before_await"] / cnt, d["send_queue"] / cnt, d["send_wire"] / cnt,
               d["response_hdr_wait_spine_side"] / cnt, d["worker_queue_wait"] / cnt,
               d["worker_decode"] / cnt, d["worker_dispatch_approx"] / cnt,
               d["worker_resp_send"] / cnt, d["response_body_wait"] / cnt]
        cells = ["%.0f" % v if isinstance(v, float) else str(v) for v in row]
        cells.append(fmt_spine_decode(d["spine_decode"], d["spine_decode_n"]))
        print("  " + " ".join("%-16s" % c for c in cells))
    if any(per_layer[l]["spine_decode_n"] < per_layer_n[l] for l in per_layer):
        print("  * spine_decode needs WP_DS4_LAYER_TRACE=1 on the spine as well; without it "
              "ns_unpack is not a real clock reading (see receive_partial()'s decode_started).")

    print("\nAggregated over the whole run (mean ns per request/chunk, n=%d):" % n)
    for k in ["spine_busy_before_await", "send_queue", "send_wire", "response_hdr_wait_spine_side",
              "worker_queue_wait", "worker_decode", "worker_dispatch_approx", "worker_resp_send",
              "response_body_wait"]:
        print("  %-28s %.0f ns" % (k, totals[k] / n))
    print("  %-28s %s ns" % ("spine_decode", fmt_spine_decode(totals["spine_decode"], totals["spine_decode_n"])))

    # --- Which worker's response completes last per layer, and by how much
    print("\nSlowest worker per layer (by await_end_ns, spine-clock; only meaningful "
          "with >1 worker):")
    by_layer_finish = defaultdict(list)
    for s, w, wr in joined:
        by_layer_finish[s["layer"]].append((s["worker_index"], s["await_end_ns"]))
    for layer in sorted(by_layer_finish):
        entries = by_layer_finish[layer]
        if len({e[0] for e in entries}) < 2:
            continue
        last = max(entries, key=lambda e: e[1])
        earliest_others = min(e[1] for e in entries if e[0] != last[0])
        print("  layer %d: worker %d finished last, %.0f ns after the next-slowest worker"
              % (layer, last[0], last[1] - earliest_others))

    return 0


if __name__ == "__main__":
    sys.exit(main())
