#!/usr/bin/env python3
"""Fail-closed real-shape benchmark harness for the DSWS fp8 GEMM kernel.

Offline mode only reads existing logs. Live mode is deliberately explicit and
dispatches each supported shape through one separate gpu_run.sh invocation.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent

# Recovered verbatim as data from commit 652053c69. No parser or execution logic
# from the deleted sweep is reused.
SHAPES = (
    ("ml8_dense_ffn_gate_up", 2048, 9216, 2560),
    ("ml8_dense_ffn_gate_up", 512, 9216, 2560),
    ("ml8_dense_ffn_down", 2048, 2560, 9216),
    ("ml8_dense_ffn_down", 512, 2560, 9216),
    ("ml8_dense_attn_q", 2048, 4096, 2560),
    ("ml8_dense_attn_q", 512, 4096, 2560),
    ("ml8_dense_attn_kv", 2048, 1024, 2560),
    ("ml8_dense_attn_kv", 512, 1024, 2560),
    ("ml8_dense_attn_o", 2048, 2560, 4096),
    ("ml8_dense_attn_o", 512, 2560, 4096),
    ("ml8_moe_ffn_gate_up", 64, 512, 2048),
    ("ml8_moe_ffn_gate_up", 512, 512, 2048),
    ("ml8_moe_ffn_down", 64, 2048, 512),
    ("ml8_moe_ffn_down", 512, 2048, 512),
    ("ml8_moe_attn_q", 64, 4096, 2048),
    ("ml8_moe_attn_q", 512, 4096, 2048),
    ("ml8_moe_attn_kv", 64, 512, 2048),
    ("ml8_moe_attn_kv", 512, 512, 2048),
    ("ml8_moe_attn_o", 64, 2048, 4096),
    ("ml8_moe_attn_o", 512, 2048, 4096),
    ("mlmf_mamba_in_proj", 4096, 4200, 768),
    ("mlmf_in_proj_ML8PAD", 4096, 4208, 768),
    ("mlmf_MoE_expert_fc1", 512, 1536, 768),
    ("mlmf_MoE_expert_fc2", 512, 768, 1536),
    ("mlmf_lm_head", 4096, 32000, 768),
    ("mlmf_mamba_out_proj", 4096, 768, 1536),
    ("mlmf_attn_o_proj", 4096, 768, 768),
    ("mlmf_router_down_proj", 4096, 256, 768),
    ("mlmf_router_MLP", 4096, 256, 256),
    ("mlmf_attn_linear_k", 4096, 192, 768),
    ("mlmf_attn_val_proj1", 4096, 96, 768),
    ("mlmf_router_out", 4096, 8, 256),
    ("mlmf_routerout_ML8PAD", 4096, 16, 256),
)

LIVE_G = 6
LIVE_FM = 1
# Set by the LIVE loop to the (label, real_m, real_n, k) it is about to dispatch, so attribution uses
#   ground truth instead of reverse-mapping padded geometry. None in offline re-parse, where the log is
#   all we have. See matching_real_shapes().
EXPECTED_SHAPE = None
LIVE_FN = 4
LIVE_TM = LIVE_G * 16 * LIVE_FM
LIVE_TN = LIVE_FN * 16
LIVE_SEGK = 256
ROUNDING_TOLERANCE = 0.050000001


class LogRejected(Exception):
    """A log failed a required invariant and cannot produce throughput."""


@dataclasses.dataclass(frozen=True)
class ParsedRun:
    path: str
    shape_labels: tuple[str, ...]
    real_m: int
    padded_m: int
    real_n: int
    padded_n: int
    n: int
    k: int
    super_tile_m: int
    reps: int
    chunks: int
    ticks: int
    tick_mhz: float
    spread_percent_reported_1dp: float | None
    padded_tflops: float
    real_flop_corrected_tflops: float


def one_match(pattern: re.Pattern[str], text: str, field: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise LogRejected(f"PARSE_{field}: expected exactly one emission, found {len(matches)}")
    return matches[0]


RE_CLOCK = re.compile(r"^\s*GPU clock-counter freq ~= (?P<mhz>[0-9]+(?:\.[0-9]+)?) MHz\b", re.MULTILINE)
RE_CONFIG = re.compile(
    r"^\s*G=(?P<g>[0-9]+) SEGK=(?P<segk>[0-9]+) FM=(?P<fm>[0-9]+) FN=(?P<fn>[0-9]+)\b",
    re.MULTILINE,
)
RE_ORACLE_SHAPE = re.compile(
    r"^\s*oracle shape (?P<m>[0-9]+)x(?P<n>[0-9]+)x(?P<k>[0-9]+)\s+"
    r"\(super-tile (?P<tm>[0-9]+)x(?P<tn>[0-9]+),",
    re.MULTILINE,
)
RE_GEOMETRY = re.compile(
    r"^\s*\[dsws2\] (?P<m>[0-9]+)x(?P<n>[0-9]+)x(?P<k>[0-9]+)\s+"
    r"super-tile=(?P<tm>[0-9]+)x(?P<tn>[0-9]+).*?"
    r"TOTAL=(?P<total>[0-9]+) TOTAL_super=(?P<total_super>[0-9]+)\b",
    re.MULTILINE,
)
RE_COMPLETION = re.compile(
    r"^\s*\[dsws2 completion\] occ\[0\]\(live\)=(?P<live>[0-9]+)\b", re.MULTILINE
)
RE_WORK_EXACT = re.compile(
    r"^\s*\[dsws2 WORK-EXACT\] computed == G\*TOTAL_super == (?P<expected>[0-9]+)\b",
    re.MULTILINE,
)
RE_THROUGHPUT = re.compile(
    r"^\s*\[dsws2 THROUGHPUT\] (?P<m>[0-9]+)x(?P<n>[0-9]+)x(?P<k>[0-9]+)\s+"
    r"TF=(?P<tf>[0-9]+(?:\.[0-9]+)?)\s+"
    r"\((?P<pct>[0-9]+(?:\.[0-9]+)?)% of 307 TF fp8 peak\)\s+"
    r"span=(?P<ticks>[0-9]+) ticks / (?P<chunks>[0-9]+) chunk\(s\) @ "
    r"(?P<mhz>[0-9]+(?:\.[0-9]+)?) MHz\s*$",
    re.MULTILINE,
)
RE_SUSTAINED = re.compile(
    r"^\s*\[dsws2 SUSTAINED\] reps=(?P<reps>[0-9]+)\s+TF=(?P<tf>[0-9]+(?:\.[0-9]+)?) mean\s+"
    r"\(per-rep (?P<lo>[0-9]+(?:\.[0-9]+)?)-(?P<hi>[0-9]+(?:\.[0-9]+)?), "
    r"spread (?P<spread>[0-9]+(?:\.[0-9]+)?)%\)",
    re.MULTILINE,
)
RE_ORACLE = re.compile(
    r"^\s*\[dsws2 oracle\] ok=(?P<ok>[0-9]+) bad=(?P<bad>[0-9]+)\b", re.MULTILINE
)
RE_CLEAN = re.compile(r"^\s*dsws2 oracle CLEAN\s*$", re.MULTILINE)
RE_CHUNK_PLAN = re.compile(
    r"^\s*\[dsws2\] compositor-safe: [0-9]+ (?:tiles|super-tiles)/dispatch x (?P<chunks>[0-9]+) chunks\b",
    re.MULTILINE,
)
RE_SINGLE_CHUNK = re.compile(r"^\s*\[dsws2\] \*\*\* WARNING: SINGLE CHUNK\b", re.MULTILINE)
RE_CANNOT_EVALUATE = re.compile(r"WORK-EXACT: CANNOT-EVALUATE", re.MULTILINE)
RE_FATAL_MARKERS = re.compile(
    r"WORK-INEXACT|dsws2 INCOMPLETE|WARN chunk .*ABORT|\*\*\*[^\n]*(?:TIMEOUT|FATAL|REFUSE)|"
    r"DSWS2 ADDRESS BOUNDS GATE FAILED|oracle \*\*\* BAD|CANARY[^\n]*(?:FAIL|DIRTY)",
    re.IGNORECASE,
)


def rounded_value_agrees(derived: float, rendered: float) -> bool:
    return math.isfinite(derived) and abs(derived - rendered) <= ROUNDING_TOLERANCE


def matching_real_shapes(padded_m: int, padded_n: int, k: int, super_tile_m: int) -> tuple[int, int, tuple[str, ...]]:
    # The log reports the PADDED geometry (that is what actually ran), so both M and N must be
    #   matched through the same padding the dispatcher applied. Matching raw shape_n against a
    #   padded n silently rejected every N-padded shape as NOT_A_REAL_SHAPE.
    candidates = [
        (label, real_m, shape_n)
        for label, real_m, shape_n, shape_k in SHAPES
        if ((shape_n + LIVE_TN - 1) // LIVE_TN) * LIVE_TN == padded_n
        and shape_k == k
        and ((real_m + super_tile_m - 1) // super_tile_m) * super_tile_m == padded_m
    ]
    real_ms = {real_m for _, real_m, _ in candidates}
    real_ns = {shape_n for _, _, shape_n in candidates}
    if not candidates:
        raise LogRejected("NOT_A_REAL_SHAPE: geometry does not match the recovered 33-shape inventory")
    # DISAMBIGUATION BY GROUND TRUTH (added 2026-07-26). This function REVERSE-MAPS the log's padded
    #   geometry back to a real shape, which is unavoidable when re-parsing arbitrary logs offline --
    #   but in LIVE mode the caller already KNOWS which shape it just dispatched, so reverse-mapping
    #   there was throwing away information it had.
    #   IT HALTED A 30-SHAPE SWEEP: mlmf_mamba_in_proj (N=4200) and mlmf_in_proj_ML8PAD (N=4208) both
    #   pad to N=4224 at LIVE_TN=64, so the padded geometry genuinely cannot distinguish them and the
    #   function refused (correctly -- attributing a TF number to the wrong shape would be worse).
    #   The kernel run itself was CLEAN (WORK-EXACT, oracle bad=0); only the attribution was ambiguous.
    #   EXPECTED_SHAPE lets live mode say which one it ran. We still VERIFY it is among the candidates
    #   rather than trusting it blindly -- if the hint disagrees with the geometry that is a real bug
    #   and must not be papered over.
    if EXPECTED_SHAPE is not None and len(real_ns) > 1:
        exp_label, exp_m, exp_n, exp_k = EXPECTED_SHAPE
        hinted = [c for c in candidates if c[1] == exp_m and c[2] == exp_n]
        if not hinted:
            raise LogRejected(
                f"EXPECTED_SHAPE_MISMATCH: dispatched {exp_label} (M={exp_m} N={exp_n} K={exp_k}) "
                f"but the log's padded geometry does not admit it -- geometry bug, not an attribution problem")
        return exp_m, exp_n, (exp_label,)
    if len(real_ms) != 1:
        raise LogRejected("AMBIGUOUS_REAL_M: shape inventory maps this padded geometry to multiple real M values")
    if len(real_ns) != 1:
        raise LogRejected("AMBIGUOUS_REAL_N: padding collapses two distinct real N onto one padded N "
                          "(offline re-parse cannot disambiguate; live mode sets EXPECTED_SHAPE)")
    return next(iter(real_ms)), next(iter(real_ns)), tuple(sorted({label for label, _, _ in candidates}))


def parse_log(path: Path) -> ParsedRun:
    resolved = path.expanduser().resolve()
    try:
        text = resolved.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise LogRejected(f"READ_ERROR: {exc}") from exc

    fatal = RE_FATAL_MARKERS.search(text)
    if fatal:
        token = " ".join(fatal.group(0).split())
        raise LogRejected(f"INVALID_RUN_MARKER: {token}")

    # A run whose correctness gate COULD NOT EVALUATE is refused under its own name. Without this it
    # would still be refused -- but as a bare "WORK_EXACT: expected 1 match, found 0", which reads as
    # a parser/format problem and invites someone to relax the regex. It is not: the gate is absent
    # because occ[71] is emitted by cnt_flush, so a STAGINSTR=0 bin has no verdict to report. Build
    # CNTLEAN=1 instead -- it trims the retire flush while keeping the gate's two inputs.
    if RE_CANNOT_EVALUATE.search(text):
        raise LogRejected(
            "WORK_EXACT_CANNOT_EVALUATE: STAGINSTR counters absent (STAGINSTR=0), so work-exactness "
            "was never checked. Absence of a verdict is not a pass; rebuild with CNTLEAN=1."
        )

    clock = one_match(RE_CLOCK, text, "CLOCK")
    config = one_match(RE_CONFIG, text, "CONFIG")
    header = one_match(RE_ORACLE_SHAPE, text, "ORACLE_SHAPE")
    geometry = one_match(RE_GEOMETRY, text, "GEOMETRY")
    completion = one_match(RE_COMPLETION, text, "COMPLETION")
    work = one_match(RE_WORK_EXACT, text, "WORK_EXACT")
    throughput = one_match(RE_THROUGHPUT, text, "THROUGHPUT")
    oracle = one_match(RE_ORACLE, text, "ORACLE")
    one_match(RE_CLEAN, text, "CLEAN_VERDICT")

    if int(completion["live"]) != 0:
        raise LogRejected("INCOMPLETE_RUN: completion live counter is nonzero")
    if int(oracle["bad"]) != 0 or int(oracle["ok"]) <= 0:
        raise LogRejected("ORACLE_FAILED: require ok>0 and bad=0")

    padded_m = int(geometry["m"])
    n = int(geometry["n"])
    k = int(geometry["k"])
    super_tile_m = int(geometry["tm"])
    dims = (padded_m, n, k)
    header_dims = (int(header["m"]), int(header["n"]), int(header["k"]))
    throughput_dims = (int(throughput["m"]), int(throughput["n"]), int(throughput["k"]))
    if dims != header_dims or dims != throughput_dims:
        raise LogRejected("GEOMETRY_MISMATCH: header, dispatch, and timing dimensions differ")
    if super_tile_m != int(header["tm"]):
        raise LogRejected("GEOMETRY_MISMATCH: super-tile M differs between header and dispatch")
    if padded_m % super_tile_m != 0:
        raise LogRejected("GEOMETRY_INVALID: padded M is not a multiple of the emitted super-tile M")

    g = int(config["g"])
    total_super = int(geometry["total_super"])
    expected = int(work["expected"])
    work_per_rep = g * total_super
    if work_per_rep <= 0 or expected % work_per_rep != 0:
        raise LogRejected("REPETITION_INCOHERENT: WORK-EXACT count is not G*TOTAL_super*integer_reps")
    reps = expected // work_per_rep
    if reps < 1:
        raise LogRejected("REPETITION_INCOHERENT: derived repetition count is zero")

    sustained_matches = list(RE_SUSTAINED.finditer(text))
    spread: float | None = None
    sustained_tf: float | None = None
    if reps == 1:
        if sustained_matches:
            raise LogRejected("REPETITION_INCOHERENT: SUSTAINED line exists for a single repetition")
    else:
        if len(sustained_matches) != 1:
            raise LogRejected(
                f"PARSE_SUSTAINED: reps={reps} requires exactly one quality emission, found {len(sustained_matches)}"
            )
        sustained = sustained_matches[0]
        if int(sustained["reps"]) != reps:
            raise LogRejected("REPETITION_INCOHERENT: WORK-EXACT and SUSTAINED repetition counts differ")
        spread = float(sustained["spread"])
        sustained_tf = float(sustained["tf"])

    chunks = int(throughput["chunks"])
    chunk_plan = list(RE_CHUNK_PLAN.finditer(text))
    single_chunk = bool(RE_SINGLE_CHUNK.search(text))
    if len(chunk_plan) > 1:
        raise LogRejected("PARSE_CHUNK_PLAN: multiple chunk plans found")
    chunks_per_rep = int(chunk_plan[0]["chunks"]) if chunk_plan else 1
    if not chunk_plan and not single_chunk:
        raise LogRejected("PARSE_CHUNK_PLAN: no compositor-safe or single-chunk emission")
    if chunks != chunks_per_rep * reps:
        raise LogRejected("CHUNK_COUNT_INCOHERENT: timed chunks do not equal chunks_per_rep*reps")

    ticks = int(throughput["ticks"])
    if ticks <= 0 or chunks <= 0:
        raise LogRejected("TIMING_INVALID: ticks and chunks must both be positive")
    tick_mhz = float(clock["mhz"])
    line_mhz = float(throughput["mhz"])
    if tick_mhz <= 0 or abs(tick_mhz - line_mhz) > 0.500000001:
        raise LogRejected("CLOCK_INCOHERENT: clock header and throughput-line MHz differ")

    # Direct transcription of occ_dispatch.cpp:2441-2443. sumSpan is the sum
    # of per-chunk occ[3]-occ[2] spans across every completed repetition.
    padded_tflops = 2.0 * padded_m * n * k * reps * (tick_mhz * 1e6) / ticks / 1e12
    printed_tf = float(throughput["tf"])
    printed_pct = float(throughput["pct"])
    if not rounded_value_agrees(padded_tflops, printed_tf):
        raise LogRejected("SELF_VALIDATION_MISMATCH: ticks/geometry/reps disagree with rendered TF")
    if sustained_tf is not None and not rounded_value_agrees(padded_tflops, sustained_tf):
        raise LogRejected("SELF_VALIDATION_MISMATCH: derived TF disagrees with SUSTAINED mean")
    if not rounded_value_agrees(padded_tflops / 307.0 * 100.0, printed_pct):
        raise LogRejected("SELF_VALIDATION_MISMATCH: derived peak percentage disagrees with rendered percentage")

    # `n` parsed from the log IS the padded N (that is the geometry that ran).
    padded_n = n
    real_m, real_n, shape_labels = matching_real_shapes(padded_m, padded_n, k, super_tile_m)
    # Correct BOTH axes back to real FLOP so padding always counts AGAINST us, never for us.
    real_tflops = padded_tflops * (real_m / padded_m) * (real_n / padded_n)
    return ParsedRun(
        path=str(resolved),
        shape_labels=shape_labels,
        real_m=real_m,
        padded_m=padded_m,
        real_n=real_n,
        padded_n=padded_n,
        n=n,
        k=k,
        super_tile_m=super_tile_m,
        reps=reps,
        chunks=chunks,
        ticks=ticks,
        tick_mhz=tick_mhz,
        spread_percent_reported_1dp=spread,
        padded_tflops=padded_tflops,
        real_flop_corrected_tflops=real_tflops,
    )


def success_record(run: ParsedRun) -> dict[str, object]:
    quality = (
        {"repetitions": 1, "spread_percent_reported_1dp": None, "assessment": "single shot; no spread data"}
        if run.reps == 1
        else {
            "repetitions": run.reps,
            "spread_percent_reported_1dp": run.spread_percent_reported_1dp,
            "assessment": "repeated",
        }
    )
    return {
        "status": "PASS",
        "reason": None,
        "provenance_log": run.path,
        "shape_candidates": list(run.shape_labels),
        "geometry": {
            "real_m": run.real_m,
            "padded_m": run.padded_m,
            "n": run.n,
            "k": run.k,
            "super_tile_m": run.super_tile_m,
        },
        "timing_inputs": {
            "ticks": run.ticks,
            "tick_mhz": run.tick_mhz,
            "chunks": run.chunks,
            "repetitions": run.reps,
        },
        "quality": quality,
        "throughput_tflops": {
            "padded_m_measured_work": run.padded_tflops,
            "real_flop_corrected": run.real_flop_corrected_tflops,
            "self_validation": "PASS: derived value agrees with kernel rounded TF and peak percentage",
        },
    }


def failure_record(path: Path | str, reason: str, status: str = "FAIL") -> dict[str, object]:
    path_text = str(Path(path).expanduser().resolve()) if path else ""
    return {
        "status": status,
        "reason": reason,
        "provenance_log": path_text or None,
        "shape_candidates": [],
        "geometry": None,
        "timing_inputs": None,
        "quality": None,
        "throughput_tflops": None,
    }


def parse_paths(paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        try:
            records.append(success_record(parse_log(path)))
        except LogRejected as exc:
            records.append(failure_record(path, str(exc)))
    return records


def shape_supported(shape: tuple[str, int, int, int]) -> tuple[bool, str, int, int]:
    _, real_m, n, k = shape
    padded_m = ((real_m + LIVE_TM - 1) // LIVE_TM) * LIVE_TM
    # *** N PADDING (2026-07-26, kmbandy: "we can't just have shapes that don't run"). ***
    #   M has ALWAYS been padded up to the super-tile and the TF divided back down by
    #   real_m/padded_m so the waste counts AGAINST us. N had no such branch -- it was simply
    #   REFUSED on n % 64, which silently excluded 6 of 33 real shapes (18% of the workload),
    #   including mlmf_mamba_in_proj, i.e. HALF the Mamba MIMO GEMM path. That was a gap in this
    #   harness, never a kernel limitation: the kernel only ever sees NTL = N/64, so a padded N
    #   is just a wider B and C with garbage columns we do not read -- exactly what M padding is.
    #   The tell that this was an oversight: mlmf_in_proj_ML8PAD exists at N=4208, i.e. someone
    #   already padded 4200 -- to a 16-multiple (ml8's alignment) instead of 64 (DSWS's N tile).
    #   N padding is NOT always cheap (router_out N=8 -> 64 is 8x waste) and it SHOULD look bad:
    #   the real_n/padded_n correction makes that cost visible instead of hiding the shape.
    padded_n = ((n + LIVE_TN - 1) // LIVE_TN) * LIVE_TN
    reasons = []
    if k % LIVE_SEGK:
        reasons.append(f"K%{LIVE_SEGK}={k % LIVE_SEGK}")
    # n_kseg == 1 hits the kernel's DOCUMENTED ZLOCK fail-safe (.Lflow_feed_empty: the boundary
    # lock is bit 0 of DA_ZDONE and needs n_kseg >= 2). The kernel retires clean with computed=0,
    # which is WORK-INEXACT + a bad oracle, which latches gpu_run.sh and halts the whole sweep.
    # 2026-07-21: mlmf_router_MLP (K=256, SEGK=256 -> n_kseg=1) halted a 4-arm matrix this way
    # after 26 good shapes. Declare it UNSUPPORTED with its reason -- it is printed, never skipped.
    #   2026-07-26: SEGK is now a SWEPT KNOB in {64,128,256} (kmbandy sanctioned the range), so
    #   n_kseg=1 is no longer a property of the shape -- it is a property of the SEGK you chose.
    #   mlmf_router_MLP (K=256) is n_kseg=1 at SEGK=256 but n_kseg=2 at SEGK=128. Say so, so the
    #   reason names the fix instead of reading as "this shape cannot run".
    n_kseg = k // LIVE_SEGK if LIVE_SEGK else 0
    if not reasons and n_kseg < 2:
        fix = next((s for s in (128, 64) if s < LIVE_SEGK and k % s == 0 and k // s >= 2), None)
        hint = f" -- rerun with --segk {fix}" if fix else " -- no legal SEGK in {64,128,256} fixes this"
        reasons.append(f"n_kseg={n_kseg}<2 at SEGK={LIVE_SEGK} (ZLOCK needs >=2){hint}")
    return not reasons, ", ".join(reasons), padded_m, padded_n


def shape_inventory(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    inventory = []
    for label, real_m, n, k in SHAPES:
        supported, reason, padded_m, padded_n = shape_supported((label, real_m, n, k))
        matching_pass_logs = []
        matching_records = []
        for record in records:
            path = str(record.get("provenance_log") or "")
            geometry = record.get("geometry")
            filename_matches = label in Path(path).name and f"_M{real_m}_" in Path(path).name
            if filename_matches:
                matching_records.append(
                    {"status": record["status"], "reason": record["reason"], "provenance_log": path}
                )
            if label in Path(path).name and isinstance(geometry, dict):
                if geometry.get("real_m") == real_m and geometry.get("n") == n and geometry.get("k") == k:
                    matching_pass_logs.append(path)
        if not supported:
            archive_status = "UNSUPPORTED_LIVE"
        elif matching_pass_logs:
            archive_status = "PASS_LOG_AVAILABLE"
        elif matching_records:
            archive_status = "NO_ACCEPTED_LOG"
        else:
            archive_status = "NO_ARCHIVED_LOG"
        inventory.append(
            {
                "label": label,
                "real_m": real_m,
                "padded_m_live": padded_m,
                "n": n,
                "k": k,
                "live_status": "SUPPORTED" if supported else "UNSUPPORTED",
                "reason": None if supported else reason,
                "archive_status": archive_status,
                "matching_pass_logs": matching_pass_logs,
                "matching_records": matching_records,
            }
        )
    return inventory


def render_table(
    records: Sequence[dict[str, object]], summary: dict[str, int], inventory: Sequence[dict[str, object]]
) -> str:
    lines = [
        "DSWS real-shape log validation",
        f"inputs={summary['inputs']} pass={summary['pass']} fail={summary['fail']} other={summary['other']}",
        "",
        "status real/pad M       N      K  reps spread% padded_TF real_TF provenance / reason",
        "------ ---------- ------- ------ ----- ------- --------- ------- -------------------",
    ]
    for record in records:
        status = str(record["status"])
        path = str(record.get("provenance_log") or "-")
        if status == "PASS":
            geometry = record["geometry"]
            quality = record["quality"]
            throughput = record["throughput_tflops"]
            assert isinstance(geometry, dict)
            assert isinstance(quality, dict)
            assert isinstance(throughput, dict)
            spread = quality["spread_percent_reported_1dp"]
            spread_text = "n/a" if spread is None else f"{float(spread):.1f}"
            lines.append(
                f"PASS   {geometry['real_m']:>4}/{geometry['padded_m']:<4} "
                f"{geometry['n']:>7} {geometry['k']:>6} {quality['repetitions']:>5} {spread_text:>7} "
                f"{throughput['padded_m_measured_work']:>9.6f} "
                f"{throughput['real_flop_corrected']:>7.6f} {path}"
            )
        else:
            lines.append(f"{status:<6} {'-':>10} {'-':>7} {'-':>6} {'-':>5} {'-':>7} {'-':>9} {'-':>7} {path} :: {record['reason']}")
    lines.append("")
    lines.append("TF columns are emitted only for PASS records. Every TF row names its exact source log.")
    lines.append("real_TF = padded_TF * real_M / padded_M. spread=n/a means reps=1.")
    lines.extend(
        [
            "",
            "Recovered real-shape inventory (live config: M tile=96, N tile=64, SEGK=256)",
            "live_status label                       real/pad M       N      K reason / passing archived logs",
            "----------- --------------------------- ---------- ------- ------ ------------------------------",
        ]
    )
    for item in inventory:
        reason = item["reason"] or (
            f"{item['archive_status']}; passing_archived_logs={len(item['matching_pass_logs'])}"
        )
        lines.append(
            f"{item['live_status']:<11} {item['label']:<27} "
            f"{item['real_m']:>4}/{item['padded_m_live']:<4} {item['n']:>7} {item['k']:>6} {reason}"
        )
    return "\n".join(lines) + "\n"


def summarize(records: Sequence[dict[str, object]]) -> dict[str, int]:
    passed = sum(record["status"] == "PASS" for record in records)
    failed = sum(record["status"] == "FAIL" for record in records)
    return {"inputs": len(records), "pass": passed, "fail": failed, "other": len(records) - passed - failed}


def write_outputs(
    records: Sequence[dict[str, object]], json_path: Path, table_path: Path, mode: str
) -> dict[str, int]:
    summary = summarize(records)
    payload = {
        "schema": "dsws-realshape-bench-v1",
        "mode": mode,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formula": "TF = 2*padded_M*N*K*repetitions*tick_hz/summed_ticks/1e12; real_TF = TF*real_M/padded_M",
        "summary": summary,
        "records": records,
        "shape_inventory": shape_inventory(records),
    }
    json_path = json_path.expanduser().resolve()
    table_path = table_path.expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = render_table(records, summary, payload["shape_inventory"])
    table_path.write_text(table, encoding="utf-8")
    sys.stdout.write(table)
    print(f"JSON: {json_path}")
    print(f"table: {table_path}")
    return summary


def collect_offline_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).expanduser() for value in args.logs]
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser()
        for pattern in args.glob:
            paths.extend(log_dir.glob(pattern))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise SystemExit("offline mode found no input logs")
    return unique


def run_live(args: argparse.Namespace) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    halted_after: str | None = None
    gpu_run = SCRIPT_DIR / "gpu_run.sh"
    # Apply the requested dispatch geometry. LIVE_TM is the super-tile M that shape_supported()
    # pads against, so it MUST track G or the padding correction would be computed for a
    # different tiling than the one actually launched.
    # FM ADDED 2026-07-26. It was a hardcoded LIVE_FM=1 while --g was a flag, so this harness could
    #   only ever dispatch DSWS2_FM=1. Running it against an FM=2 bin would have launched FM=1
    #   geometry at an FM=2 kernel: the super-tile is G*16*FM rows, so every shape's M-padding would
    #   have been computed for a 96-row tile while the bin wanted 128 -- a silent geometry mismatch of
    #   exactly the kind that wedged the card earlier today. A knob that is a flag on one axis and a
    #   constant on a coupled axis is a trap, not a default.
    # FN ADDED 2026-07-29, for exactly the reason the FM note above gives. FN became an env knob that
    #   day (DSWS2_FN / build_flow.sh FN); LIVE_TN = LIVE_FN*16 is what every shape's N-PADDING is
    #   computed against. Left hardcoded at 4, `--fn 2` would have padded N for a 64-col panel while
    #   the bin wanted 32 -- the identical silent geometry mismatch the FM comment describes, on the
    #   other axis. DSWS2_FN is also passed to the dispatcher below: without it the host defaults
    #   FNc=4 and disagrees with the bin (fail-loud via the oracle, but there is no reason to rely on
    #   that).
    global LIVE_G, LIVE_FM, LIVE_TM, LIVE_FN, LIVE_TN
    LIVE_G = args.g
    LIVE_FM = args.fm
    LIVE_FN = args.fn
    LIVE_TM = LIVE_G * 16 * LIVE_FM
    LIVE_TN = LIVE_FN * 16

    # ---- CONFIG-OF-RECORD PRE-FLIGHT (2026-07-29). Fail BEFORE the first dispatch, not after it. ----
    # gpu_run.sh refuses a non-standard geometry, and this sweep halts on the first nonzero return --
    # so without DSWS_ALLOW_NONSTD the run dies one shape in, having burned a card claim, with the
    # reason buried in a subprocess log. Worse, this script never passed the flag AT ALL, so the
    # 1 WG/CU config (--pool 64) could not be swept even deliberately.
    # The flag is NOT auto-emitted on detected deviation: the entire point of DSWS_ALLOW_NONSTD is that
    # deviating is an EXPLICIT act, and a harness that quietly sets it for you is the exact failure the
    # guard exists to prevent.
    deviations = []
    if args.waves != 16:
        deviations.append(f"FLOW_WAVES={args.waves} (standard 16)")
    if args.pool != 128:
        deviations.append(f"ML8_POOL={args.pool} (standard 128 = 2 WG/CU; 64 = 1 WG/CU)")
    if args.segk not in (64, 128, 256):
        deviations.append(f"DSWS2_SEGK={args.segk} (sanctioned {{64,128,256}})")
    if deviations and not args.allow_nonstd:
        sys.stderr.write(
            "REFUSED: this geometry deviates from the config of record and --allow-nonstd was not passed:\n"
            + "".join(f"    - {d}\n" for d in deviations)
            + "  gpu_run.sh would refuse every shape. Re-run with --allow-nonstd and name the reason\n"
              "  in --tag, so the logs record WHY the sweep deviated.\n"
        )
        raise SystemExit(2)   # NOT `return 2`: run_live() -> list[dict], an int would reach write_outputs()
    if deviations and args.tag == "rs":
        sys.stderr.write(
            "REFUSED: --allow-nonstd with the default --tag 'rs'. The standing rule is that a deviation\n"
            "  must be NAMED IN THE LOGNAME. Pass something like --tag secondary_1wgcu.\n"
        )
        raise SystemExit(2)
    for shape in SHAPES:
        label, real_m, n, k = shape
        # Ground truth for attribution: we KNOW which shape we are about to dispatch, so the parser must
        #   not have to reverse-map padded geometry to guess it. Two shapes (mlmf_mamba_in_proj N=4200 and
        #   mlmf_in_proj_ML8PAD N=4208) both pad to N=4224 and are indistinguishable from the log alone.
        global EXPECTED_SHAPE
        EXPECTED_SHAPE = (label, real_m, n, k)
        supported, reason, padded_m, padded_n = shape_supported(shape)
        if not supported:
            records.append(failure_record("", f"UNSUPPORTED_GEOMETRY: {reason}", status="UNSUPPORTED"))
            records[-1]["shape_candidates"] = [label]
            records[-1]["geometry"] = {"real_m": real_m, "padded_m": padded_m, "n": n, "k": k}
            continue
        if halted_after is not None:
            records.append(
                failure_record("", f"NOT_RUN: sweep halted after nonzero gpu_run.sh return for {halted_after}", status="NOT_RUN")
            )
            records[-1]["shape_candidates"] = [label]
            records[-1]["geometry"] = {"real_m": real_m, "padded_m": padded_m, "n": n, "k": k}
            continue

        command = [
            str(gpu_run),
            f"{args.tag}_{label}_M{real_m}",
            "--",
        ]
        if args.allow_nonstd:
            # gpu_run.sh scans the post-`--` kv list for the literal DSWS_ALLOW_NONSTD=1 (its case arm
            #   at :83), so it must be spelled exactly and must sit in this list, not the parent env.
            command.append("DSWS_ALLOW_NONSTD=1")
        if args.chunk:
            command.append(f"ML8_COOP_CHUNK={args.chunk}")
        if args.chunk_maxs:
            # ADDED 2026-07-26. The compositor-safety cap defaults to 0.75s and FM=2 measures 0.81s per
            #   chunk, so without this every FM=2 shape would ABORT ("chunk wall > cap -> ABORT remaining")
            #   and the sweep would report 30 failures that are not kernel failures at all.
            #   NOTE the cap is NOT tile-proportional -- measured 0.81s at BOTH ML8_COOP_CHUNK=512 and 256
            #   -- so lowering --chunk cannot substitute for raising this. Unlike DEADMAN_TICKS (an
            #   anti-brick floor that must never move) this is a designed knob: occ_dispatch.cpp:1599 names
            #   raising it as the remedy, and the check is reactive (measured AFTER the chunk completes).
            command.append(f"ML8_COOP_CHUNK_MAXS={args.chunk_maxs}")
        # ---- AUTO-POOL (2026-07-29). ML8_POOL DERIVED FROM THE SHAPE, NOT PINNED. ----
        # MEASURED: on ml8_moe_ffn_gate_up M64 (8 output tiles) the runtime is 100% per-dispatch FIXED
        #   COST, and that cost SCALES WITH WORKGROUP COUNT, not with work. Sweeping ML8_POOL
        #   64/16/8/4/2/1 gave real TF 0.1/0.4/0.5/0.5/0.4/0.2 and span/chunk 116k/38k/27.5k/24.8k/
        #   35k/61k ticks -- a clear interior optimum at 4-8 and a 5x win. Reproduced on
        #   ml8_moe_ffn_down M64 (different N AND K): 0.111 -> 0.5 TF. Launching all 64 CUs for 8 tiles
        #   means most WGs ramp up, find nothing, and retire -- AND THAT RAMP IS THE RUNTIME.
        # THE RULE: keep >=TILES_PER_WG super-tiles per workgroup, capped by --pool.
        #   Fits both measured ends: M64 (TOTAL_super=64) -> 6, M2048 (TOTAL_super=11520) -> 64.
        # NOTE TOTAL_super is INDEPENDENT of superM for the small shapes (M=64 gives MTLsuper=1 at both
        #   superM=256 and 64), so this changes ONLY pool and stays comparable to the pinned-pool table.
        # NOT ON BY DEFAULT: --pool-auto must be passed. A harness that silently retunes the dispatch
        #   geometry per shape would make every cross-run comparison meaningless.
        pool_used = args.pool
        if args.pool_auto:
            mtl_super = padded_m // LIVE_TM
            ntl = padded_n // LIVE_TN
            n_kseg = k // args.segk if args.segk else 0
            total_super = mtl_super * ntl * n_kseg
            pool_used = max(1, min(args.pool, total_super // args.tiles_per_wg))
            print(f"  [auto-pool] {label} M{real_m}: TOTAL_super={total_super} "
                  f"(MTLsuper={mtl_super} NTL={ntl} n_kseg={n_kseg}) -> ML8_POOL={pool_used} "
                  f"(cap {args.pool}, >={args.tiles_per_wg}/WG)", flush=True)
        command += [
            f"SSWIN={args.sswin}",
            f"FLOW_WAVES={args.waves}",
            f"ML8_POOL={pool_used}",     # pinned to --pool, or derived per shape under --pool-auto
            "DSWS2_FLOW=1",
            f"DSWS2_FM={LIVE_FM}",
            f"DSWS2_FN={LIVE_FN}",   # ADDED 2026-07-29 -- host default is FNc=4; unsent, it disagrees with an FN!=4 bin
            f"DSWS2_G={LIVE_G}",
            f"DSWS2_ACC_N={args.acc_n}",
            "FLOW_POOL_N=1",
            f"DSWS2_SEGK={args.segk}",
            f"DSWS2_K={k}",
            f"DSWS2_ORACLE_MTL={padded_m // LIVE_TM}",
            f"DSWS2_ORACLE_NTL={padded_n // LIVE_TN}",
            f"DSWS2_ORACLE_STRIDE={args.stride}",
            f"DSWS2_TARGET_SECS={args.target_secs}",
            "STAGINSTR=1",
            "FORENSICS=0",
            "TFPROBE=1",
            "./occ_dispatch",
            "--dsws2",
        ]
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sys.stdout.write(completed.stdout)
        log_match = re.search(r"\[gpu_run\].*?\blog=(\S+)", completed.stdout)
        log_path = Path(log_match.group(1)) if log_match else None
        if completed.returncode != 0:
            reason_text = f"GPU_RUN_NONZERO: rc={completed.returncode}; whole sweep halted"
            records.append(failure_record(log_path or "", reason_text))
            records[-1]["shape_candidates"] = [label]
            records[-1]["geometry"] = {"real_m": real_m, "padded_m": padded_m, "n": n, "k": k}
            halted_after = f"{label}_M{real_m}"
            continue
        if log_path is None:
            records.append(failure_record("", "PROVENANCE_MISSING: gpu_run.sh did not emit its log path"))
            halted_after = f"{label}_M{real_m}"
            continue
        try:
            records.append(success_record(parse_log(log_path)))
        except LogRejected as exc:
            records.append(failure_record(log_path, str(exc)))
            halted_after = f"{label}_M{real_m}"
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    offline = subparsers.add_parser("offline", help="parse archived logs without dispatching")
    offline.add_argument("logs", nargs="*", help="explicit log paths")
    offline.add_argument("--log-dir", help="directory searched with each --glob")
    offline.add_argument("--glob", action="append", default=[], help="glob under --log-dir; repeatable")
    offline.add_argument("--json", type=Path, required=True, help="machine-readable output path")
    offline.add_argument("--table", type=Path, required=True, help="human-readable output path")

    live = subparsers.add_parser("live", help="dispatch every shape through gpu_run.sh; requires human approval")
    live.add_argument("--target-secs", type=float, default=1.5)
    live.add_argument("--stride", type=int, default=8)
    # SEGK is a sanctioned knob in {64,128,256} (kmbandy 2026-07-26). It is what makes the
    #   n_kseg>=2 shapes reachable: mlmf_router_MLP (K=256) is n_kseg=1 at 256, n_kseg=2 at 128.
    live.add_argument("--segk", type=int, default=256, choices=(64,128,256), help="DSWS2_SEGK; must match the bin")
    # Dispatch config. These ONLY affect what is launched; the parser, the tick-derivation and
    # every self-validation check are untouched by them. G also sets the super-tile M (G*16*FM),
    # which is what the padding correction is computed from.
    live.add_argument("--g", type=int, default=LIVE_G, help="DSWS2_G (super-tile M = G*16*FM)")
    live.add_argument("--fm", type=int, default=LIVE_FM, choices=(1, 2, 4, 8),
                      help="DSWS2_FM; MUST match the bin (super-tile M = G*16*FM). Was hardcoded to 1 until 2026-07-26. "
                           "Widened 1,2 -> 1,2,4,8 on 2026-07-29: the real limiter is the dyn-VGPR grow target "
                           "NFV = roundup16(32 + 8*FM*FN + 2*FM + 2*FN) <= 128, which build_flow.sh and occ_dispatch.cpp "
                           "both now enforce (FM=4 FN=4 needs 176 and is refused; FM=4 FN=2 needs 112 and is fine).")
    live.add_argument("--fn", type=int, default=LIVE_FN, choices=(1, 2, 4, 8),
                      help="DSWS2_FN; MUST match the bin (N-panel = FN*16). Hardcoded to 4 until 2026-07-29, while "
                           "LIVE_TN = FN*16 silently drove every shape's N-PADDING -- the same coupled-axis trap "
                           "the --fm note describes. Power-of-two only: the DSWS2_PREFETCH P2 block decode is a shift.")
    live.add_argument("--acc-n", type=int, default=3, help="DSWS2_ACC_N (GROUPS = G/ACC_N)")
    live.add_argument("--sswin", type=int, default=32, help="SSWIN control-window depth")
    # CONFIG OF RECORD (kmbandy 2026-07-26): 2 WG/CU = 16 waves x 128 WGs = 2048 resident.
    #   ML8_POOL WAS NEVER PASSED BY THIS SCRIPT, so every sweep it has ever run launched 64 WGs
    #   (1 WG/CU) regardless of intent -- the dispatcher's silent default. gpu_run.sh now refuses
    #   that, but the fix belongs here too: pass it explicitly, always, and record it in the JSON.
    live.add_argument("--waves", type=int, default=16, help="FLOW_WAVES (selects occ_dsws2_w<N>_flow_gd.bin)")
    live.add_argument("--pool", type=int, default=128, help="ML8_POOL = number of WGs (128 = 2 WG/CU, the config of record)")
    live.add_argument("--tag", type=str, default="rs", help="log-name prefix, so arms do not overwrite each other")
    # ADDED 2026-07-29. This script never passed DSWS_ALLOW_NONSTD at all, so gpu_run.sh refused every
    #   shape at --pool 64 and the 1 WG/CU config (the fastest ever measured, +63%) could not be swept
    #   even deliberately. NOT auto-set on detected deviation: the whole point of the flag is that
    #   deviating is an EXPLICIT act, so a harness that quietly sets it defeats the guard it satisfies.
    # ADDED 2026-07-29. See the AUTO-POOL block in run_live() for the measurements behind the rule.
    live.add_argument("--pool-auto", action="store_true",
                      help="derive ML8_POOL per shape as min(--pool, TOTAL_super/--tiles-per-wg) instead of pinning it. "
                           "Measured 5x on ml8_moe_ffn_gate_up M64 and ml8_moe_ffn_down M64, where the runtime is "
                           "100%% per-dispatch fixed cost that scales with WORKGROUP COUNT, not with work.")
    live.add_argument("--tiles-per-wg", type=int, default=10,
                      help="minimum super-tiles per workgroup under --pool-auto (default 10; fits both measured ends)")
    live.add_argument("--allow-nonstd", action="store_true",
                      help="pass DSWS_ALLOW_NONSTD=1 to gpu_run.sh, permitting a geometry that deviates from the "
                           "config of record (e.g. --pool 64 = 1 WG/CU). Requires a non-default --tag naming the reason.")
    # ML8_COOP_CHUNK bounds tiles per dispatch. Unset => the dispatcher's 512-tile compositor-safe
    # default. A large value collapses the problem to ONE chunk, which is what the pre-2026-07-21
    # broken cap did; the 0.75s abort CANNOT fire mid-chunk, only between chunks/reps. RULE 7.
    live.add_argument("--chunk", type=int, default=0, help="ML8_COOP_CHUNK tiles/dispatch; 0 = dispatcher default (512)")
    live.add_argument("--chunk-maxs", type=float, default=0.0,
                      help="ML8_COOP_CHUNK_MAXS compositor-safety cap in seconds; 0 = dispatcher default (0.75). "
                           "FM=2 needs ~0.85 (its chunk measures 0.81s, and the cost is NOT tile-proportional).")
    live.add_argument("--json", type=Path, required=True, help="machine-readable output path")
    live.add_argument("--table", type=Path, required=True, help="human-readable output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "offline":
        if not args.logs and (not args.log_dir or not args.glob):
            parser.error("offline mode needs explicit logs or --log-dir with at least one --glob")
        records = parse_paths(collect_offline_paths(args))
    else:
        if args.target_secs <= 0 or args.stride <= 0:
            parser.error("live --target-secs and --stride must be positive")
        records = run_live(args)
    summary = write_outputs(records, args.json, args.table, args.mode)
    return 0 if summary["fail"] == 0 and summary["other"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
