#!/usr/bin/env bash
# gpu_run.sh — the ONLY sanctioned way to put work on the R9700.
#
# Enforces the rules in CLAUDE.md mechanically, because on 2026-07-14 I proved that
# "I know the rules" is not a control. The R9700 drives the displays and hosts other
# agents' live sessions; a brick costs ~1M tokens of someone else's context.
#
#   usage:  ./gpu_run.sh <logname> -- <ENV=v ...> ./occ_dispatch --dsws2
#
# It will:
#   1. REFUSE to run if the previous run hung (latch file). Only a human clears it.
#   2. REFUSE to run if DEADMAN=0 or DEADMAN_TICKS is raised above the 0.5s guard.
#   3. REFUSE to run if the .bin is older than the kernel source (stale-bin trap).
#   4. Log to REAL DISK (~/dsws_gpu_logs) — survives a MODE1 reset.
#   5. Snapshot the GPU reset count before/after and SHOUT if the card reset.
#   6. Capture the kernel journal after every run — the only record that survives a brick.
#   7. Latch a hang so the NEXT invocation is blocked (rule 3: a hang is a full stop).
set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HOME/dsws_gpu_logs"
LATCH="$DIR/.gpu_last_hang"
KSRC="$DIR/occ_kernel_dsws_flow.s"

die() { echo "  [gpu_run] REFUSED: $*" >&2; exit 2; }

[ $# -ge 3 ] || die "usage: ./gpu_run.sh <logname> -- <ENV...> ./occ_dispatch --dsws2"
NAME="$1"; shift
[ "$1" = "--" ] || die "expected '--' after <logname>"; shift

# ---- RULE 3: a hang is a FULL STOP. The previous hang must be cleared by a human. ----
if [ -f "$LATCH" ]; then
  echo "  [gpu_run] *** BLOCKED: the previous run HUNG ***" >&2
  cat "$LATCH" >&2
  die "a hang is a full stop (CLAUDE.md rule 3). Go offline, root-cause it.
             A human clears this with:  rm $LATCH"
fi

# ---- RULE 4: the deadman is the brick guard, not a knob. ----
for kv in "$@"; do
  case "$kv" in
    DEADMAN=0)        die "DEADMAN=0 removes the anti-brick guard (rule 4)";;
    DEADMAN_TICKS=*)  v="${kv#DEADMAN_TICKS=}"
                      [ "$v" -gt 50000000 ] 2>/dev/null && \
                        die "DEADMAN_TICKS=$v > 0.5s guard. Raising it caused 3 bricks on 2026-07-14.
             A false kill means a MISSING deadman_progress site -- fix that, not the threshold (rule 4)";;
  esac
done

# ---- RULE 7: bandwidth-risky knobs must be tried SMALL first. A kernel that saturates HBM
#      starves the compositor and kills the desktop WITHOUT ever resetting the GPU -- so none of
#      the reset/hang checks above will catch it. (NTLOAD=1 at full scale did exactly that.)
BW_RISK=""; CHUNK=""
for kv in "$@"; do
  case "$kv" in
    NTLOAD=1)               BW_RISK="$kv";;
    ML8_COOP_CHUNK=*)       CHUNK="${kv#ML8_COOP_CHUNK=}";;
  esac
done
if [ -n "$BW_RISK" ]; then
  if [ -z "$CHUNK" ] || [ "$CHUNK" -gt 1024 ] 2>/dev/null; then
    die "$BW_RISK raises HBM traffic and can STARVE THE COMPOSITOR (kills the desktop with NO
             GPU reset -- nothing else here catches it). Run it small first:
             ML8_COOP_CHUNK<=1024  (you passed: ${CHUNK:-unset})"
  fi
fi

# ---- stale-bin trap: never measure a binary you did not just build ----
BIN=$(ls -t "$DIR"/occ_dsws2_*_flow_gd.bin 2>/dev/null | head -1)
[ -n "$BIN" ] || die "no flow .bin found -- did build_flow.sh fail? (it rm -f's its bin on failure)"
[ "$BIN" -nt "$KSRC" ] || die "$(basename "$BIN") is OLDER than the kernel source -- STALE BIN. Rebuild."

mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)
LOG="$LOGDIR/${NAME}_${TS}.log"

resets() { journalctl -b 0 -k --no-pager 2>/dev/null | grep -c "GPU reset begin"; }
R0=$(resets)

echo "  [gpu_run] bin=$(basename "$BIN")  log=$LOG  resets_before=$R0"
env "$@" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

R1=$(resets)

# ---- RULE 6/7: capture forensics; the journal is the ONLY record that survives a brick ----
journalctl -b 0 -k --no-pager --since "-3 min" 2>/dev/null \
  | grep -iE "amdgpu|gfxhub|page fault|MES|MODE1|VRAM lost|Resetting wave" > "$LOG.journal" || true

if [ "$R1" -gt "$R0" ]; then
  echo "  [gpu_run] *** THE GPU RESET DURING THIS RUN ($((R1-R0)) reset(s)) -- see $LOG.journal ***" >&2
fi

# A host-side REFUSE prints INCOMPLETE but NEVER DISPATCHED -- it is not a hang, do not latch it.
if grep -qE "\*\*\* REFUSE:" "$LOG"; then
  echo "  [gpu_run] host REFUSED the geometry before dispatch (no GPU work) -- not latching." >&2
  exit 4
fi
# ---- WORK-EXACTNESS IS A FULL STOP (2026-07-20). Same class as a hang. ----
#   An audit of 31 runs on the oracle shape found 10 that SILENTLY DROPPED WORK -- and the worst of
#   them ALSO printed "oracle CLEAN", because the oracle samples 32 of 16384 tiles (0.2%). A run that
#   drops work and reports clean will be logged as a result unless something refuses it. Dropped work
#   also FLATTERS TF (less work, same span), so this invalidates the perf number, not just correctness.
if grep -q "WORK-INEXACT" "$LOG"; then
  { echo "  WORK-INEXACT (dropped/duplicated work) run: $LOG"; echo "  at: $(date)"; echo "  cmd: $*";
    grep -A4 "WORK-INEXACT" "$LOG" | sed 's/^/  /'; } > "$LATCH"
  echo "  [gpu_run] *** WORK-INEXACT LATCHED -- the kernel dropped or double-counted work." >&2
  echo "  [gpu_run] *** THE THROUGHPUT NUMBER IS INVALID. Further dispatches BLOCKED until a human clears $LATCH ***" >&2
  exit 5
fi
# A run that never reached the STAGINSTR verdict cannot be trusted either -- absence of the gate is
#   not a pass. (STAGINSTR=0 builds have no counters; those must not be used for perf verdicts.)
# NOTE the greps below match "[dsws2 WORK-EXACT]" IN BRACKETS -- only the PASS path prints that. The
#   host's CANNOT-EVALUATE notice also contains the substring "WORK-EXACT", so a loose grep would let
#   an unverdicted run masquerade as a verdicted one. Absence of a gate must never read as a pass.
if grep -q "WORK-EXACT: CANNOT-EVALUATE" "$LOG"; then
  echo "  [gpu_run] *** NO CORRECTNESS VERDICT: the STAGINSTR counters are absent (STAGINSTR=0)." >&2
  echo "  [gpu_run] *** Work-exactness was NOT checked. Not latching (this is a coverage gap, not a" >&2
  echo "  [gpu_run] *** detected fault) -- but DO NOT quote this run's throughput as validated." >&2
  echo "  [gpu_run] *** For an instrumentation ablation use CNTLEAN=1, which KEEPS the verdict." >&2
elif grep -qE "\[dsws2 STAGINSTR\]" "$LOG" && ! grep -qE "\[dsws2 WORK-EXACT\]|WORK-INEXACT" "$LOG"; then
  echo "  [gpu_run] *** WARNING: STAGINSTR ran but no WORK-EXACT verdict -- host is stale, rebuild it. ***" >&2
fi
if grep -qE "INCOMPLETE|WARN chunk .* -> ABORT|INVALID RUN" "$LOG"; then
  { echo "  hung/invalid run: $LOG"; echo "  at: $(date)"; echo "  cmd: $*"; } > "$LATCH"
  echo "  [gpu_run] *** HANG/INVALID LATCHED -- further dispatches BLOCKED until a human clears $LATCH ***" >&2
  exit 3
fi

exit "$RC"
