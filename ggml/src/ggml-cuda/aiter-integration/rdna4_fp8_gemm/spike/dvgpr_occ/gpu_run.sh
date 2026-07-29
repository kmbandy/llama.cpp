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

# ---- CONFIG OF RECORD (kmbandy, 2026-07-26). MECHANICAL, because "I know the config" is not a control.
#      On 2026-07-26 an entire POLLSTAGE campaign was measured at 64 WGs / 1920 waves and WITHOUT the
#      prefetch, because the launch geometry was reconstructed from a stale doc instead of the standard.
#      That is the SECOND documented instance of silently departing from the config of record.
#      THE STANDARD: 2 WG/CU -- FLOW_WAVES=16 + ML8_POOL=128 (128 WGs x 16 waves = 2048 waves).
#      DSWS2_SEGK is a FREE knob within {64,128,256}: all three publish 13,824B LDS, so 2x13,824
#      = 27,648B < 65,536 and 2x16 = 32/32 wave slots -- every one of them fits 2 WG/CU.
#      Deviating must be an EXPLICIT act: pass DSWS_ALLOW_NONSTD=1 and say why in the logname.
STD_WAVES=16; STD_POOL=128
W=""; P=""; SEGK=""; NONSTD=""
for kv in "$@"; do
  case "$kv" in
    FLOW_WAVES=*)        W="${kv#FLOW_WAVES=}";;
    ML8_POOL=*)          P="${kv#ML8_POOL=}";;
    DSWS2_SEGK=*)        SEGK="${kv#DSWS2_SEGK=}";;
    DSWS_ALLOW_NONSTD=1) NONSTD=1;;
  esac
done
if [ -z "$NONSTD" ]; then
  [ "$W" = "$STD_WAVES" ] || die "FLOW_WAVES=${W:-unset} but the config of record is $STD_WAVES (2 WG/CU).
             128 WGs x 16 waves is the standard kmbandy set. If you MEAN to deviate, pass
             DSWS_ALLOW_NONSTD=1 and name the reason in the logname -- deviation is an explicit act."
  [ "$P" = "$STD_POOL" ] || die "ML8_POOL=${P:-unset} but the config of record is $STD_POOL (2 WG/CU, 128 WGs).
             ML8_POOL unset silently means 64 WGs = 1 WG/CU -- that is how 2026-07-26 was lost.
             If you MEAN to deviate, pass DSWS_ALLOW_NONSTD=1."
  case "$SEGK" in
    64|128|256) ;;
    *) die "DSWS2_SEGK=${SEGK:-unset} is outside the sanctioned range {64,128,256} (kmbandy 2026-07-26).
             All three fit 2 WG/CU at 13,824B LDS. Pass DSWS_ALLOW_NONSTD=1 to override.";;
  esac
fi

# ---- stale-bin trap: never measure a binary you did not just build ----
BIN=$(ls -t "$DIR"/occ_dsws2_*_flow_gd.bin 2>/dev/null | head -1)
[ -n "$BIN" ] || die "no flow .bin found -- did build_flow.sh fail? (it rm -f's its bin on failure)"
[ "$BIN" -nt "$KSRC" ] || die "$(basename "$BIN") is OLDER than the kernel source -- STALE BIN. Rebuild."

# ---- .bin / .lds SIDECAR PAIRING (added 2026-07-26 after I RESET THE GPU) ----
#   ON 2026-07-26 17:08 I COPIED fm2.bin -> occ_dsws2_w16_flow_gd.bin AND LEFT THE .lds BEHIND.
#   build_flow.sh emits TWO artifacts per build -- $tag.bin and $tag.lds (a 4-byte u32 holding the
#   LDS byte count the host must allocate). They are SEPARATE FILES. The bin was the FM=2 tile
#   (needs 17,920B); the .lds was from a build 14 minutes earlier (13,824B). The host allocated
#   13,824B for a kernel needing 17,920B, the kernel ran past its allocation, MES went
#   unrecoverable -> MODE1 reset -> VRAM lost.
#
#   EVERY OTHER GUARD IN THIS FILE PASSED. The stale-bin trap only compares the .bin to the kernel
#   SOURCE -- a freshly-copied bin is newer than the source, so it sails through. Nothing checked
#   the two halves of the build against EACH OTHER.
#
#   The host DID print the defect ("host reconstruction says 67072B but the BIN PUBLISHES 13824B")
#   -- but that line prints on EVERY run and is normally benign, so it carries no signal. THAT IS
#   THE REAL LESSON: a warning that fires unconditionally is not a control. Hence this is a hard
#   refusal, NOT a warning, and deliberately NOT fail-soft like the claim check above -- a missing
#   or stale sidecar is never ambiguous and never someone else's outage.
LDSFILE="${BIN%.bin}.lds"
[ -f "$LDSFILE" ] || die "$(basename "$BIN") has NO .lds sidecar. build_flow.sh emits the pair;
      you copied a bin without it. The host would allocate the WRONG LDS size -> GPU RESET. Rebuild."
if [ "$BIN" -nt "$LDSFILE" ]; then
  die "SIDECAR MISMATCH: $(basename "$BIN") is NEWER than its .lds sidecar.
      bin  $(stat -c %y "$BIN"   | cut -c1-19)
      lds  $(stat -c %y "$LDSFILE" | cut -c1-19)  (says $(od -An -tu4 -N4 "$LDSFILE" | tr -d ' ')B)
      They are written together by one build, so the bin can never legitimately be newer.
      You copied/cached a .bin without its .lds. THIS EXACT MISTAKE RESET THE GPU ON 2026-07-26.
      Rebuild, or copy BOTH files together."
fi
echo "  [gpu_run] LDS sidecar OK: $(od -An -tu4 -N4 "$LDSFILE" | tr -d ' ')B, paired with $(basename "$BIN")"

# ---- DO I STILL HOLD THE CLAIM? (added 2026-07-26 after I collided with another session) ----
#   ON 2026-07-26 MY CLAIM EXPIRED AT ITS 3h TTL AT 11:22:10. The board correctly promoted the queued
#   mlambaformer session at 11:22:16. My dispatch driver had NO notion of claim validity and kept
#   going -- 8 more dispatches, 11:22:09 -> 11:25:19, on a card someone else legitimately held.
#   Every other guard in this file passed. A long driver can silently outlive its own TTL and
#   collide, and nothing in the stack notices. "I know I have the card" is not a control.
#
#   FAIL-SOFT vs FAIL-CLOSED, deliberately asymmetric:
#     - board unreachable / no session id / no claims parsed -> WARN and PROCEED. A board outage
#       must never halt GPU work, and absence of evidence is not evidence of a conflict.
#     - board POSITIVELY reports a different holder                -> REFUSE. That is a real collision.
#   Escape hatch: DSWS_SKIP_CLAIM_CHECK=1 (say why in the logname).
if [ "${DSWS_SKIP_CLAIM_CHECK:-0}" != "1" ]; then
  MCP="${MAD_LAB_MCP_HTTP:-http://100.102.191.30:18800}"
  CLAIM_JSON=$(curl -s -m 5 "$MCP/board/check?machine=mad-lab-main&resource=gpu%3AR9700" 2>/dev/null)
  if [ -z "$CLAIM_JSON" ]; then
    echo "  [gpu_run] NOTE: board unreachable ($MCP) -- cannot verify claim. Proceeding." >&2
  elif [ -z "${CLAUDE_CODE_SESSION_ID:-}" ]; then
    echo "  [gpu_run] NOTE: CLAUDE_CODE_SESSION_ID unset -- cannot verify claim. Proceeding." >&2
  else
    HOLDER=$(printf '%s' "$CLAIM_JSON" | python3 -c "
import json,sys
try: c=json.load(sys.stdin).get('claims') or []
except Exception: sys.exit(0)
print(c[0]['holder'] if c else '')" 2>/dev/null)
    if [ -n "$HOLDER" ] && [ "$HOLDER" != "$CLAUDE_CODE_SESSION_ID" ]; then
      die "gpu:R9700 is held by ANOTHER SESSION ($HOLDER), not you ($CLAUDE_CODE_SESSION_ID).
             This is the 2026-07-26 collision guard: your claim probably EXPIRED mid-campaign and the
             board handed the card on. Re-claim (board_claim) before dispatching, and size ttl_hours
             to the whole campaign. Override only with DSWS_SKIP_CLAIM_CHECK=1."
    fi
  fi
fi

mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)
LOG="$LOGDIR/${NAME}_${TS}.log"

resets() { journalctl -b 0 -k --no-pager 2>/dev/null | grep -c "GPU reset begin"; }
R0=$(resets)

echo "  [gpu_run] bin=$(basename "$BIN")  log=$LOG  resets_before=$R0"

# ---- DRY RUN: exercise every guard, touch NO hardware. GPU_RUN_DRY=1 ----
#   ADDED 2026-07-26 FOR A CONCRETE REASON. The claim guard above has a FAIL-SOFT branch whose
#   whole behaviour is "proceed and dispatch". I tried to TEST that branch by pointing the board
#   URL at a dead address and running for real -- which launched occ_dispatch against a card
#   another session held, i.e. I reproduced the exact collision the guard exists to prevent,
#   minutes after apologising for it. (It was inert only by luck: the test command omitted
#   DSWS2_K/ORACLE_MTL/ORACLE_NTL so occ_dispatch opened the KFD node and exited before queuing
#   any PM4. No kernel ran, no reset, the other session was untouched.)
#
#   THE LESSON, WHICH GENERALISES: A FAIL-SOFT PATH CANNOT BE VERIFIED BY EXECUTING IT AGAINST
#   LIVE HARDWARE -- "proceed" IS the behaviour under test. Guard changes get verified HERE,
#   with GPU_RUN_DRY=1, or by reading the code. Never by firing a real dispatch to see what happens.
if [ "${GPU_RUN_DRY:-0}" = "1" ]; then
  echo "  [gpu_run] *** DRY RUN -- all guards passed, NOTHING DISPATCHED. ***"
  echo "  [gpu_run] would run: env $*"
  rm -f "$LOG"
  exit 0
fi

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
