#!/usr/bin/env bash
# build_flow.sh — FIX 1 (flow economy) bin (occ_kernel_dsws_flow.s). N-deep pool + ROLE mailbox +
#   coordinator. Bin name matches occ_dispatch.cpp DSWS2_FLOW path: occ_dsws2_<c>c<a>a<b>b_flow_gd.bin
#   OFFLINE/CPU only. Usage: ./build_flow.sh [NCOMP NAFEED NBFEED]
#   Env: POOL_N=3 PHASEPROBE={0|1} NOCFLUSH={0|1} CSTORE={0|1} SLEEPN=N COORD_PERIOD=N DIAG=0
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
fail=0

# ============================ CONFIG OF RECORD (kmbandy, 2026-07-26) ============================
# WHY THIS EXISTS: the recurring defect on this project is not a bad number, it is a SILENTLY WRONG
# CONFIG -- a mechanism left out, a geometry reconstructed from a stale doc, a superseded knob still
# read. On 2026-07-26 the whole POLLSTAGE campaign was measured at 1 WG/CU with DSWS2_PREFETCH=0,
# because the profile was copied from a findings doc instead of being defined in one place.
# So the standard now lives HERE, in the tool, and applies unless you explicitly opt out.
#
# These are DEFAULTS ONLY (`:=` sets only when unset), so an explicit env still wins and every
# existing ablation/A-B invocation keeps working unchanged.
#
#   *** THESE VALUES WERE STALE UNTIL 2026-07-31 AND THIS BLOCK WAS THE THING ASSERTING THEM. ***
#   They still read "2 WG/CU / WAVES=16 / ML8_POOL=128 / 13,824B LDS" long after the config of record
#   became WAVES=6 at 34,304B and 1 WG/CU. A bare `./build_flow.sh` therefore built WAVES=16 G=6 FM=1
#   ACC_N=3 and PRINTED "CONFIG OF RECORD" over it -- precisely the silently-wrong-config failure this
#   block exists to prevent, committed by the block itself. Corrected below; see DSWS_TESTING_LOG.md 82.
#
#   1 WG/CU        WAVES=6       (dispatch with ML8_POOL=64 -> 64 WGs x 6 = 384 waves)
#                                34,304B LDS x 2 = 68,608 > 65,536, so 2 WG/CU is REFUSED at this geometry
#                                (occ_dispatch.cpp host gate). Occupancy is a retired axis: 1 WG/CU
#                                measured FASTER than 3 on 2026-07-31 (log 79/85).
#   SEGK           {64,128,256}  LDS is SEGK-independent under SELFSERVE (ACC_STRIDE = FM*FN*1024 has no
#                                SEGK term), so all three fit identically. SEGK is the strongest measured
#                                axis: 256/128/64 -> 17.5/10.2/5.6 TF (log 85).
#   JDEPTH         PINNED TO 1   -- SELFSERVE requires it. The ksi%J lead-gate ("k-slice filter") is
#                                NOT AVAILABLE under SELFSERVE at any SEGK. Restoring it is a DESIGN
#                                change, not a knob. Verified 2026-07-26 across SEGK 64/128/256.
#   POOL_N         INERT         -- verified byte-identical at 1/2/4 (SELFSERVE dead-staging fix, 07-25).
#   MAXFAT         INERT         -- the FATTOK token layer is compiled to no-ops under BATONGATE.
#   FN             KNOB AS OF 2026-07-29 -- was a hard-coded literal `-defsym,FN=4` in mkflow (and a
#                                hard-coded `const int FNc = 4` on the host). Unlocked to test the
#                                feed-loads/WMMA = (FM+FN)/(FM*FN) axis at constant super-tile M.
#                                *** THE POINT OF THE KNOB IS THE CONTROL ARM FM=4 FN=2 ***: its feed
#                                ratio is 0.750, IDENTICAL to the current FM=2 FN=4, and its NFV is
#                                112, also identical. If it does NOT measure the same TF as FM=2 FN=4,
#                                the feed-ratio model is wrong and the frag-grid extrapolation dies.
: ${WAVES:=6}    ; : ${SEGK:=256} ; : ${G:=8}      ; : ${FM:=2}    ; : ${ACC_N:=4}  ; : ${FN:=4}
: ${JDEPTH:=1}   ; : ${POOL_N:=1} ; : ${SSWIN:=32} ; : ${KMAJOR:=0}; : ${CFASSIGN:=0}
: ${SELFSERVE:=1}; : ${DECENTASN:=1}; : ${BANKZERO:=1}; : ${BATONGATE:=1}; : ${STAGGER:=1}
: ${DSWS2_OVERLAP:=1}; : ${DSWS2_PREFETCH:=1}
: ${DEADMAN:=1}  ; : ${STAGINSTR:=1}; : ${TFPROBE:=1}
# DSWS2_FUNNEL -- ADDED TO THIS BLOCK 2026-07-27, and the reason is the whole point of this block.
#   The funnel is the boundary-advance readiness PRE-GATE. On 2026-07-27 we measured that the wall is
#   producer-side frontier starvation: only 1.6% of boundary entries ADVANCE, 76.5% lose the ZLOCK
#   election outright, and 93% of the waves that DO win are blocked by the GSTORED C-store gate. The
#   funnel pre-checks that exact gate READ-ONLY before the CAS -- i.e. it was built for precisely this
#   failure -- and it had NEVER BEEN COMPILED INTO A SINGLE RUN IN PROJECT HISTORY, because it was
#   absent from this block and so defaulted to 0 in mkflow's `${DSWS2_FUNNEL:-0}`. That is exactly the
#   silently-wrong-config failure this block exists to prevent, and the block failed at it. It is now
#   listed and PRINTED, so "off" is a visible choice rather than an omission.
#   SPIN_N default is 1, NOT the kernel's .ifndef 1024: see the polarity note at
#   occ_kernel_dsws_flow.s:.Lflow_da_funnel_notready. SPIN_N is currently INERT (the counter test has
#   inverted SCC polarity, so the funnel bails on the first not-ready = a pure check-once pre-gate).
#   If that polarity is ever corrected, 1 keeps the spin bounded instead of opening a 1024-deep spin
#   that re-reads 4 LDS words per iteration on a ~2.15M-entry hot path.
# *** DEFAULT FLIPPED 0 -> 1 ON 2026-07-27 (kmbandy). READ THE EVIDENCE BEFORE TRUSTING IT. ***
#   WHAT IS PROVEN (unthrottled counters, one run, ml8_dense_ffn_down M2048 K9216 @ FM=2):
#     occ[97] boundary drain / C-store-gate bails: 513,443 -> 0. The funnel completely eliminates
#       take-the-lock-then-bail-on-GSTORED churn. It does exactly what it was built to do.
#     occ[86] feed-path bails 29,928,830 -> 24,585,102 (-18%); iters per reserve 129.9 -> 106.7 (-18%).
#     Correctness audited by Codex against the RDNA4 ISA: read-only (no lds_put/CAS/atomic, only
#       private s54-s56), bails BEFORE the ownership CAS so it cannot abandon a boundary, and every
#       gate is RE-READ after winning ZLOCK (so no TOCTOU). Oracle bad=0, WORK-EXACT.
#   *** WHAT IS NOT PROVEN: THAT IT HELPS THROUGHPUT. *** TF 3.3 -> 3.2, i.e. unmoved//marginally
#     worse, and the empty-frontier share of bails got slightly WORSE (96.1% -> 97.6%). It costs
#     ~8.6M extra SERIALIZED LDS loads (lds_get = ds_load_b32 + s_wait_dscnt 0). It has been run on
#     exactly ONE shape, ONCE, and that A/B spanned ~3h on a shared box so the TF half is confounded.
#   So this default is justified by ELIMINATING MEASURED WASTE and by a clean correctness audit --
#   NOT by a demonstrated speedup. VALIDATE ON THE FULL 30-SHAPE SWEEP before relying on it, and if a
#   shape regresses, DSWS2_FUNNEL=0 restores the old behaviour byte-identically.
: ${DSWS2_FUNNEL:=1}; : ${DSWS2_FUNNEL_SPIN_N:=1}
export WAVES SEGK G FM FN ACC_N JDEPTH POOL_N SSWIN KMAJOR CFASSIGN \
       SELFSERVE DECENTASN BANKZERO BATONGATE STAGGER DSWS2_OVERLAP DSWS2_PREFETCH \
       DEADMAN STAGINSTR TFPROBE DSWS2_FUNNEL DSWS2_FUNNEL_SPIN_N

# ---- Guard the standard. Deviating must be an EXPLICIT act, not an omission. ----
if [ "${DSWS_ALLOW_NONSTD:-0}" != "1" ]; then
  case "$SEGK" in
    64|128|256) ;;
    *) echo "  build_flow.sh REFUSED: SEGK=$SEGK is outside the sanctioned range {64,128,256}"; \
       echo "     (kmbandy 2026-07-26). Set DSWS_ALLOW_NONSTD=1 to override."; exit 3;;
  esac
  for m in SELFSERVE DECENTASN BANKZERO BATONGATE STAGGER DSWS2_OVERLAP DSWS2_PREFETCH; do
    if [ "$(eval echo \$$m)" != "1" ]; then
      echo "  build_flow.sh REFUSED: $m=$(eval echo \$$m) -- that is a CORE MECHANISM of the config of record."
      echo "     Turning one off silently is exactly how 07-26 was lost. If this is a deliberate A/B arm,"
      echo "     set DSWS_ALLOW_NONSTD=1 and say so in the run name."; exit 3
    fi
  done
  if [ "$JDEPTH" != "1" ]; then
    echo "  build_flow.sh REFUSED: JDEPTH=$JDEPTH -- SELFSERVE requires JDEPTH=1 and the assembler will"
    echo "     refuse anyway. The k-slice lead-gate is a DESIGN change, not a knob. DSWS_ALLOW_NONSTD=1 to try."; exit 3
  fi
fi
# ---- dyn-VGPR GROW-TARGET GATE (added 2026-07-29). THIS ONE PREVENTS A HANG, NOT A DEVIATION. ----
# NFV is the kernel's dyn-VGPR grow target, mirrored EXACTLY from occ_dispatch.cpp:3246 and the kernel:
#     NFV = roundup16( 32 + 8*FM*FN + 2*FM + 2*FN )
# Verified against disassembly 2026-07-29: FM1FN4 -> 80 (0x50), FM2FN4 -> 112 (0x70), FM4FN4 -> 176 (0xb0).
# The per-wave cap is (SQ_DYN_VGPR.MAX_BLOCK_ALLOC+1) * BLOCK_SIZE = (7+1)*16 = 128 by default.
# ASKING FOR MORE THAN THE CAP IS NOT A SLOW RUN -- s_alloc_vgpr fails on EVERY wave, permanently, and
# the kernel makes no progress. That is a rule-3 full stop. So this gate is NOT bypassed by
# DSWS_ALLOW_NONSTD (which is a POLICY override for deliberate A/B deviations). Raising it requires
# DSWS2_VGPR_CAP=256 explicitly, which ALSO requires the volatile `sudo umr` BLOCK_SIZE=1 flip to have
# been applied -- and that flip reverts on idle, so it must be re-checked immediately before dispatch.
: ${DSWS2_VGPR_CAP:=128}
_nfv_raw=$(( 32 + 8*FM*FN + 2*FM + 2*FN ))
_nfv=$(( (_nfv_raw + 15) / 16 * 16 ))
if [ "$_nfv" -gt "$DSWS2_VGPR_CAP" ]; then
  echo "  build_flow.sh REFUSED: FM=$FM FN=$FN needs NFV=$_nfv VGPRs but the dyn-VGPR cap is $DSWS2_VGPR_CAP."
  echo "     s_alloc_vgpr would fail on EVERY wave, permanently -- that is a HANG, not a slow run."
  echo "     Legal at cap 128: FM*FN <= 10 (e.g. 2x4=112, 4x2=112, 1x4=80, 2x2=80, 2x5=128, 1x8=128)."
  echo "     To go higher you must FIRST apply the umr BLOCK_SIZE=1 flip (cap 256, VOLATILE - reverts"
  echo "     on idle), then build with DSWS2_VGPR_CAP=256. DSWS_ALLOW_NONSTD does NOT bypass this."
  exit 3
fi
echo "== CONFIG OF RECORD: WAVES=$WAVES (1 WG/CU, dispatch ML8_POOL=64) SEGK=$SEGK G=$G FM=$FM FN=$FN ACC_N=$ACC_N"
_feed=$(awk "BEGIN{printf \"%.3f\", ($FM+$FN)/($FM*$FN)}")
echo "   NFV=$_nfv/$DSWS2_VGPR_CAP VGPR (dyn grow target)  super-tile M=$((G*16*FM))  feed-loads/WMMA=$_feed"
echo "   SELFSERVE=$SELFSERVE DECENTASN=$DECENTASN BANKZERO=$BANKZERO BATONGATE=$BATONGATE STAGGER=$STAGGER OVERLAP=$DSWS2_OVERLAP PREFETCH=$DSWS2_PREFETCH${DSWS_ALLOW_NONSTD:+  [NONSTD OVERRIDE ACTIVE]}"
if [ "$DSWS2_FUNNEL" = "0" ]; then _fnl="   *** PRE-GATE DISABLED (NON-DEFAULT) ***"; else _fnl="   (pre-gate active)"; fi
echo "   FUNNEL=$DSWS2_FUNNEL (boundary-advance readiness pre-gate) SPIN_N=$DSWS2_FUNNEL_SPIN_N$_fnl"
# ===============================================================================================
mkflow() { # EMERGENT economy: no mix args. Env: WAVES VBUDGET G SEGK POOL_N ACC_N ...
  local tag="occ_dsws2_w${WAVES:-16}_flow_gd"
  nice -19 ionice -c3 "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
     -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=${FM:-2} -Wa,-defsym,FN=${FN:-4} -Wa,-defsym,G=${G:-6} -Wa,-defsym,SEGK=${SEGK:-64} \
     -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=${DIAG:-0} -Wa,-defsym,POOL_N=${POOL_N:-3} -Wa,-defsym,ACC_N=${ACC_N:-1} -Wa,-defsym,WOFLUSH=${WOFLUSH:-0} \
     -Wa,-defsym,WAVES=${WAVES:-16} -Wa,-defsym,VBUDGET=${VBUDGET:-1536} \
     -Wa,-defsym,PHASEPROBE=${PHASEPROBE:-0} -Wa,-defsym,PHSHIFT=${PHSHIFT:-8} -Wa,-defsym,PHSPLIT=${PHSPLIT:-0} -Wa,-defsym,NOCFLUSH=${NOCFLUSH:-0} -Wa,-defsym,KMAJOR=${KMAJOR:-0} -Wa,-defsym,JDEPTH=${JDEPTH:-1} -Wa,-defsym,STAGGER=${STAGGER:-0} ${RELSTART:+-Wa,-defsym,RELSTART=${RELSTART}} ${BATONGATE:+-Wa,-defsym,BATONGATE=${BATONGATE}} ${GRELAX:+-Wa,-defsym,GRELAX=${GRELAX}} ${BATON_SEED:+-Wa,-defsym,BATON_SEED=${BATON_SEED}} -Wa,-defsym,MAXFAT=${MAXFAT:-0} -Wa,-defsym,STAGERS=${STAGERS:-4} -Wa,-defsym,DUTYPROBE=${DUTYPROBE:-0} -Wa,-defsym,NTLOAD=${NTLOAD:-0} -Wa,-defsym,RBU=${RBU:-1} -Wa,-defsym,NOFEED=${NOFEED:-0} -Wa,-defsym,MULTISLOT=${MULTISLOT:-0} -Wa,-defsym,MSCOMP=${MSCOMP:-${MULTISLOT:-0}} -Wa,-defsym,MSSCAN=${MSSCAN:-${MSCOMP:-${MULTISLOT:-0}}} -Wa,-defsym,MSDRAIN=${MSDRAIN:-${MSCOMP:-${MULTISLOT:-0}}} -Wa,-defsym,MSFEED=${MSFEED:-${MULTISLOT:-0}} -Wa,-defsym,BATCHASN=${BATCHASN:-0} -Wa,-defsym,DECENTASN=${DECENTASN:-0} -Wa,-defsym,CFASSIGN=${CFASSIGN:-0} -Wa,-defsym,DSWS2_RCONV=${DSWS2_RCONV:-0} -Wa,-defsym,DSWS2_RCONV_COAST_N=${DSWS2_RCONV_COAST_N:-64} -Wa,-defsym,SELFSERVE=${SELFSERVE:-0} -Wa,-defsym,SSWIN=${SSWIN:-8} -Wa,-defsym,PHIST=${PHIST:-0} -Wa,-defsym,NOBLOAD=${NOBLOAD:-0} -Wa,-defsym,NODSADD=${NODSADD:-0} -Wa,-defsym,NOWMMA=${NOWMMA:-0} -Wa,-defsym,BNDPROBE=${BNDPROBE:-0} -Wa,-defsym,BNDSPLIT=${BNDSPLIT:-0} -Wa,-defsym,DSWS2_FUNNEL=${DSWS2_FUNNEL:-0} -Wa,-defsym,DSWS2_FUNNEL_SPIN_N=${DSWS2_FUNNEL_SPIN_N:-1024} -Wa,-defsym,RESVPROBE=${RESVPROBE:-0} -Wa,-defsym,BATCH=${BATCH:-1} -Wa,-defsym,INITBAR=${INITBAR:-1} -Wa,-defsym,TERMFIX=${TERMFIX:-1} -Wa,-defsym,DUTY_EVERY=${DUTY_EVERY:-64} -Wa,-defsym,CSTORE=${CSTORE:-0} \
     -Wa,-defsym,DSWS2_ADVPROBE=${DSWS2_ADVPROBE:-0} -Wa,-defsym,DSWS2_BNDTIME=${DSWS2_BNDTIME:-0} -Wa,-defsym,DSWS2_PASSTIME=${DSWS2_PASSTIME:-0} \
     -Wa,-defsym,DSWS2_BURSTCNT=${DSWS2_BURSTCNT:-0} \
     -Wa,-defsym,DSWS2_KDBUF=${DSWS2_KDBUF:-0} \
     -Wa,-defsym,DSWS2_WTBUDGET=${DSWS2_WTBUDGET:-0} -Wa,-defsym,WTB_THR=${WTB_THR:-64} \
     -Wa,-defsym,DSWS2_GAP=${DSWS2_GAP:-0} \
     -Wa,-defsym,DSWS2_POLLSTAGE=${DSWS2_POLLSTAGE:-0} -Wa,-defsym,DSWS2_POLLSTAGE_EVERY=${DSWS2_POLLSTAGE_EVERY:-64} \
     -Wa,-defsym,DSWS2_OVERLAP=${DSWS2_OVERLAP:-0} -Wa,-defsym,OVERLAP=${OVERLAP:-2} \
     -Wa,-defsym,DSWS2_ROLEFLOW=${DSWS2_ROLEFLOW:-0} -Wa,-defsym,DSWS2_ROLEFLOW_BACK_N=${DSWS2_ROLEFLOW_BACK_N:-2} \
     -Wa,-defsym,DSWS2_PREFETCH=${DSWS2_PREFETCH:-0} -Wa,-defsym,PREFETCH_LINES=${PREFETCH_LINES:-4} \
     -Wa,-defsym,DSWS2_PF_COUNTERS=${DSWS2_PF_COUNTERS:-0} -Wa,-defsym,PREFETCH_BATCHES_PER_VISIT=${PREFETCH_BATCHES_PER_VISIT:-16} \
     -Wa,-defsym,SLEEPN=${SLEEPN:-2} -Wa,-defsym,COORD_PERIOD=${COORD_PERIOD:-64} -Wa,-defsym,TFPROBE=${TFPROBE:-0} -Wa,-defsym,DEADMAN=${DEADMAN:-1} -Wa,-defsym,DEADMAN_TICKS=${DEADMAN_TICKS:-50000000} -Wa,-defsym,STAGINSTR=${STAGINSTR:-0} -Wa,-defsym,CNTLEAN=${CNTLEAN:-0} -Wa,-defsym,SPANFLIP=${SPANFLIP:-0} -Wa,-defsym,TRACE=${TRACE:-0} \
     -Wa,-defsym,FORENSICS=${FORENSICS:-0} -Wa,-defsym,BANKZERO=${BANKZERO:-1} -Wa,-defsym,FATGAUGE=${FATGAUGE:-0} \
     -c occ_kernel_dsws_flow.s -o "$tag.o" 2>/tmp/flow_build.err \
   && { "$L/llvm-objcopy" -O binary --only-section=.text "$tag.o" "$tag.bin"; \
        "$L/llvm-objcopy" -O binary --only-section=.lds_total "$tag.o" "$tag.lds" 2>/dev/null || rm -f "$tag.lds"; \
        echo "  OK   $tag.bin ($(wc -c < "$tag.bin")B .text)  [POOL_N=${POOL_N:-3} SSWIN=${SSWIN:-8} PHASEPROBE=${PHASEPROBE:-0}] LDS=$(od -An -tu4 -N4 "$tag.lds" 2>/dev/null | tr -d ' ')B"; } \
   || { echo "  FAIL $tag"; rm -f "$tag.bin" "$tag.o"; sed -n '1,25p' /tmp/flow_build.err; fail=1; }   # DELETE the stale bin -- a failed build must never leave a runnable artifact behind
}
echo "== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=${WAVES:-16} G=${G:-6} FM=${FM:-2} SEGK=${SEGK:-64} POOL_N=${POOL_N:-3} VBUDGET=${VBUDGET:-1536}) =="
mkflow
echo "flow build done. fail=$fail"
exit $fail
