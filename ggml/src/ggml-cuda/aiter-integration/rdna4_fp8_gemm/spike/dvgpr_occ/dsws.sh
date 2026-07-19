#!/usr/bin/env bash
# ============================================================================
# dsws.sh — THE canonical DSWS run. ALL values hard-coded. Do NOT re-derive.
#
#   *** JDEPTH=1 IS LOCKED (kmbandy, 2026-07-18). DEEP-J (JDEPTH>1) IS RETIRED. ***
#   Deep-J in this kernel = ONE carrier wave computes J K-slices SERIALLY in its own
#   registers (walking pool slots, WAITING for each to stage = jwait), flush once.
#   That LOSES on committed-clock runs (2s): J=1=9.5 TF vs J=2=8.7 TF @ product 256.
#   The CORRECT design (kmbandy's) is J=1 BANKED: each wave computes ONE slice ->
#   ds_add_f32 into the shared LDS bank -> the TILEDONE completer (a free wave) does
#   the ONE C-store. PARALLEL compute, DECOUPLED delivery. (The 2026-07-16 "higher-J
#   = better" numbers were ALL sub-0.5s idle-clock -- refuted by the 2s runs.)
#
# THE CONFIG (J=1 banked; the correctness-proven canonical):
#   JDEPTH=1               *** deep-J OFF -- the locked decision ***
#   DECENTASN=1            intra-WG decentralized assign (B-addr 64-bit fix IN, 2026-07-18)
#   BANKZERO=1 WOFLUSH=0   BANKED LDS reduce + TILEDONE completer (= the correct design)
#   STAGGER=1 BATONGATE=1  baton machinery present but INERT (grow-fail=0 -> no regime; retired)
#   DYNVGPR=1 (default)    dyn-VGPR grow/shrink per rowblk-burst
#   SEGK=64               proven-correct segment (full-oracle bad=0). *** SEGK IS THE THROUGHPUT
#                          LEVER: bigger SEGK = fewer flushes; J=1/SEGK=256/ACC_N=3 = 9.5 TF (best
#                          measured, needs a correctness gate). The WALL is FEED/staging. ***
#   G=6 ACC_N=6 (GROUPS=1) POOL_N=2 WAVES=30 FM=1 FN=4 MSDRAIN=1 RBU=1
#   MEASUREMENT RULE: NEVER quote TF from <2s. Feed to >=2s clock-committed via BIG M (many
#   tiles, low RAM); deep-K OOMs (~30GB) at 2s. See DSWS_TESTING_LOG.md 2026-07-18.
#
# USAGE:
#   ./dsws.sh              # FED deep-K run (steady state, the architecture measurement)
#   ./dsws.sh correct      # bounded K + FULL stride=1 oracle (correctness gate)
# Dispatches ONLY via ./gpu_run.sh (the sanctioned wrapper: latch/deadman/stale-bin guards).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
MODE="${1:-fed}"

# ---- BUILD (all defsyms baked) --------------------------------------------
echo "== building canonical DSWS bin =="
DECENTASN=1 STAGGER=1 BATONGATE=1 BANKZERO=1 WOFLUSH=0 JDEPTH=1 \
FM=1 G=6 ACC_N=6 POOL_N=2 SEGK=64 WAVES=30 MSDRAIN=1 RBU=1 \
STAGINSTR=1 TFPROBE=1 ./build_flow.sh || exit 1
echo "  bin sha: $(sha256sum occ_dsws2_w30_flow_gd.bin | cut -c1-8)"

# ---- HOST geometry (MUST match the bin) -----------------------------------
GEOM="FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=6 FLOW_POOL_N=2 DSWS2_SEGK=64"
SHAPE="DSWS2_ORACLE_MTL=6 DSWS2_ORACLE_NTL=64"

if [ "$MODE" = "correct" ]; then
  # bounded K + FULL stride=1 oracle (feeding-independent correctness gate)
  echo "== CORRECTNESS run (bounded K, full stride=1 oracle) =="
  ./gpu_run.sh dsws_correct -- $GEOM $SHAPE \
    DSWS2_K=2048 DSWS2_ORACLE_STRIDE=1 ML8_COOP_CHUNK=96 ML8_COOP_CHUNK_MAXS=3.0 \
    STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
else
  # FED: deep-K single chunk (n_kseg=2097152/64=32768) -> ~1s clock-committed steady state.
  #   ADD WORK VIA DEEPER K (the proven high-TF feed: dj32_committed 07-16 used exactly this K=2097152,
  #   1 chunk, computed=75497472 work-exact, NO DMFAT), NOT reps (reps relaunch -> slow reps trip the
  #   fat-carrier deadman = DMFAT). Same EXACT config as the baseline; ONLY DSWS2_K changed. compositor-
  #   safe bounded chunk; sampled oracle. Expected computed = 9437184*8 = 75497472.
  echo "== FED run (deep-K K=2097152 single chunk, ~1s clock-committed) =="
  ./gpu_run.sh dsws_fed -- $GEOM $SHAPE \
    DSWS2_K=2097152 DSWS2_ORACLE_STRIDE=4096 ML8_COOP_CHUNK=384 ML8_COOP_CHUNK_MAXS=3.0 \
    STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
fi
