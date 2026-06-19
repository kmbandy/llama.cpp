#!/usr/bin/env bash
# build.sh  (dvgpr_occ P3) -- RAM-capped build of the dyn-VGPR GEMM-occupancy de-risk harness.
#
# Produces:
#   occ_n{8,16}_d{0,1}.bin  - the gfx1201 throughput-kernel matrix (NACC x DYNVGPR via -defsym)
#   test_oracle             - CPU fp8 e4m3 oracle self-test (built + run here)
#   occ_dispatch            - the KFD PM4 throughput/de-risk harness (links libhsakmt.a)
#
# RAM safety: the harness compile is wrapped in a systemd-run --user scope capped
# at 4G (host has ~15G; never run an uncapped build here).
set -euo pipefail
cd "$(dirname "$0")"

ROCM=/opt/rocm
L="$ROCM/llvm/bin"
PM4=../dvgpr_pm4
MEMMAX="${MEMMAX:-4G}"

run_capped() {
    if command -v systemd-run >/dev/null 2>&1; then
        systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 "$@"
    else
        echo "WARN: systemd-run unavailable; running uncapped" >&2
        "$@"
    fi
}

echo "[1/3] assembling occ_kernel.s -> occ_n{8,12,16}_d{0,1}.bin (+ fed-12) (gfx1201)"
for nacc in 8 12 16; do for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=$nacc -c occ_kernel.s -o occ_n${nacc}_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_n${nacc}_d${dv}.o occ_n${nacc}_d${dv}.bin
    echo "      occ_n${nacc}_d${dv}.bin: $(wc -c < occ_n${nacc}_d${dv}.bin) bytes"
done; done
# FEED variants: operand-feed gap (re-fetch B each iter) to test whether dyn-VGPR occupancy
# converts to throughput when there's a gap to hide. NACC=12 (128 VGPR) fits the current dyn cap;
# NACC=16 (160 VGPR = the real acc[4][4] tile) needs SQ_DYN_VGPR.BLOCK_SIZE=1 (cap 128->256).
for nacc in 12 16; do for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=$nacc -Wa,-defsym,FEED=1 -c occ_kernel.s -o occ_n${nacc}fed_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_n${nacc}fed_d${dv}.o occ_n${nacc}fed_d${dv}.bin
    echo "      occ_n${nacc}fed_d${dv}.bin: $(wc -c < occ_n${nacc}fed_d${dv}.bin) bytes"
done; done

# COMBINED kernel: all levers stacked (UNROLL x NACC x FEED x dyn-VGPR). UNROLL=8.
COMB_UNROLL=8
echo "[1b] combined kernel (unroll=$COMB_UNROLL x ILP x feed x dyn) -> occ_comb_n{8,16}_d{0,1}"
for nacc in 8 16; do for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=$nacc -Wa,-defsym,FEED=1 -Wa,-defsym,UNROLL=$COMB_UNROLL \
        -c occ_kernel_combined.s -o occ_comb_n${nacc}_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_comb_n${nacc}_d${dv}.o occ_comb_n${nacc}_d${dv}.bin
    echo "      occ_comb_n${nacc}_d${dv}.bin: $(wc -c < occ_comb_n${nacc}_d${dv}.bin) bytes"
done; done
# no-feed ceiling reference (NACC=8, unrolled): the operands-in-registers throughput ceiling.
for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=8 -Wa,-defsym,FEED=0 -Wa,-defsym,UNROLL=$COMB_UNROLL \
        -c occ_kernel_combined.s -o occ_combnf_n8_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_combnf_n8_d${dv}.o occ_combnf_n8_d${dv}.bin
    echo "      occ_combnf_n8_d${dv}.bin: $(wc -c < occ_combnf_n8_d${dv}.bin) bytes"
done

# timer-check kernel: measures the actual s_sendmsg REALTIME tick rate (validates freq_hz).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c occ_timercheck.s -o occ_timercheck.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_timercheck.o occ_timercheck.bin
echo "      occ_timercheck.bin: $(wc -c < occ_timercheck.bin) bytes"

# MICRO-BATCH dynamic-queue kernel: persistent waves pull tiles from an atomic queue; per tile
# dyn-VGPR grows (s_alloc 96) -> computes -> ships -> shrinks (s_alloc 32). NACC=8 fits the
# default 128 dyn cap (no umr). static (d0) reserves 96 for life; dyn (d1) lean-32 + grow/shrink.
echo "[1c] micro-batch dynamic-queue kernel -> occ_mb_d{0,1}.bin (NACC=8)"
for dv in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=8 -c occ_kernel_mb.s -o occ_mb_d${dv}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_mb_d${dv}.o occ_mb_d${dv}.bin
    echo "      occ_mb_d${dv}.bin: $(wc -c < occ_mb_d${dv}.bin) bytes"
done

# FED FAT-TILE micro-batch GEMM: the micro-batch architecture on a REAL fp8 GEMM (real K-stream
# A direct + B global_load_tr feed). FM/FN = accumulator-tile fragment dims (reuse lever). Stage 1
# = FM=FN=1 (the fed-plumbing correctness gate). Static (d0) reserves the fat block for life; dyn
# (d1) lean-32 + s_alloc grow/ship/shrink.
echo "[1d] FED fat-tile micro-batch GEMM -> occ_mbgemm_{shape}_b{batch}_d{0,1}.bin (reuse x batch-grab)"
# shape: FMxFN accumulator tile (1x1..2x4 fit the 128 dyn cap; 4x4=192 needs SQ_DYN_VGPR.BLOCK_SIZE=1).
# batch: tiles claimed per atomic grab (amortizes the contended work-queue counter + grow/shrink).
for shape in 1x1 2x2 2x4 4x4 5x4; do
    fm=${shape%x*}; fn=${shape#*x}
    for batch in 1 8 32; do for dv in 0 1; do
        "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
            -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,FM=$fm -Wa,-defsym,FN=$fn -Wa,-defsym,BATCH=$batch \
            -c occ_kernel_mbgemm.s -o occ_mbgemm_${shape}_b${batch}_d${dv}.o
        "$L/llvm-objcopy" -O binary --only-section=.text occ_mbgemm_${shape}_b${batch}_d${dv}.o occ_mbgemm_${shape}_b${batch}_d${dv}.bin
    done; done
    echo "      occ_mbgemm_${shape}_b{1,8,32}_d{0,1}.bin"
done
# NOFEED isolation probe: 2x4 dyn batch32, operands loaded ONCE (no per-K feed) -> framework ceiling.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,DYNVGPR=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,BATCH=32 -Wa,-defsym,NOFEED=1 \
    -c occ_kernel_mbgemm.s -o occ_mbgemm_2x4_b32_nf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_mbgemm_2x4_b32_nf.o occ_mbgemm_2x4_b32_nf.bin
echo "      occ_mbgemm_2x4_b32_nf.bin (NOFEED probe)"
# PROFILE variants: in-kernel REALTIME per-phase timing breakdown (fed=prof0, nofeed=prof1), 2x4 b32.
for nf in 0 1; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,BATCH=32 -Wa,-defsym,PROFILE=1 -Wa,-defsym,NOFEED=$nf \
        -c occ_kernel_mbgemm.s -o occ_mbgemm_2x4_b32_prof${nf}.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_mbgemm_2x4_b32_prof${nf}.o occ_mbgemm_2x4_b32_prof${nf}.bin
done
echo "      occ_mbgemm_2x4_b32_prof{0,1}.bin (PROFILE phase timers)"

# MERGE lockstep-stagger sweep: 4x4 static, BATCH=1 (so all pool waves stay active = neighbors for the
# stagger to interleave), vary STAGGER to break the persistent-wave K-lockstep (KG 50147c07 -- feed stalls
# fire in a synchronized burst; phase-offset by TGID_X so they interleave and the existing occupancy hides
# them). st0 == the un-staggered baseline.
echo "[1e] merge lockstep-stagger -> occ_mbgemm_4x4_b1_st{0,4,16,64}_d0.bin"
for st in 0 4 16 64; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,DYNVGPR=0 -Wa,-defsym,FM=4 -Wa,-defsym,FN=4 -Wa,-defsym,BATCH=1 -Wa,-defsym,STAGGER=$st \
        -c occ_kernel_mbgemm.s -o occ_mbgemm_4x4_b1_st${st}_d0.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_mbgemm_4x4_b1_st${st}_d0.o occ_mbgemm_4x4_b1_st${st}_d0.bin
done
echo "      occ_mbgemm_4x4_b1_st{0,4,16,64}_d0.bin"

# WAVE-GROUP skeleton (MAD-305 Phase 1): 4-wave (128-thread) cooperative workgroup, grid-stride TGID
# tile decode, SMOKE per-wave decode mark. Proves the workgroup forms + lane/wave mapping + coverage.
echo "[1f] wave-group skeleton smoke -> occ_wggemm_smoke.bin (TWM=2 TWN=2 SMOKE=1)"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,TWM=2 -Wa,-defsym,TWN=2 -Wa,-defsym,SMOKE=1 \
    -c occ_kernel_wggemm.s -o occ_wggemm_smoke.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm_smoke.o occ_wggemm_smoke.bin
echo "      occ_wggemm_smoke.bin: $(wc -c < occ_wggemm_smoke.bin) bytes"

# WAVE-GROUP atomic-claim + LDS-broadcast smoke (MAD-305 Phase 1 pivot): raw-PM4 TGID is unavailable,
# so the leader atomic-claims a tile and broadcasts it to the 4 waves via LDS+barrier. Proves LDS
# alloc (RSRC2.GRANULATED_LDS_SIZE) + barrier + atomic distribution. Also the SGPR probe kernel.
echo "[1g] wave-group LDS-broadcast smoke + SGPR probe -> occ_wglds.bin, occ_wgdiag.bin"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,TWM=2 -Wa,-defsym,TWN=2 \
    -c occ_kernel_wglds.s -o occ_wglds.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wglds.o occ_wglds.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c occ_kernel_wgdiag.s -o occ_wgdiag.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wgdiag.o occ_wgdiag.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c occ_kernel_ldsbound.s -o occ_ldsbound.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_ldsbound.o occ_ldsbound.bin
echo "      occ_wglds.bin: $(wc -c < occ_wglds.bin) bytes, occ_wgdiag.bin: $(wc -c < occ_wgdiag.bin) bytes, occ_ldsbound.bin: $(wc -c < occ_ldsbound.bin) bytes"

# Phase 2/3: 4-wave cooperative fp8 GEMM compute (A-LDS + B-global_load_tr + static 4x4). G2 vehicle.
# STORE=1 (default) = full 16-frag diagnostic store for the oracle (--wggemm-compute).
# STORE=0 = minimal acc[0][0]-only store for perf (--wggemm-perf) so the 16x diagnostic store traffic
#           doesn't mask compute throughput at the G2 parity gate.
echo "[1h] wave-group fp8 GEMM compute -> occ_wggemm2.bin (STORE=1) + occ_wggemm2_perf.bin (STORE=0)"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c occ_kernel_wggemm2.s -o occ_wggemm2.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2.o occ_wggemm2.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -c occ_kernel_wggemm2.s -o occ_wggemm2_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_perf.o occ_wggemm2_perf.bin
echo "      occ_wggemm2.bin: $(wc -c < occ_wggemm2.bin) B, occ_wggemm2_perf.bin: $(wc -c < occ_wggemm2_perf.bin) B"
# matched FEEDONLY on the real DBUF==1 path: identical feed (loads/prefetch/barriers/waits), WMMA skipped
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,FEEDONLY=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_feedonly_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_feedonly_perf.o occ_wggemm2_feedonly_perf.bin
echo "      occ_wggemm2_feedonly_perf.bin: $(wc -c < occ_wggemm2_feedonly_perf.bin) B (DBUF==1 feed, no WMMA)"
# A-LDS-ring K-WINDOW: amortize the publish barrier over KWIN K-tiles (GPT structural lever). NTILES%KWIN==0.
for U in 2 4 8; do
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=$U -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin$U.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin$U.o occ_wggemm2_kwin$U.bin
done
# B-prefetch-one-ahead (KWINBPF=1): 2 B slots, hide the per-slice B wait inside the window
for U in 2 4; do
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=$U -Wa,-defsym,KWINBPF=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin${U}_bpf.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin${U}_bpf.o occ_wggemm2_kwin${U}_bpf.bin
done
# 2-wide publish (KWINPUB2=1): overlap 2 A-slices' loads per wait -> halve publish A-load exposure
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPUB2=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin4_pub2.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin4_pub2.o occ_wggemm2_kwin4_pub2.bin
# single-reuse-barrier (KWINNOTAIL=1): drop the tail barrier. STORE=1 oracle (tail + notail) for the RACE gate; STORE=0 perf.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin4_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin4_st1.o occ_wggemm2_kwin4_st1.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINNOTAIL=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin4nt_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin4nt_st1.o occ_wggemm2_kwin4nt_st1.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINNOTAIL=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin4_notail.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin4_notail.o occ_wggemm2_kwin4_notail.bin
echo "      occ_wggemm2_kwin{2,4,8}.bin + kwin{2,4}_bpf.bin + kwin4_pub2.bin + kwin4{_st1,nt_st1,_notail}.bin built"

# [1h+] SATURATED SUSTAIN sweep binaries (the --sustain feed-lever A/B). publish-width grid (KWINPW) + per-tile
#   NOFEED ceilings. KWINPW slots: w0->v16 w1->v176 w2->v192 w3->v200 (field-26 headroom). 128x128 = 4-wave 2x2-grid.
for PW in 2 4; do
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=$PW -c occ_kernel_wggemm2.s -o occ_wggemm2_kwin4_pw$PW.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_kwin4_pw$PW.o occ_wggemm2_kwin4_pw$PW.bin
done
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,NOFEED=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_nofeed4_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_nofeed4_perf.o occ_wggemm2_nofeed4_perf.bin
# LARGER TILE (MAD-305 step): TWN=4 -> 8-wave 128x256 cooperative tile (A-strip reused by 4 N-waves; A-feed halved).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,NOFEED=1 -Wa,-defsym,TWN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_tw4_nofeed.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_tw4_nofeed.o occ_wggemm2_tw4_nofeed.bin
for PW in 2 4; do
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=$PW -Wa,-defsym,TWN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_tw4_kwin4_pw$PW.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_tw4_kwin4_pw$PW.o occ_wggemm2_tw4_kwin4_pw$PW.bin
done
echo "      sustain bins: kwin4_pw{2,4} + nofeed4_perf + tw4_{nofeed,kwin4_pw2,kwin4_pw4} (TWN=4 128x256) built"

# REUSE-TILE 8x2 (MAD-305 B-feed lever): FM=8 FN=2 @ TWN=4 -> per-wave 128x32 quadrant, 256x128 claimed tile.
#   HALVES the binding per-wave B global_load_tr feed (B-tr/MAC 0.25->0.125) at the cost of 2x A-LDS reads.
#   acc=128 VGPR (v32-159), A-frags FA=v160-191, B-frags FB=v192-199, pub-slots v200-207 -> max v207 = field26
#   (occupancy-matched to 4x4@TWN4). LDS = KWIN*ATILE = 4*(256*32) = 32768 (+ti) -> harness passes ldsBytes 32772.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_pw4.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_pw4.o occ_wggemm2_82_tw4_kwin4_pw4.bin
# STORE=1 full-fragment oracle bins (4x4@TWN4 sanity + 8x2@TWN4 gate): all FM*FN frags stored for CPU wmma_ref check
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_tw4_kwin4_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_tw4_kwin4_st1.o occ_wggemm2_tw4_kwin4_st1.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_st1.o occ_wggemm2_82_tw4_kwin4_st1.bin
# 8x2 wall-attribution variants: FEEDONLY (KWIN feed, no WMMA) + NOFEED (WMMA ceiling, no feed)
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,FEEDONLY=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_pw4_feedonly.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_pw4_feedonly.o occ_wggemm2_82_tw4_kwin4_pw4_feedonly.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,NOFEED=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_nofeed.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_nofeed.o occ_wggemm2_82_tw4_nofeed.bin
# REUSE-TILE 8x2 @ TWN=2 (MAD-305 residency lever): same FM=8 FN=2 (B-tr/MAC=0.125) but 4-wave WGs -> +50% resident
#   waves (768 vs 512 @ TWN4) to re-hide the WMMA exposed at TWN4 (FED/FO 0.887). A-fill = NBANDS=FM/TWN=4 bands
#   (general band-sequential publish path). Same LDS = KWIN*ATILE = 32768 (+ti) -> ldsBytes 32772 (ATILE TWN-invariant).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=2 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_kwin4_pw4.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_kwin4_pw4.o occ_wggemm2_82_kwin4_pw4.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=2 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_kwin4_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_kwin4_st1.o occ_wggemm2_82_kwin4_st1.bin
echo "      reuse-tile bins: 82_tw4_kwin4_pw4 (8x2 perf) + {tw4,82_tw4}_kwin4_st1 (oracle) + 82_tw4_{kwin4_pw4_feedonly,nofeed} (wall) built"
echo "      reuse-tile TWN=2 bins: 82_kwin4_pw4 (8x2 perf, NBANDS=4) + 82_kwin4_st1 (oracle) built"
# REUSE-TILE 8x2 @ TWN=4 KWIN=2 (MAD-305 OCCUPANCY lever): HALVE the A-LDS ring (KWIN 4->2 -> LDS=KWIN*ATILE=16384,
#   ldsBytes 16388) to ~2x resident WGs (64->~128 -> ~1024 resident waves) and re-hide the WMMA exposed at the 182
#   FEEDONLY ceiling. KWINPW=2 (must divide KWIN). NBANDS=FM/TWN=2 -> proven 2-band fast path. Tradeoff: 2x A-publish
#   barrier frequency. perf (STORE=0) + STORE=1 oracle (gates the KWIN=2 publish/consume).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=2 -Wa,-defsym,KWINPW=2 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin2_pw2.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin2_pw2.o occ_wggemm2_82_tw4_kwin2_pw2.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=2 -Wa,-defsym,KWINPW=2 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin2_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin2_st1.o occ_wggemm2_82_tw4_kwin2_st1.bin
echo "      reuse-tile KWIN=2 bins: 82_tw4_kwin2_pw2 (8x2 perf, half LDS ring) + 82_tw4_kwin2_st1 (oracle) built"
# REUSE-TILE 8x2 @ TWN=4 + B-IN-LDS DEDUP (MAD-305 binding-feed lever): wave_m==0 loads B (global_load_tr) -> LDS
#   B-ring; BOTH wave_m read B from LDS in the consume. Both wave_m of a wave_n load IDENTICAL B columns today ->
#   dedup HALVES the binding global B-tr feed (B-tr/MAC 0.125->0.0625). No VGPR change, no umr. KWINBPF must be 0.
#   LDS = KWIN*ATILE + KWIN*BTILE = 32768 + 4*4096 = 49152 (+ti) -> ldsBytes 49156. perf (STORE=0) + STORE=1 oracle.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -Wa,-defsym,BLDS=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_blds.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_blds.o occ_wggemm2_82_tw4_kwin4_blds.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -Wa,-defsym,BLDS=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_blds_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_blds_st1.o occ_wggemm2_82_tw4_kwin4_blds_st1.bin
echo "      reuse-tile B-in-LDS bins: 82_tw4_kwin4_blds (8x2 perf, dedup B) + 82_tw4_kwin4_blds_st1 (oracle) built"
# REUSE-TILE 8x2 @ TWN=4 + KWINBPF (MAD-305 CDNA-rung-7 DOUBLE-BUFFER equivalent): B-prefetch-one-ahead. Issue next
#   slice's B (global_load_tr) into the OTHER of 2 ping-pong slots (FB / FB+4FN) while WMMA runs on the current
#   slice -> overlaps the binding B-load latency behind compute, hiding the 162->182 WMMA-exposure. NO new instrs,
#   NO extra barrier (A stays on dscnt, B-only loadcnt with s_wait_loadcnt 8 descending). Slots symbolized for 8x2
#   (FB=192 -> 192/200, no FA collision). perf (STORE=0) + STORE=1 oracle.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf.o occ_wggemm2_82_tw4_kwin4_bpf.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf_st1.o occ_wggemm2_82_tw4_kwin4_bpf_st1.bin
echo "      reuse-tile KWINBPF bins: 82_tw4_kwin4_bpf (8x2 perf, B-prefetch double-buffer) + _bpf_st1 (oracle) built"
# 4x4 @ TWN=4 + KWINBPF (tile x lever interaction: does the double-buffer help a higher-B-feed tile MORE?)
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,TWN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_tw4_kwin4_bpf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_tw4_kwin4_bpf.o occ_wggemm2_tw4_kwin4_bpf.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,TWN=4 -c occ_kernel_wggemm2.s -o occ_wggemm2_tw4_kwin4_bpf_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_tw4_kwin4_bpf_st1.o occ_wggemm2_tw4_kwin4_bpf_st1.bin
echo "      4x4 KWINBPF bins: tw4_kwin4_bpf (perf) + _bpf_st1 (oracle) built"
# 8x2 + KWINBPF + SETPRIO (MAD-305 CDNA rung-9 scheduling equiv, STACKS on the 165 winner): s_setprio 1 around the
#   per-slice WMMA burst, s_setprio 0 during feed -> bias the issue port toward WMMA-phase waves (denser back-to-back
#   WMMA, the gap-filler version of CDNA wave ping-pong). perf (STORE=0) + STORE=1 oracle.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,SETPRIO=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf_sp.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf_sp.o occ_wggemm2_82_tw4_kwin4_bpf_sp.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,SETPRIO=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf_sp_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf_sp_st1.o occ_wggemm2_82_tw4_kwin4_bpf_sp_st1.bin
echo "      KWINBPF+SETPRIO bins: 82_tw4_kwin4_bpf_sp (perf) + _bpf_sp_st1 (oracle) built"
# 8x2 + KWINBPF + SETPRIO + ALD2 (MAD-305 issue-slot lever, the "fewer dispatch slots on feed" axis): wide A-read via
#   ds_load_2addr_stride64_b64 loads 2 M-frags/instr (offset*512 == mi*512 frag stride) -> A-reads 16->8/slice ->
#   more dispatch bandwidth to WMMA. Same LDS addresses/bytes (oracle-identical). perf (STORE=0) + STORE=1 oracle.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,SETPRIO=1 -Wa,-defsym,ALD2=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf_sp_a2.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf_sp_a2.o occ_wggemm2_82_tw4_kwin4_bpf_sp_a2.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,SETPRIO=1 -Wa,-defsym,ALD2=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=8 -Wa,-defsym,FN=2 -c occ_kernel_wggemm2.s -o occ_wggemm2_82_tw4_kwin4_bpf_sp_a2_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_82_tw4_kwin4_bpf_sp_a2_st1.o occ_wggemm2_82_tw4_kwin4_bpf_sp_a2_st1.bin
echo "      wide-A-read bins: 82_tw4_kwin4_bpf_sp_a2 (perf) + _bpf_sp_a2_st1 (oracle) built"

# LDS-FREE A (ANOLDS): per-wave global A, no LDS, no barriers (issue-density structural test). perf (STORE=0) + oracle (STORE=1).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,ANOLDS=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_anolds_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_anolds_perf.o occ_wggemm2_anolds_perf.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,ANOLDS=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_anolds_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_anolds_st1.o occ_wggemm2_anolds_st1.bin
# COALESCING DIAGNOSTIC: ANOLDS with v11=lane*8 (coalesced A access, WRONG DATA) -> isolates strided-access cost.
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,ANOLDS=1 -Wa,-defsym,ACOAL=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_anolds_coal_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_anolds_coal_perf.o occ_wggemm2_anolds_coal_perf.bin
echo "      anolds bins: occ_wggemm2_anolds_{perf,st1,coal_perf}.bin (LDS-free A + coalescing diag) built"
# LDS-FREE A via global_load_tr (COALESCED A-shuf) -- the real fix. perf (STORE=0) + oracle (STORE=1).
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,ANOLDSTR=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_anoldstr_perf.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_anoldstr_perf.o occ_wggemm2_anoldstr_perf.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=1 -Wa,-defsym,ANOLDSTR=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_anoldstr_st1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_anoldstr_st1.o occ_wggemm2_anoldstr_st1.bin
echo "      anoldstr bins: occ_wggemm2_anoldstr_{perf,st1}.bin (LDS-free A via global_load_tr) built"

# MAD-305 Step A: FEED-ONLY depth-P pipeline bandwidth probe. No WMMA/LDS/barrier; keeps PDEPTH slices of
# FRAGS b64 loads in flight via s_wait_loadcnt((P-1)*FRAGS). (P-1)*FRAGS must be <=63 -> F=8 for P<=8, F=4 @ P=16.
echo "[1i] feed-only depth-P pipeline bandwidth probe -> occ_feedpipe_p{1,2,4,8}_f8 + p16_f4"
for P in 1 2 4 8; do
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
        -Wa,-defsym,PDEPTH=$P -Wa,-defsym,FRAGS=8 -c occ_kernel_feedpipe.s -o occ_feedpipe_p${P}_f8.o
    "$L/llvm-objcopy" -O binary --only-section=.text occ_feedpipe_p${P}_f8.o occ_feedpipe_p${P}_f8.bin
done
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,PDEPTH=16 -Wa,-defsym,FRAGS=4 -c occ_kernel_feedpipe.s -o occ_feedpipe_p16_f4.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedpipe_p16_f4.o occ_feedpipe_p16_f4.bin
echo "      occ_feedpipe_p{1,2,4,8}_f8.bin + occ_feedpipe_p16_f4.bin"

# MAD-305 Step A localization ladder: add 4-wave / barrier / LDS-A-share couplings back onto the 123 GB/s
# baseline one at a time (GPT rungs 1-5). Same body; WAVES/BARRIER/LDSMODE toggles. barrier=s_barrier_signal/wait.
echo "[1j] Step A localization ladder -> occ_feedladder_r{1..5}.bin"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,WAVES=1 -Wa,-defsym,BARRIER=0 -Wa,-defsym,LDSMODE=0 -c occ_kernel_feedladder.s -o occ_feedladder_r1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedladder_r1.o occ_feedladder_r1.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,WAVES=4 -Wa,-defsym,BARRIER=0 -Wa,-defsym,LDSMODE=0 -c occ_kernel_feedladder.s -o occ_feedladder_r2.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedladder_r2.o occ_feedladder_r2.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,WAVES=4 -Wa,-defsym,BARRIER=1 -Wa,-defsym,LDSMODE=0 -c occ_kernel_feedladder.s -o occ_feedladder_r3.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedladder_r3.o occ_feedladder_r3.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,WAVES=4 -Wa,-defsym,LDSMODE=1 -c occ_kernel_feedladder.s -o occ_feedladder_r4.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedladder_r4.o occ_feedladder_r4.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,WAVES=4 -Wa,-defsym,LDSMODE=2 -c occ_kernel_feedladder.s -o occ_feedladder_r5.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_feedladder_r5.o occ_feedladder_r5.bin
echo "      occ_feedladder_r{1,2,3,4,5}.bin"

# MAD-305 Step A rung 6: is B's global_load_tr_b64 the FED wall? 6a tr+synthetic, 6b tr+real Bshuf addr,
# 6d plain global_load_b64 + real Bshuf (negative control). (6c = 6b binary at real VGPR/LDS residency.)
echo "[1k] rung 6 B-transpose-load probe -> occ_btr_6{a,b,d}.bin"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,TR=1 -Wa,-defsym,BADDR=0 -c occ_kernel_btr.s -o occ_btr_6a.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_btr_6a.o occ_btr_6a.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,TR=1 -Wa,-defsym,BADDR=1 -c occ_kernel_btr.s -o occ_btr_6b.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_btr_6b.o occ_btr_6b.bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,TR=0 -Wa,-defsym,BADDR=1 -c occ_kernel_btr.s -o occ_btr_6d.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_btr_6d.o occ_btr_6d.bin
echo "      occ_btr_6{a,b,d}.bin"

echo "[1k+] Lever A micro-oracle: global_load_tr_b128 fp8 fragment semantics -> occ_btr128.bin"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c occ_kernel_btr128.s -o occ_btr128.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_btr128.o occ_btr128.bin
echo "      occ_btr128.bin: $(wc -c < occ_btr128.bin) bytes"

# MAD-305 Step A phase timers: PROFILE build of the real BLADDER FEEDONLY kernel (sampled in-kernel
# realtime timers around the K-loop; 1 profiler wave -> per-phase tick-sums in occ[8..15]). Non-PROFILE
# builds are byte-identical (all timer code is under .if PROFILE).
echo "[1l] real FED phase-timer build -> occ_wggemm2_prof.bin (DEFAULT DBUF==0 path = the 1.4 TF FED; PROFILE=1 STORE=0)"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,PROFILE=1 -c occ_kernel_wggemm2.s -o occ_wggemm2_prof.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_prof.o occ_wggemm2_prof.bin
echo "      occ_wggemm2_prof.bin: $(wc -c < occ_wggemm2_prof.bin) B"

# [1m] rung 8: inert per-WG phase-stagger sweep (DBUF==1 path; STORE=0, STAGGER=1, NON-PROFILE).
#   STAGGER=0 build is byte-identical to occ_wggemm2_perf.bin (gate verified by cmp below). MASK=0 is the
#   delay==0 control (stagger code present, zero loop iters). Sweep MASK at SHIFT5 + two SHIFT probes @MASK15.
echo "[1m] rung 8 inert per-WG stagger sweep -> occ_wggemm2_stag_m{MASK}_s{SHIFT}.bin"
build_stag() {  # $1=MASK $2=SHIFT
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,STORE=0 -Wa,-defsym,STAGGER=1 -Wa,-defsym,STAGGER_MASK=$1 -Wa,-defsym,STAGGER_SHIFT=$2 \
    -c occ_kernel_wggemm2.s -o occ_wggemm2_stag_m$1_s$2.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_stag_m$1_s$2.o occ_wggemm2_stag_m$1_s$2.bin
}
for M in 0 3 7 15 31; do build_stag $M 5; done
build_stag 15 4
build_stag 15 6
# gate: STAGGER=0 must be byte-identical to the perf bin (proves the inert path doesn't change non-stagger codegen)
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,STAGGER=0 -c occ_kernel_wggemm2.s -o occ_wggemm2_stag0.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_stag0.o occ_wggemm2_stag0.bin
if cmp -s occ_wggemm2_stag0.bin occ_wggemm2_perf.bin; then echo "      STAGGER=0 byte-identical to perf bin: OK"; else echo "      *** STAGGER=0 DIFFERS from perf bin -- gate FAILED ***"; fi
echo "      stagger bins: $(ls -1 occ_wggemm2_stag_m*_s*.bin | wc -l) built ($(wc -c < occ_wggemm2_stag_m15_s5.bin) B each)"

# [1n] rung 9: bisect the PROFILE 70x. Each PB variant adds ONE PROFILE ingredient to the real DBUF==1 path
#   (STORE=0, non-PROFILE, non-STAGGER). PB=0 (default) == occ_wggemm2_perf.bin (inherits the byte-id gate).
#   PB1 = per-K sendmsg+kmcnt (all-wave); PB2 = per-tile leader token atomic; PB3 = per-K cmp/branch skeleton
#   (all-wave); PB4 = per-K inert busy-loop (all-wave, control vs PB1).
echo "[1n] rung 9 PROFILE-70x bisection -> occ_wggemm2_pb{1,2,3,4}.bin"
for P in 1 2 3 4; do
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,PB=$P \
    -c occ_kernel_wggemm2.s -o occ_wggemm2_pb$P.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_wggemm2_pb$P.o occ_wggemm2_pb$P.bin
done
echo "      pb bins: $(ls -1 occ_wggemm2_pb*.bin | wc -l) built (pb1=$(wc -c < occ_wggemm2_pb1.bin) pb2=$(wc -c < occ_wggemm2_pb2.bin) pb3=$(wc -c < occ_wggemm2_pb3.bin) pb4=$(wc -c < occ_wggemm2_pb4.bin) B)"

# [1o] STACK LADDER: rebuild the fast feed from a known-good core, +1 obligation/rung (TF/GB/s/proof each).
echo "[1o] stack ladder rung 1 (load-only truthful base) -> occ_stack_r1.bin"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,RUNG=1 -c occ_kernel_stack.s -o occ_stack_r1.o
"$L/llvm-objcopy" -O binary --only-section=.text occ_stack_r1.o occ_stack_r1.bin
echo "      occ_stack_r1.bin: $(wc -c < occ_stack_r1.bin) B"

# [1p] CLEAN PM4 streaming bandwidth probe -> occ_bw_{read,copy,write}_b{32,64,128}.bin (prove near-spec BW)
echo "[1p] clean PM4 bandwidth probe -> occ_bw_*.bin"
bw() {  # $1=MODE $2=LDW $3=UNROLL $4=name
  "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,MODE=$1 -Wa,-defsym,LDW=$2 -Wa,-defsym,UNROLL=$3 -c occ_kernel_bw.s -o occ_bw_$4.o
  "$L/llvm-objcopy" -O binary --only-section=.text occ_bw_$4.o occ_bw_$4.bin
}
bw 0 4  32 read_b32
bw 0 8  16 read_b64
bw 0 16 8  read_b128
bw 1 16 8  copy_b128
bw 2 16 8  write_b128
bw 0 8  4  read_b64_u4    # MLP-depth scrutiny: is 45/85 GB/s MLP-limited? (b64 read, UNROLL 4 vs 8 vs 16)
bw 0 8  8  read_b64_u8
echo "      bw bins: $(ls -1 occ_bw_*.bin | wc -l) built (read_b128=$(wc -c < occ_bw_read_b128.bin) B)"

echo "[2/3] oracle self-test"
clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle
./test_oracle

echo "[3/3] building occ_dispatch (MemoryMax=$MEMMAX)"
run_capped clang++ -std=c++17 -O2 -Wall -Wno-unused \
    -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I "$ROCM/include" \
    occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" \
    "$ROCM/lib/libhsakmt.a" \
    -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt \
    -o occ_dispatch

echo "OK -> ./occ_dispatch [--prong1|--prong2]   (SUPERVISED: raw PM4 on the gfx12 node)"
echo "      prong1 (occupancy->throughput curve)     : timeout 40 ./occ_dispatch --prong1"
echo "      prong2 (dyn vs static heavy, KDEPTH sweep): timeout 60 ./occ_dispatch --prong2"
echo "      correctness A/B (KDEPTH=1)               : timeout 30 ./occ_dispatch"
