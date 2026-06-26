#!/usr/bin/env bash
# ws_rga_sweep.sh — WS-T5 offline RGA gate for the lean wave-specialized fp8 GEMM.
#
# Sweeps tile (FM x FN) x DYNVGPR {static, dyn} through RGA (livereg peak + spills + alloc),
# all offline (no GPU). NLOAD is intentionally NOT swept here: loaders stay lean (LEANREG)
# and the compute body dominates NFV, so NLOAD does not move the single-kernel register
# picture — it is a GPU-occupancy axis for the T6 dispatch, not an RGA axis.
#
# Gate per cell: 0 VGPR spills, 0 SGPR spills, and dyn grow target NFV <= 128 => no umr
# cap-lift needed (4x4 deliberately crosses that line as the fat stretch cell).
set -euo pipefail
ROCM=/opt/rocm
L="$ROCM/llvm/bin"
RGA=/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga
KSRC=occ_kernel_wavespec.s
NCOMP=4

# tile cells: "FM FN"  (lean four + 4x4 stretch). FN in {1,2,4}; grow NFV = 32+8*FM*FN+2*FM+2*FN+16
CELLS=( "2 1" "2 2" "2 4" "4 2" "4 4" )

printf '%-7s %-7s %-6s %-9s %-9s %-7s %-7s %-7s %-8s\n' \
  tile dyn NFV alloc-vgpr live-peak vspill sspill LDS isa-bytes
echo "-----------------------------------------------------------------------------------------"

for cell in "${CELLS[@]}"; do
  set -- $cell; FM=$1; FN=$2
  NFV=$(( 32 + 8*FM*FN + 2*FM + 2*FN + 16 ))
  LDS=$(( NCOMP*FM*256 + FN*256 + 4 ))
  for dv in 0 1; do
    [ "$dv" = 1 ] && tag=dyn || tag=static
    OUT="rga_out/ws_${FM}x${FN}_${tag}"; mkdir -p "$OUT"
    "$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
      -Wa,-defsym,RGADESC=1 -Wa,-defsym,STORE=0 \
      -Wa,-defsym,FM=$FM -Wa,-defsym,FN=$FN -Wa,-defsym,NLOAD=1 -Wa,-defsym,NCOMP=$NCOMP \
      -Wa,-defsym,DYNVGPR=$dv -c "$KSRC" -o "$OUT/k.o" 2>"$OUT/asm.err" || {
        printf '%-7s %-7s %-6s  *** ASSEMBLE FAILED (see %s) ***\n' "${FM}x${FN}" "$tag" "$NFV" "$OUT/asm.err"; continue; }
    "$L/ld.lld" -shared "$OUT/k.o" -o "$OUT/k.co"
    "$RGA" -s bin --isa "$OUT/isa.txt" -a "$OUT/stats.csv" \
           --livereg "$OUT/lr.txt" --livereg-sgpr "$OUT/lr_sgpr.txt" --co "$OUT/k.co" >/dev/null 2>&1
    CSV=$(ls "$OUT"/*stats*.csv 2>/dev/null | head -1)
    LR=$(ls "$OUT"/*lr.txt 2>/dev/null | head -1)
    # stats.csv columns vary; pull by header name robustly
    getcol() { awk -F, -v want="$1" 'NR==1{for(i=1;i<=NF;i++){h=$i;gsub(/^ +| +$/,"",h);if(h==want)c=i}} NR==2{v=$c;gsub(/^ +| +$/,"",v);print v}' "$CSV"; }
    VGPR=$(getcol "VGPRs" 2>/dev/null); [ -z "$VGPR" ] && VGPR=$(getcol "USED_VGPRS" 2>/dev/null)
    VSPILL=$(getcol "VGPR spills" 2>/dev/null); [ -z "$VSPILL" ] && VSPILL=$(getcol "VGPR_SPILLS" 2>/dev/null)
    SSPILL=$(getcol "SGPR spills" 2>/dev/null); [ -z "$SSPILL" ] && SSPILL=$(getcol "SGPR_SPILLS" 2>/dev/null)
    ISA=$(getcol "ISA size" 2>/dev/null); [ -z "$ISA" ] && ISA=$(getcol "ISA_SIZE" 2>/dev/null)
    PEAK=$(grep -i 'maximum' "$LR" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    printf '%-7s %-7s %-6s %-9s %-9s %-7s %-7s %-7s %-8s\n' \
      "${FM}x${FN}" "$tag" "$NFV" "${VGPR:-?}" "${PEAK:-?}" "${VSPILL:-?}" "${SSPILL:-?}" "$LDS" "${ISA:-?}"
  done
done
echo "-----------------------------------------------------------------------------------------"
echo "gate: vspill=0 & sspill=0 everywhere; NFV<=128 => no umr (4x4 NFV=192 needs SQ_DYN_VGPR.BLOCK_SIZE=1)"
