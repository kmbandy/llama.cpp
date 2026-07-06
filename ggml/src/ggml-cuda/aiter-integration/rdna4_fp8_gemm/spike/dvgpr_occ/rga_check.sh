#!/usr/bin/env bash
# rga_check.sh — STANDARD post-change static-analysis gate (MAD-305).
#
# Run this after EVERY kernel code change/addition. It builds a linked code object with the
# RGADESC analysis descriptor, runs the Radeon GPU Analyzer in binary mode, and reports
# VGPR/SGPR/LDS/spills + ISA size — the offline occupancy picture, BEFORE any GPU dispatch.
#
# Usage:  ./rga_check.sh <label> [DEFSYM=val ...]
#   winner : ./rga_check.sh winner STORE=0 KWIN=4 KWINPW=4 KWINBPF=1 SETPRIO=1 TWN=4 FM=8 FN=2
#   lean   : ./rga_check.sh lean_4x2 STORE=0 ANOLDSTR=1 TWM=1 TWN=1 FM=4 FN=2
#
# NOTE: the real RGA is the RDTS build below — NOT /usr/bin/rga, which is ripgrep-all (name clash).
set -euo pipefail
ROCM=/opt/rocm
L="$ROCM/llvm/bin"
RGA=/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga
KSRC="${KSRC:-occ_kernel_wggemm2.s}"   # override to gate a different kernel, e.g. KSRC=occ_kernel_coop.s

LABEL="${1:?usage: rga_check.sh <label> [DEFSYM=val ...]}"; shift || true
OUT="rga_out/$LABEL"; mkdir -p "$OUT"

DEFS=( -Wa,-defsym,RGADESC=1 )
for d in "$@"; do DEFS+=( -Wa,-defsym,"$d" ); done

echo "[rga_check] $LABEL : ${*:-(no extra defsyms)}"
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 "${DEFS[@]}" -c "$KSRC" -o "$OUT/k.o"
"$L/ld.lld" -shared "$OUT/k.o" -o "$OUT/k.co"
"$RGA" -s bin --isa "$OUT/isa.txt" -a "$OUT/stats.csv" \
       --livereg "$OUT/lr.txt" --livereg-sgpr "$OUT/lr_sgpr.txt" --co "$OUT/k.co" >/dev/null

echo "=== RGA stats ($LABEL) ==="
cat "$OUT"/*stats*.csv
# livereg peak (max VGPRs simultaneously live) — the occupancy-limiting number
LR=$(ls "$OUT"/*lr.txt 2>/dev/null | head -1)
[ -n "${LR:-}" ] && echo "--- livereg: $(grep -i 'maximum' "$LR" | head -1)"
