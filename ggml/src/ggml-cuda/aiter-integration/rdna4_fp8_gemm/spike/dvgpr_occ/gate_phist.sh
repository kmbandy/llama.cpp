#!/bin/bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
CC=/opt/rocm/llvm/bin/clang; OC=/opt/rocm/llvm/bin/llvm-objcopy; OD=/opt/rocm/llvm/bin/llvm-objdump
BASE=(-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=1 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=256
  -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,POOL_N=1 -Wa,-defsym,ACC_N=3
  -Wa,-defsym,WOFLUSH=0 -Wa,-defsym,WAVES=30 -Wa,-defsym,VBUDGET=1536 -Wa,-defsym,PHASEPROBE=0
  -Wa,-defsym,NOCFLUSH=0 -Wa,-defsym,JDEPTH=1 -Wa,-defsym,STAGGER=1 -Wa,-defsym,KMAJOR=0
  -Wa,-defsym,MAXFAT=0 -Wa,-defsym,STAGERS=4 -Wa,-defsym,NTLOAD=0 -Wa,-defsym,DUTYPROBE=0
  -Wa,-defsym,RBU=1 -Wa,-defsym,NOFEED=0 -Wa,-defsym,MULTISLOT=0 -Wa,-defsym,MSCOMP=0
  -Wa,-defsym,MSSCAN=0 -Wa,-defsym,MSDRAIN=0 -Wa,-defsym,MSFEED=0 -Wa,-defsym,BATCHASN=0
  -Wa,-defsym,DECENTASN=1 -Wa,-defsym,DUTY_EVERY=64 -Wa,-defsym,CSTORE=0 -Wa,-defsym,SLEEPN=2
  -Wa,-defsym,COORD_PERIOD=64 -Wa,-defsym,TRACE=0 -Wa,-defsym,BANKZERO=1 -Wa,-defsym,FATGAUGE=0
  -Wa,-defsym,SELFSERVE=1 -Wa,-defsym,FORENSICS=0 -Wa,-defsym,STAGINSTR=1 -Wa,-defsym,TFPROBE=1)
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT; rc=0

echo "== P1: PHIST=1 SSWIN=8 assembles, ZERO spill =="
$CC "${BASE[@]}" -Wa,-defsym,DEADMAN=1 -Wa,-defsym,DEADMAN_TICKS=50000000 \
    -Wa,-defsym,PHIST=1 -Wa,-defsym,SSWIN=8 -c occ_kernel_dsws_flow.s -o "$D/p.o" 2>&1 | head -3
if [ -f "$D/p.o" ]; then
  $OC -O binary --only-section=.text "$D/p.o" "$D/p.bin"
  sp=$($OD -d "$D/p.o" | grep -ci 'scratch_store\|scratch_load')
  echo "   sha $(sha256sum $D/p.bin | cut -c1-16)  spill=$sp"
  [ "$sp" = "0" ] || { echo "   *** FAIL spill"; rc=1; }
else echo "   *** FAIL assemble"; rc=1; fi

echo "== P2: PHIST=1 must REFUSE at DEADMAN=0 (throttle is the anti-brick mechanism) =="
e=$($CC "${BASE[@]}" -Wa,-defsym,DEADMAN=0 -Wa,-defsym,DEADMAN_TICKS=50000000 \
    -Wa,-defsym,PHIST=1 -c occ_kernel_dsws_flow.s -o /dev/null 2>&1 | grep -oE "error:.*PHIST requires DEADMAN=1" | head -1)
[ -n "$e" ] && echo "   PASS refused" || { echo "   *** FAIL: assembled with no deadman throttle"; rc=1; }

echo "== P3: EVERY phist_bump call site must be ACC-DEAD (no bump between grow and shrink) =="
#   The kernel's ACC-live regions are bounded by s_alloc_vgpr grow -> s_alloc_vgpr shrink.
#   A phist_bump there writes v3/v4 and flips exec -> corrupts C (:2056, the 2026-07-13 STAGINSTR bug).
python3 - <<'PY'
import re,sys
src=open('occ_kernel_dsws_flow.s').read().split('\n')
bad=[];depth=0
for i,l in enumerate(src,1):
    c=l.split('//')[0]
    if re.search(r'\bs_alloc_vgpr\b',c):
        # NFV grow raises, lean shrink lowers; track by the immediate operand where visible
        depth = 1 if re.search(r's_alloc_vgpr\s+(?!.*\b32\b)',c) else 0
    if re.match(r'\s*phist_bump\b',c) and depth==1:
        bad.append((i,l.strip()))
print("   ACC-live phist_bump sites:", len(bad))
for i,l in bad: print(f"   *** FAIL {i}: {l}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && echo "   PASS all bumps ACC-dead" || rc=1

echo "== P4: no phist_bump in the FAT jwait spin (holds ACC by design) =="
if awk '/^\.Lflow_jwait:/{f=1;next} f&&/^ *phist_bump/{print "   *** FAIL: bump in jwait at line "NR; bad=1} f&&/s_branch \.Lflow_jwait/{f=0} END{exit bad?1:0}' occ_kernel_dsws_flow.s; then
  echo "   PASS jwait clean"; else rc=1; fi

echo "== P5: every phist_bump is followed by the storecnt drain (macro-level) =="
if grep -A20 '^\.macro phist_bump' occ_kernel_dsws_flow.s | grep -q 's_wait_storecnt 0x0'; then
  echo "   PASS drain present"; else echo "   *** FAIL no drain -> s_alloc_vgpr register-file corruption"; rc=1; fi

echo "== P6: throttle present (s71==0) -- the anti-brick gate =="
if grep -A6 '^\.macro phist_bump' occ_kernel_dsws_flow.s | grep -q 's_cmp_eq_u32 s71, 0'; then
  echo "   PASS throttled"; else echo "   *** FAIL unthrottled -> hung-WG store storm -> MODE1"; rc=1; fi

echo "gate_phist rc=$rc"; exit $rc
