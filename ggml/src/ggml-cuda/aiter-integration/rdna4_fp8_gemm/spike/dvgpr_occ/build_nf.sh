#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin
D="-x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,STORE=0 -Wa,-defsym,GENDIV=1 -Wa,-defsym,KWIN=4 -Wa,-defsym,KWINPW=4 -Wa,-defsym,KWINBPF=1 -Wa,-defsym,TWN=4 -Wa,-defsym,FM=4 -Wa,-defsym,FN=2"
$L/clang $D -Wa,-defsym,NOFEED=1   -c occ_kernel_wggemm2.s -o nf.o
$L/llvm-objcopy -O binary --only-section=.text nf.o occ_wggemm2_42_tw4_kwin4_nofeed_gd.bin
echo "NOFEED OK $(wc -c < occ_wggemm2_42_tw4_kwin4_nofeed_gd.bin)B"
$L/clang $D -Wa,-defsym,FEEDONLY=1 -c occ_kernel_wggemm2.s -o fo.o
$L/llvm-objcopy -O binary --only-section=.text fo.o occ_wggemm2_42_tw4_kwin4_feedonly_gd.bin
echo "FEEDONLY OK $(wc -c < occ_wggemm2_42_tw4_kwin4_feedonly_gd.bin)B"
