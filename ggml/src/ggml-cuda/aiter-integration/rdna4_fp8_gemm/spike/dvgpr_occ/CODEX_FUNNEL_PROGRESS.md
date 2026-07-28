# 1. Spin-counter SGPR chosen

`s56` is the bounded-spin counter. The boundary handler's live registers are
`s44-s47`, `s51`, `s53`, and `s66-s68`; persistent `s50` is the RCONV coast
counter and is not touched. A whole-file occurrence audit confirms `s56` is
dead across this boundary: its last use before `.Lflow_da_boundary` is in the
coordinator path at lines 3224-3258, and its next occurrence after the boundary
handler is a fresh definition (`s_add_u32 s56, s2, s22`) in the compute path.
There is no intervening read, and both the ready path and the not-ready flow-off
path may clobber it safely.

# 2. Reads-only and hard-bounded spin loop

The repeated readiness path contains these instructions:

```asm
lds_get s54, DRAIN_HEAD_OFF
lds_get s55, ASSIGN_HEAD_OFF
s_cmp_lt_u32 s54, s55
s_cbranch_scc1 .Lflow_da_funnel_notready
lds_get s54, DA_ZDONE_OFF
s_and_b32 s54, s54, ~ZLOCK
s_lshr_b32 s54, s54, s68
lds_get s55, GSTORED_OFF
s_cmp_lt_u32 s55, s54
s_cbranch_scc1 .Lflow_da_funnel_notready
```

The shared not-ready tail is:

```asm
s_sub_u32 s56, s56, 1
s_cbranch_scc0 .Lflow_feedmt_sleep
s_branch .Lflow_da_funnel_ready
```

The loop performs only the four `lds_get` reads plus scalar ALU, comparisons,
and branches. It contains no store, CAS, atomic, `s_sendmsg`, or other
side-effecting instruction. `s56` is initialized from
`DSWS2_FUNNEL_SPIN_N`, whose assembly and `build_flow.sh` defaults are both
1024. Each not-ready pass consumes one budget count; SCC clears when subtracting
1 from zero, which branches to `.Lflow_feedmt_sleep`. Thus the initial readiness
test has at most 1024 bounded retry re-reads and can never spin indefinitely.
When both checks pass, execution falls directly into the unchanged ZLOCK
election. The ZLOCK election, drain gate, and GSTORED post-gate were not edited.

# 3. Offline gate results

`DSWS2_FUNNEL=0` A1 canonical profile: MATCH

Command:

```text
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_FUNNEL=0 ./build_flow.sh
```

Verbatim output:

```text
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
```

Command:

```text
sha256sum occ_dsws2_w30_flow_gd.bin
```

Verbatim output:

```text
cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```

The digest matches canonical prefix `cac3ff7c2338e73f` byte-for-byte.

`DSWS2_FUNNEL=1 DSWS2_RCONV=1` A1 assemble/link: PASS

Command:

```text
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_FUNNEL=1 DSWS2_FUNNEL_SPIN_N=1024 DSWS2_RCONV=1 ./build_flow.sh
```

Verbatim output:

```text
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32560B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
```

The equivalent complete A1 defsym command with `RGADESC=1` assembled with exit
code 0 and empty output. `/opt/rocm/llvm/bin/ld.lld -shared k.o -o k.co`
linked with exit code 0 and empty output. Resulting artifact output:

```text
/tmp/codex_funnel_gate.uvs2xD/k.o 44280 bytes
/tmp/codex_funnel_gate.uvs2xD/k.co 45608 bytes
```

RGA spill result: BLOCKED-by-readonly-.rga

Command:

```text
/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin --isa /tmp/codex_funnel_gate.uvs2xD/isa.txt -a /tmp/codex_funnel_gate.uvs2xD/stats.csv --livereg /tmp/codex_funnel_gate.uvs2xD/lr.txt --livereg-sgpr /tmp/codex_funnel_gate.uvs2xD/lr_sgpr.txt --co /tmp/codex_funnel_gate.uvs2xD/k.co
```

Verbatim output:

```text
/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga

Failed opening file /home/kmbandy/.rga/GPUOpen/rga/rga_cli21-20260723-165546.log for writing: Read-only file system
```

RGA returned exit code 0 but created no ISA, stats, or livereg output; the
temporary directory contains only `k.o` and `k.co`. Therefore the RGA spill
gate is blocked and 0-spill is not claimed.
