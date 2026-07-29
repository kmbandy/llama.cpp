# ADVPROBE progress

Write mode was confirmed for `occ_kernel_dsws_flow.s`, `occ_dispatch.cpp`, and
`build_flow.sh`. Work was offline only. No GPU executable or dispatch path was
run. No file was staged or committed, and `occ_kernel_coop.s` was not touched.

## Occ slots

- `occ[131]`, byte offset 524 (`ADVP_TICKS_OFF`): total sampled ZLOCK
  critical-section ticks.
- `occ[132]`, byte offset 528 (`ADVP_COUNT_OFF`): sampled successful GROUP or
  TILE advances.

The whole-file slot audit found no pre-existing read, write, or named offset for
either word. They follow PHIST `occ[104..115]`, BNDPROBE `occ[116..126]`, and
BNDSPLIT `occ[127..130]`. The host allocates a 4096-byte, 1024-word occ buffer.
The per-chunk clear covers only the first 0x100 bytes, so both new accumulators
survive and sum across chunks and repetitions.

## SGPR audit

- `s58:s59` hold the start `GET_REALTIME` result. There is no use or definition
  of either register from `advprobe_start` through either `advprobe_end`.
  `zero_banks` clobbers `s45` and vector scratch only. Later compute-path uses
  of `s58:s59` are fresh definitions, so the pair is dead at both DA_ZDONE
  completion sites.
- `s62:s63` hold the end `GET_REALTIME` result and `s64` holds the modular
  low-32-bit delta. All three are dead scratch at both completion sites.
- `s49` is the local exec-save for the two lane-0 atomics. The preceding
  `lds_put` restores it, and it is dead on return from the probe.
- The handler's live `s44-s47`, `s51`, `s53`, and `s66-s68` are untouched.
  The reserved `s50`, `s54-s57`, and persistent `s69` are also untouched.

ACC is dead on the whole lean boundary path, including both `zero_banks` calls
and both probe endpoints.

## Throttle confirmation

Both `advprobe_start` and `advprobe_end` begin with:

```text
s_cmp_eq_u32 s71, 0
s_cbranch_scc0 <skip>
```

The only new realtime reads are `s_sendmsg_rtn_b64 ... MSG_RTN_GET_REALTIME`
inside those gates. The two occ atomics are also inside the end gate. `s71` is
not modified during a boundary pass, so a sampled start pairs with its sampled
successful end. Drain-gate and C-store-gate bails never execute
`advprobe_end`. An assembly guard rejects `DSWS2_ADVPROBE=1 DEADMAN=0`.
No `s_memtime` instruction was added.

Enabled-object disassembly showed the start gate followed by the
`s[58:59]` realtime read, and both end copies showed the `s71==0` gate, the
`s[62:63]` realtime read, modular subtract, and atomics at offsets 524 and 528.

## Offline gates

### DSWS2_ADVPROBE=0 A1 byte identity: PASS

Command:

```text
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_ADVPROBE=0 ./build_flow.sh
sha256sum occ_dsws2_w30_flow_gd.bin
cmp -s occ_dsws2_w30_flow_gd.bin /tmp/codex_advprobe_baseline.FujF7b/canonical.bin
```

Verbatim output:

```text
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
ADVPROBE_OFF_CMP_EXIT=0
```

### DSWS2_ADVPROBE=1 A1 + DSWS2_RCONV=1 build: PASS

`build_flow.sh` assemble/objcopy output:

```text
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32596B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
3280ef6df0153ff1fcc3a3602a0960779e1573d14e696278c00ac4c674033aa3  occ_dsws2_w30_flow_gd.bin
occ_dsws2_w30_flow_gd.o 43888 bytes
occ_dsws2_w30_flow_gd.bin 32596 bytes
```

The equivalent full A1 `RGADESC=1 DSWS2_RCONV=1
DSWS2_ADVPROBE=1` command assembled and linked with:

```text
ADVPROBE_ENABLED_ASSEMBLE_EXIT=0
ADVPROBE_ENABLED_LINK_EXIT=0
/tmp/codex_advprobe_gate.hymA1e/k.o 44440 bytes
/tmp/codex_advprobe_gate.hymA1e/k.co 45768 bytes
ADVPROBE_GATE_DIR=/tmp/codex_advprobe_gate.hymA1e
```

### Host compile: PASS

The repository `build.sh` reached its host step but its `systemd-run --user`
wrapper could not connect to the sandbox's user bus. The identical host compile
was rerun under `ulimit -v 4194304` and exited 0. It produced 23 pre-existing
warnings at unrelated lines and no warning in the ADVPROBE block. Final output:

```text
/tmp/codex_advprobe_occ_dispatch 636184 bytes
```

### DEADMAN guard: PASS

Verbatim output:

```text
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  FAIL occ_dsws2_w30_flow_gd
occ_kernel_dsws_flow.s:1016:3: error: DSWS2_ADVPROBE requires DEADMAN=1: both realtime reads and both atomics throttle on s71==0.
  .error "DSWS2_ADVPROBE requires DEADMAN=1: both realtime reads and both atomics throttle on s71==0."
  ^
flow build done. fail=1
ADVPROBE_DEADMAN0_GUARD_EXIT=1
```

### RGA spill gate: BLOCKED-by-readonly-.rga

Command:

```text
/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin --isa /tmp/codex_advprobe_gate.hymA1e/isa.txt -a /tmp/codex_advprobe_gate.hymA1e/stats.csv --livereg /tmp/codex_advprobe_gate.hymA1e/lr.txt --livereg-sgpr /tmp/codex_advprobe_gate.hymA1e/lr_sgpr.txt --co /tmp/codex_advprobe_gate.hymA1e/k.co
```

Verbatim output:

```text
/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga

Failed opening file /home/kmbandy/.rga/GPUOpen/rga/rga_cli22-20260723-173105.log for writing: Read-only file system
RGA_EXIT=0
k.co 45768 bytes
k.o 44440 bytes
```

RGA created no ISA, stats, VGPR-livereg, or SGPR-livereg output. Zero spill is
not claimed.

GitNexus pre-edit impact was LOW for the host `main` (zero modeled upstream
dependents) and UNKNOWN for the assembly and shell targets, which are not
modeled symbols. Its required final change detector could not run:

```text
Error: Git diff failed: spawnSync git EPERM
```
