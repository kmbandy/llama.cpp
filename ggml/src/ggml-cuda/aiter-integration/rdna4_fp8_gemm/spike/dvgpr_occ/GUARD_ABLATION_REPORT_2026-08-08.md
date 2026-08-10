# DSWS guard ablation offline report

Date: 2026-08-08

This report covers the offline phase only. No GPU dispatch, `gpu_run.sh`, `occ_dispatch`,
rocprof, or hipBLASLt was run.

## Contract and baseline

The config of record was the bare `./build_flow.sh` configuration: `WAVES=6 G=8 FM=2
ACC_N=4 FN=4 SEGK=256`, with `SELFSERVE=1 DECENTASN=1`. The pre-change source was
`occ_kernel_dsws_flow.s` SHA `57ab3100c9450ad6`. The OFF build reproduced the required
bin exactly:

```
58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c  occ_dsws2_w6_flow_gd.bin
```

OFF build facts: `.text=28852`, `LDS=34304`, `SGPR=72`, zero spills.

## Site audit

### LEANGUARD

Symbol: `LEANGUARD`, default `0` in both source and `build_flow.sh`.

Converted macros:

- `lds_put`
- `lds_inc`
- `lds_inc_r`
- both `lds_put_r` definitions

These are lane-0 accessors. Their audited callers are wave-uniform entry paths or
lane-0 publication paths; no converted macro contains a first-active-lane fetch or
read while a selected mask is installed. The save/restore through `s49` remains in
every converted macro, and the skip labels remain present.

Excluded macros:

- `lds_fetch_add` and `lds_fetch_add_r`: excluded because `SELFSERVE=1` uses
  `s_ff1_i32_b32` and reads the result under the selected active-lane mask.
- `lds_cmpstore_adv` and `lds_cas_rtn`: excluded for the same first-active-lane
  behavior in the self-serve configuration, and because their callers include
  arbitration paths where lane 0 is not established by this audit.

The exclusion is intentional; no first-active path was changed.

### GUARDHOIST

Symbol: `GUARDHOIST`, default `0` in both source and `build_flow.sh`.

The source scan found this adjacent-run distribution (comments and assembler
conditionals ignored):

```
run length  1: 103
run length  2:   6
run length  4:   3
run length  8:   1
run length 20:   1
```

The top regions were the coordinator initialization runs (20 and 8 calls), followed
by four-call frontier-update runs. The implemented region is the hot group-frontier
update, the four-store run around the `DA_ZDONE_OFF` release. It uses
`lds_run_begin`/`lds_run_end` and `lds_put_nog`; no global rewrite was attempted.
`GUARDHOIST=1 LEANGUARD=0` assembles independently. With LEANGUARD on, the bracket
uses the single `s_mov_b32 exec_lo, 1` guard.

### LEANMARSH

Symbol: `LEANMARSH`, default `0` in both source and `build_flow.sh`.

The arm reserves `v15`, initializes it once to `DRAIN_HEAD_OFF`, and uses it for the
three active constant-offset `lds_put` expansions at that site. Runtime-address `_r`
macros are excluded. `TRACE=1` is rejected because that optional path uses `v15` as
trace scratch. RGA reports peak livereg 83 versus baseline 82, with zero spills, so
the resident register is free within the config-of-record lean target of 112.

## Disassembly census

Counts are from `llvm-objdump -d --mcpu=gfx1201` on each built object.

| build | `v_cmp_eq_u32` | `s_cbranch_execz` | `s_and_b32` | `v_mov_b32` | `s_mov_b32` | text bytes |
|---|---:|---:|---:|---:|---:|---:|
| baseline / all OFF | 370 | 370 | 405 | 956 | 797 | 28852 |
| LEANGUARD | 29 | 29 | 64 | 956 | 1138 | 26124 |
| GUARDHOIST | 367 | 367 | 402 | 956 | 791 | 28792 |
| LEANMARSH | 370 | 370 | 405 | 954 | 797 | 28844 |
| LEANGUARD + GUARDHOIST | 29 | 29 | 64 | 956 | 1129 | 26088 |

LEANGUARD prediction: 341 converted guarded blocks. Observed drops are exactly
`370-29=341` for both `v_cmp_eq_u32` and `s_cbranch_execz`; this gate passes.

All ON builds retain `LDS=34304`, `SGPR=72`, and zero assembler-reported spills.

## RGA

RGA: `/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga`
v2.14.2.8. The analysis used the source's `RGADESC` flow and the config-of-record
defsym set. RGA LDS/VGPR USED fields are descriptor artifacts and are not used here.

| build | livereg peak | SGPR | SGPR spills | VGPR spills | RGA ISA size |
|---|---:|---:|---:|---:|---:|
| baseline / all OFF | 82 | 72 | 0 | 0 | 27452 |
| LEANGUARD | 82 | 72 | 0 | 0 | 24756 |
| GUARDHOIST | 82 | 72 | 0 | 0 | 27392 |
| LEANMARSH | 83 | 72 | 0 | 0 | 27444 |
| LEANGUARD + GUARDHOIST | 82 | 72 | 0 | 0 | 24720 |

RGA's ISA size includes the analysis descriptor and therefore differs from the
shipped `.text` size; the table reports it only as an RGA metric.

## Binary proofs

The OFF proof is the exact required SHA above. ON binary prefixes, recorded from the
CPU-only builds, were:

```
LEANGUARD                 b1f912df317ce78c8786ae0991af4b48d67773f2952b1e77529d624e48fc2898
GUARDHOIST                1326ec8353ed5727acbf6fba354e9d0c9c90d23fe20ec973bec09c008438045b
LEANMARSH                 430b0590f29376ce50de6e1a9e1d39cd89cc8a10c988adbc25f98e08c32e381d
LEANGUARD+GUARDHOIST     aead44160f72c3465e2c1022b6dbc401bbfe262e697fd6f4ce80b203d1c6412a
```

## Pre-registered silicon predictions

These are predictions for the later silicon session, not behavioral verification.
The denominator is the brief's approximately 1,800 pure guard-bookkeeping
instructions. `f` is the fraction of the 20.8 ns fixed term that this bookkeeping
could own; `f=1` is the upper bound.

| arm | removed bookkeeping instructions | fraction of 1800 | upper-bound fixed-term prediction |
|---|---:|---:|---|
| LEANGUARD | 682 | 37.89% | `20.8 - 7.89*f ns` |
| GUARDHOIST | 15 | 0.83% | `20.8 - 0.17*f ns` |
| LEANMARSH | 3 address-marshalling instructions | 0.17% proxy | `20.8 - 0.035*f ns` |
| LEANGUARD + GUARDHOIST | 691 | 38.39% | `20.8 - 7.98*f ns` |

The LEANMARSH percentage is explicitly a proxy: its removed instructions are address
marshalling, not the exec-guard idiom itself. No correctness claim is made in this
offline report; correctness and the fixed-term refit require the later silicon phase.
