# SEGK per-segment cost attribution - offline report

Date: 2026-08-08

This is the offline phase only. No GPU dispatch, `gpu_run.sh`, `occ_dispatch`, RTC
probe, or silicon result was produced here. The `segk_fit.py` script accepts only
logs with an explicit `WORK-EXACT` verdict and derives time from span ticks and the
reported counter frequency. It never reads rendered TF.

## Baseline proof

Bare `./build_flow.sh` after the change still produces the required binary:

```
58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c  occ_dsws2_w6_flow_gd.bin
```

The build is `WAVES=6 G=8 FM=2 ACC_N=4 FN=4 SEGK=256`, `LDS=34304`, `.text=28852`.
The OFF arm is default 0 and is byte-identical. No new counter was added, so no
new SGPR is needed; C1 reuses existing `s50`. The conditional free-SGPR audit
requirement is therefore not triggered. The C1 arm adds no counter or memory
emission and retains the recorded `SGPR=72`, zero-spill allocation.

## Mechanism and switch inventory

| component | source mechanism and multiplicity | isolation | offline result |
|---|---|---|---|
| C1 dyn-VGPR grow/shrink | `s_alloc_vgpr NFV` at `occ_kernel_dsws_flow.s:5260`, followed by the burst and `s_alloc_vgpr 32` at `:5686`; the same shrink is used by the exhausted path at `:5863`. One grow/shrink cycle is the normal segment-burst path. | New `SEGK_STAYFAT=1`, source default 0 and build default 0. It skips repeated grow/shrink only while the compute wave immediately continues; it shrinks on no-work and role change. | Bites: `.text` 28852 -> 28884. Measurement-only and ORACLE-INVALID until silicon correctness is independently gated. It does not touch `JDEPTH`, `SEGK`, or `DUTYGUARD`. |
| C2 bank flush | Banked live path at `:5575`; `FM*FN*8=128` `ds_add_f32` instructions are gated at `:5588-5598`, then `s_wait_dscnt` at `:5615`. This repeats for each computed segment. | Existing `NODSADD=1`. | Bites: 128 `ds_add_f32` -> 0; `.text` 28852 -> 27824. Wrong C by construction: **ORACLE-INVALID** everywhere used. |
| C3 operand staging quantum | The self-serve operand burst repeats the B-load sequence in the K-step loop; `NOBLOAD` gates the B load at the existing B-load arm. The census confirms 132 -> 68 `global_load_tr_b64` at this geometry. | Existing `NOBLOAD=1`. | Bites: `.text` 28852 -> 28148. Wrong C by construction: **ORACLE-INVALID** everywhere used. |
| C4 reservation/claim | The live reservation is `lds_cas_rtn s47, ASSIGN_HEAD_OFF, s44, s45` at `:6192-6197`, one reservation per rowblk-segment in the ordinary path. | No clean removal. `CFASSIGN=1` changes the reservation mechanism and count rather than removing it; it is not a one-variable C4 isolation. | Residual only. Do not use CFASSIGN as a C4 slope attribution. The log's modern-geometry result was slower and is not a removal experiment. |
| C5 boundary transitions | `DA_ZDONE` is read and tested at `:6050-6082`; boundary ownership/advance is the `.Lflow_da_boundary` path. `BNDPROBE` counts exact group/tile transitions in existing `occ[116..126]` fields. | Existing `BNDPROBE=1`, instrument only; no removal arm exists. | Bites as an instrument: `.text` 28852 -> 29356. Use its counts to normalize C5, not as a timing arm. |

`NOCFLUSH=1` was also checked. It bites in the current banked C-store (`.text`
27444) but removes the final C-store, not the per-segment `ds_add` reduction, and
therefore is not a C2 arm. It is wrong-C and **ORACLE-INVALID** if timed.

## C1 arm details and safety boundary

`SEGK_STAYFAT` uses `s50` as a per-wave state only in the config-of-record path,
where the default self-serve role-flow alternatives do not use it. At role adoption
the state is reset. After the first successful grow it is set, and a following
compute burst branches over the grow. The end-of-burst path branches over the
shrink while the state is set. The no-work shrink resets it. The assembler's
`DUTYGUARD` remains unchanged and still checks `JDEPTH*SEGK <= DUTY_KMAX`.

The build refuses `SEGK_STAYFAT=1` without `DSWS_ALLOW_NONSTD=1`. This arm is a
measurement instrument only, not a configuration candidate. Any silicon row using
it must be labeled **ORACLE-INVALID** until the full oracle and WORK-EXACT gate
prove otherwise; its intended slope is the grow/shrink price, not a correctness or
shipping claim.

## Static census and pre-registered predictions

All counts below are from fresh `llvm-objdump -d --mcpu=gfx1201` disassembly of the
current builds. All builds retain `LDS=34304` and the assembler reports zero spills.

| arm | SHA-256 prefix | text bytes | ds_add_f32 | B global loads | WMMA |
|---|---|---:|---:|---:|---:|
| baseline OFF | `58e965a46f3e162d` | 28852 | 128 | 132 | 256 |
| C1 `SEGK_STAYFAT=1` | `73b8456d96b35378` | 28884 | 128 | 132 | 256 |
| C2 `NODSADD=1` | `92dc2a3e0b51735b` | 27824 | 0 | 132 | 256 |
| C3 `NOBLOAD=1` | `e3301d4d2c3f0b1a` | 28148 | 128 | 68 | 256 |
| C5 `BNDPROBE=1` | `dfda4c3e7e4b7229` | 29356 | 128 | 132 | 256 |

Predictions were recorded before silicon. Baseline is the known 3-point slope
`b0 ~= 0.108 ms/n_kseg`.

| arm | pre-registered slope prediction | interpretation |
|---|---|---|
| C1 stay-fat | `b1 = 0.065..0.080 ms/n_kseg` (25-40% drop) | grow/shrink owns a material share of the per-segment slope. If the slope is flat, the old-geometry 40% result does not transfer. **ORACLE-INVALID** pending gate. |
| C2 NODSADD | `b2 = 0.080..0.100 ms/n_kseg` (8-26% drop) | bank reduction latency/traffic owns a measurable share. **ORACLE-INVALID**; this arm is wrong C. |
| C3 NOBLOAD | `b3 = 0.103..0.108 ms/n_kseg` (0-5% drop) | the prior base effect was about 2%; a flat slope is expected attribution evidence. **ORACLE-INVALID**; this arm is wrong C. |
| C4 residual | `b4 = b0 - (C1+C2+C3)` | reservation share is not independently isolated; report only the residual after valid slope fits. |
| C5 normalized | no pre-registered removal slope | use `BNDPROBE` transition counts to report cost per transition; no timing claim from the instrument build. |

Decision rule: use the fitted slope drop, not an intercept move. Intercepts are
reported by `segk_fit.py` but are not attributed to C1-C5.

## Silicon run matrix (ready to paste)

The contract's wording says "3 arms x 3 SEGK plus baseline 3"; that expands to
12 cells. The table below preserves that explicit design: 9 ablation cells plus 3
baseline controls. Every run must use the same host shape and fixed-rep regime as
the §87 reproduction. `SSWIN=32` is mandatory. Build the selected binary before
each row; `DSWS_ALLOW_NONSTD=1` is required for core-mechanism deviations. The
NODSADD and NOBLOAD rows are **ORACLE-INVALID** for any correctness/TF claim, but
must still pass the WORK-EXACT gate before their span is considered a diagnostic.

```sh
# fixed geometry and host contract for every row
export SSWIN=32 DSWS2_REPS=125 DSWS2_TARGET_SECS=0
export DSWS2_G=8 DSWS2_FM=2 DSWS2_FN=4 DSWS2_ACC_N=4 DSWS2_K=9216

# baseline controls: SEGK=256,128,64
SEGK=256 DSWS_ALLOW_NONSTD=0 ./build_flow.sh
SEGK=128 DSWS_ALLOW_NONSTD=0 ./build_flow.sh
SEGK=64  DSWS_ALLOW_NONSTD=0 ./build_flow.sh

# C1: measurement-only, ORACLE-INVALID until separately gated
SEGK=256 SEGK_STAYFAT=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=128 SEGK_STAYFAT=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=64  SEGK_STAYFAT=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh

# C2: wrong C by construction, ORACLE-INVALID
SEGK=256 NODSADD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=128 NODSADD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=64  NODSADD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh

# C3: wrong C by construction, ORACLE-INVALID
SEGK=256 NOBLOAD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=128 NOBLOAD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
SEGK=64  NOBLOAD=1 DSWS_ALLOW_NONSTD=1 ./build_flow.sh
```

The silicon operator must dry-run-check the host header before each dispatch and
must not use `DSWS2_TARGET_SECS` as the measurement regime. Use the exact span and
computed lines from each resulting log with `python3 segk_fit.py ...`; do not copy
rendered TF into the fit.

## Residual accounting

After accepted baseline/C1/C2/C3 slope fits, compute:

```
C4_residual = b_baseline - drop_C1 - drop_C2 - drop_C3
```

This is a residual, not a claim that all remainder is reservation. C5 is reported
separately as span per exact `BNDPROBE` transition. If C1-C3 are all flat, the
result is the pre-registered counter-free coordination-fabric outcome: the
per-segment slope is not owned by the three ablatable mechanisms.
