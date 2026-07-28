# DSWS harness — the ONLY sanctioned way to build and run. READ THIS FIRST.

> ## ⚠ CONFIG SUPERSEDED 2026-07-26 — THE PRIMARY CONFIG IS NOW **FM=2 G=4 ACC_N=2**
>
> **Every `FM=1 G=6 ACC_N=3` / `WAVES=30` command below is the OLD config and is NO LONGER PRIMARY.**
> They are kept only as historical provenance for old logs. Do not build or dispatch from them.
>
> **BUILD:** `FM=2 G=4 ACC_N=2 ./build_flow.sh`
> → sha `a581c7b8b8825392…`, `.text` 30,812 B, **LDS 17,920 B**, GROUPS=2, super-tile M=128 rows.
> **DISPATCH adds:** `DSWS2_FM=2 DSWS2_G=4 DSWS2_ACC_N=2` and **`ML8_COOP_CHUNK_MAXS=0.85`**
> (REQUIRED — FM=2 runs 0.81 s/chunk against the 0.75 s default compositor cap, so every dispatch
> aborts without it. This is a designed knob, NOT `DEADMAN_TICKS`, which never moves.)
> **SWEEP adds:** `--fm 2 --g 4 --acc-n 2 --chunk-maxs 0.85`
>
> **WHY:** `s_alloc_vgpr` grow-fail is the ONLY admission throttle under `SELFSERVE=1 BATONGATE=1`
> (kernel asserts this at `:1338`). It was **exactly 0 on every run in project history** — the dyn-VGPR
> moat never engaged. FM=2 doubles `ACC_STRIDE` and it finally binds (140.7 M events across the sweep).
> **At FM=1 the moat is inert and the design cannot be evaluated.**
> Validated 30/30 real shapes, oracle bad=0, WORK-EXACT.
>
> **Full detail: `DSWS_BRIEF_2026-07-27_AM.md` and `DSWS_TESTING_LOG.md` §6–8.**

There is exactly **one** dispatch path and **one** sweep wrapper. Everything else was deleted
2026-07-23 because a stale "canonical run" script (`dsws.sh`, baked to an OLD config) caused a
multi-day confusion about which harness/config was real. **Do not recreate ad-hoc run scripts.**

## The four files that matter

| File | Role |
|------|------|
| `gpu_run.sh` | **THE dispatch funnel.** Every GPU run goes through it. Enforces the latch / deadman / stale-bin guards. Single dispatch = `./gpu_run.sh <name> -- <ENV...> ./occ_dispatch --dsws2`. |
| `dsws_realshape_bench.py` | **The all-shapes sweep** (`live` subcommand). Fail-closed: asserts oracle bad=0 + WORK-EXACT or rejects. It wraps `gpu_run.sh` — one invocation per shape. Use for the full real-shape matrix, NOT a single run. |
| `build_flow.sh` | **The kernel builder** (parametric defsyms → `occ_dsws2_w<WAVES>_flow_gd.bin`). |
| `build.sh` | **The host builder** (compiles the `occ_dispatch` PM4 harness). |

Nothing else dispatches. If you need a one-off run, call `gpu_run.sh` directly — never write a
new wrapper that invokes `./occ_dispatch` (that bypasses the anti-brick guards).

## ⛔ CONFIG OF RECORD — 2 WG/CU (kmbandy, 2026-07-26). THIS SUPERSEDES THE A1 SECTION BELOW.

**The standard is `WAVES=16` + `ML8_POOL=128` = 128 WGs × 16 waves = 2048 resident waves (2 WG/CU),
with prefetch and overlap ON.** It is now enforced *mechanically* in `build_flow.sh` and
`gpu_run.sh`, because "I know the config" has failed twice as a control — most recently on
2026-07-26, when an entire POLLSTAGE campaign was measured at 64 WGs / 1 WG/CU **and with
`DSWS2_PREFETCH=0`**, after the profile was copied out of a findings doc.

```
./build_flow.sh                       # standard, SEGK=256  -> 585d287e
SEGK=128 ./build_flow.sh              #                     -> 62001b24
SEGK=64  ./build_flow.sh              #                     -> bc75d341

./gpu_run.sh <name> -- FLOW_WAVES=16 ML8_POOL=128 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 \
  DSWS2_ACC_N=3 FLOW_POOL_N=1 DSWS2_SEGK=256 DSWS2_K=2560 SSWIN=32 \
  DSWS2_ORACLE_MTL=22 DSWS2_ORACLE_NTL=144 DSWS2_ORACLE_STRIDE=1 \
  ML8_COOP_CHUNK=96 STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```

`build_flow.sh` now applies the standard as **defaults** (`:=`, so explicit env still wins) and
REFUSES if a core mechanism is off. `gpu_run.sh` REFUSES a non-standard launch geometry.
**Deviating is allowed but must be EXPLICIT: `DSWS_ALLOW_NONSTD=1`, and name the reason in the
logname.** Both tools print the active config, so every log records what actually ran.

| knob | standard | status (verified 2026-07-26) |
|---|---|---|
| `WAVES` / `ML8_POOL` | **16 / 128** | 2 WG/CU. `2×13,824=27,648B < 65,536` ✓, `2×16 = 32/32` wave slots ✓ |
| `SEGK` | **{64, 128, 256}** | all sanctioned; LDS is 13,824B at **all three** (operands are L2-only under SELFSERVE) |
| `DSWS2_PREFETCH` / `DSWS2_OVERLAP` | **1 / 1** | LIVE. Prefetch requires OVERLAP=1; OVERLAP is free since the 07-25 staging fix |
| `SELFSERVE` `DECENTASN` `BANKZERO` `BATONGATE` `STAGGER` | **1** | mandatory — the assembler refuses without them |
| `JDEPTH` | **1 (pinned)** | **SELFSERVE requires JDEPTH=1.** The `ksi%J` lead-gate ("k-slice filter") is NOT available at ANY SEGK. Restoring it is a **design change**, not a knob |
| `POOL_N` | any (**inert**) | byte-identical at 1/2/4 since the 07-25 SELFSERVE dead-staging fix |
| `MAXFAT` | any (**inert**) | the `FATTOK` token layer is compiled to no-ops under `BATONGATE` |
| `CFASSIGN` / `KMAJOR` | **0 / 0** | required by `DSWS2_PREFETCH` |

**Compiled-in ≠ working.** Two standard mechanisms are in the binary but measure inert at runtime:
the **baton** (fills dyn-VGPR budget valleys; `grow-fail = 0` in every run, including at 2048 waves,
so it has never had anything to do) and the **`MAXFAT` cap**. Do not cite either as active.

---

## Canonical config = A1 → baseline sha `cac3ff7c2338e73f` (verified 2026-07-23) — SUPERSEDED, kept for reproducing old measurements (use `DSWS_ALLOW_NONSTD=1`)

**A remembered sha is not a sha. Identify builds by COMMIT + DEFSYMS.** The A1 config string
(`G=6 ACC_N=3 SEGK=256 POOL_N=1 WAVES=30 SSWIN=32 CFASSIGN=1 …`) is a *summary*; the byte-exact
build needs the full profile below. Plain `./build_flow.sh` defaults (`STAGGER=0`) do **NOT**
reproduce `cac3ff7c` — they give a different kernel (`55a6983d`). Always pass the full profile.

### Build the baseline (RCONV off) → `cac3ff7c2338e73f`
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 \
  SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 ./build_flow.sh
```
Add `DSWS2_RCONV=1` (and optionally `DSWS2_RCONV_COAST_N=<N>`) for the runtime-role-conversion
kernel → `53a309f76a9bbea7`.

### Dispatch (single shape, ffn_gate_up M2048 = 2112×9216×2560), through gpu_run.sh
```
./gpu_run.sh <name> -- FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 \
  FLOW_POOL_N=1 DSWS2_SEGK=256 DSWS2_K=<k> \
  DSWS2_ORACLE_MTL=<padded_m/96> DSWS2_ORACLE_NTL=<n/64> DSWS2_ORACLE_STRIDE=<1|8> \
  DSWS2_TARGET_SECS=<secs> [ML8_COOP_CHUNK=<tiles, default 512>] \
  STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```
**The host GEOM (`DSWS2_ACC_N` / `DSWS2_SEGK` / `FLOW_POOL_N` / `FLOW_WAVES` / `DSWS2_G` /
`DSWS2_FM`) MUST match the bin's build defsyms**, or the run is wrong / wedges. Expected for the
established fed A1 run at this shape: `computed=760320` WORK-EXACT, `oracle bad=0`.

For PHASEPROBE profile runs: build with `PHASEPROBE=1` (+ `PHSHIFT`/`PHSPLIT` as needed), same
`gpu_run.sh` dispatch plus host `ML8_PHSHIFT`/`ML8_PHSPLIT`. `bench.py` has **no** PHASEPROBE path.

## Rule
`build_flow.sh` builds → `gpu_run.sh` dispatches (or `dsws_realshape_bench.py live` for the sweep,
which itself calls `gpu_run.sh`). That is the whole harness. Nothing else.
