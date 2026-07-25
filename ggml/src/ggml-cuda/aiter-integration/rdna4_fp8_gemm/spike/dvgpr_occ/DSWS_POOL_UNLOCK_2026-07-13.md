# ⚠️ SUPERSEDED — READ `DSWS_THE_FLUSH_IS_THE_KERNEL_2026-07-13.md` INSTEAD

> **This document is WRONG about what matters, and its "Next steps" are ALL dead ends.**
> Written on 2026-07-13 morning; refuted by measurement the same evening.
>
> | this doc claims | measured truth |
> |---|---|
> | "`grow-fail=1588`, the pool is binding, **BUILD THE STAGGER**" | `grow-fail = 0` on a clean build. The 1588 was an artifact of `PHASEPROBE=1`, which slows the machine **~44x**. **There is no VGPR contention. The stagger is dead.** |
> | "POOL_N is the whole flow economy" | span **identical** at POOL_N=1/2/3/4. **Not a throughput lever at all.** |
> | "WOFLUSH=1 is the win (frees LDS)" | WOFLUSH is **6.7x SLOWER**. It replaces the LDS reduction with **1.21 BILLION global atomics**. |
> | "FLUSH is 0.2% of runtime" | **The flush is 97.4% of runtime.** That 0.2% was measured on a 33ms toy that never reached steady state. |
>
> **Root cause of every error above: nothing here was measured on a run longer than 33ms.**
> The wave economy never spun up. See §1 of the superseding doc.
>
> **Still valid in this doc:** the POOL_N slot-stride constant bug (`s_lshl_b32 sX,sY,14`),
> the RDNA4 dyn-VGPR hazard, the guard-page brick fix, and the geometry constraints.
> Everything about *priorities* and *throughput* is superseded.

---

# DSWS — THE POOL UNLOCK, and the moat engaging for the first time
**2026-07-13 (all day + night).** Read this before touching the flow kernel.

**Headline:** the flow economy was implemented, correct in design, and **disabled by a hardcoded
constant**. Fixing it made the kernel bit-exact at `POOL_N=3`, dropped FOLLOW_WAIT from 66.8% to 19.4%,
and produced **`grow-fail = 1588` — the per-SIMD VGPR budget binding for the first time in the project's
history.** The dyn-VGPR moat has never engaged in any run we have ever taken. It has now.

Throughput has **not** moved yet (0.3 TF vs hipBLASLt's 70.4). The new bottleneck is **GROW at 46% with
1588 failed grows** — fat waves colliding on `s_alloc_vgpr`. That is precisely the problem the
"traveling peak" / staggered grow-shrink handshake was designed to solve. **We have finally reached the
actual architecture instead of fighting the scaffolding.**

---

## 0. TL;DR — what changed, what to do next

| | |
|---|---|
| **Fixed** | 3x hardcoded `s_lshl_b32 sX, sY, 14` (slot*16384) -> `s_mul_i32 sX, sY, OPSTRIDE` |
| **Fixed** | `LDS_TOTAL_FLOW` reserved 48KB of banks even under `WOFLUSH` (never touched) |
| **Fixed** | `build_flow.sh` now DELETES the stale `.bin` on a failed build |
| **Wired** | `CNT_FEED` (never had a call site), `CNT_FEEDMT`, and the whole `phase_*` profiler |
| **Found** | The RDNA4 dyn-VGPR **VGPR-write hazard** (see §5) |
| **NEXT** | (1) fix `PH_WMMA`/`PH_FLUSH` under WOFLUSH, (2) **the stagger** — grow collisions |

**The one-line reproduce (bit-exact, real shape):**
```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
WAVES=16 G=6 SEGK=32 ACC_N=6 POOL_N=3 WOFLUSH=1 BANKZERO=0 STAGINSTR=1 PHASEPROBE=0 \
  DEADMAN=1 TFPROBE=1 DIAG=0 ./build_flow.sh
ML8_YIELD_MS=5 ML8_YIELD_EVERY_MS=50 DSWS2_FLOW=1 DSWS2_SEGK=32 DSWS2_ACC_N=0 DSWS2_G=6 \
FLOW_POOL_N=3 FLOW_WAVES=16 DSWS2_ORACLE_MTL=3 DSWS2_ORACLE_NTL=64 DSWS2_NKSEG=64 \
DSWS2_ORACLE_STRIDE=32 ML8_POOL=64 ML8_COOP_CHUNK=64 ML8_COOP_STREAM=1 ML8_COOP_CHUNK_MAXS=0.9 \
timeout 300 ./occ_dispatch --dsws2
```
**`DSWS2_ACC_N=0` is REQUIRED under WOFLUSH** — it tells the host to stop reserving bank bytes (§3.2).

---

## 1. THE BUG THAT NAILED `POOL_N` TO 1

`occ_kernel_dsws_flow.s` computed a pool slot's operand base as:
```asm
s_lshl_b32 s52, s45, 14        // slot * 16384
s_add_u32  s52, s52, OP_BASE
```
**16384 is `OPSTRIDE` at SEGK=64** (`FN*16*64 + G*16*FM*64 = 4096 + 12288`).
**At SEGK=32, `OPSTRIDE` is 8192.** So slot 1 read its operands from **double the correct offset** —
the wrong address entirely.

=> **`POOL_N > 1` HAD NEVER WORKED AT SEGK=32.** And SEGK=32 is the **only** size that fits LDS at
ACC_N=6 (SEGK=64 needs 65792B > 65536).

That single constant is why:
- `POOL_N` was pinned to 1 in every invocation,
- someone wrote *"POOL_N=1 required for correctness"* (a **workaround**, recorded as if it were a design),
- `slot_of` returns literal `0` at POOL_N==1,
- **FOLLOW_WAIT was 81%** — a depth-1 pipeline is a blocking handshake,
- **`grow-fail` was 0 forever** — with one assignment live, only ACC_N waves are ever fat, so the VGPR
  budget can never be contended, so **the moat can never engage.**

**FIX:** `s_mul_i32 sX, sY, OPSTRIDE` at all three sites (was `:1842`, `:2107`, `:2159`).

### 1a. The design was never wrong
`build_flow.sh`'s own header: *"FIX 1 (flow economy) ... **N-deep pool** + ROLE mailbox + coordinator"*.
Its own **default is `POOL_N=3`**. kmbandy's 3-core-fixes plan (2026-07-04) says fix #1 was *"waves always
flow ... kills FOLLOW_WAIT"*. **The fix shipped. We ran it with the pool switched off for two weeks.**

---

## 2. RESULTS

### 2.1 Correctness (576x512x2048, FULL oracle, `WOFLUSH=1`)
| POOL_N | oracle |
|---|---|
| 1 | `ok=1152 bad=0 max_rel=0` **bit-exact** |
| 2 | `ok=1152 bad=0 max_rel=0` **bit-exact** |
| 3 | `ok=1152 bad=0 max_rel=0` **bit-exact** |

### 2.2 Real shape — ml8 `moe attn_q` (M=576 K=2048 N=4096). hipBLASLt = **70.4 TF**
| POOL_N | oracle | TF | FOLLOW_WAIT | GROW | SHRINK | **grow-fail** |
|---|---|---|---|---|---|---|
| 1 | ok=288 bad=0 | 0.3 | **66.8%** | 9.6% | 23.6% | **0** |
| 2 | ok=288 bad=0 | 0.3 | 21.4% | — | — | 0 |
| 3 | ok=288 bad=0 | 0.3 | **19.4%** | **46.0%** | 34.6% | **1588** |

**LDS: 57600B -> 8448B** (WOFLUSH, POOL_N=1). 7x less.

### 2.3 >>> `grow-fail = 1588` IS THE RESULT <<<
The per-SIMD VGPR budget is **binding**. Fat waves are genuinely contending for the register file.
This has **never happened before** in any DSWS run. The moat — the one thing HIP structurally cannot
express (`s_alloc_vgpr` via raw PM4/KFD) — is finally being exercised.

### 2.4 TF has NOT moved (0.3). Do not oversell this.
The wait is gone but throughput hasn't followed, because the bottleneck **moved** to grow collisions.
That is progress, not success.

---

## 3. THE OTHER TWO THINGS THAT WERE CAPPING THE POOL

### 3.1 `LDS_TOTAL_FLOW` reserved banks that WOFLUSH never touches
```asm
.set LDS_TOTAL_FLOW,(ACC_BASE + ACC_N*ACC_STRIDE)   // UNCONDITIONAL
```
Under `WOFLUSH=1` the kernel has **no LDS accumulator banks at all** (each burst
`global_atomic_add_f32`s ACC straight to C) — yet this still reserved `ACC_N*8KB` = 48KB, which alone
capped `POOL_N`. Now `WOFLUSH`-aware.

### 3.2 The HOST does not know about WOFLUSH — **you must pass `DSWS2_ACC_N=0`**
`occ_dispatch.cpp:1823`:
```cpp
uint32_t ldsBytesRaw = 256u + poolSlots * operandBytes + accBytes;
```
`accBytes` comes from `DSWS2_ACC_N`. With a WOFLUSH bin and `DSWS2_ACC_N=6` the host requests **65792B >
65536** -> **the workgroup silently never launches** (every counter reads 0; it *looks* like a hang, it
is a dispatch that could not fit). Pass `DSWS2_ACC_N=0`.

### 3.3 `OP_BASE` is a HOST+KERNEL CO-CHANGE — do not touch it alone
`OP_BASE=256` is hardcoded in **both** the kernel and `occ_dispatch.cpp:1823`. Raising it (to allow
`POOL_N>3`, since `SLOTC_BASE=148 + POOL_N*32 <= OP_BASE`) **requires changing the host too**.
I raised it to 512 kernel-side, broke the LDS agreement, and reverted. **POOL_N is capped at 3 until the
host is changed.**

---

## 4. THE NEW BOTTLENECK: GROW COLLISIONS -> **THE STAGGER**

`GROW = 46%` of compute-wave time, `grow-fail = 1588`. Waves hit `s_alloc_vgpr NFV`, **fail**, retry.
Their fat peaks are **colliding**.

This is exactly the problem the **traveling-peak / staggered grow-shrink handshake** exists to solve
(KG: the rolling dyn-VGPR architecture). The governing rule: *the SUM of all resident waves'
INSTANTANEOUS allocations must stay under the per-SIMD budget at every instant* — phase-offset the peaks
so the trapezoids interleave instead of stacking.

**This is the next piece of work, and it is the real DSWS.**

### 4.1 CAUTION — a previous stagger attempt was measured DEAD
KG `50147c07` / the 2026-06-17 handoff: a `TGID_X*STAGGER` phase-offset on *persistent single waves* was
measured dead-flat (0.3-0.4 TF across a 3x3 pool x stagger grid). **That was a different mechanism**
(phase-offsetting independent single waves to interleave FEED stalls, on an occupancy-capped
micro-batch). It does **not** pre-refute the grow/shrink handshake here — but it is a warning that
"stagger" has failed once before, and the *mechanism* must be argued, not assumed.

---

## 5. THE RDNA4 dyn-VGPR HAZARD (undocumented — this is a real hardware finding)

> **A VALU VGPR WRITE adjacent to `s_alloc_vgpr` CORRUPTS THE REGISTER FILE.**

Realloc moves the wave's VGPR base out from under a pending write. Symptom: **partial, low-magnitude,
NON-DETERMINISTIC** corruption of C, with counters that still read perfectly (the writes DO land — they
corrupt the register file on the way out).

**Proven by elimination — every row a measured GPU run:**
| probe | result |
|---|---|
| gauge with ONLY the exec save/mask/restore (no VGPR write, no atomic) | **CLEAN 2/2** |
| same + ONE `v_mov_b32 v3, 1`, **still no memory op at all** | **bad=1152** <- TRIGGER |
| same but writing **v7** instead of v3 | **bad=1152** (not register-specific) |
| a bare `s_sleep 2` at the identical sites (same latency, no VGPR/exec/mem) | **CLEAN 3/3** (not timing) |
| the SAME gauge in the C-store path, far from any `s_alloc_vgpr` | **CLEAN** (it is PROXIMITY) |
| 3 identical runs of the failing bin | bad = **527 / 758 / 647** (non-deterministic) |
| `s_wait_storecnt 0x0` before `s_alloc_vgpr` | **NO HELP** |
| `s_waitcnt_depctr depctr_va_vdst(0)` before `s_alloc_vgpr` | **NO HELP** |

**ISA context:** `S_ALLOC_VGPR`'s pseudocode (RDNA4 ISA line 14366) is
`WaitIdleExceptStoreCnt(); n = ReallocVgprs(...)` — it drains everything EXCEPT storecnt. But draining
storecnt does NOT fix this, and neither does `va_vdst`. **The hazard is not in the ISA's wait-state
tables. It is UNDOCUMENTED.**

**WORKAROUND IN PLACE:** `FORENSICS` defaults to **0** (in *both* the kernel `.ifndef` AND
`build_flow.sh` — the script's own default silently overrides the kernel's; that bit me once).
The two guilty gauges are `flow_gauge` at the SHRINK/TASHRINK sites, which bracket `s_alloc_vgpr`.
`flow_snapshot` is **INNOCENT** (clean in isolation). The C-store gauge is **INNOCENT**.
**Proper fix later:** relocate the spin gauges away from any `s_alloc_vgpr`.

**RULE: never put a VALU VGPR write next to `s_alloc_vgpr`.** `phase_stamp` is safe there — it is pure SALU.

---

## 6. INSTRUMENTATION — what is now wired, and what is still lying to you

### 6.1 Newly wired (all had ZERO call sites before today)
- **`CNT_FEED`** — a completed feed stage (at the `STAGE_HEAD` advance). `feed-stages=0` was **never a
  measurement**; the counter did not exist. Feed waves run hard: **100k+ stages**.
- **`CNT_FEEDMT`** — a feed wave that found nothing to stage.
- **The whole `phase_*` profiler** (`phase_reset` / 6x `phase_stamp` / `phase_flush`). The macros and the
  host printout (`[dsws2 PHASE breakdown]`) both existed. Nobody ever plugged them in.

### 6.2 STILL BROKEN — do not misread these
1. **`PH_WMMA` and `PH_FLUSH` read 0.0% under WOFLUSH.** Their call sites live inside the `.else`
   (banked) branch and are **compiled out**. The real compute time is misattributed to SHRINK.
   **The GROW/SHRINK split is NOT trustworthy until this is fixed. FIX THIS FIRST.**
2. **`PHASEPROBE=1` perturbs timing enough to open a LATENT RACE.** `max_rel` goes 0 -> 0.02-0.9 and bad
   counts vary run to run. The kernel is **bit-exact with `PHASEPROBE=0`**. There IS a real latent race
   in the kernel; it is narrow. **Separate finding, needs its own hunt.**
3. **There is still NO runtime ROLE CENSUS.** The host's `16c0a0b` is a **launch label**
   (`occ_dispatch.cpp:1828`, inherited from the coop kernel's STATIC-mix naming). The flow kernel is
   EMERGENT — roles live in `ROLE[wid]` in LDS. **I misread that label as telemetry and reported "zero
   feed waves." It was false.** Add a real census.

---

## 7. GEOMETRY CONSTRAINTS (these block the real shapes)

- **`SEGK` is whitelisted to {32, 64}** by the host, and **only 32 fits LDS** at ACC_N=6.
- **`n_kseg` must be a POWER OF TWO** (`sti = (t<<shift)|ksi`, `shift = ff1(n_kseg)`).
- => **K must be a power of two, <= 2048.**
  **EXCLUDED: K=2560, K=9216** (most of ml8 dense) and **K=768 / K=1536** (BOTH mlambaformer MoE
  experts — *the 56%-of-GEMM-time target*).
- **M must be a multiple of 192** (`G*16*FM`), **N a multiple of 64** (`FN*16`).
- **Cost of fixing the power-of-2 limit:** `DECODE_STI` shift+mask (2 SALU) -> magic-div (3 SALU), once
  per rowblk-segment = **+0.06 instructions per WMMA**. The machinery already exists in the `KMAJOR`
  path. **Essentially free — do it.**

---

## 8. THE BANKED PATH (`WOFLUSH=0`) — why it cannot pipeline

`acc_base_of` = `ACC_BASE + r*ACC_STRIDE` — **no slot term AND no group term.** The LDS accumulator banks
are shared across **both** slots and groups. Per-slot banks would need `POOL_N * G * 8KB` = **96KB** at
POOL_N=2. The card has 64.

I built a `BANKZERO` path (zero the banks once per tile; every ksi becomes a pure `ds_add_f32`, killing
the `ksi==0` fresh-write that forced ordering) and a **tile-scoped completer** (`TILEDONE[group]`; whoever
brings it to `ACC_N*n_kseg` owns the C-store — valid because all ksi of a tile share `t`, so `mblk`/`tcol`
are identical across them). **`BANKZERO=1` is correct at POOL_N=1** but the banked path still cannot
pipeline, because of the **group** sharing. Both are left in, gated, `BANKZERO` default 1, inert under
WOFLUSH.

**=> WOFLUSH is the path forward.** No banks, no sharing, no ordering, deep pool, 7x less LDS.
Its cost (n_kseg x more C-write traffic as global atomics) is a trade worth making: **FLUSH measured only
6.4%** of wave time when FOLLOW_WAIT was 81%.

---

## 9. PROCESS FAILURES TODAY (all mine — do not repeat)

1. **I ran ~40 dispatches with `FLOW_POOL_N=1`** because I copied an invocation out of a handoff doc and
   never asked why the pool was 1 when the build script defaults it to 3. **Then I measured that
   behavior and started designing a fix for a knob I was holding wrong.**
2. **STALE BIN.** A `POOL_N=4` sweep ran against a stale `POOL_N=3` bin: the overlap `.error` fired,
   I swallowed it with `>/dev/null 2>&1`, and `cp` copied the previous iteration's artifact. **This is
   the SAME stale-bin failure that poisoned the baselines two days ago.**
   **FIXED: `build_flow.sh` now `rm -f`s the `.bin` and `.o` on a failed build.**
3. **I theorized wrong four times** (accumulator clobber; in-flight-store — which matched the ISA text
   *exactly*; latent race; "the economy is starved"). **Every one was killed by a measurement.**
   The bisects earned everything; the reasoning earned nothing.
4. **I misread a printf as telemetry** (`16c0a0b`) and reported "zero feed waves" as a finding.

**STANDING RULES:** verify the baseline before localizing against it. Read the callee before calling
something a bug. A failed build must never leave a runnable artifact. Measure, then reason.

---

## 10. NEXT SESSION — in order

1. **Fix `PH_WMMA` / `PH_FLUSH` for the WOFLUSH path.** The GROW/SHRINK split is untrustworthy until
   then, and it is the split that tells us how bad the grow collisions really are.
2. **THE STAGGER.** `grow-fail=1588`, `GROW=46%`. Phase-offset the fat peaks so they interleave instead
   of colliding. **This is the actual DSWS architecture, and we can finally reach it.**
   Heed §4.1: a previous stagger was measured dead — argue the mechanism, do not assume it.
3. **Add a runtime role census** (§6.2.3). We are flying blind on the role economy.
4. **Kill the power-of-2-K limit** (§7). It is ~free and it unblocks the real ml8/mlambaformer shapes —
   including the two MoE experts that are 56% of mlambaformer's GEMM time.
5. **Hunt the latent race** (§6.2.2).
6. Host co-change for `OP_BASE` -> `POOL_N > 3` (§3.3).

**Do NOT** touch SEGK/flush cost, vector width, or tile size. The hipBLASLt teardown
(`HIPBLASLT_TEARDOWN_2026-07-13.md`) argued those hard — and then the phase profile measured FLUSH at
**6.4%** while FOLLOW_WAIT was **81%**. They are all downstream of a machine that was not running.

---

## 11. TREE STATE (all uncommitted)
- `occ_kernel_dsws_flow.s` — slot-stride fix, `WOFLUSH`-aware LDS, `BANKZERO` + `zero_banks` +
  tile-scoped completer, `FORENSICS`/`FATGAUGE` split, SALU counters (`cnt_*`), wide C-store,
  `CNT_FEED`/`CNT_FEEDMT`, phase profiler wired.
- `build_flow.sh` — `FORENSICS`/`FATGAUGE`/`BANKZERO` flags; **deletes the stale bin on failure**.
- `occ_dispatch.cpp` — the **guard-page fix** (`IsaMapBytes`, all 21 ISA alloc sites). Independent
  safety fix; zero bricks across ~60 dispatches today. **Commit-worthy on its own.**
- Docs: this file + `HIPBLASLT_TEARDOWN_2026-07-13.md`.
- Logs/bins: `~/dsws_gpu_logs/iso_20260713/` (every variant, hashed).
