# ⛔ REJECTED — DO NOT IMPLEMENT (2026-07-20)

**Rejected by Codex gpt-5.6-sol adversarial review, on source evidence. The premise is false.**
Kept as a record of the error; the caveats section is still accurate and reusable.

**Why it is wrong — `MSSCAN` is NOT the concurrency cap in the config of record:**

1. **The success criterion was logically impossible.** Self-serve waves attempt `s_alloc_vgpr` BEFORE
   entering any ring path (`:3937`); only a GROW FAILURE falls through to the ring stamp (`:3951`). The
   ring path IS the failure path. Unpinning it cannot "make grow-fail non-zero for the first time" —
   ring work only exists *because* a grow-fail already happened.
2. **"8 reservations stamped and staged" is FALSE.** Successful carry-through reservations are published
   as PRE-COMPLETED control sentinels and use direct global operand loads (`:4009`, `:4057`). They are
   not claimable work being ignored. Ring-staged operands remain only `POOL_N=1` deep.
3. **The `[DRAIN, STAGE)` scan is bounded by `POOL_N`, not `SSWIN`** (`:3074`). At `POOL_N=1` it contains
   only the head, so unpinning yields ZERO extra concurrency — while burning a `wid % window`
   repeated-subtraction loop with `window==1`.
4. **The plan's own safety gate is a DAM.** `lds_get_r` = `ds_load_b32` + `s_wait_dscnt 0x0`, a BLOCKING
   read — four lines after the plan prohibits adding blocking reads.
5. **A pre-claim `SL_GEN == cursor` read is not an atomic pin.** `SL_GEN` and `SL_RBNEXT` are separate LDS
   words; the source attributes compute safety to `RB_PENDING` + the won CAS, not to a generation gate.
6. Widening the ring window to `SSWIN` would be a **silent wrong-C** path: `SSWIN` widens only the control
   array; the feeder deliberately refuses to stage beyond `POOL_N` from DRAIN (`:3684`), so multiple
   logical slots would overwrite the same resident operand buffer.
7. The guard text at `:923` is itself STALE: `occ[95]` is a lane-0-inactive diagnostic, not a "straddle
   observer", and `s46` there is `exec_lo & 1`, never compared against DRAIN. I took the comment at face
   value instead of reading the instructions.

**Confirmed sound and NOT to be disturbed:** `drain_advance` head-walk; the stamp ordering
(`RB_PENDING` -> `SL_STI` -> `SL_GEN` last); the `DA_ZDONE`/`ZLOCK`/`GSTORED`/`TILEDONE` boundary interlock.
**Critical warning:** those are all COUNT-based gates — they cannot detect duplicated work or wrong
operands. A numerically wrong contribution still produces exact `TILEDONE`/`DRAIN`/`GSTORED` counts.

**Where the investigation must go instead (Codex):** *"A valid next investigation must start from the
self-serve reserve/grow/direct-compute path, because that is where compute admission actually occurs."*
Since `grow-fail == 0`, every wave that GETS a reservation grows and computes — so admission is gated at
GETTING a reservation, not at the grow.

---

# PLAN — remove the head-pin: let compute claim across the staged window (2026-07-20)

## Goal

Delete the last **software** cap on concurrent compute waves, so the only throttle is the **physical**
`s_alloc_vgpr` grow-fail. This is the river principle: gates that bias the flow, never a dam that stops it.

## What is measured, not assumed

Five ablations on the config of record (`SSWIN=8 ACC_N=3 SEGK=256`, span 1.704 s, oracle CLEAN):

| ablated | span | delta |
|---|---|---|
| control | 1.704 s | — |
| every C store (`NOCFLUSH=1`) | 1.710 s | +0.3% |
| every B load, 768 GB (`NOBLOAD=1`) | 1.747 s | +2.5% |
| **all WMMA math** (`NOWMMA=1`) | 1.711 s | **+0.4%** |

**Loads, math and stores are all free.** The entire compute burst can be deleted and the kernel still
takes 1.7 s. All of it is coordination latency, and it is not being hidden because too few waves are
in flight: 2167 ns per rowblk-segment per WG / `ACC_N=3` concurrent => **~6.5 us of unhidden round-trip**,
with **192 of 1920 resident waves computing (10%)**.

## The cap, precisely

It is **`MSSCAN=0`** (head-pinned compute), not the banks and not `ACC_N` as such.

- `acc_base_of` is `bank*ACC_STRIDE + ACC_BASE` — **no slot term, by design**. Many in-flight slots
  holding different `ksi` of the same tile all `ds_add_f32` into the same bank; that IS the split-K sum,
  and `BANKZERO=1` makes every `ksi` a pure `ds_add` so it is concurrency-safe. **Banks cap nothing.**
- `MAXFAT=0` — the software fat-token cap is off; the source itself says the hardware `s_alloc_vgpr`
  is the real concurrent-fat cap.
- `VBUDGET=1536` — the physical budget, which *should* bind and currently never gets the chance.
- **`MSSCAN=0` pins every wave to the single DRAIN-head slot**, which offers exactly `ACC_N` rowblk
  claims (`s_cmp_ge_u32 s47, ACC_N` on `SL_RBNEXT`). 3 waves compute, 27 coast.

`SSWIN=8` already keeps 8 reservations stamped and staged. **The work is present; the compute path
refuses to look at it.**

## Scope

`occ_kernel_dsws_flow.s` only. The `MSSCAN=1` body already exists (`:3067`, "spread the waves across the
STAGED WINDOW [dh, sh)"); it is *refused* under `DECENTASN` by the guard at `:923`. The task is to make
that combination correct and remove the refusal — **not** to invent a new mechanism.

Gated so `MSSCAN=0` remains **bit-identical** to the current default bin (`f36c06a0`).
No host change. No GPU execution — offline assembly only.

## What has to happen

- The two stated blockers in the `:923` guard are **register aliasing**, not architecture: the straddle
  observer (`occ[95]`) and the post-grow slot re-derivation both compare against DRAIN in `s46`, and
  `MSSCAN` reassigns `s46` to a spread cursor. Re-point both at the wave's actual chosen cursor.
- A wave that picks slot `c` from the window must validate `SL_GEN == c` before touching that slot's
  `SL_STI`/`SL_RBNEXT` — the v3 gate that already exists on the feed and coast picks. Spreading does not
  need a new safety property; it needs the existing one applied on this path.
- Remove the `:923` refusal only once the above holds, and say plainly which invariant now covers it.
- Expected consequence: concurrency rises from `ACC_N` toward `SSWIN*ACC_N`, `s_alloc_vgpr` starts
  failing, and `door4 GROW-FAIL` goes **non-zero for the first time in this project's history**.

## Caveats and nuances (each cost a real run or a real day)

- **THE RIVER: gates, never dams.** Do not add a cap, a wait, a blocking read, or a hard role split to
  make spreading safe. Every previous attempt to bound concurrency in software (`MAXFAT<ACC_N`, the
  `.Lflow_batonwait` spin, the NCARR carrier/feeder split) broke the kernel. If spreading cannot be made
  correct without a wait, **report that instead of adding the wait**.
- **SILENT WRONG C IS THE FAILURE MODE HERE**, and it is worse than a hang. A wave computing against the
  wrong slot's `STI` produces a wrong C with a clean-looking run. **The oracle samples 32 of 16384 tiles
  (0.2%)** — it can easily miss this. Bring-up MUST use full-check or a much denser stride; a "CLEAN"
  oracle at stride 512 is NOT evidence for this change.
- **`grow-fail` going non-zero is the SUCCESS signal, not a regression.** It means the physical budget
  finally binds. Waves that fail to grow coast and retry — that is the design working.
- **THE DUTY-CYCLE INVARIANT still holds.** Time at peak ~ `JDEPTH*SEGK`; this change must not extend the
  fat window. The `DUTYGUARD` assembler check will refuse an attempt and is right to. Do not "while we're
  here" raise `SEGK` or `JDEPTH`.
- **Never disable coast->feed.** A wave that cannot compute must still do the next productive thing.
- **`GSTORED`, not `DRAIN`, is the real bank-reuse barrier.** The boundary gates on both.
- **`s49` is the reserved `exec_lo` save** for every `lds_*` macro; never hold a live value across one.
- **THERE ARE ZERO FREE SGPRs** — full audit of `.set` aliases AND literal `sNN` AND `s[a:b]` ranges,
  s0..s105. `s91/s92/s93` look free in the `.set` table and are NOT (JDEPTH counter, fat scratch).
  Grep **both** spellings; either alone yields a false "free" and corrupts live state.
- **`s_alloc_vgpr` does NOT drain VMEM stores.** Drain before every reallocation.
- **Exec-masked atomics must target the first ACTIVE lane**, result read while that mask is installed.
- **Instrumentation cost scales with TIMES EXECUTED.** PHIST measured **~220%**, not the ~10% budgeted.
- **Ablation switches must be proven to bite.** `NOCFLUSH` was inert on the banked path and would have
  produced a fabricated "the flush is free" finding; it now has a negative control. Same discipline here.

## Gates (all offline, no GPU)

Config of record: `WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1
BANKZERO=1 STAGGER=1 SELFSERVE=1 SSWIN=8 PHIST=0 NOCFLUSH=0 NOBLOAD=0 NOWMMA=0 FORENSICS=0 STAGINSTR=1
TFPROBE=1 DEADMAN=1`.

- `MSSCAN=0` **bit-identical** to `f36c06a0` — the proof it is inert until switched on.
- `SELFSERVE=0` still byte-identical to `43beb082`.
- `MSSCAN=1` assembles with **zero** scratch/spill; `gate_lds.sh` still agrees with the published `.lds`.
- `gate_duty.sh` 10/10 (this change must not move `J*SEGK`), `gate_phist.sh` 6/6, `gate_sswin.sh` 6/6.
- The `:923` refusal is either gone with a stated justification, or still fires — not silently weakened.

**Bring-up is ONE dispatch then STOP** (rule 2). Expected `computed` = **50331648 exactly**; a short count
means work was dropped, and dropped work flatters TF.

## Falsification criterion (set in advance)

If `door4 GROW-FAIL` stays **0** after unpinning, then concurrency did not actually rise and the cap is
somewhere else — **report that and stop**; do not add machinery to force it. If `grow-fail` rises but span
does not improve, the coordination latency is not concurrency-hideable and that is a different (and more
interesting) finding — report it rather than tuning around it.
