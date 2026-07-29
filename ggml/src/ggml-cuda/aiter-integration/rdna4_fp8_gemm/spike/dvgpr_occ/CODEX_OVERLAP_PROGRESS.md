# DSWS2_OVERLAP build pass — 2026-07-24 (Sonnet builder)

**Status: STOP AND REPORT.** The scaffolding (defsyms, budget confirmation, offline gates) is done and
green. The actual overlap mechanism from `DESIGN_OVERLAP_PREFETCH_2026-07-24.md` SS5 is **not implemented**
because reading live source found the design's trigger condition does not measure what SS2/SS5 assume, and
correctly fixing that forces the same class of complexity SS6 explicitly says to avoid. Details below.

## What changed (file:line)

- `occ_kernel_dsws_flow.s`, inserted after the `DSWS2_TICKET_SELFTEST` defsym (originally ~line 422,
  now ~line 424-441 after the shared tree's other concurrent edits): a `DSWS2_OVERLAP` (default 0) and
  `OVERLAP` (default 2) `.ifndef`/`.set` pair, with a comment block recording this STOP. **No `.if
  DSWS2_OVERLAP` code exists anywhere in the file** — these are inert scaffolding only, requested by the
  build task so a follow-on design pass has the knob names reserved. They emit zero bytes at every value.
- `build_flow.sh`, one line added to `mkflow()`'s defsym list: `-Wa,-defsym,DSWS2_OVERLAP=${DSWS2_OVERLAP:-0}
  -Wa,-defsym,OVERLAP=${OVERLAP:-2}`, plumbing the two new knobs through exactly like the existing
  `DSWS2_RCONV` knob. No other change to this file.
- `occ_dispatch.cpp`: **untouched by me.** (The tree is shared with a live session per the working-dir
  `CLAUDE.md`; `git diff` shows unrelated pre-existing uncommitted BNDSPLIT/ADVPROBE host-print additions
  from that other session, already present before I opened the file. My diff against that baseline is
  zero lines in this file — confirmed by grepping the diff for my own markers.)

No git add/commit/stash was performed; nothing was staged. No file outside this spike dir was touched.
No GPU dispatch was run at any point (no `./gpu_run.sh`, `./occ_dispatch`, no `systemd-run`'d dispatch).

## Offline gates

### Gate 1 — DSWS2_OVERLAP=0 byte-identical to baseline

```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
  flow build done. fail=0
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match to the canonical baseline sha in `HARNESS.md`. Expected: since no `.if
DSWS2_OVERLAP` code exists, this is byte-identity by construction, not by luck.

### Gate 2 — DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 assembles + links

```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32428B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
  flow build done. fail=0
sha256sum occ_dsws2_w30_flow_gd.bin
  53a309f76a9bbea792aebb18e5116c6f692afbff92af1f4d40f69e63eafd1e4f  occ_dsws2_w30_flow_gd.bin
```
This sha and `.text` size are **identical to the plain `DSWS2_RCONV=1` build documented in `HARNESS.md`**
(`53a309f76a9bbea7`) — i.e. `DSWS2_OVERLAP=1 OVERLAP=2` truly adds zero bytes, confirming the scaffolding
is inert exactly as intended. **This gate does not exercise any new mechanism** (there is none to exercise);
it only proves the plumbing doesn't break assembly.

Linked `.co` for RGA (mirroring build_flow.sh's `clang -c` step + `ld.lld -shared`, same full A1+RCONV+
OVERLAP defsym set plus `RGADESC=1`):
`rga_out/overlap_on/k.co` (45,672B), from `rga_out/overlap_on/k.o` (44,336B; `.text`-only extract is
32,492B, sha `0929a31c14fd182fbc925d6b20748e10034e02a45913ee3f11ddd2006ce60cbe` — the RGADESC descriptor
bytes account for the size delta vs. the 32,428B non-RGADESC `.bin` above).

RGA (`/home/kmbandy/Downloads/rdts/.../rga`) **was runnable** in this sandbox (not blocked) and ran clean:
```
DEVICE,SCRATCH_MEM,THREADS_PER_WORKGROUP,WAVEFRONT_SIZE,AVAILABLE_LDS_BYTES,USED_LDS_BYTES,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,0,N/A,32,65536,65536,106,72,0,256,256,0,...,30792
Maximum # VGPR used  50, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  54, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** (Expected — again, no new mechanism is present to spill on; this is
essentially re-confirming the already-proven `DSWS2_RCONV=1` occupancy picture.)

### Gate 3 — host `occ_dispatch.cpp` compiles

`./build.sh` ran to completion (`OK -> ./occ_dispatch [...]`), 23 pre-existing `-Wformat` warnings
(`%u` vs `uint64_t`, unrelated to this work, present before I touched anything), 0 errors. No
`systemd-run` failure in this sandbox; ran under the script's own `run_capped` wrapper unmodified.
Expected: I made zero changes to `occ_dispatch.cpp`, so this gate could only fail from something
already broken in the shared tree, which it isn't.

### Gate 4 — LDS budget confirmation (design SS7)

Traced at live source (not re-derived from the design doc's numbers):
- `BRES_BYTES = FN*16*SEGK = 4*16*256 = 16384`
- `ARES_BYTES = G*16*FM*SEGK = 6*16*1*256 = 24576`
- `OPSTRIDE = BRES_BYTES + ARES_BYTES = 40960` ; `POOL_N=1` so the whole operand pool is 40960B
- `ACC_BASE = OP_BASE(512) + POOL_N*OPSTRIDE = 41472` ; `ACC_STRIDE = FM*FN*1024 = 4096`
- `ACC_N*ACC_STRIDE = 3*4096 = 12288` -> `LDS_TOTAL_FLOW = 41472+12288 = 53760`
- `SELFSERVE && SSWIN(32) > POOL_N(1)` -> `+ SSWIN*SLOTC_STRIDE = 32*32 = 1024` -> **54784B total**,
  matching both the design doc's stated "current ~54,784B" and the `build_flow.sh` `.lds` readout above
  (`LDS=54784B`) exactly.
- If B-L2-only removed `BRES_BYTES` from the operand stride and a second `ACC_N*ACC_STRIDE` generation
  were added: `OPSTRIDE' = ARES_BYTES = 24576` -> `ACC_BASE' = 512+24576 = 25088` -> two gens
  `2*12288=24576` -> `+SSWIN*32=1024` -> **50688B total**, matching the design doc's "~50,688B" estimate
  exactly, comfortably under the 65536B WGP limit.

**The LDS arithmetic in the design doc is correct and would work.** The budget was never the blocker —
see below.

## Self-audit against design SS6 and the SS9 open questions

### SS9 Q1 — is B-staging (BRES) the only consumer of that 16KB, and is B-L2-only mechanically clean?

**Yes, mechanically.** Traced two consumers of the LDS B-staging region:
- The self-serve fast path (`.Lflow_da_ss_rowblk`, ~4690-4767) **already never reads BRES** — it
  self-loads B via `global_load_tr_b64` straight from L2 (line ~4755), exactly the pattern the design
  wants generalized.
- The ring/fallback compute path (`.Lflow_jloop`, ~3605-3651) **does** read staged B via `ds_load_b64`
  from `BRES_ROFF` (line 3632), fed by the `BSTAGE_R` macro (def'd ~1706, called at two feed call sites,
  ~4082 and ~5080). A-staging (`ARES`/`ASTAGE_R`) is architecturally separate and would be **kept**
  under the design (only B moves to L2-only).
- This part of the design (SS9 Q1) is cleanly separable and, on its own, does **not** touch the
  correctness crux below — removing `BSTAGE_R`'s two call sites and swapping the ring path's `ds_load_b64`
  (:3632) for a `global_load_tr_b64` mirroring the self-serve idiom is a mechanical, independently
  reviewable change. **I did not make this change** because, per the task framing, B-L2-only only earns
  its complexity as *funding* for the second accumulator generation (SS4) — with the accumulator-doubling
  blocked (below), there is no LDS pressure to relieve, and shipping a B-L2-only change with nothing
  consuming the freed space would be a change with no purpose behind it, not a partial win.

### SS9 Q2 / SS6 correctness crux — can TILEDONE/GSTORED/zero_banks be made per-generation without a second operand slot?

**TILEDONE and GSTORED, in isolation, are tractable.** `TILEDONE_BASE` is already a `GROUPS`-sized array
(line 740); making it 2-entry (per accumulator generation, not per absolute group index) and having
`zero_banks` reset only the *newly-opened* generation's entry (not all `GROUPS` at once, as it does today
at line 1238-1243) is a small, mechanical change. `GSTORED` (currently one monotonic per-WG counter,
line 819) could become 2 entries similarly, or could stay monotonic if the boundary math is re-derived
per generation. Neither of these is the blocker.

**The blocker is the frontier itself (`ASSIGN_HEAD`/`DRAIN_HEAD`/`DA_ZDONE`), which the design's own SS5
step 2 assumes can be advanced early without restructuring — and it cannot, for two independent,
source-verified reasons:**

**(a) The CFASSIGN cohort math is single-field, and early-advancing `DA_ZDONE` corrupts it for any wave
still inside the old field.** Under `CFASSIGN` (part of the A1 profile), `ASSIGN_HEAD` is kept identical
to `DA_ZDONE` (=`z`) at all times (set together at every boundary advance, e.g. lines 4406-4411,
4444-4449) — this matches the design doc's own description ("ASSIGN == z ... frozen there"). The peek's
cohort computation (lines 4179-4197) derives `field_start = z - field_width` **from the single scalar
`z`**, then computes `progress = DRAIN - field_start` to find which WAVES-wide cohort is "current." If
`DA_ZDONE` is advanced early to open field g+1 while `DRAIN` is still tracking work in field g, then any
wave that peeks recomputes `field_start` from the **new** `z`, and `progress = DRAIN - field_start` either
underflows (huge unsigned wraparound, if DRAIN is genuinely still behind the old field) or silently
reinterprets DRAIN's position against the wrong field's origin. There is no field-generation tag in this
arithmetic today; it assumes exactly one field is ever open. Making this safe for two live fields is not a
bank change — it is a second cohort-math generation, i.e. exactly the same species of dual-state
bookkeeping the design's own guard tries to keep out of the *operand* pool, just relocated to the
*frontier*.

**(b) Under `SELFSERVE=1` (the profile this design targets), `DRAIN_HEAD` is not a real-completion
signal, so the design's literal trigger `ASSIGN - DRAIN <= OVERLAP` does not measure what SS2/SS5 say it
measures.** Traced at `.Lflow_da_ss_decode` (~4638-4658): the self-serve fast path publishes a
**pre-completed sentinel** for its reservation (`SL_RBDONE = ACC_N` at line 4645-4646, `SL_GEN = r`
released last at 4657-4658) **before running the real compute burst** — the comment at 4638-4639 says so
explicitly ("ring feed/compute never owns the item, while TILEDONE/GSTORED remain the actual
compute/bank-reuse gates"). `drain_advance` is then called at `.Lflow_da_ss_stage_done` (line 4682),
**before** `.Lflow_da_ss_rowblk` (line 4690, the actual `v_wmma`/`ds_add_f32` burst) even starts, and it
succeeds immediately (its gate is exactly `SL_GEN==DRAIN_HEAD && SL_RBDONE>=ACC_N`, both already true from
the pre-completed stamp). So `DRAIN_HEAD` races to `ASSIGN_HEAD` on **reservation bookkeeping alone**,
independent of real accumulation progress. The **real** completion signal is `GSTORED`, which only
increments after a real completer's C-store `s_wait_storecnt` drains (line 3971-3972), gated by
`TILEDONE[group]` reaching its target — and `TILEDONE` is bumped only after each rowblk's *actual*
`ds_add_f32` lands (self-serve: line 4828-4829, after the real burst; ring: line 3783-3793, same pattern).
The existing boundary handler already reflects this: it checks `DRAIN>=ASSIGN` first (line 4340-4343)
*and then, separately*, `GSTORED >= z>>shift` (line 4350-4353) — the comment at 819-823 says outright that
"`DRAIN==ASSIGN` alone does NOT exclude" the real race window, which is exactly what I re-derived from the
self-serve stamp order. Given DRAIN catches ASSIGN almost immediately after a cohort is claimed, **the
design's own measured "90% dead gap" (SS1) is waves spinning on the `GSTORED` gate, not the `DRAIN` gate**
— meaning `ASSIGN - DRAIN` is already ~0 for essentially the *entire* dead gap, not just its tail. Gating
early-advance on it would not fire "when nearly drained" as SS2/SS5 intend; it would fire immediately upon
cohort-claim, indistinguishable from "always overlap."

**Why (a)+(b) together are a hard stop, not just an inconvenience:** a `GSTORED`/`TILEDONE`-relative
trigger (checking real per-rowblk progress instead of `DRAIN`) is a valid *fix* for (b) in isolation. But
using it to safely bound the overlap at **exactly 1-deep** (SS6's hard requirement) requires **splitting**
today's single atomic boundary action — currently one thing, done by one ZLOCK-elected wave: check
drain-gate, check C-store-gate, `zero_banks`, rebase, advance — into **two independent, separately
synchronized actions**: (1) *open the next field* (early, gated on the new real-progress-relative trigger,
zeroing the *other* generation's banks) and (2) *retire the current field / free its generation* (still
gated on real `GSTORED` completion, but now must NOT re-trigger a second early-open — it only marks "my
generation is free for two fields from now"). That is a genuine state-machine redesign of the frontier,
not an assembly-level realization of settled prose — and it reintroduces, by construction, the exact
correctness shape SS6 names as the load-bearing risk (two generations' worth of live completer/zero/C-store
state that must never cross-contaminate, the same species as the POOL_N>1 `bad=96/116` completer race),
now spread across the cohort/frontier layer instead of the operand layer. Per the task's explicit
instruction — "if it forces pool-like slot indexing on operands, STOP" — I read this as the same
prohibition one layer over: it forces pool-like *dual-generation* indexing on the **frontier's own
cohort/drain math**, which the design's SS9 Q2 asks me to rule out for operands but which turns out to be
required for the frontier regardless of what happens to operands.

### SS9 Q3 — is the trigger readable without clobbering live handler registers?

Yes, mechanically — `s51`(z)/`s45`(DRAIN) are already live at the peek (line 4171/4176) and `s44`(ASSIGN)/
`s46`(DRAIN) at the boundary (line 4340-4341), so a comparison against `OVERLAP` costs one more `s_sub`/
`s_cmp` with free scratch at both sites. **This was never the hard part** — the hard part is that the
value being compared doesn't mean what the design's prose assumes it means (Q2/SS6 above), and the fix
requires restructuring, not just reading it in a new place.

### Ping-pong index correctness, B-L2-only/feeder coherence

Not evaluated in depth beyond SS9 Q1's mechanical check above, since the accumulator-doubling and
frontier-doubling this would ride on are themselves blocked. No point auditing ping-pong arithmetic for a
mechanism I'm not building.

## What I recommend

The design's *funding* argument (SS4, LDS budget) and its *goal* (close the measured dead gap) both hold up
under source review — I found nothing wrong there, and confirmed the numbers exactly. The part that needs
a second design pass is SS5 step 2 (the trigger + the single-scalar-`z` frontier it rides on) and SS6's
implicit assumption that only the accumulator needs a second generation. A follow-up design should either:
(i) replace the `ASSIGN-DRAIN<=OVERLAP` trigger with a `TILEDONE`/`GSTORED`-relative one and explicitly
design the split "open-next-field" / "retire-this-field" two-action frontier (with its own 1-deep-only
proof, analogous to SS6's), or (ii) find a formulation where the *existing* single-`z` boundary handler
can still fire exactly once per field (no early trigger) but the **handler itself** is made cheaper/faster
so the dead gap shrinks without overlap (attacking the `GSTORED` wait directly instead of prefetching
around it) — the ADVPROBE data already collected (SS1: "~264 ticks of ~2600, ~90% dead gap") would be the
right starting instrument for either path. Either is a kmbandy/claude__main design decision, not something
I should improvise into a concurrency-critical hand-tuned kernel I cannot dispatch to verify.
