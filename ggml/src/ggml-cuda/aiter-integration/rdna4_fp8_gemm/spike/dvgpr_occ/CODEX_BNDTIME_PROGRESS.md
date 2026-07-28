# DSWS2_BNDTIME — boundary LOSING-pass timer (2026-07-24)

**OFFLINE ONLY.** No GPU dispatch, no `./gpu_run.sh`, no `./occ_dispatch`. Every artifact below
came from assemble / link / objcopy / RGA / host-compile. Nothing was staged or committed
(`git diff --cached` empty). Files touched: `occ_kernel_dsws_flow.s`, `occ_dispatch.cpp`,
`build_flow.sh`. `occ_kernel_coop.s` shows **no diff**.

## What it measures and why

ADVPROBE (`occ[131..132]`) times the WIN path only: ZLOCK election win → `DA_ZDONE` store, ~264
ticks. BNDSPLIT (`occ[127..130]`) counts boundary outcomes but prices none of them. The 2026-07-24
PHIST census put **78.9% of all loop passes** inside `.Lflow_da_boundary`, and BNDSPLIT put
**93.1% of those on the losing side** of the election — so ~73% of every loop pass in this kernel
is a wave entering the boundary and leaving with nothing. **That pass has never been priced.**
BNDTIME prices it: `occ[133]` = summed ticks in non-advancing passes, `occ[134]` = count of them,
host prints ticks/lost-visit.

## Where the code is

| what | file:line |
|---|---|
| defsym `DSWS2_BNDTIME`, default 0 | `occ_kernel_dsws_flow.s:150` |
| `.error` guard: requires `DEADMAN=1` | `occ_kernel_dsws_flow.s:1193` |
| `.error` guard: refuses `PHASEPROBE=1` (SGPR conflict) | `occ_kernel_dsws_flow.s:1196` |
| `BNDT_LOSTTICKS_OFF` = 532 → `occ[133]` | `occ_kernel_dsws_flow.s:2609` |
| `BNDT_LOSTCOUNT_OFF` = 536 → `occ[134]` | `occ_kernel_dsws_flow.s:2610` |
| `bndtime_start` macro | `occ_kernel_dsws_flow.s:2941` |
| `bndtime_end_lost` macro | `occ_kernel_dsws_flow.s:2950` |
| **START** at boundary entry | `occ_kernel_dsws_flow.s:4931` |
| LOST 1/5 redirect (election CAS) | `occ_kernel_dsws_flow.s:4957` → stub `:5090`, end `:5091` |
| LOST 2/5 + 3/5 (drain gate, C-store gate) | `occ_kernel_dsws_flow.s:5097` (in `.Lflow_da_bnd_bail:5094`) |
| LOST 4/5 (terminal) | `occ_kernel_dsws_flow.s:5114` (in `.Lflow_da_bnd_term:5104`) |
| LOST 5/5 (`DSWS2_FUNNEL` not-ready) | `occ_kernel_dsws_flow.s:5681` |
| host print, SUCCESS path | `occ_dispatch.cpp:2577` (block), `:2590` (printf) |
| defsym passthrough | `build_flow.sh:17` |

Slot audit: the highest previously named occ offset in the whole file is 528 (`ADVP_COUNT_OFF`,
`occ[132]`). 532/536 are the first free words, have no pre-existing read or write anywhere, and sit
**above** the host's `0x100` per-chunk `memset` (`occ_dispatch.cpp:2112`) so they accumulate across
chunks and reps, exactly like ADVPROBE's. The occ buffer is `0x1000` B = 1024 words.

## COMPLETE enumeration of boundary-handler exit paths

Every control-flow edge that leaves `.Lflow_da_boundary` (`:4930`), taken from the actual branch
listing of the handler body, not from memory:

**Non-advancing exits — ALL FIVE instrumented:**

1. **Lost the ZLOCK election CAS** (`:4957`). Was `s_cbranch_scc0 .Lflow_feedmt_sleep`; under
   BNDTIME it branches to a new out-of-line stub `.Lflow_da_bnd_lost_zcas` (`:5090`) that times and
   then branches to the identical target. This is the ~93% case.
2. **Drain-gate bail** (`DRAIN < ASSIGN`) → `.Lflow_da_bnd_bail`.
3. **C-store/GSTORED-gate bail** (`GSTORED < z>>shift`) → `.Lflow_da_bnd_bail`.
   2 and 3 share one end site at `:5097`.
4. **Terminal** (`t_new >= chunkHi`) → `.Lflow_da_bnd_term`, end site `:5114`.
5. **`DSWS2_FUNNEL` not-ready, spin budget exhausted** → `.Lflow_da_funnel_notready`, end site
   `:5681`. `DSWS2_FUNNEL=0` in both profiles of record, so this arm is dead there — but it is
   instrumented anyway so that turning the funnel on cannot silently create an unpaired start, and
   it was **built and disassembled** (see EXTRA A) rather than shipped as untested text.

**Advancing exits — deliberately NOT timed as lost (already timed by `advprobe_end`):**

6. GROUP advance → `s_branch .Lflow_da_peek`.
7. TILE advance → `s_branch .Lflow_da_peek`.

Internal-only branches inside the handler (not exits): the five BNDPROBE divergence-detector
branches, the `s_cbranch_scc1 .Lflow_da_bnd_tile` dispatch, and the `s_cbranch_execz` around the
tile-claim atomic. There is no fall-through out of the handler: `:5090`'s stub, `.Lflow_da_bnd_bail`
and `.Lflow_da_bnd_term` all end in unconditional branches, and the block preceding the stub ends in
`s_branch .Lflow_da_peek`.

## Pairing argument (exactness of start/end)

**Claim: under the throttle, every sampled START has exactly one sampled END, and every END has a
START. No path is double-counted or missed.**

1. **Domination.** All five END sites are reachable *only* from inside the handler.
   `.Lflow_da_bnd_bail` has exactly two predecessors, both the gate branches at `:4969`/`:4979`.
   `.Lflow_da_bnd_term` has exactly one, `:5064`. `.Lflow_da_funnel_notready` has exactly two, both
   inside the handler. `.Lflow_da_bnd_lost_zcas` is new text with exactly one predecessor, `:4957`.
   The START at `:4931` is the first instruction after the label, so it dominates all of them.
   ⇒ **no END without a START.**
2. **`s71` is invariant across a pass.** The only writers of `s71` are `deadman_stamp` (prologue)
   and `deadman_check`/`deadman_check_fat` at `:3433`, `:3447` (the `.Lflow_loop` head), `:4102`
   and `:4114` (the FAT J-carrier spin, unreachable from this LEAN handler, which holds no slot),
   and `:5663` (`.Lflow_da_drain`, reached only *after* the terminal END site has already fired).
   **None lies between the entry latch and any exit site.** So the START gate and the END gate read
   the same `s71` value: sampled start ⇒ sampled end, unsampled start ⇒ unsampled end. Verified in
   the disassembly: every one of the three emitted END sites is immediately preceded by
   `s_cmp_eq_u32 s71, 0` / `s_cbranch_scc0` (3/3 at both profiles).
3. **No double counting.** Each pass executes at most one END: sites 1, 4, 5 each end in an
   unconditional branch out of the handler, and 2/3 are the *same* site reached by mutually
   exclusive branches. A pass that reaches an advance takes edge 6 or 7 and passes through **no**
   `bndtime_end_lost` at all, so a winning pass can never be counted as lost.
4. **A START with no END is harmless.** After a win the handler branches to `.Lflow_da_peek`, which
   may re-enter `.Lflow_da_boundary` and simply **re-latch** `s[78:79]`. The latch is a register
   write that nothing consumes unless an END fires; a stale latch is overwritten, never accumulated.
   Only an *unpaired END* could corrupt the average, and (1)+(2) exclude that.
5. **Terminal is inside the count, on purpose.** It is a non-advancing exit, so the "every
   non-advancing exit" rule covers it, and including it makes pairing **total** rather than
   almost-total. Magnitude bound: at most one terminal per WG for the whole run (≤~64 events),
   then 1/64-sampled ⇒ ~1 sample against ~10^6. It cannot move ticks/lost-visit. The host print
   states this explicitly so `occ[134]` is never misread as "bails only".

## SGPR liveness audit

Registers used by BNDTIME: **`s[78:79]`** (entry RTC, must survive the whole pass), **`s[80:81]`**
(exit RTC), **`s82`** (delta), **`s83`** (exec save). Both `s_sendmsg_rtn_b64` destinations are
even-aligned.

- All six are **PHASEPROBE-only**. `phase_reset` zeroes `s78..s83`, `phase_stamp` accumulates into
  them, `phase_flush` emits them; `s77` is PHASEPROBE's last-stamp. A grep of every occurrence of
  `s74`–`s83` in the file shows **no other writer or reader** of `s78..s83`. `PHASEPROBE=1` is
  refused at build time (`:1196`) — this is the run #8 lesson (`:2670`) applied: an SGPR reused
  through a `.set` alias corrupted live state, not just a counter, so the audit was done against
  the `.set` table, not the register spelling.
- **Untouched, handler-live:** `s44`, `s45`, `s46`, `s47` (ASSIGN / scratch / TOTAL / CAS result),
  `s51` (clean z), `s52`, `s53` (z−base, t_new), `s66` (COUNT), `s67` (mask), `s68` (shift).
  Also `s15`, `s16`, `s24`, `s69` (chunkHi).
- **Untouched, reserved elsewhere:** `s49` (the `lds_*` macros' exec save — deliberately *not*
  reused, since `lds_put`/`lds_cas_rtn` run between the latch and the ends), `s50` (RCONV coast
  counter), `s54`–`s57` (`flow_snapshot`/`phist_bump`/`flow_gauge` scratch), `s58`/`s59`/`s62`/`s63`/
  `s64` (ADVPROBE — disjoint, so BNDTIME and ADVPROBE compose; proven by EXTRA B), `s70`/`s71`
  (deadman), `s72`/`s73` (TRACE / `BATCH>1`), `s75` (**live in the CF0 build** — DSWS2_ROLEFLOW
  hysteresis), `s76` (KMAJOR magic_TOTAL), `s77` (PHASEPROBE stamp), `s84`–`s105` (`cnt_inc`
  counters and the `FATHELD`/`DM_PROG` flags).
- SGPR range: `RSRC1.SGPRS = 0` ⇒ all 106 SGPRs allocated, and `cnt_inc` already uses `s84..s105`,
  so `s78..s83` are comfortably in range. RGA reports **0 SGPR spills** at both profiles.
- **VGPR / ACC:** the sites write `v3` (scratch, as `phist_bump`/`bnd_bump`/`advprobe_end` already
  do at these same sites) and read `v4` (the occ base address) and `v2` (lane id). ACC is **dead on
  the whole lean boundary path** — the handler runs on a wave holding no slot and no accumulator;
  the pre-existing BNDPROBE/BNDSPLIT/ADVPROBE sites in this same handler are documented ACC-DEAD for
  exactly this reason. RGA livereg is **unchanged** by the probe: max VGPR 50 both with and without
  it at A1.
- **SCC:** `bndtime_start` clobbers SCC at `.Lflow_da_boundary`, where SCC is already dead (all
  three entries arrive via a taken branch, and the `s_or_b32 s45, s51, ZLOCK` below recomputes it —
  the same argument `phist_bump PH_BOUNDARY` already relies on at that label). Every
  `bndtime_end_lost` site is immediately followed by an unconditional branch.

## Anti-brick argument (CLAUDE.md rule 5)

- **Every** new `s_sendmsg_rtn` and **every** new atomic is behind `s_cmp_eq_u32 s71, 0` /
  `s_cbranch_scc0`, the deadman's 1-in-64 boundary — verified in the disassembly, not just the
  source (3/3 END sites gated at both profiles; the single START site likewise).
- `DEADMAN=0` is refused at assembly time.
- **Rate bound:** the probe adds 2 RTC reads per *sampled* boundary pass. Boundary entries are 78.9%
  of loop passes and `deadman_check` itself already performs 1 sampled RTC read per loop pass at the
  same 1/64 gate ⇒ the added message-bus traffic is ≈1.6× the traffic the kernel already generates
  and has run with. This is the same order as the accepted baseline, not an unthrottled storm
  (DUTYPROBE's failure mode).
- **Lock hygiene:** the drain/C-store END sits **after** `lds_put DA_ZDONE_OFF, s51` releases ZLOCK,
  so two RTC reads and two device atomics never run inside the boundary critical section and cannot
  lengthen it for other waves. The election-lost END holds no lock (its CAS failed). The terminal
  END holds ZLOCK by TERMFIX design, once per WG.

## Gates — verbatim output

### GATE 1 — `DSWS2_BNDTIME=0` byte-identical to `cac3ff7c...` at A1: **PASS**

```
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  [POOL_N=1 SSWIN=32 PHASEPROBE=0] LDS=54784B
flow build done. fail=0
cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```

### GATE 2 — ON build assembles + links 0-spill (RGA) at BOTH profiles: **PASS**

```
### GATE 2 (ON, both profiles, RGA 0-spill) ###
assemble+link OK
  a1   SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 ISA_SIZE=30916
       starts=1 ends=3 atomics532=3 atomics536=3 gated_ends=3
assemble+link OK
  cf0  SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 ISA_SIZE=27520
       starts=1 ends=3 atomics532=3 atomics536=3 gated_ends=3
```

RGA ran for real here (the previous ADVPROBE session recorded this gate as
BLOCKED-by-readonly-`.rga`; `~/.rga` is writable in this session). Full stats rows, with the
`DSWS2_BNDTIME=0` A1 control for comparison:

```
a1_off    SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 USED_VGPR=256 ISA=30692
a1        SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 USED_VGPR=256 ISA=30916
cf0       SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 USED_VGPR=256 ISA=27520
funnel    SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 USED_VGPR=256 ISA=31124
all       SCRATCH=0 SGPR_SPILLS=0 VGPR_SPILLS=0 USED_VGPR=256 ISA=31704
--- a1_off livereg: Maximum # VGPR used  50, VGPRs allocated by HW:  96 (74 requested)
--- a1     livereg: Maximum # VGPR used  50, VGPRs allocated by HW:  96 (74 requested)
--- cf0    livereg: Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
```

Emitted A1 END site, disassembled (RGA ISA) — throttle, RTC, modular subtract, lane-0 mask, the two
atomics at 532/536, storecnt drain, exec restore:

```
_L391:
	s_cmp_eq_u32 s71, 0
	s_cbranch_scc0 _L481
	s_sendmsg_rtn_b64 s[80:81], sendmsg(MSG_RTN_GET_REALTIME)
	s_wait_kmcnt 0x0
	s_sub_co_u32 s82, s80, s78
	s_mov_b32 s83, exec_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz _L482
	v_mov_b32_e32 v3, s82
	global_atomic_add_u32 v4, v3, s[0:1] offset:532 scope:SCOPE_DEV
	v_mov_b32_e32 v3, 1
	global_atomic_add_u32 v4, v3, s[0:1] offset:536 scope:SCOPE_DEV
	s_wait_storecnt 0x0
_L482:
	s_mov_b32 exec_lo, s83
_L481:
	s_branch _L329
```

**EXTRA A — `DSWS2_FUNNEL=1 DSWS2_BNDTIME=1`** (proves the 5th arm is real code, not untested text):
`starts=1 ends=4 atomics532=4 gated_ends=4`, 0 spill. Exactly one extra END appears when the funnel
is enabled.
**EXTRA B — `DSWS2_BNDTIME=1 DSWS2_ADVPROBE=1 BNDSPLIT=1 BNDPROBE=1` together**: assembles, links,
0 spill — the register sets are disjoint and the probes compose.

### GATE 3 — host compiles; print on the SUCCESS path: **PASS**

```
HOST_COMPILE_EXIT=0  warnings=23  bytes=636184
```

23 warnings, all pre-existing and at unrelated lines; none in the BNDTIME block. Compiled to a
scratch path so the checked-in `./occ_dispatch` binary was not overwritten.

**Success-path proof:** the BNDTIME block is `occ_dispatch.cpp:2577`. The last `return res;` before
it is at `:2268`, and it lives inside the failure branch `if (!allok) {` at `:2266`. Everything from
`:2270` on — including the existing ADVPROBE print at `:2568`, immediately above mine — is
unconditionally on the success path. This file has wired a diagnostic into the failure branch twice;
that is why the check is stated as a line-number claim you can verify rather than an assertion.

### GATE 4 — `DEADMAN=0` refuses to build: **PASS**

```
== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 VBUDGET=1536) ==
  FAIL occ_dsws2_w30_flow_gd
occ_kernel_dsws_flow.s:1194:3: error: DSWS2_BNDTIME requires DEADMAN=1: both realtime reads and both atomics throttle on s71==0. The START probe sits at .Lflow_da_boundary, which the 2026-07-24 PHIST census measured at 78.9% of ALL loop passes -- unthrottled that is an s_sendmsg_rtn storm off idle coast waves, i.e. CLAUDE.md rule 5 verbatim (DUTYPROBE bricked the box exactly this way).
flow build done. fail=1
BNDTIME_DEADMAN0_GUARD_EXIT=1
```

Bonus guard, `PHASEPROBE=1` (SGPR conflict) also refuses:

```
occ_kernel_dsws_flow.s:1197:3: error: DSWS2_BNDTIME uses s78:s79 / s80:s81 / s82 / s83, which ARE the PHASEPROBE per-phase accumulators (phase_reset zeroes s78..s83, phase_flush emits them). Build PHASEPROBE=0. (See the run #8 post-mortem at :2670: an SGPR reused through a .set alias corrupted LIVE state, not just a counter.)
flow build done. fail=1
BNDTIME_PHASEPROBE_GUARD_EXIT=1
```

### Final on-disk state (baseline restored as the LAST build)

```
cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
065da39a8cbb9b4252c4f2c81b580076616c72cc927933c545ed529e1e700917  occ_dsws2_w16_flow_gd.bin
```

## STOP / caveats for the operator

1. **I overwrote and then reconstructed `occ_dsws2_w16_flow_gd.bin`.** It pre-existed at 32388 B
   from an earlier session. Bisecting defsyms identified it as the CF0 profile with
   `DSWS2_ADVPROBE=1`; rebuilt at `DSWS2_BNDTIME=0` it is again exactly 32388 B. Because
   `DSWS2_BNDTIME=0` emits zero bytes, that `.text` is byte-identical to what the pre-change source
   produced. The sibling `.o` differs in size (44024 vs 43896) — that is debug/symbol sections
   tracking the grown source file; only the objcopy'd `.text` `.bin` is dispatched. **If you
   intended a different w16 profile, rebuild before dispatching.**
2. **Never run, never measured.** Every number above is static. This kernel has not executed.
   Per CLAUDE.md rule 2 the CF0/WAVES=16 BNDTIME build is a *bring-up*: one dispatch, then stop.
3. **Probe perturbation is unknown and probably not small.** BNDTIME fires on ~100% of *sampled*
   boundary entries whereas ADVPROBE fires only on sampled wins (~7% of entries), so expect a
   larger slowdown than ADVPROBE's. **Do not quote TF from a BNDTIME build** (CLAUDE.md measurement
   rules). Ratios and ticks/lost-visit are what this build is for.
4. **Read `occ[134]` as "non-advancing passes", not "bails".** Terminals are included (bounded at
   ≤1 per WG). The host print says so.
5. **The `DSWS2_FUNNEL` arm has never been exercised at runtime** — it is off in both profiles. It
   assembles and disassembles correctly (EXTRA A), but if you enable the funnel, that is new
   behaviour independent of this probe.
6. Board shows the R9700/6900XT claimed by another agent for ~1.5h of P0 validation. Irrelevant to
   this task (offline only), but relevant to whoever schedules the bring-up run.
