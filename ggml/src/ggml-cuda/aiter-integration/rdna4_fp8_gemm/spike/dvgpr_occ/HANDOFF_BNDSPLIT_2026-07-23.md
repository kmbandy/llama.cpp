# HANDOFF — DSWS boundary sub-gate counters (split SS_WAIT: herd vs handler-latency)

**Author:** claude__main, 2026-07-23. **Builder:** Codex gpt-5.6-terra. **Kernel:** `occ_kernel_dsws_flow.s`.
**Config:** A1 canonical (`STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1`).

> **No code in this handoff.** Mechanism, exact file:line insertion points, invariants, gates. You
> write the assembly. If live source contradicts anything, **STOP and report** — do not improvise.

## 0. Why (measured 2026-07-23)

Runtime role conversion (`DSWS2_RCONV=1`) collapsed RING_WAIT 56%→0.3%, but the wall moved to
SS_WAIT (self-serve reservation) 50.7%. A PHIST run localized it precisely: **81.9% of every wave's
loop passes enter the tile-group boundary handler** (`.Lflow_da_boundary`), only **2.7% win a
reservation** (36:1 try:win churn), and it is **NOT the window** (`gatefull=0`, POOL_N/SSWIN is not
the bind) and **NOT the terminal drain** (`drainwait≈0`). So the stall is inside the boundary
interlock — but we **cannot yet split WHY** waves leave the boundary, because the relevant PHIST
doors (`occ[107] zlock`, `occ[109] bnd-lost`, etc.) have **no bump sites**; they all funnel into
`feedmt`. This handoff wires four cheap counters that split the boundary interlock into its causes.

## 1. What this builds — a new probe flag, four counters

Add a defsym — suggested **`BNDSPLIT`**, default 0. When `BNDSPLIT=1`, four counters record what
happens to a wave that enters `.Lflow_da_boundary`. When `BNDSPLIT=0`, **byte-identical to `cac3ff7c`**.

The four are mutually exclusive outcomes of one boundary entry — they should sum (modulo throttling)
to `occ[110] boundary` (the existing entry count):

1. **`ZLOCK_LOST` (the herd):** the wave lost the ZLOCK election CAS — another wave is already
   handling the boundary. Insertion: the taken branch of the election test right after the
   `lds_cas_rtn DA_ZDONE_OFF` — currently `occ_kernel_dsws_flow.s:4241–4243`
   (`s_cmp_eq_u32 s47,s51 ; s_cbranch_scc0 .Lflow_feedmt_sleep`). Count on the **scc0-taken** path
   (CAS failed = lost).
2. **`DRAINGATE_BAIL` (fields are serial):** the wave WON ZLOCK but bailed because the current field
   has not fully drained (`DRAIN < ASSIGN`). Insertion: the taken branch at `:4249–4250`
   (`s_cmp_lt_u32 s46,s44 ; s_cbranch_scc1 .Lflow_da_bnd_bail`). Count on the **scc1-taken** path.
3. **`CSTOREGATE_BAIL` (C-store latency on the critical path):** the wave WON ZLOCK, the field
   drained, but the finishing group's C-store has not drained (`GSTORED < z>>shift`). Insertion: the
   taken branch at `:4258–4259` (`s_cmp_lt_u32 s47,s46 ; s_cbranch_scc1 .Lflow_da_bnd_bail`). Count on
   the **scc1-taken** path. **Counters #2 and #3 must be separate** — they both jump to
   `.Lflow_da_bnd_bail` but they are different fixes, so bump BEFORE the shared bail label at the two
   distinct compare sites.
4. **`BOUNDARY_WIN` (real progress):** the wave WON ZLOCK, passed both gates, and actually advanced
   the boundary (zeroed the next group's banks / re-based, bumped `DA_ZDONE`, cleared ZLOCK).
   Insertion: once on the advance path past both gates — the group-advance at `:4303+` and/or the
   tile-advance `.Lflow_da_bnd_tile` (`:4302`). If group and tile advance are separate blocks, bump
   the same counter in each (or split into `BOUNDARY_WIN_GROUP`/`_TILE` if trivial — your call, note it).

## 2. Mechanism — this path is ACC-DEAD, so a direct lane-0 atomic is safe

The boundary handler is reached from the reservation peek on the **lean/coast path** — the wave has
no live WMMA accumulator (it never grew fat here). So the 2026-07-13 instrumentation invariant
(no memory/VGPR/exec writes while ACC is live) **does not bind here**: a direct lane-0
`global_atomic_add_u32` to a free `occ[]` slot is safe, exactly as the existing DECENTASN claim-diag
counters do (`occ[95]/occ[96]/occ[97]`, look at those as the pattern). **Confirm ACC is dead at each
of the four sites before emitting** (it should be — the wave is lean on the whole DA-peek/boundary
path); if any site is somehow ACC-live, STOP and report.

- Use the **DECENTASN-diag pattern** (`occ[95–97]`): guard `exec` to lane 0, `v_mov_b32 v,1`,
  `global_atomic_add_u32 ... offset:<SLOT>`, restore exec. Do NOT use the register-accumulate +
  cnt_flush path unless you have a reason — direct atomic is fine on this cold, rare-ish path (it
  fires only on a boundary entry, ~82% of *lean* passes, never inside a compute burst).
- Pick **four free `occ[]` slots** and print them in a host block (mirror the DECENTASN-diag or PHIST
  print in `occ_dispatch.cpp`). State which slots you chose and confirm they are not written elsewhere.
  Do NOT reuse the PHIST slots `occ[104–115]` (they must stay PHIST's).
- Counts are RAW (unthrottled) like the DECENTASN diag — the atomics are cheap and this must be
  quantitatively trustworthy (we do NOT want another PHIST-style 220% contaminated read).

## 3. Flag + byte-identity contract

- Defsym `BNDSPLIT`, default 0. All new code inside `.if BNDSPLIT`.
- **`BNDSPLIT=0` MUST assemble byte-identical to `cac3ff7c2338e73f`** at the A1 canonical profile —
  re-check the sha after every edit (canonical build line in `HARNESS.md`). Any divergence = an edit
  leaked outside `.if BNDSPLIT`.
- `BNDSPLIT` composes with `DSWS2_RCONV=1` (the diagnostic run uses both). It does NOT require RCONV.
  Add no guard coupling them.
- No new SGPR pressure concern if you use direct atomics (v-regs + exec save, no persistent SGPR).
  If you do need an SGPR, the RCONV coast counter took `s50`; find a genuinely dead one and say which.

## 4. Safety (this kernel has bricked the box)

- **Rule 5:** the atomic fires on a boundary ENTRY (lean path, rare relative to the compute burst),
  never inside a WMMA/ACC-live region, never a per-iteration hot-loop store beyond what `occ[95–97]`
  already do. One atomic per counted event. No `s_sendmsg_rtn`.
- **No `s_alloc_vgpr`, no barrier** on this path.
- **You do NOT dispatch to the GPU.** Offline gates only. Never touch `occ_kernel_coop.s`. Only the
  spike dir; the tree is shared with a live weight-pager session — stage nothing, flag before `git diff`.

## 5. Offline gates (all green before you report done)

- `BNDSPLIT=0` sha == `cac3ff7c2338e73f` at the A1 canonical profile (after every edit).
- `BNDSPLIT=1 DSWS2_RCONV=1 CFASSIGN=1` assembles + links **0-spill** (RGA). Use the RDTS RGA at
  `/home/kmbandy/Downloads/rdts/.../rga` (the `/usr/bin/rga` name-clash is ripgrep-all). If `~/.rga`
  is read-only in your sandbox, report the spill gate as BLOCKED (claude__main will run it) — do NOT
  claim 0-spill you didn't measure.
- Record in `CODEX_BNDSPLIT_PROGRESS.md`: the four occ slots chosen, confirmation ACC is dead at each
  of the four sites, the exact live line numbers (they drift — RCONV added ~lines), and the sha result.

## 6. STOP-AND-REPORT if

- Any of the four insertion points does not match live source at the cited line (re-verify — RCONV
  edits shifted line numbers; the boundary handler was at `:4237–4306` in the RCONV working tree).
- Any site is ACC-live (would make a direct atomic unsafe).
- The byte-identity check fails and the cause isn't an obvious `.if BNDSPLIT` leak.
- `BNDSPLIT=1` spills.

**GPU is claude__main's, not yours.** After your offline gates are green, claude__main builds
`BNDSPLIT=1 DSWS2_RCONV=1` and runs one clean dispatch (cheap counters, so full 512 chunk, reps=4,
steady state) to read the four-way split — herd (`ZLOCK_LOST`) vs handler latency
(`DRAINGATE_BAIL` / `CSTOREGATE_BAIL`) vs progress (`BOUNDARY_WIN`) — which decides the fix.
