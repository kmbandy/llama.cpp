# GUARD ABLATION — DESIGN + PREREGISTRATION (2026-08-08)

**Purpose:** settle `DSWS_BRIEF_2026-08-01_AM.md` §1c by the §3 falsifier. Question, verbatim:
*is the 20.8 ns fixed per-event cost the `lds_*` exec-guard idiom, or something else that merely
happens to be a similar fraction?* The falsifier: make the guard cheaper on a subset of `lds_*`
sites and re-fit `ns_per_event`. If the fixed term falls roughly in proportion to guard
instructions removed, the link is causal. If it does not move, that is a *result*.

**This document covers the OFFLINE phase only.** Build the arms, prove them byte-identical when
off, census the instruction deltas, RGA them. **NO GPU DISPATCH in this phase** — both cards are
claimed by the weight-pager session (board claim `6202d8a7`, R9700). The silicon re-fit is a
separate, later session under its own board claim.

---

## 0. Ground truth to build against

- Source: `occ_kernel_dsws_flow.s`, sha `57ab3100c9450ad6`.
- Config of record: bare `./build_flow.sh` → bin **`58e965a46f3e162d`**
  (WAVES=6 G=8 FM=2 ACC_N=4 FN=4 SEGK=256, LDS 34,304 B, `.text` 28,852, SGPR=72, 0 spills).
- The guard idiom (macro block `occ_kernel_dsws_flow.s:1633–1731`, plus `lds_cmpstore_adv`
  :1838, `lds_cas_rtn` :1861, `lds_put_r` :2040):
  ```
  s_mov_b32   s49, exec_lo          ; save
  v_cmp_eq_u32 vcc_lo, 0, v2       ; lane test        <- arm-1 target
  s_and_b32   exec_lo, exec_lo, vcc_lo                 <- arm-1 target
  s_cbranch_execz .Lskip\@         ; guard branch     <- arm-1 target (becomes dead)
  v_mov_b32   v[RP_A], \off        ; marshal addr     <- arm-3 target
  v_mov_b32   v[RP_D], \ssrc       ; marshal data     <- arm-3 target
  ds_store_b32 ...
  s_wait_dscnt 0x0
  .Lskip\@:
  s_mov_b32   exec_lo, s49          ; restore
  ```
- Static census at config of record (07-31 static matrix): `v_cmp_eq_u32` = 360,
  `s_cbranch_execz` = 360 (exactly equal → all from this family), 267 source sites,
  ~1,800 pure-bookkeeping instructions = 34% of the kernel.
- **`lds_get`/`lds_get_r` are NOT guarded** (all-lanes load + `v_readfirstlane`) — they are not
  arm-1 targets; they ARE arm-3 candidates (the `v_mov` address marshal).

## 1. The three arms — each its own defsym, each default-off, each byte-identical when off

Discipline (standing, §86a): every arm is ONE symbol. If a variant needs two conditions to agree,
derive both from the one symbol — never two independently-set symbols.

### Arm 1 — `LEANGUARD` (cheapest, least invasive)
Where lane-0 is *unconditional*, replace the 3-instruction mask dance
(`v_cmp_eq_u32` + `s_and_b32` + `s_cbranch_execz`) with `s_mov_b32 exec_lo, 1`.
The exec save/restore pair (`s49`) stays. Expected saving: 2–3 of the 5 bookkeeping
instructions per guarded block.

**Safety constraints — these are the whole job:**
1. **Semantic difference to audit per macro:** the current idiom *skips the access entirely*
   if lane 0 is inactive in the caller's exec; `s_mov exec_lo, 1` *forces* lane 0 active.
   These agree ONLY where callers are in wave-uniform control flow with lane 0 live.
   Audit call sites per macro; apply LEANGUARD **per-macro**, only to macros whose every call
   site is provably full-exec or lane-0-live. Document any excluded macro and why.
2. **The `SELFSERVE` variants of `lds_fetch_add`/`lds_fetch_add_r` are first-ACTIVE-lane**
   (`s_ff1_i32_b32` on the saved mask, :1662–1667, :1699–1704) — NOT unconditional lane 0.
   Config of record builds with SELFSERVE on ⇒ **exclude the SELFSERVE arms of the fetch_add
   family from LEANGUARD** unless the audit proves every caller has lane 0 active (do not
   assume; if excluded, say so in the report — the delta math must reflect it).
3. **Exec-restore must still pair** on every path, including the now-dead skip label
   (leave the label; dead labels are free) and the SELFSERVE `v_readfirstlane` placement
   (it reads *while the selected mask is installed* — arm 1 must not move it).
4. `v_readfirstlane_b32` semantics: reads the first *active* lane. With exec forced to 1 the
   active lane is 0 — same result. Confirm per use, don't assert globally.

### Arm 2 — `GUARDHOIST` (amortize save/restore across adjacent runs)
Many sites appear in back-to-back runs of `lds_*` calls. Hoist the `s_mov s49, exec_lo` /
restore pair (and under LEANGUARD, the single `s_mov exec_lo, 1`) around the *run* instead of
per call.

**Scope control:** first ENUMERATE — static scan of the source for adjacent `lds_put`/`lds_inc`/
`lds_fetch_add`/`_r` runs (adjacent = no intervening instruction that reads/writes exec or
branches). Report the run-length distribution and the top regions. Then implement ONLY the top
few regions (the poll/dispatch hot path first) via explicit `lds_run_begin`/`lds_run_end`
bracket macros + `_nog` (no-guard) inner variants. Do NOT attempt a global rewrite — call-site
surgery across 267 sites is not worth it before the causal question is answered.
Arm 2 composes with arm 1 (`GUARDHOIST` implies the lean guard inside the bracket); it must
still be independently buildable (`GUARDHOIST=1 LEANGUARD=0` allowed or explicitly rejected at
assembly time with a clear error — pick one, document it).

### Arm 3 — `LEANMARSH` (kill re-marshalling)
The `v_mov v[RP_A], \off` address marshal re-materializes per call. Where the same LDS address
(or a small set: the fixed control-word offsets) is used repeatedly, keep it resident in a
dedicated vreg (initialized once in the prologue) and have `ds_store`/`ds_add` take it
directly. Compile-time-constant `\off` sites only; runtime-`saddr` (`_r`) variants excluded.
This arm needs a free VGPR in the *lean* (feed) register budget — verify livereg impact via RGA
before claiming it free. If no vreg is free in every wave class, report that and stop this arm
(that is a finding, not a failure).

## 2. Verification gates — all offline, all mandatory per arm

1. **OFF ⇒ byte-identical:** with the arm's symbol at 0, bare `./build_flow.sh` reproduces
   `58e965a46f3e162d` exactly (sha the bin, not the .o). Any arm that cannot be made
   byte-identical off is wrong by construction.
2. **ON ⇒ assembles clean at config of record:** 0 spills, SGPR count reported, LDS unchanged
   (34,304), `.text` size reported.
3. **Instruction census via disassembly** (`/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1201`
   on the built `.o`): counts of `v_cmp_eq_u32`, `s_cbranch_execz`, `s_and_b32`, `v_mov_b32`,
   `s_mov_b32` per arm vs baseline. The arm-1 prediction to check: `v_cmp_eq_u32` falls by
   exactly the number of converted guarded blocks; `s_cbranch_execz` falls identically.
   **A mismatch between predicted and observed census is a stop-and-report, not a shrug.**
4. **RGA static pass** — the real RGA is `~/Downloads/rdts/…/rga` (v2.14.2.8);
   `/usr/bin/rga` is ripgrep-all. Use the harness's RGADESC flow so the analysis object is the
   shipped clang line. Take ONLY livereg/SGPR/spills/ISA-size from RGA (its LDS/VGPR "USED"
   figures are analysis-descriptor artifacts).
5. **No behavioral verification is claimed in this phase.** Correctness (full-stride oracle,
   work-exactness) happens on silicon, later, arm by arm. Nothing in this phase may be
   described as "verified correct" — only "assembles, byte-identical off, census as predicted".

## 3. Deliverables

1. Edits to `occ_kernel_dsws_flow.s` (guarded by the three symbols, default 0) and
   `build_flow.sh` plumbing (follow the existing `:=` env-override pattern).
2. `GUARD_ABLATION_REPORT_2026-08-08.md` in this directory:
   - per-arm: symbol, sites touched, sites EXCLUDED and why (the audit trail is the product),
   - the census table (baseline vs each arm, plus arm1+arm2 combined if built),
   - RGA numbers,
   - the byte-identity proof lines (shas),
   - **the pre-registered silicon prediction**: for each arm, guard instructions removed as %
     of the 1,800, and the proportional expected drop in the 20.8 ns fixed term IF causal
     (e.g. arm removes 40% of guard bookkeeping ⇒ fixed term predicted ≈ 20.8 − 0.40×20.8×f,
     with f = the fraction of the fixed term the idiom could plausibly own; state f=1 as the
     upper bound and let silicon discriminate). These numbers get written BEFORE any run.
3. NO log entry in `DSWS_TESTING_LOG.md` (that log is for measured runs; the silicon session
   appends there).

## 4. Tree + safety discipline (non-negotiable)

- Touch ONLY files under `spike/dvgpr_occ/`. The repo carries other sessions' uncommitted WIP
  (pipeline/wp-worker files) — never `git add -A`, never `git commit`, never `git checkout`/
  `restore`/`stash`. No commits at all in this phase; kmbandy decides when to commit.
- NO GPU work of any kind: no `gpu_run.sh`, no `occ_dispatch` execution, no rocprof, no
  hipBLASLt. Assembly + objdump + RGA are CPU-only and allowed.
- Do not touch `occ_kernel_dsws.s` (the pre-flow kernel), `occ_kernel_coop.s`, `fp8_oracle.*`
  (shared/not-ours), or any canonical `.bin`.
- `\@` label uniqueness: new macro variants must keep `\@`-suffixed labels so repeated
  expansion assembles.
- If an arm turns out to need a paired condition (à la `NCOMPUTE < 2` / `BATON_MAGIC`),
  derive both from the ONE symbol — §86a's coupling-hazard rule.

## 5. What this phase does NOT decide

- Whether any arm ships. They are ablation instruments.
- The SEGK per-segment cost (+27%/+53% above the per-event model) — separate investigation.
- The counter-free assign design (07-21) — complementary (event COUNT vs event COST); the
  silicon result here decides which lever is worth more.
