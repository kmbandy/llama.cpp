> **SUPERSEDED (2026-06-18).** The "issue-port ledger / throughput = WMMA fraction (48%)" model below was
> OVERTURNED by the feed-only proof (FED == FEEDONLY => WMMA is free, the wall is the per-wave FEED, not the
> issued-instruction mix). The 48% coincidence was just that. Current model + winner: see RESULT_WGGEMM.md
> "BREAKTHROUGH: 8x2 reuse tile = 162 TF" and KG a8ea0196 / 4ef267cd. The reuse ratio splits into B-feed/MAC
> (=1/FM, binding) + A-feed/MAC (=1/FN, non-binding); grow FM. Kept below for historical method record only.

# Instruction ledger — KWIN=4 pw4 steady-state window (gfx1201 fp8 WMMA cooperative GEMM)

**Date:** 2026-06-18. **Kernel:** `occ_kernel_wggemm2.s`, defsyms `STORE=0 KWIN=4 KWINPW=4` (the ~145–150 TF
baseline). **Method:** assembled with `clang -target amdgcn-amd-amdhsa -mcpu=gfx1201`, disassembled with
`llvm-objdump -d`, opcode histogram of the steady-state `.Lkt_loop` body (disasm addr `0x36c`..backedge).

The whole kernel emits **exactly 128 WMMA = one KWIN=4 window** (4 K-slices × 32 WMMA). gfx12 inserted
**no** `s_delay_alu`/`s_nop` in the loop (the WMMA chain covers its own latency), so the disassembly *is*
the issue-slot ledger — 1 line = 1 issued instruction.

## The window (per 128 WMMA)

| class | instrs | per-WMMA | what it is |
|---|---:|---:|---|
| **`v_wmma_f32_16x16x16_fp8_fp8`** | **128** | **1.00** | the useful work |
| `ds_load_b64` (A frags, consume) | 32 | 0.25 | LDS→reg, 8/slice (2kk×4mi) |
| `global_load_tr_b64` (B frags, consume) | 32 | 0.25 | global→reg transposed, 8/slice |
| address/control SALU+VALU | 43 | 0.34 | s_add_co 17, s_addc 8, s_lshl 8, v_add 8, cmp/branch 2 |
| `global_load_b128`+`ds_store_b128` (A publish) | 16 | 0.125 | KWIN-amortized A staging (8+8) |
| `s_wait_loadcnt`+`s_wait_dscnt` | 10 | 0.078 | 5+5 |
| `s_barrier_signal`+`s_barrier_wait` | 4 | 0.031 | KWIN-amortized (publish + tail) |
| **TOTAL** | **265** | **2.07** | |

**WMMA fraction = 128/265 = 48.3%.** Measured FED wall = ~148/307 = **48.2%**. The ledger total
*is* the wall: at ~1 instr issued/cycle, throughput = WMMA's share of issued instructions. (Vector-memory-
only fraction would be 128/216 = 59% → ~182 TF, which we do NOT see — so SALU/waits/barriers count too.)

## Non-WMMA = 137 slots, ranked

1. **address/control — 43 (31%)** — largest bucket, but diffuse; much of it (B pointer advance/addr) scales *with* B loads.
2. **A frag loads `ds_load_b64` — 32 (23%)**
3. **B frag loads `global_load_tr_b64` — 32 (23%)**
4. A publish (b128 load+store) — 16 (12%) — already KWIN-amortized.
5. waits — 10 (7%); barriers — 4 (3%) — near-tapped (KWIN already amortized these).

## HIP-winner comparison (normalize to feed-loads per WMMA)

- **This kernel:** 8 A + 8 B frag-loads feed 32 WMMA/slice = **0.50 feed-loads/WMMA**.
- **HIP 155-winner (ISSUE_PROBE, KG 2601d691):** "8 feed instrs feed 16 WMMA" = **0.50 feed-loads/WMMA** — *same ratio*.

We are tied with the HIP winner on feed-instruction density, which is exactly why we sit at ~145 next to
their ~155. **To beat them we must push below 0.50 feed-loads/WMMA.** Two independent paths do that, and
they MULTIPLY:

## The two levers (instruction-count hypotheses)

### Lever A — wider loads (NO dyn-VGPR, no occupancy cost). TRY FIRST.
`ds_load_b128` and `global_load_tr_b128` both encode on gfx1201 (verified). Loading 2 frags/instruction
halves the frag-load count if the register+LDS layout packs 2 frags contiguously (correctness = oracle gate;
the transposed-b128 layout must reproduce the WMMA frag — Risk 1).
- A loads 32→16, B loads 32→16. Window 265→233. **128/233 = 54.9% → ~169 TF.** Beats HIP 161 with zero occupancy risk.
- Reaches the *same* 0.25 feed-loads/WMMA as the 8×8 tile, but by moving more bytes/instr rather than reusing more.

**A2 — B `global_load_tr_b128`: micro-oracle RESULT (2026-06-18, `--btr128`) = NOT a drop-in.**
Assumption-free check (`b128 per-lane == tr_b64(tile0) ++ tr_b64(tile1)`?) → **NO**. `tr_b64` group0 holds
K={0,1,4,5} (interleaved single-tile swizzle); `tr_b128` group0 holds K={0..7} contiguous + groups 2-3 hold
K=16-31 — structurally different lane→data map. Bonus: `tr_b64` on *plain* row-major fp8 ≠ `pack_B` →
`global_load_tr` is a **16-bit-granule** transpose; the kernel's fp8 correctness is **preshuffle (Bshuf) + tr
as a package**, not tr alone. So halving B feed via b128 is **not free** — needs a *custom* B preshuffle (not
ruled out: b128's contiguous-K layout could be a valid 32-K-deep B load under the right col map, but no longer
drop-in). **Per GPT's gate: B-b128 OUT as drop-in.**

**A1 — A `ds_load_b128` is the live cheap lever:** independent of the transpose issue (LDS is a plain
contiguous read, no 16-bit-granule transpose). 16B/lane = 2 A-frags if the LDS A-tile lays adjacent K-frags
contiguously and they land in 4 consecutive VGPRs. A frag-loads 32→16 → window 265→249 → **128/249 = 51.4% →
~158 TF**, zero occupancy cost. This is the next step.

### Lever B — register-blocked reuse tile (NEEDS dyn-VGPR for ≥6×6). The headline; COMPOUNDS with A.
Per-wave frag grid FM×FN: feed-loads/WMMA = (FM+FN)/(FM·FN). 4×4 = 0.50 → 6×6 = 0.33 → 8×8 = 0.25.
- 8×8 alone: frag-loads 64→32 (+ B address math ~halves). Window ~265→~223. **128/223 = 57.4% → ~176 TF.**
- VGPR: 8×8 = 64 acc×8 = 512 VGPR (+frags) > 256 → **dyn-VGPR via PM4 RSRC2 bit6** (armable, MAD-304). 6×6 = 288 also >256. 5×5 = 200 fits static but only 0.40 ratio (weak). **The meaningful reuse needs dyn-VGPR** — this is the quantified issue-density ↔ dyn-VGPR link.
- **A × B compound:** 8×8 tile + b128 loads → 0.125 feed-loads/WMMA → window ~265→~191 → 128/191 = 67% → ~206 TF.

### Lever C — SALU address fold (cheap, diffuse). Fold B kk1 pointer recompute (8 slots) + hoist publish addr → address 43→~28 → 128/250 = 51.2% → ~157 TF (~+5%).

### Lever D — waits/barriers (14): near-tapped by KWIN. Skip.

## Recommended sequence (methodology: instruction-count hypothesis → implement → measure → confirm)
1. **Lever A (wider A+B loads)** — cheapest, no occupancy risk, oracle-gated. Validates the ledger→TF mapping on silicon. Predicted ~169 TF.
2. **Lever C (SALU fold)** — cheap cleanup. Predicted ~157 alone; compounds with A.
3. **Lever B (dyn-VGPR 8×8 reuse tile)** — the headline, once A/C confirm the ledger predicts TF. Predicted ~176 alone, ~206 compounded with A.

Mandatory result columns for every change: TF | %peak | fill% | full-oracle | resident_waves | Δinstrs (ledger).
