# L4 — Lean Persistent Single-Wave Register-Blocked fp8 GEMM (design)

Date: 2026-06-19. MAD-305 lever **L4**. Companion to `MAD305_LEVER_CATALOG.md` (read §0–§3 there first).
Target: gfx1201 (RDNA4 / R9700, wave32), fp8 e4m3, training-throughput GEMM.

## 1. Why (the one unmeasured point on the map)

The campaign wall is **FED 164 vs NOFEED 282 TF** — feed instructions sitting in the compute wave's
issue stream, not hidden, because only **~64 waves are resident**. Two facts frame L4:

- The 64-wave cap is **workgroup *shape***, not a per-wave resource. This session's probes (VGPR
  216→240, LDS-trim, B128 feed, wide-A) all came back **flat** → the cooperative **8-wave barrier-WG**
  locks a WGP slot, capping resident WGs. (Supersedes the catalog's older "VGPR-capped at 64" note.)
- The **4-wave confound**: a leaner 4-wave tile reached **192 resident but only ~149 TF** — *more
  occupancy, weaker B-reuse, net loss*. So occupancy alone is not the win.

**L4 measures the corner we have never measured: high occupancy AND strong B-reuse together.**
The bet: one wave per WG (no barrier, no co-residency lock) lets WGs pack to the VGPR/SIMD limit, and a
register-blocked M0×N0 tile keeps B-reuse high — so the feed-in-stream penalty is hidden across the many
resident waves. Win condition: `residWv ≫ 64` **and** `TF > 165.7`. Informative failure: `residWv` climbs
but `TF` stalls ≈149 → the confound is fundamental → route to L3 (dyn-VGPR) / L2 (wave-spec).

## 2. The crux: persistent, not static-grid

If each WG statically owns one M0×N0 output tile, a bigger (more-reuse) tile means **fewer WGs** →
*lower* occupancy. Register-blocking and occupancy fight for a fixed problem size.

**Resolution — persistent:** launch exactly **W** single-wave WGs (W = target occupancy, a launch
param). Each WG loops claiming output tiles off a shared counter (or a static stride). Now the two levers
are independent and separately sweepable:

- **W** (grid size) → occupancy. Read `residWv` back to confirm.
- **M0×N0** (register block) → B-reuse vs VGPR-leanness.

The existing BAND-claim machinery and the MICROBATCH persistent-pull dispatch (`occ_dispatch.cpp:311`)
are exactly this pattern, so they port.

## 3. Architecture

- **One wave per WG** (`dim3(32,1,1)`), wave32. → the **safe regime**: no barriers, no co-residency,
  **zero 16-wave brick risk** (§6 of the catalog).
- Each wave holds an **M0×N0 grid of 16×16 f32 accumulators** (8 VGPR each), live across the K-loop.
- Per 16-wide K-step: load **M0 A-frags** (2 VGPR ea) + **N0 B-frags** (2 VGPR ea) → issue **M0·N0 WMMA**.
  Each B-frag reused across M0 rows, each A-frag across N0 cols. (Frag footprints confirmed from the live
  kernel: A=2 VGPR, B=2 VGPR, C=8 VGPR.)
- **Direct-global feed** (default): A from A-shuf, B from B-shuf, both `global_load_tr_b64` — no LDS, no
  barriers. This is the existing **ANOLDSTR** feed (`occ_kernel_wggemm2.s:642`), reused verbatim in form.
- WMMA emission form: `v_wmma_f32_16x16x16_fp8_fp8 v[ACC+(mi*N0+ni)*8 +:8], v[AF+mi*2 +:2],
  v[BF+ni*2 +:2], v[ACC+(mi*N0+ni)*8 +:8]` (the register-blocked form already at line 1321).

## 4. Register budget (drives occupancy)

`VGPR ≈ 8·M0·N0 (acc) + 2·M0 (A) + 2·N0 (B) + ~20 (addr/scratch)`, ×2 on feed regs if B-prefetch on.

| (M0,N0) | acc | feed | ≈VGPR | tile | B-reuse | A-reuse |
|---|---|---|---|---|---|---|
| (4,1) | 32 | 10 | ~70  | 64×16 | 4× | 1× |
| (2,2) | 32 | 8  | ~64  | 32×32 | 2× | 2× |
| (4,2) | 64 | 12 | ~100 | 64×32 | 4× | 2× |
| (2,4) | 64 | 12 | ~100 | 32×64 | 2× | 4× |
| (6,2) | 96 | 16 | ~136 | 96×32 | 6× | 2× |

(4,1) and (4,2) are the lean-AND-B-stationary sweet spots that directly answer the confound.

## 5. Configurable axes

| Knob | Where | Default | Sweep |
|---|---|---|---|
| **W** (resident single-wave WGs) | launch arg | fill machine | 64 / 128 / 256 / 512 |
| **M0 × N0** (register block) | defsym | 4×2 | (4,1)(2,2)(4,2)(2,4)(6,2) |
| **Feed path** | defsym | direct-global | LDS-A (later) |
| **B-prefetch depth** | defsym | 0 | 1 (double-buffer) |
| **Claim mode** | defsym | static stride | atomic BAND |

## 6. Components

- **New kernel** `occ_kernel_lean.s` — clean single-wave register-blocked body. Recommended over a
  `WAVES=1` branch in `wggemm2.s` (the cooperative LDS/claim/barrier scaffolding there is dead weight and
  an interaction risk). **The lean kernel adopts the existing wggemm2 SGPR/kernarg contract** (s0:1
  counter, s2:3 A-shuf, s4:5 B-shuf, s6:7 C, s8 K, s9 NT·256, s10 mask, s11 log2, s12 NTILES, s13 TOTAL,
  s14 MT·256) so the harness binds it unchanged.
- **Reuse unchanged:** `mbg_preshuffle_A` / `mbg_preshuffle_B` (frag-ready A-shuf/B-shuf), the CPU oracle
  `wmma_ref_16x16x16`, and the perf wall timing in `run_wggemm_compute` / `run_wggemm_perf`.
- **New dispatch mode `--lean`** in `occ_dispatch.cpp`: workgroup size = 32 (DIM_X = W·32), grid = W, M0/N0
  passed for the oracle's per-WG tile geometry, descriptor audit (wave32, LDS≈0, VGPR field). Mirrors the
  flat diagnostic store so the existing every-frag oracle verifies it.
- **build.sh**: `occ_kernel_lean.s` variants over (M0,N0), STORE∈{0,1}, feed/prefetch defsyms.

## 7. Correctness & safety

- **Gate:** 512³ oracle (STORE=1 full-fragment, every frag vs chained `wmma_ref`) **before any timed run** —
  the same floor held all session.
- **Safety:** WAVES=1 → cannot deadlock on co-residency, no barriers → **the safe regime**. Still WARN
  before any GPU dispatch (catalog §6). **Never** WAVES=16. Harness targets KFD Node 1 (gfx1201); leave
  Node 2 (6900XT desktop) alone.

## 8. What we sweep / report

Sweep **W × (M0,N0)**; per cell report `residWv`, FED vs NOFEED, TF, oracle PASS/FAIL. The matrix isolates
occupancy (W) from reuse (M0×N0) — the decomposition the 4-wave confound denied us.
