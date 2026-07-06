# MAD-305 / DSWS — Master Pickup Doc (GPU kernel campaign)

**Single source of truth for the dynamic-VGPR fp8 GEMM work.** Consolidated
2026-06-26 before a context compact. Read this first to resume cold. Detailed
history lives in the sibling docs + KG notes pointed to at the bottom.

---

## 1. North star + the moat

- **Goal:** 250–300 TF fp8 GEMM on **gfx1201** (AMD R9700 / Navi48, RDNA4, wave32).
- **The moat — dynamic VGPR (`s_alloc_vgpr`)** is armable on a compute queue **only
  via raw PM4** writing **COMPUTE_PGM_RSRC2 bit 6** on a KFD queue (libhsakmt,
  bypassing ROCr). HIP/the toolchain cannot emit it for compute. This is why the
  whole harness dispatches via raw PM4 (`occ_dispatch.cpp` + `dvgpr_pm4/`), not HIP.
  Confirmed: bit clear → reads 0; bit 6 set → `DYN_VGPR_EN` reads back. (MAD-304.)

## 2. Current state (the numbers)

- **fp8 WMMA ceiling = 307 TF** (15.9 WMMA/cyc); **272 TF** feed-free. One WMMA =
  16×16×16 = 8192 FLOP. R9700 = 64 CU × 2 SIMD32 = **128 SIMDs**.
- **Best real kernel ≈ 165.7 TF** (8×2 tile, KWINBPF double-buffer + s_setprio +
  wide-A feed) on **square** shapes = **~52% of ceiling**, **8.34 WMMA/cyc**.
  HIP 4×4-dbuf reference = 161 TF / 52.5% (VGPR 183/256, 16 frags = 128 acc VGPR,
  occ 5 blocks = 20 waves).
- **On real ml8 dims = ~113 TF** (latest tile tuning, 3 verified fixes; commit
  `99f063e9c`). ml8 shapes: **`down`** (ffn_down) M=2048 K=9216 N=2560;
  **`down_pf`** (prefill) M=512. *Always bench on these, not throwaway squares.*

## 3. THE WALL — it's the VALU issue port, NOT occupancy

- Measured **~31 non-WMMA instructions issued per 32 WMMAs** (square shapes) →
  caps throughput at **8.34 / 15.9 WMMA/cyc = 52%** of 307 TF. The issue port,
  not memory and not occupancy, is the binding constraint.
- **Occupancy is FLAT**: minWaves 4→8 barely moves TF (~160 TF either way). So
  raising occupancy is NOT the lever.
- **The lever = cut non-WMMA issues per WMMA** — via (a) more fragment reuse, or
  (b) wave specialization so the math wave issues ~only WMMAs while other waves do
  the address/feed VALU. This is what DSWS targets.
- Caveat: the 165 TF winner + the 31:32 ratio are **square** shapes. ml8 shapes
  (tall-skinny `down`) differ — re-measure the issue mix ON ml8 with `--att`.

## 4. SOLVED this week (both real, keep)

### POOLTERM — pool≥2 teardown brick (FIXED, confirmed on silicon 2026-06-26)
- **Root cause:** the compute terminal in `occ_kernel_coop.s` was a **POOL=1-only
  diagnostic stub** — compute counted tiles it processed (s57) and exited at
  s57==TOTAL. Valid only when ONE WG owns all tiles. At pool≥2 the WGs SPLIT tiles
  via the shared global atomic claim, so each compute gets <TOTAL → count never
  reaches TOTAL → compute spins forever at `.Lwait_epoch` → WG never retires → EOP
  RELEASE_MEM fence never fires → queue never IDLE → any reclaim wedges the GPU.
- **Fix (gated behind `POOLTERM` defsym):** the feed already broadcasts a per-WG
  terminal `ti≥TOTAL` into LDS `TI_OFF` and bumps epoch before `.Lfeed_exit`;
  compute now exits when it sees that broadcast, checked on the **RAW ti BEFORE the
  SAFEPROBE clamp** (the clamp pins ti to TOTAL-1 and would hide it). POOLTERM=0
  keeps static d0 **byte-identical** to the reference (1716B).
- **Silicon-confirmed:** first-ever clean pool=2 run — compPh=8 (was stuck 7),
  fence=FIRED (was `--`), clean teardown, oracle ok=256 bad=0, no brick, no dmesg
  faults. (KG `dac0bb8c` root cause, `0a2cea44` confirmation.)

### Compositor-safe chunking (in `occ_dispatch.cpp`)
- The R9700 drives the user's monitors, so long (>~1s) persistent dispatches starve
  the compositor's gfx ring → brick. Fix: bound each dispatch to `ML8_COOP_CHUNK`
  output tiles (claim `occW[5]=base`, `userdata[11]/s11=hi` terminal), drain on the
  EOP fence, YIELD between chunks, `ML8_COOP_CHUNK_MAXS` guard. Sub-second bounded
  dispatches run the real ml8 GEMM **imperceptibly** on the display GPU.

## 5. DSWS — the design (the actual north-star architecture)

**"Dynamic-Split Wave-Specialization" / adaptive wave-role economy.** ONE kernel
that re-balances its mix of **{fat compute waves (large VGPR, hold fp32 WMMA
accumulators) / lean A-feed waves / lean B-feed waves}** to the bottleneck **at
runtime**, using `s_alloc_vgpr` to make the VGPR budget **fungible across roles
asymmetrically** (shrink one fat compute ~112–256 VGPR → fund several lean feeders,
or vice-versa). Bottleneck sensed by the prod/cons ring counters already in
`occ_kernel_coop.s`.

- **Split-K is the headroom-creator, NOT an occupancy play.** Full-K GEMM holds the
  fp32 accumulators (~64 of 112 VGPR) at peak for the ENTIRE K-loop (~95% duty —
  square wave, not trapezoid), so staggering buys nothing. Split-K creates brief
  accumulator-peak windows = the room to reallocate VGPR between roles.
- **Why this beats static:** the issue-port wall (§3) is broken by letting the math
  wave issue ~only WMMAs while feed waves absorb the address/feed VALU — and tuning
  that ratio per-shape (and within a kernel) to whatever the runtime says is short.

## 6. Prior art + the research — VERDICT (Murmur prior-art-scan, landed 2026-06-27)

**Workflow `5ec8a958` (42 scouts on 6900xt + Qwen3.6-27B captain sift). Verdict:
the RUNTIME-ADAPTIVE piece is NOVEL; the lean/fat mechanism is established (borrow
it).** Ranked leads:

1. **Runtime-adaptive producer:consumer wave-role rebalancing with dynamic per-wave
   register reallocation is NOT in any known GPU GEMM/compute kernel** (HIGH conf —
   unanimous negative across ~40 scouts: arXiv 2023–2026, AMD GPUOpen, NVIDIA
   CUTLASS/CuTe, Triton, LLVM/ACO, forums). → DSWS's core (in-kernel ring-counter
   sensing + runtime `s_alloc_vgpr` rebalancing) is **architecturally novel**.
2. **CUTLASS warp-spec = STATIC `setmaxnreg` / `warpgroup_reg_alloc` at launch** —
   producer(DMA/TMA):consumer(MMA) split + register budgets fixed for the whole
   kernel. The direct baseline to differentiate from; register fungibility is a
   compile/launch-time optimization, not a runtime control loop.
3. **AMD `s_alloc_vgpr` (RDNA3.5/4) = static per-wave; no in-kernel sensing / runtime
   shifting.** The primitive we repurpose; the ISA lacks native runtime-rebalancing
   semantics → DSWS supplies its own coordination layer (= the contribution).
4. **Adjacent dynamic work (Stream-K, persistent kernels, WaveTune) balances at the
   CTA/workgroup level, NOT intra-CTA wave roles or mid-flight register realloc**
   (HIGH conf). Rules out the nearby paradigm a reviewer might conflate us with.

**GAPS / must verify against primary sources before we stake the claim:**
- Exact CUTLASS `setmaxnreg` register ranges + granularity (the "24..256 step 8"
  numbers) and documented HW restrictions — NOT extracted by any scout.
- Precise producer:consumer warp ratios in CUTLASS / CK-rocWMMA / Triton persistent.
- AMD `s_alloc_vgpr` HW rules re: barriers, occupancy, wave-launch sync in COMPUTE
  shaders (vs the RT path) — needs ISA-level confirmation. (Ties to our own
  barrier-vs-dyn-VGPR deadlock history — KG `8a9ce97f`/`17f209af`.)
- Novelty is partly absence-of-evidence; one citation (`WaveTune arXiv:2604.10187`)
  is suspect (possible 8B-scout hallucination) — do not cite unverified.

Full LEADS doc: handoff `141924b9` result in quantdb on mad-lab-2026. Captain banked
to claude__main KG. (This run also surfaced + fixed the engine bug that had been
truncating the captain's scout bundle — KG `22063f79`.)

## 7. Key files + bins (all on mad-lab-main, committed to fork master)

- `occ_kernel_coop.s` — cooperative kernel; **POOLTERM** fix; the prod/cons ring
  counters DSWS will sense; SAFEPROBE bounds guard.
- `occ_kernel_wavespec.s` — static loader/compute role-split prototype (WS path).
- `occ_kernel_mbgemm.s`, `occ_kernel_wggemm2.s` — the 8×2 / KWINBPF winner lineage.
- `occ_dispatch.cpp` — raw-PM4 KFD harness; compositor-safe chunking; oracle gate.
- `build_coop.sh` / `build.sh` — defsym build matrices (FM,FN,P,RINGD,BATCH,DYNVGPR).
- `../dvgpr_pm4/` — PM4 packet defs + RSRC2 bit6 arming.
- **Detailed specs:** `SPEC_WAVESPEC.md` (DSWS v1+v2, wall numbers, prior art),
  `MORNING_PICKUP.md` (day arc), `RESULTS_ml8_dynvgpr_gauntlet.md`,
  `MAD305_LEVER_CATALOG.md`, `L4_LEAN_DESIGN.md`, the `RELATED_WORK_SWEEP` +
  `NVIDIA_KERNEL_IDEAS_FOR_RDNA4` research docs.

## 8. STANDING SAFETY (hard rules — do not violate)

- **A GPU brick is a BUG, never an "accepted tax."** Recoverable-via-MODE1/reboot
  ≠ acceptable. Root-cause and fix; don't route around.
- **THE USER decides EVERY GPU dispatch** — each individual dispatch, not one
  blanket "go" for a session. A hang/timeout on an unproven kernel = full STOP +
  report, NOT auto-fire the next variant. Do max work OFFLINE first (disasm, RGA,
  static analysis); batch a diagnostic into ONE prepared dispatch.
- Only **sub-second bounded** dispatches are safe on the display GPU (R9700 drives
  the monitors). Freeze dyn dispatch on any build known to leave the queue non-idle.
- **NEVER pass `--gl2c`** (MES-crash landmine). Keep SAFEPROBE clamp + bounds gate +
  padding ON. Every run streams to disk (`ML8_COOP_STREAM=1`).
- Do NOT move displays to the eGPU 6900XT (can't init pre-login; also NOT the
  single-GPU target user — the kernel must coexist with the compositor).

## 9. NEXT STEPS (resume order for step 4)

1. **Incorporate the Murmur prior-art leads** (research output) into the DSWS
   novelty + setmaxnreg mechanics-to-copy.
2. **R0 occupancy attribution** — step pool up ONE at a time (gated, abort on any
   non-clean exit); RGA showed compute peak-live ~81 vs HW-allocated 120 → trim NFV
   toward ~96 = free occupancy.
3. **ML8_P=2** — one feed + TWO compute waves in one WG = the real reuse/throughput
   lever (reuse 2.0/2.4/2.67 for P=2/3/4). P>1 needs its own terminal review.
4. **DSWS build** — brainstorm → spec → plan → TDD, on the POOLTERM substrate,
   measured on ml8 `down`/`down_pf` with `--att` for the issue mix.

## 10. Pointers

- KG: `dac0bb8c` (POOLTERM root cause), `0a2cea44` (POOLTERM silicon confirm),
  `63583120` (DSWS v2), `21827908` (compositor chunking), `34f29c00` (full-day
  session summary), `58a41155` (GPU-dispatch discipline), `1630687a` (earlier
  brick = OOB shader access root cause + the mandatory bounds-guard lesson).
- Jira epic: MAD-305 (under the 250-300 TF north star). MAD-304 = the PM4 dyn-VGPR
  arming. MAD-300 = the WMMA ceiling + rocWMMA baseline.
- Branch/state: fork **master** (pushed `9ff961564`), all dvgpr work committed
  (`a514395ec` snapshot + the upstream merge).
