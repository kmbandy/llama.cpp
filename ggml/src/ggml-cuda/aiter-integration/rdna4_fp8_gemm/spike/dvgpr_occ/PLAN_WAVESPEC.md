# Lean Wave-Specialized fp8 GEMM — Implementation Plan (MAD-305 #323)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, from scratch, a persistent wave-specialized fp8 (e4m3) WMMA GEMM for gfx1201/wave32 that launches lean and uses dyn-VGPR to keep loader waves at 32 VGPR while compute waves carry accumulators — to test whether the freed occupancy hides the 26% B-feed wall and beats the 165 TF static winner.

**Architecture:** Clean-room hand-asm kernel `occ_kernel_wavespec.s` (persistent atomic-claim queue; per-WG lean loader waves fill a double-buffered B-ring in LDS, fat compute waves drain it + WMMA over full K). A new `WAVESPEC` dispatch mode in `occ_dispatch.cpp` (clone of `run_wggemm_perf`) arms dyn-VGPR via raw PM4 RSRC2 bit6 + the umr `SQ_DYN_VGPR.BLOCK_SIZE=1` cap-lift. A/B/barrier/WMMA/claim machinery is COPY-AND-ADAPT from named proven kernels, not greenfield.

**Tech Stack:** gfx1201 hand assembly (`clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201`), raw-PM4 KFD dispatch via libhsakmt (`occ_dispatch.cpp`), CPU C++ oracle, RGA (RDTS), rocprofv3.

**Reference docs:** `SPEC_WAVESPEC.md` (this dir), `WAVESPEC_RESEARCH.md` (NVIDIA→AMD mapping).

## Global Constraints

- Target: gfx1201 (R9700) only, wave32. ROCm clang at `/opt/rocm/lib/llvm/bin/clang` (NOT `/usr/bin/clang`). RGA = RDTS build `/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga` (NOT `/usr/bin/rga`).
- EVERY kernel config instrumented with BOTH rocprof AND RGA before deciding the next experiment. Numbers drive.
- HARD oracle gate: stored C must match the CPU e4m3 reference at every config before any TF number counts.
- dyn-VGPR is armable ONLY via raw PM4 RSRC2 bit6 (MAD-304) — no compiler path. `s_alloc_vgpr` reaches ISA only via hand-asm.
- Lean tiles only — do NOT pursue fatter tiles (big-tile+dyn thesis closed 2026-06-16).
- SAFETY: first GPU dispatch is SUPERVISED (stop for go). NEVER pass `--gl2c`. gfx1201 drives the displays — a hang resets monitors.
- Run defsym loops under `bash -c` (fish/zsh don't word-split unquoted vars).
- Work is uncommitted in the spike dir per established pattern; commit only when the user asks.

---

### Task 1: CPU tile oracle + frag layout for wavespec

**Files:**
- Create: `wavespec_oracle.h`, `wavespec_oracle.cpp` (host reference C=A·B over the wavespec tile map)
- Create: `test_wavespec_oracle.cpp`
- Reuse: `fp8_oracle.h/.cpp` (`fp8_e4m3_to_float`, `wmma_ref_16x16x16`), `frag_layout.h` (`pack_A`, `pack_B`)

**Interfaces:**
- Produces: `void wavespec_ref(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K)` — row-major A[M*K], col-major-shuffled B as the kernel consumes it (match the existing `run_wggemm` Bshuf layout), full fp32-accumulated GEMM via repeated `wmma_ref_16x16x16` over 16×16×16 blocks. This is the gate reference the dispatch compute-mode compares against.

- [ ] **Step 1: Write the failing test** — `test_wavespec_oracle.cpp`: build a tiny 32×32×32 case, all-ones fp8 (0x38) ⇒ every C entry == K (=32.0); plus a mixed case cross-checked against a direct triple-loop fp32 reference using `fp8_e4m3_to_float`. Assert `wavespec_ref` matches the triple loop bit-for-bit (`<1e-4` abs).
- [ ] **Step 2: Run to verify it fails** — `/opt/rocm/lib/llvm/bin/clang++ -O2 test_wavespec_oracle.cpp wavespec_oracle.cpp fp8_oracle.cpp -o /tmp/twso && /tmp/twso` → expect link/compile FAIL (`wavespec_ref` undefined).
- [ ] **Step 3: Implement `wavespec_ref`** — tile the M×N output into 16×16 blocks, accumulate over K in 16-steps by calling `wmma_ref_16x16x16` on each (A-block, B-block) with carried C, exactly mirroring the kernel's accumulation order. Decode the kernel's B-shuffle the same way `run_wggemm`'s host reference does (see `occ_dispatch.cpp` `mbg_preshuffle_*` / chained `wmma_ref`).
- [ ] **Step 4: Run to verify it passes** — same compile cmd → expect `PASS`.
- [ ] **Step 5** (no git commit unless user asks; leave staged in spike dir).

---

### Task 2: `occ_kernel_wavespec.s` — persistent role-split, STATIC alloc (DYNVGPR=0)

**Files:**
- Create: `occ_kernel_wavespec.s`
- Adapt from: `occ_kernel_wggemm2.s` (atomic-claim + `ds_store ti` broadcast + barrier + cooperative A-fill + WMMA block + the KWINBPF double-buffered B-ring), `occ_kernel_mbgemm.s` (PROFILE in-kernel timer block).

**Interfaces:**
- Consumes: userdata SGPR layout from Task 3 dispatch (s0:1=occ, s2:3=A, s4:5=B, s6:7=C, s8=K, s9=NT*256, s10/s11=tile decode mask/shift, s12=NTILES, s13=TOTAL).
- Produces: kernel symbol consumed by `build.sh` defsyms `FM,FN,NLOAD,NCOMP,RINGDEPTH,LEANREG,DYNVGPR,COMPSHRINK,STORE`. This task sets `DYNVGPR=0` (static), `STORE` both.
- Wave role: `wid = (TID / 32)`; `wid < NLOAD` ⇒ loader, else compute (compute-local index `cid = wid - NLOAD`).

- [ ] **Step 1: Prologue + persistent claim loop** — adapt wggemm2's leader-`atomic_add` on the global tile counter (s12/s13) → `ds_store ti` to the LDS broadcast slot → `s_barrier` → all waves read `ti`; loop until `ti >= TOTAL`. Lift verbatim, keep the same decode (s10/s11 mask/shift).
- [ ] **Step 2: Role branch + LDS layout** — `.set` LDS regions: A-tile (cooperative fill, KWIN reuse), B-ring `RINGDEPTH` slots, ti-broadcast. Branch `wid < NLOAD` to the loader body, else compute body.
- [ ] **Step 3: Loader body** — per-K: `global_load_tr_b64` a B-slice → `ds_store` into B-ring slot `(k % RINGDEPTH)` → participate in the per-K-window `s_barrier`. Run one window ahead (double-buffer, KWINBPF cadence from wggemm2).
- [ ] **Step 4: Compute body** — per-K: `s_barrier`-wait the published B-slot → `ds_load_b64` A frags + `ds_load_b64` B frags → `FM*FN` `v_wmma_f32_16x16x16_fp8` accumulating into the C-frag VGPRs → advance. After full K: if `STORE`, write C frags to global (wggemm2 C-store).
- [ ] **Step 5: PROFILE timer (optional defsym)** — lift mbgemm's realtime-tick phase block (compute / feed-wait / barrier / claim), WG0 writes occ[24..44]. `.text` byte-identical when `PROFILE=0`.
- [ ] **Step 6: Assemble** — `bash -c '/opt/rocm/lib/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,DYNVGPR=0 -Wa,-defsym,FM=2 -Wa,-defsym,FN=2 -Wa,-defsym,NLOAD=1 -Wa,-defsym,NCOMP=4 -Wa,-defsym,RINGDEPTH=2 -Wa,-defsym,STORE=1 -c occ_kernel_wavespec.s -o /tmp/ws.o'` → expect clean assemble.
- [ ] **Step 7: RGA livereg gate** — `./rga_check.sh wavespec_2x2_static "DYNVGPR=0 FM=2 FN=2 NLOAD=1 NCOMP=4 RINGDEPTH=2 STORE=1"` (set `KSRC=occ_kernel_wavespec.s`). Record peak-live (expect ~54 for 2×2 compute waves; loaders share the static fat alloc here).
- [ ] **Step 8** (oracle gate deferred to Task 3 — needs the dispatch path to run it).

---

### Task 3: `WAVESPEC` dispatch mode (STATIC first) + oracle gate

**Files:**
- Modify: `occ_dispatch.cpp` — add `WAVESPEC` to the `mode` enum (line ~2009), `--wavespec` arg (near line 2029), and `run_wavespec_compute`/`run_wavespec_perf` (clone of `run_wggemm_compute`/`run_wggemm_perf`, ~line 740).
- Modify: `build.sh` — assemble `occ_wavespec.bin` (STORE=0) + `occ_wavespec_st.bin` (STORE=1).

**Interfaces:**
- Consumes: `occ_kernel_wavespec.s` binary; `wavespec_ref` (Task 1) for the compute-mode gate.
- Produces: `--wavespec` runnable mode; launches `nWG * (NLOAD+NCOMP)*32` lanes.

- [ ] **Step 1: Clone dispatch fn** — copy `run_wggemm_perf` → `run_wavespec_perf`; change the wave count in `dims[3]` and `PM4DispatchDirectPacket` from `WAVES*32` to `(NLOAD+NCOMP)*32`; keep `BuildPgmRsrc2(false)` for now (static, no dyn). Same userdata[16] wiring.
- [ ] **Step 2: Clone compute/oracle fn** — copy `run_wggemm_compute` → `run_wavespec_compute`; after the run, compare stored C against `wavespec_ref` (Task 1), print `ORACLE OK n/n` / `MISMATCH`.
- [ ] **Step 3: Wire arg + enum** — add `WAVESPEC` enum member and `else if (!strcmp(argv[i],"--wavespec")) mode=WAVESPEC;`; in the mode dispatch, call `run_wavespec_compute` (STORE) then `run_wavespec_perf` (perf).
- [ ] **Step 4: build.sh stanza** — mirror the wggemm2 lines (151–168): assemble `occ_wavespec_st.bin` (STORE=1) and `occ_wavespec.bin` (STORE=0) with `DYNVGPR=0 FM=2 FN=2 NLOAD=1 NCOMP=4 RINGDEPTH=2`.
- [ ] **Step 5: Build harness** — run `./build.sh` (the occ_dispatch compile section); expect clean build of `occ_dispatch`.
- [ ] **Step 6: CPU-only smoke is N/A (GPU dispatch)** — defer the actual run to Task 6 (SUPERVISED). Gate here = builds clean + oracle fn compiles + `wavespec_ref` linked.

---

### Task 4: dyn-VGPR path (DYNVGPR=1) + arm in dispatch

**Files:**
- Modify: `occ_kernel_wavespec.s` (add `DYNVGPR` grow/lean + SCC guard)
- Modify: `occ_dispatch.cpp` `run_wavespec_*` (arm bit6 + umr cap-lift when a `dynvgpr` flag is set)

**Interfaces:**
- Consumes: Task 2/3 static kernel + dispatch.
- Produces: `DYNVGPR=1` byte-path: loaders `s_alloc_vgpr LEANREG(32)`; compute `s_alloc_vgpr <TILE_VGPR>` grow at entry with `s_cmp`/SCC guard branching to a safe stall on fail.

- [ ] **Step 1: Kernel dyn block** — in the role branch, `.if DYNVGPR`: loader path emits `s_alloc_vgpr 32` at entry; compute path emits `s_alloc_vgpr <FATREGS>` (round FM*FN accumulator footprint to 16-VGPR block, as mbgemm `FATREGS` does, lines ~46–49) then `s_cmp_eq_u32 scc...` guard. Lift the grow/shrink encodings from `occ_kernel_mbgemm.s:207,297`.
- [ ] **Step 2: Assemble both** — `bash -c '... -Wa,-defsym,DYNVGPR=1 ... -c occ_kernel_wavespec.s -o /tmp/ws_dyn.o'` and the `DYNVGPR=0` variant → both assemble clean.
- [ ] **Step 3: Arm in dispatch** — add a `dynvgpr` bool to `run_wavespec_*`; when true: `BuildPgmRsrc2(true)` (sets bit6) and emit the umr cap-lift wait-loop (clone line ~530: `umr -i 1 -w '*.*.regSQ_DYN_VGPR' 0x1ff`) before submit. Reduce `rsrc1` vgprField to the lean launch size when dyn (waves grow from there).
- [ ] **Step 4: RGA confirms lean loaders** — `./rga_check.sh wavespec_2x2_dyn "DYNVGPR=1 FM=2 FN=2 NLOAD=1 NCOMP=4 RINGDEPTH=2 STORE=1"` → loader-path peak-live should drop to ~32 vs compute ~54.
- [ ] **Step 5** — oracle re-gate deferred to Task 6 (needs GPU). Gate here = both variants assemble + RGA shows the lean/fat split.

---

### Task 5: `build.sh` sweep matrix + CPU/RGA gate

**Files:**
- Modify: `build.sh` — add a wavespec defsym sweep loop (FM/FN ∈ {2×2, 4×4}, NLOAD ∈ {1,2,4}, DYNVGPR ∈ {0,1}), each emitting a named `.bin`, with an RGA call per cell.

**Interfaces:**
- Consumes: Tasks 2–4 kernel + knobs.
- Produces: a matrix of `occ_wavespec_<fm>x<fn>_nl<n>_dyn<d>{,_st}.bin` + per-cell RGA livereg lines.

- [ ] **Step 1: Sweep loop** — `bash -c 'for cfg in "2 2" "4 4"; do for nl in 1 2 4; do for dv in 0 1; do ... assemble ...; done; done; done'`. Use the exact `clang -x assembler` line from Task 2 Step 6 with substituted defsyms; STORE=0 (perf) + STORE=1 (oracle) per cell.
- [ ] **Step 2: CPU-green** — re-run `test_wavespec_oracle` (Task 1) to confirm the host ref still builds/passes.
- [ ] **Step 3: RGA per cell** — loop `./rga_check.sh` over the same matrix; assert every cell assembles and record peak-live. Expect dyn cells show loader peak-live ≈ 32.
- [ ] **Step 4: Verify all `.bin` present** — `bash -c 'ls occ_wavespec_*'` matches the matrix count.

---

### Task 6: [SUPERVISED — STOP for go] first GPU dispatch + 2×2 dyn-vs-static A/B

**Files:** none (run + observe). Produces measurements.

**Interfaces:** consumes the built `.bin` matrix + `--wavespec` dispatch.

- [ ] **Step 1: STOP — get explicit user go** before any raw-PM4 KFD dispatch (gfx1201 = display GPU; hang risk).
- [ ] **Step 2: Baseline smoke (static, STORE=1)** — run `--wavespec` at 2×2 DYNVGPR=0; confirm `ORACLE OK` and a sane TF.
- [ ] **Step 3: dyn smoke (STORE=1)** — run 2×2 DYNVGPR=1 (with umr cap-lift); confirm `ORACLE OK` (dyn path correctness).
- [ ] **Step 4: A/B perf (STORE=0)** — 2×2 DYNVGPR=0 vs DYNVGPR=1 TF, plus rocprofv3 `--pmc` (occupancy/issue) and `--att` (stall %) on each. Capture feed-wait %.
- [ ] **Step 5: Record** the four numbers + occupancy/feed-wait deltas inline for the user.

---

### Task 7: [SUPERVISED] tile/NLOAD sweep + RESULT + bank

**Files:**
- Create: `RESULT_WAVESPEC.md`

**Interfaces:** consumes Task 6 go + matrix.

- [ ] **Step 1:** sweep NLOAD ∈ {1,2,4} and tile ∈ {2×2, 4×4} at DYNVGPR=1 (and the DYNVGPR=0 twin per cell for attribution), rocprof+RGA on every cell, oracle-gated.
- [ ] **Step 2:** write `RESULT_WAVESPEC.md` — table of {tile, NLOAD, DYNVGPR} → {TF, occupancy, feed-wait %, peak-live, oracle}; verdict vs the 165 TF winner and the static-role baseline.
- [ ] **Step 3:** `mneme_write` the outcome (type=decision); update Jira MAD-305 / task #323.
- [ ] **Step 4:** state the verdict plainly (beat / null / regressed) with the measured numbers — null is a clean lever-space close, reported as such.

---

## Self-Review

- **Spec coverage:** persistent claim (T2), role-split lean loaders + fat compute (T2/T4), dyn-VGPR arm + cap-lift (T4), no-fail SCC guard (T4), B-ring double-buffer sync (T2), DYNVGPR attribution knob (T3/T5), oracle gate (T1/T3/T6), rocprof+RGA every cell (T5/T6/T7), supervised first run (T6), success criteria + null-result honesty (T7) — all covered.
- **Placeholders:** kernel-asm steps name the exact source file + line to adapt (wggemm2 claim/WMMA/B-ring; mbgemm s_alloc_vgpr:207/297, FATREGS:46–49) and gate by assemble+RGA+oracle rather than fabricating unverified gfx1201 WMMA encodings — deliberate: hand-inventing 500 lines of WMMA asm into the plan would be less correct than adapting the proven blocks.
- **Type consistency:** `wavespec_ref` signature identical in T1 and T3; defsym names (`FM/FN/NLOAD/NCOMP/RINGDEPTH/LEANREG/DYNVGPR/COMPSHRINK/STORE`) identical across T2–T5; userdata SGPR map identical T2↔T3.
