# DSWS Rolling dyn-VGPR Sum-Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the multi-grower dyn-VGPR collision unreachable by routing every per-rowblk compute burst grow through the shared `vgpr_reserved` sum-envelope, and realize the rolling trapezoid (`lean → reserve → grow → WMMA burst → flush → shrink → release`) with real split-K.

**Architecture:** Off the confirmed root cause (bare `s_alloc_vgpr NFV` SCC-retry with no envelope, `occ_kernel_dsws.s:1158`), we (1) prove the enveloped reserve/spin/release semantics in the CPU reference model under a thread race, (2) transcribe them into the kernel behind a new `DSWS2_ENVELOPE` symbol that defaults to byte-identical-off, (3) gate on assemble/RGA/byte-identity offline, then (4) one supervised GPU oracle run at `n_kseg>1`. Stagger + conversion-reserve reconciliation are the full-design follow-ups after the first gate.

**Tech Stack:** GCN/RDNA4 hand-assembly (`gfx1201`, `occ_kernel_dsws.s`); C++17 CPU reference model (`dsws_ctrl_model.cpp` + std::thread test harnesses); LLVM `clang`/`llvm-objcopy` assemble + `sha256sum` byte-identity; RGA static spill analysis.

**Spec:** `docs/superpowers/specs/2026-07-02-dsws-rolling-dynvgpr-envelope-design.md`

**Working dir (all paths relative to it unless absolute):** `/home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ`

## Global Constraints

- **Byte-identity is a hard gate.** With `DSWS2_ENVELOPE=0 && DSWS2_STAGGER=0`, the `.text` MUST be bit-for-bit identical to HEAD (`7b9680508`): `DSWS2_CONV=0` → `4840B`, sha256 prefix `e5ec5e50`; `DSWS2_CONV=1 && DIAG=0` → `7920B`, sha256 prefix `049818c6`. (The earlier `e296b846` figure in the handoff/spec was stale — measured pre-instrumentation-commit; ground truth re-derived from `git show HEAD:occ_kernel_dsws.s` on 2026-07-02.) Every kernel change is verified against these before proceeding.
- **`NFV=112 < 128`** (default `SQ_DYN_VGPR` cap). Never emit a grow > 128 VGPR (no operator umr flip).
- **No `s_barrier` anywhere. Never pass `--gl2c`.**
- **NEVER modify or `git add`** `occ_kernel_coop.s`, `occ_dispatch.cpp`, `fp8_oracle.cpp`, `fp8_oracle.h` (shared/not-ours). They may be dirty; leave them.
- **No GPU dispatch is auto-run.** kmbandy greenlights EVERY GPU dispatch individually. Tasks 8 is operator-triggered.
- **Commit to git ONLY when kmbandy explicitly asks.** Steps below that say "Commit" are gated on that ask; stage only OURS files.
- **After any `DSWS2_ENVELOPE=1` build+dispatch, restore the safe `CONV=0` 4840B/`e5ec5e50` bin immediately and unconditionally** (footgun removal).
- **ALL GPU data-collection writes (occ stream, dmesg/fault capture, fire scripts) go to REAL DISK (`/home/kmbandy/dsws_gpu_logs/`, btrfs), NEVER `/tmp` (tmpfs).** A brick forces a reboot that wipes tmpfs — writing the stream there destroys the very evidence the `ML8_COOP_STREAM` safeguard exists to preserve. (Lesson: 2026-07-02 brick — stream + fire script both lost to the reboot because they were in the tmpfs scratchpad.) Build intermediates (`.o`/`.text`) may stay in scratch since they are regenerable.
- **Never confound an untested feature with the change under test in one GPU dispatch.** `n_kseg>1` split-K addressing has NEVER been GPU-run or oracle-validated; it must be validated on its own before any `n_kseg>1` dispatch. The envelope gate runs at `n_kseg=1` (proven-safe geometry).
- Constants (from the kernel, fixed at `FM=2 FN=4 G=6 SEGK=64`): `NFV=112`, `VLEAN=32`, `Δ=NFV−VLEAN=80`, `WAVES=8` (4c2a2b), `KSEG_STEPS=SEGK/16=4`.

---

## File Structure

- `dsws_ctrl_model.cpp` (MODIFY) — reference semantics. Add `reserve_spin` + `reserve_release` (the compute-burst envelope helpers). Pure, header-style; no new includes.
- `test_dsws_ctrl_model.cpp` (MODIFY) — add the single-thread envelope-invariant asserts.
- `test_dsws_envelope_race.cpp` (CREATE) — std::thread forward-progress race for the compute-burst reserve loop (mirrors `test_dsws_quiesce_race.cpp`).
- `occ_kernel_dsws.s` (MODIFY) — new symbols + `BUDGET` envelope default + assemble guard; `VRESV_OFF` init inversion; `.Lcompute_reserve`/grow/release in the claim loop; DIAG envelope telemetry. All behind `DSWS2_ENVELOPE` / `DIAG`.
- `build_dsws.sh` (MODIFY) — `mk2` passthrough for `DSWS2_ENVELOPE`/`PEAK_CONC`/`DSWS2_STAGGER`/`STAGGER_PERIOD`, defaults mirroring the in-file `.ifndef` (unset ⇒ byte-identical).
- `scratchpad/fire_envelope_oracle.sh` (CREATE, Task 8) — supervised build+dispatch+restore for the GPU gate.

Post-first-gate (full design): `occ_kernel_dsws.s` stagger phase-token (Task 9), `conv_apply` delta reconciliation (Task 10).

---

## Task 1: CPU model — compute-burst reserve/release + envelope invariant

**Files:**
- Modify: `dsws_ctrl_model.cpp` (add two helpers after `reserve_grow`, ~line 54)
- Test: `test_dsws_ctrl_model.cpp` (add asserts in `main`, before the final PASS print)

**Interfaces:**
- Consumes: `bool reserve_grow(std::atomic<uint32_t>& resv, uint32_t delta, uint32_t budget)` (existing, `dsws_ctrl_model.cpp:47`).
- Produces:
  - `void reserve_spin(std::atomic<uint32_t>& resv, uint32_t delta, uint32_t budget, uint64_t& spins)` — reserve `+delta`, spin-retry on over-budget, count backoffs in `spins`.
  - `void reserve_release(std::atomic<uint32_t>& resv, uint32_t delta)` — release `−delta` (models the post-shrink `lds_fetch_add VRESV_OFF, -Δ`).

- [ ] **Step 1: Write the failing test** — append to `test_dsws_ctrl_model.cpp` `main`, just before `printf("dsws_ctrl_model: ALL PASS\n");`:

```cpp
  // ---- envelope invariant: at most PEAK_CONC concurrent peaks fit; release frees a slot ----
  {
    const uint32_t VLEAN = 32, NFV = 112, D = NFV - VLEAN;   // Δ = 80
    const uint32_t WAVES = 8, PEAK_CONC = 2;
    const uint32_t BUDGET = WAVES * VLEAN + PEAK_CONC * D;    // 256 + 160 = 416
    std::atomic<uint32_t> resv{WAVES * VLEAN};               // init: everyone lean
    uint64_t spins = 0;
    reserve_spin(resv, D, BUDGET, spins); assert(spins == 0);   // 1st peak fits
    reserve_spin(resv, D, BUDGET, spins); assert(spins == 0);   // 2nd peak fits (== PEAK_CONC)
    assert(reserve_grow(resv, D, BUDGET) == false);             // 3rd over-budget -> rejected + undone
    assert(resv.load() == WAVES * VLEAN + PEAK_CONC * D);       // undo left counter exactly at 2 peaks
    reserve_release(resv, D);                                    // free one slot
    assert(reserve_grow(resv, D, BUDGET) == true);              // now the 3rd fits
    reserve_release(resv, D); reserve_release(resv, D);
    assert(resv.load() == WAVES * VLEAN);                       // conservation: back to all-lean
    printf("dsws_ctrl_model: envelope invariant OK\n");
  }
```

- [ ] **Step 2: Run to verify it fails** — Run: `cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ && g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/tcm && /tmp/tcm`
  Expected: FAIL — compile error `reserve_spin`/`reserve_release` not declared.

- [ ] **Step 3: Add the helpers** — in `dsws_ctrl_model.cpp`, immediately after the `reserve_grow` function (closing brace ~line 54):

```cpp
// Compute-burst reserve with spin-retry (models .Lcompute_reserve): reserve +delta against the
// sum-envelope; on over-budget, reserve_grow has already undone its add, so back off and retry.
// `spins` accumulates the backoff count (permit-starvation depth). Bounded when >=1 peak fits.
static inline void reserve_spin(std::atomic<uint32_t>& resv, uint32_t delta,
                                uint32_t budget, uint64_t& spins) {
    while (!reserve_grow(resv, delta, budget)) ++spins;
}
// Release a booked burst (models the post-shrink lds_fetch_add VRESV_OFF, -delta). Never fails.
static inline void reserve_release(std::atomic<uint32_t>& resv, uint32_t delta) {
    resv.fetch_sub(delta, std::memory_order_acq_rel);
}
```

- [ ] **Step 4: Run to verify it passes** — Run: `g++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o /tmp/tcm && /tmp/tcm`
  Expected: PASS — prints `envelope invariant OK` then `ALL PASS`.

- [ ] **Step 5: Commit** (only if kmbandy asked)

```bash
git add dsws_ctrl_model.cpp test_dsws_ctrl_model.cpp
git commit -m "feat(dsws): CPU model reserve_spin/reserve_release + envelope invariant test"
```

---

## Task 2: CPU model — forward-progress thread race

**Files:**
- Create: `test_dsws_envelope_race.cpp`

**Interfaces:**
- Consumes: `reserve_spin`, `reserve_release` (Task 1).
- Produces: standalone executable; exit 0 == no stall reproduced.

- [ ] **Step 1: Write the failing test** — create `test_dsws_envelope_race.cpp`:

```cpp
// Forward-progress race for the DSWS compute-burst sum-envelope (.Lcompute_reserve/grow/release).
// NCOMP threads each loop {claim rowblk -> reserve_spin(+Δ) -> bounded burst -> reserve_release(-Δ)}.
// A watchdog trips if any thread fails to finish its rowblks (models a permit-starvation hang on CPU
// what cannot be safely bisected on the compositor-attached gfx1201). Target: 0 stalls for PEAK_CONC>=1.
#include "dsws_ctrl_model.cpp"
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>

static constexpr uint32_t VLEAN = 32, NFV = 112, D = NFV - VLEAN;   // Δ = 80
static constexpr uint32_t WAVES = 8, NCOMP = 4;
static constexpr uint32_t ROWBLKS = 64;                              // rowblks each compute wave completes
static constexpr uint64_t SPIN_LIMIT = 50'000'000ull;

static bool run_once(uint32_t peak_conc) {
    const uint32_t budget = WAVES * VLEAN + peak_conc * D;
    std::atomic<uint32_t> resv{WAVES * VLEAN};
    std::atomic<uint32_t> done{0};
    std::atomic<bool> stalled{false};
    std::vector<std::thread> ts;
    for (uint32_t w = 0; w < NCOMP; ++w) {
        ts.emplace_back([&]{
            for (uint32_t r = 0; r < ROWBLKS && !stalled.load(); ++r) {
                uint64_t spins = 0;
                while (!reserve_grow(resv, D, budget)) {
                    if (++spins > SPIN_LIMIT) { stalled.store(true); return; }
                    std::this_thread::yield();
                }
                // bounded "burst": a few atomic touches, then release
                for (int k = 0; k < 8; ++k) done.fetch_add(0);
                reserve_release(resv, D);
            }
            done.fetch_add(1);
        });
    }
    for (auto& t : ts) t.join();
    bool ok = !stalled.load() && done.load() >= NCOMP && resv.load() == WAVES * VLEAN;
    if (!ok) printf("  STALL peak_conc=%u  resv=%u done=%u stalled=%d\n",
                    peak_conc, resv.load(), done.load(), (int)stalled.load());
    return !ok;
}

int main() {
    const int TRIALS = 200;
    int fails = 0;
    for (uint32_t pc = 1; pc <= 3; ++pc) {
        int f = 0;
        for (int i = 0; i < TRIALS; ++i) if (run_once(pc)) f++;
        printf("[peak_conc=%u] %d/%d trials stalled\n", pc, f, TRIALS);
        fails += f;
    }
    if (fails == 0) printf("dsws_envelope_race: NO STALL — envelope guarantees forward progress\n");
    else            printf("dsws_envelope_race: STALL REPRODUCED (%d)\n", fails);
    return fails ? 1 : 0;
}
```

- [ ] **Step 2: Run to verify it builds & passes** — Run: `g++ -std=c++17 -O2 -pthread test_dsws_envelope_race.cpp -o /tmp/ter && /tmp/ter; echo "exit=$?"`
  Expected: `[peak_conc=1] 0/200 … [peak_conc=3] 0/200`, `NO STALL`, `exit=0`. (If it stalls at `peak_conc=1`, the envelope math is wrong — stop and re-derive before touching asm.)

- [ ] **Step 3: Commit** (only if kmbandy asked)

```bash
git add test_dsws_envelope_race.cpp
git commit -m "test(dsws): forward-progress thread race for the compute-burst sum-envelope"
```

---

## Task 3: Kernel — new symbols + BUDGET envelope default + assemble guard

**Files:**
- Modify: `occ_kernel_dsws.s` (symbol block near `:125`; `BUDGET` block `:441`; guard `:447`)

**Interfaces:**
- Produces symbols: `DSWS2_ENVELOPE` (0), `PEAK_CONC` (2), `DSWS2_STAGGER` (0), `STAGGER_PERIOD` (`NCOMP`); envelope-mode `BUDGET` default.

- [ ] **Step 1: Add the symbols** — in `occ_kernel_dsws.s`, alongside the other `.ifndef` symbol defaults (after the `DSWS2_CONV`/`CONV_COOLDOWN` block, ~`:130`):

```asm
.ifndef DSWS2_ENVELOPE
  .set DSWS2_ENVELOPE, 0        // 1 = route the per-rowblk compute burst grow through the vgpr_reserved
.endif                          //     sum-envelope. 0 = HEAD (bare .Lcompute_grow) -> .text byte-identical.
.ifndef PEAK_CONC
  .set PEAK_CONC, 2             // concurrent compute peaks the budget admits (R3 sweep). Used iff ENVELOPE=1.
.endif
.ifndef DSWS2_STAGGER
  .set DSWS2_STAGGER, 0        // 1 = lock-free phase-token stagger (Task 9). 0 -> emergent envelope stagger.
.endif
.ifndef STAGGER_PERIOD
  .set STAGGER_PERIOD, NCOMP   // phase slots in the stagger ring (R3 sweep). Used iff STAGGER=1.
.endif
```

- [ ] **Step 2: Make BUDGET envelope-aware** — replace the `BUDGET` `.ifndef` block (`:441`):

```asm
.ifndef BUDGET
.if DSWS2_ENVELOPE
  .set BUDGET, (WAVES*VLEAN + PEAK_CONC*(NFV-VLEAN))   // rolling: lean floor + concurrent-peak headroom
.else
  .set BUDGET, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)     // static-fat (HEAD) — unchanged
.endif
.endif
```

- [ ] **Step 3: Add the forward-progress guard** — inside the existing `.if DSWS2_CONV` guard block (`:445-450`, next to the `WAVES*VLEAN > BUDGET` `.error`):

```asm
.if DSWS2_ENVELOPE
.if (WAVES*VLEAN + (NFV-VLEAN)) > BUDGET
  .error "ENVELOPE: BUDGET admits < 1 concurrent peak — forward progress impossible"
.endif
.endif
```

- [ ] **Step 4: Byte-identity gate** — assemble at the byte-identical config (`ENVELOPE` unset ⇒ 0) and confirm sha unchanged. Run:

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
L=/opt/rocm/llvm/bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
  -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=1 -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 \
  -Wa,-defsym,DSWS2_CONV=0 -c occ_kernel_dsws.s -o /tmp/bi.o \
&& "$L/llvm-objcopy" -O binary --only-section=.text /tmp/bi.o /tmp/bi.text \
&& echo "CONV=0: $(wc -c < /tmp/bi.text)B $(sha256sum /tmp/bi.text | cut -c1-8) (expect 4840B e5ec5e50)"
```
  Expected: `CONV=0: 4840B e5ec5e50`. If it differs, a symbol block emitted bytes — a `.set`/`.ifndef` must be inert; fix before proceeding.

- [ ] **Step 5: Commit** (only if kmbandy asked)

```bash
git add occ_kernel_dsws.s
git commit -m "feat(dsws): envelope symbols + BUDGET envelope-mode default + forward-progress guard"
```

---

## Task 4: Kernel — VRESV init inversion under ENVELOPE

**Files:**
- Modify: `occ_kernel_dsws.s` (`VRESV_OFF` seed, `:744`)

- [ ] **Step 1: Invert the init under envelope mode** — replace the single `lds_put VRESV_OFF, …` at `:744`:

```asm
.if DSWS2_ENVELOPE
    lds_put VRESV_OFF, (WAVES*VLEAN)                          // rolling: everyone lean; counter books peaks
.else
    lds_put VRESV_OFF, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)    // HEAD (unchanged)
.endif
```

- [ ] **Step 2: Byte-identity gate** — re-run the Task 3 Step 4 block (ENVELOPE unset). Expected unchanged: `CONV=0: 4840B e5ec5e50`. (The `.else` arm is the only one emitted when `ENVELOPE=0`, and it is textually identical to the original `lds_put`.)

- [ ] **Step 3: Commit** (only if kmbandy asked)

```bash
git add occ_kernel_dsws.s
git commit -m "feat(dsws): invert VRESV init to lean-floor under DSWS2_ENVELOPE"
```

---

## Task 5: Kernel — reserve/grow/release in the compute claim loop

**Files:**
- Modify: `occ_kernel_dsws.s` (`.Lcompute_claim`/`.Lcompute_grow` `:1151-1161`; `.Lcompute_shrink` `:1221-1224`)

**Interfaces:**
- Consumes: `reserve_try delta, won` macro (`:497`), `VRESV_OFF=52`, `NFV`/`VLEAN`, `SLEEPN`.
- Registers: `s54` = reserve `won`; `s62/s63` = `reserve_try` internal scratch (dead here — conv runs only at `.Lcompute_drained`). No new persistent reg.

- [ ] **Step 1: Add reserve-before-grow** — in the `.if DYNVGPR` block between `s_wait_storecnt 0x0` (`:1157`) and `.Lcompute_grow` (`:1158`), insert:

```asm
.if DSWS2_ENVELOPE
.Lcompute_reserve:
    reserve_try +(NFV-VLEAN), s54          // s54 = won (1 = booked, prev+Δ ≤ BUDGET)
    s_cmp_eq_u32 s54, 0
    s_cbranch_scc0 .Lcompute_grow          // won -> grow (SCC==0 when s54!=0)
    s_sleep SLEEPN                          // over budget -> back off AT LEAN (reserve_try already undid its add)
    s_branch .Lcompute_reserve
.endif
```
  (The existing `.Lcompute_grow:` label and its `s_alloc_vgpr NFV` SCC-retry stay exactly as-is, immediately below.)

- [ ] **Step 2: Add release-after-shrink** — in the `.if DYNVGPR` block, after the `.Lcompute_shrink` SCC-retry (`:1223`) and before `lds_inc ROWBLK_DONE_OFF` (`:1225`):

```asm
.if DSWS2_ENVELOPE
    lds_fetch_add s54, VRESV_OFF, -(NFV-VLEAN)   // release −Δ; wave is lean-32 here (v-temps ≤v15, OOR-safe)
.endif
```

- [ ] **Step 3: Byte-identity gate (ENVELOPE=0)** — re-run the Task 3 Step 4 block. Expected unchanged: `CONV=0: 4840B e5ec5e50`. Also verify `CONV=1/DIAG=0`:

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
L=/opt/rocm/llvm/bin
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
  -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 \
  -Wa,-defsym,DSWS2_CONV=1 -c occ_kernel_dsws.s -o /tmp/bi1.o \
&& "$L/llvm-objcopy" -O binary --only-section=.text /tmp/bi1.o /tmp/bi1.text \
&& echo "CONV=1/DIAG=0: $(sha256sum /tmp/bi1.text | cut -c1-8) (expect 049818c6)"
```
  Expected: `CONV=1/DIAG=0: 049818c6`.

- [ ] **Step 4: Assemble the ENVELOPE=1 bin (smoke)** — confirm the new path assembles clean and grows are ≤128:

```bash
"$L/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
  -Wa,-defsym,DSWS2=1 -Wa,-defsym,FM=2 -Wa,-defsym,FN=4 -Wa,-defsym,G=6 -Wa,-defsym,SEGK=64 \
  -Wa,-defsym,SAFEPROBE=1 -Wa,-defsym,DIAG=0 -Wa,-defsym,NCOMP=4 -Wa,-defsym,NAFEED=2 -Wa,-defsym,NBFEED=2 \
  -Wa,-defsym,DSWS2_CONV=1 -Wa,-defsym,DSWS2_ENVELOPE=1 -Wa,-defsym,PEAK_CONC=2 \
  -c occ_kernel_dsws.s -o /tmp/env.o && echo "ENVELOPE=1 assembles OK"
```
  Expected: `ENVELOPE=1 assembles OK` (no `.error`; guard passes since `336 ≤ 416`).

- [ ] **Step 5: Commit** (only if kmbandy asked)

```bash
git add occ_kernel_dsws.s
git commit -m "feat(dsws): envelope the per-rowblk compute burst grow (.Lcompute_reserve/release)"
```

---

## Task 6: Kernel — DIAG envelope telemetry

**Files:**
- Modify: `occ_kernel_dsws.s` (DIAG wedge-frame block `:893-931`; reserve/release sites for event counters)

**Interfaces:**
- Free occ slots (verified against used set `{24,28,40,44,60,76,84,88,104,108,112,116}`): occ[8]=32, occ[9]=36, occ[12]=48, occ[13]=52, occ[14]=56.

> **Scope decision (2026-07-02, implemented):** Task 6 landed the **claimer-side readback only** — instantaneous `vgpr_reserved` → occ[8] and `PEAK_CONC` → occ[14], inside the existing lane-0 DIAG wedge block (reuses `s52`). This is the headline diagnostic: a wedge with `occ[8]==BUDGET` is permit-starvation, anything below points elsewhere. It touches only the claimer's cold advance-gate poll — **not** the compute hot path — avoiding a fresh heisenbug (recall: adding stores to the compute poll loop perturbed timing and hid the original race). The per-wave reserve-spin / grow / release counters (occ[9]/[12]/[13]) are **deferred** to a follow-up: they require carrying scalars across the WMMA burst (clobber risk) or LDS accumulation (hot-path atomics), for resolution the `vgpr_reserved` readback already covers at the first gate. Add them only if the first GPU gate wedges ambiguously.

- [ ] **Step 1: Add a per-wave reserve-spin + event counters** — carry three scalars across the burst (choose from the compute-loop dead band; `s55`/`s56`/`s57` are free outside `.Lcompute_drained`). At `.Lcompute_reserve` increment a spin counter on each backoff; at grow and release increment event counters. In the reserve loop (Task 5 Step 1), under `.if DSWS2_ENVELOPE && DIAG` only, bump `s55` (reserve-spin) before `s_sleep`, and bump `s56` (grow events) after a win / `s57` (release events) at the release site. Keep these purely additive and DIAG-gated so `DIAG=0` stays byte-identical.

```asm
; in .Lcompute_reserve, before s_sleep SLEEPN:
.if DSWS2_ENVELOPE && DIAG
    s_add_u32 s55, s55, 1                 ; reserve-spin depth (saturating in practice)
.endif
; at the win fall-through (just before/after .Lcompute_grow's alloc), and at the release:
.if DSWS2_ENVELOPE && DIAG
    s_add_u32 s56, s56, 1                 ; (grow events)  — at grow
    s_add_u32 s57, s57, 1                 ; (release events) — at release
.endif
```

- [ ] **Step 2: Publish envelope telemetry in the wedge frame** — in the claimer DIAG block (`:912-929`, inside the lane-0 exec mask, alongside the existing `global_store_b32` markers), under `.if DSWS2_ENVELOPE && DIAG` add reads of `VRESV_OFF` (high-water via running max in a scalar) and stores of the envelope slots:

```asm
.if DSWS2_ENVELOPE && DIAG
    lds_get s52, VRESV_OFF
    v_mov_b32 v14, s52
    global_store_b32 v4, v14, s[0:1] offset:32 scope:SCOPE_DEV   // occ[8]  vgpr_reserved (peak Σ readback)
    v_mov_b32 v14, PEAK_CONC
    global_store_b32 v4, v14, s[0:1] offset:56 scope:SCOPE_DEV   // occ[14] PEAK_CONC echo (config readback)
.endif
```
  (Per-wave `s55`/`s56`/`s57` — reserve-spin / grow / release — are published to occ[9]=36 / occ[12]=48 / occ[13]=52 from the compute wave's own lane-0 path at drain, same idiom as `epoch_mark`.)

- [ ] **Step 3: Byte-identity gate (DIAG=0)** — re-run Task 3 Step 4 (CONV=0) and Task 5 Step 3 (CONV=1/DIAG=0). Expected unchanged: `4840B e5ec5e50` and `7920B 049818c6`. (All new stores are under `.if …&& DIAG`.)

- [ ] **Step 4: Assemble ENVELOPE=1/DIAG=1 (smoke)** — re-run Task 5 Step 4 with `-Wa,-defsym,DIAG=1`. Expected: assembles OK.

- [ ] **Step 5: Commit** (only if kmbandy asked)

```bash
git add occ_kernel_dsws.s
git commit -m "feat(dsws): DIAG envelope telemetry (vgpr_reserved HWM, reserve-spin, grow/release counts)"
```

---

## Task 7: build_dsws.sh passthrough + assemble matrix + RGA gate

**Files:**
- Modify: `build_dsws.sh` (`mk2`, add envelope defsym passthrough, defaults mirroring the in-file `.ifndef`)

- [ ] **Step 1: Add passthrough to `mk2`** — in `build_dsws.sh` `mk2` (the `clang` invocation, `:30-36`), append (mirroring the existing `DSWS2_FORCE`/`BUDGET` passthrough pattern so an unset env is byte-identical):

```bash
     -Wa,-defsym,DSWS2_ENVELOPE=${DSWS2_ENVELOPE:-0} -Wa,-defsym,PEAK_CONC=${PEAK_CONC:-2} \
     -Wa,-defsym,DSWS2_STAGGER=${DSWS2_STAGGER:-0} -Wa,-defsym,STAGGER_PERIOD=${STAGGER_PERIOD:-4} \
```
  Also make the `mk2` `budget` local respect envelope mode: when `DSWS2_ENVELOPE=1` and `$BUDGET` unset, default to `$(( 8*32 + ${PEAK_CONC:-2}*80 ))`; else the existing `$(( $1*112 + ($2+$3)*32 ))`.

- [ ] **Step 2: Assemble matrix** — build all four corners and confirm the byte-identity ones and that the envelope ones assemble:

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
DIAG=1 DSWS2_ENVELOPE=0 ./build_dsws.sh norga         # baseline path (byte-identical family)
DIAG=1 DSWS2_ENVELOPE=1 PEAK_CONC=2 ./build_dsws.sh norga
```
  Expected: both print `OK … occ_dsws2_4c2a2b_gd.bin`. (Note: `mk2`'s output bin name is fixed; the envelope build OVERWRITES it — re-run the ENVELOPE=0 build or restore the safe bin before any GPU work.)

- [ ] **Step 3: RGA 0-spill gate** — run the static spill analysis on the ENVELOPE=1 build (bar: 0 spill, max-live ≤ prior 84):

```bash
KSRC=occ_kernel_dsws.s DIAG=0 DSWS2=1 DSWS2_CONV=1 DSWS2_ENVELOPE=1 PEAK_CONC=2 \
  ./rga_check.sh env_4c2a2b FM=2 FN=4 G=6 SEGK=64 NCOMP=4 NAFEED=2 NBFEED=2 SAFEPROBE=1 2>&1 \
  | grep -E "gfx1201,|livereg|spill" || true
```
  Expected: `livereg`/spill line shows 0 spill; max-live ≤ 84. If spill > 0, the reserve/release added live pressure — investigate the scratch assignment before the GPU gate.

- [ ] **Step 4: Restore the safe CONV=0 bin** (footgun removal) — rebuild the safe bin so no envelope bin is left installed:

```bash
DIAG=1 DSWS2_ENVELOPE=0 ./build_dsws.sh norga
echo "installed: $(wc -c < occ_dsws2_4c2a2b_gd.bin)B $(sha256sum occ_dsws2_4c2a2b_gd.bin | cut -c1-8)"
```
  Note: the default `mk2` build is `CONV=0`-equivalent only if the bin matches the safe sha; if `build_dsws.sh` builds `DIAG=1` the bin won't be the 4840B CONV=0 — confirm the exact safe bin (`e5ec5e50`, 4840B, `DIAG=1 CONV=0`) is what's installed, matching `fire_wedge_diag.sh`'s restore.

- [ ] **Step 5: Commit** (only if kmbandy asked)

```bash
git add build_dsws.sh
git commit -m "build(dsws): mk2 passthrough for DSWS2_ENVELOPE/PEAK_CONC/STAGGER"
```

---

## Task 8: Supervised GPU gate — envelope ISOLATED at n_kseg=1 (OPERATOR-TRIGGERED)

> **This task is NOT auto-run. kmbandy greenlights it individually.** The agent prepares the script and STOPS. A hang/fault can MODE1-reset the desktop — accepted, but only with the safeguards intact.
>
> **REVISED after the 2026-07-02 brick.** The first attempt confounded the envelope with `n_kseg=2` (never validated) and page-faulted (OOB global address, gfxhub → MES unrecoverable → MODE1). The envelope is LDS-atomic only and cannot cause a global page fault; the fault was the untested split-K addressing. **This gate now isolates the envelope on proven-safe `n_kseg=1`** — the only change under test is the collision fix. `n_kseg>1` is validated separately (Task 8b) before any `n_kseg>1` dispatch.

**Files:**
- Create: `/home/kmbandy/dsws_gpu_logs/fire_envelope_oracle.sh` (REAL DISK — not scratchpad/tmpfs).

- [ ] **Step 1: Write the fire script** — build `ENVELOPE=1/CONV=1/DIAG=0 PEAK_CONC=2`, run the safeguarded dispatch at **`n_kseg=1`** (`DSWS2_NKSEG=1`), stream occ + dmesg to **real disk** (`~/dsws_gpu_logs/`), then UNCONDITIONALLY restore the safe `CONV=0` 4840B/`e5ec5e50` bin. Safeguards: quiesce gate ON, chunk watchdog 0.75s, `timeout 30`, restore-with-sha-check. (Script already written at `~/dsws_gpu_logs/fire_envelope_oracle.sh`, default `NKSEG=1 DIAG_RUN=0`.)

```bash
ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 \
  DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS2_NKSEG=1 \
  timeout 30 ./occ_dispatch --dsws2 4c2a2b 2>"$STREAM"   # $STREAM on /home (btrfs), NOT /tmp
```

- [ ] **Step 2: STOP — request greenlight.** Present the script path and exact config. Do not run it.

- [ ] **Step 3 (on greenlight): run + read out.** Expected: `ok=1536 bad=0`, `occ[0]=0`, dmesg delta 0. DIAG=0 first so a clean pass proves the *envelope* (not DIAG perturbation) fixed the collision. If it wedges (forward-progress hang, NOT a page fault): re-run `DIAG_RUN=1` to read `occ[8]` (`==416` ⇒ permit-starvation). If it *page-faults*: that's an addressing bug, not the envelope — investigate offline.

- [ ] **Step 4: Confirm safe bin restored** — `sha256sum occ_dsws2_4c2a2b_gd.bin | cut -c1-8` == `e5ec5e50`, size 4840B. Confirm the stream persisted to `~/dsws_gpu_logs/`.

## Task 8b: Validate n_kseg>1 split-K addressing (SEPARATE, before any n_kseg>1 dispatch)

> `n_kseg>1` addressing has never been validated. The 2026-07-02 page fault was in the `ksi`-dependent A/B base math (`A += ksi*SEGK`, `B += ksi*KSEG_STEPS*NT*256`) — and `SAFEPROBE` clamps the per-lane *vaddr* (v8/v9/v10) + `ti`, NOT the scalar segment base, so it did not catch it.

- [ ] Build a CPU model of the `n_kseg>1` (t,ksi) decode + A/B/C address computation; assert every computed global offset stays within the host-allocated buffer bounds (`occ_dispatch` `Amax`/`Bmax` at :1374/:1377) for all `sti ∈ [0,TOTAL_super)`. Only after that passes offline, gate one supervised `n_kseg=2` oracle dispatch (LOOSE tier) — ISOLATED from the envelope (i.e. `ENVELOPE=0` first, to prove the addressing alone).

---

## Task 9 (post-first-gate): Stagger phase-token — full design

> Build only after Task 8 is green. The envelope alone fixes the collision; the stagger is the occupancy/feed-fungibility refinement (R3). Whether explicit stagger beats emergent envelope staggering is an empirical sweep — this task adds the mechanism; the sweep (Task 8 Step 5 sweep, extended) decides if it earns its bytes.

**Files:**
- Modify: `occ_kernel_dsws.s` — `STAGGER_TOK_OFF=100` LDS word (extend the overlap guard `:184`); a `.Lcompute_phase` poll before `.Lcompute_reserve`, gated `.if DSWS2_STAGGER`.
- Modify: `dsws_ctrl_model.cpp` + a stagger-fairness test — model the phase ring keeps `≈PEAK_CONC` peaks active and phase-spread.

- [ ] Add `STAGGER_TOK_OFF=100` + guard; CPU-model the phase-ring fairness (TDD); transcribe the lock-free `lds_fetch_add STAGGER_TOK_OFF,1` mod `STAGGER_PERIOD` poll; byte-identity at `STAGGER=0`; assemble/RGA; (greenlit) GPU sweep `STAGGER ∈ {0,1} × STAGGER_PERIOD`.

---

## Task 10 (post-first-gate): Conversion-reserve reconciliation

> Land before conversions go live WITH the envelope. Under `DSWS2_ENVELOPE=1`, a feed↔compute conversion books zero peak VGPR (both roles lean at rest); the burst reserve is the only peak booking.

**Files:**
- Modify: `occ_kernel_dsws.s` — the four `conv_apply` grow call sites (`:1010/1019/1084/1093`) pass `delta=0` and `alloc_sz=32` under `.if DSWS2_ENVELOPE`; shrink sites (`:1253/1257/1267/1274`) already lean.
- Modify: `dsws_ctrl_model.cpp` + test — model that conversion is delta-neutral under envelope mode and the burst reserve carries all peak accounting; assert no double-booking.

- [ ] TDD the delta-neutral conversion in the CPU model; transcribe the `.if DSWS2_ENVELOPE` delta/alloc change; byte-identity at `ENVELOPE=0`; assemble/RGA; (greenlit) GPU gate with conversions LIVE + envelope at `n_kseg>1`.

---

## Self-Review

- **Spec coverage:** §2 accounting inversion → Task 4 (+ Task 1 model). §4.1 envelope reserve/release → Task 5 (+ Tasks 1-2 model). §4.2 split-K → Task 8 (`DSWS2_NKSEG=2`; no kernel change needed). §4.3 stagger → Task 9. §4.4 conversion reconciliation → Task 10. §4.5 DIAG telemetry → Task 6. §3 symbols/BUDGET → Task 3. §7 CPU-model → Tasks 1-2. §8 validation ladder → Tasks 1-2 (model) → 7 (assemble/RGA/byte-identity) → 8 (GPU) → 9 (sweep). §9 safety → Global Constraints + Task 8 gating. All covered.
- **Placeholder scan:** Task 6 Step 1 describes the counter placement in prose with the exact `.if …&& DIAG` guards and register choices (`s55/s56/s57`) rather than a single fixed diff, because the three bump sites are in distinct blocks; the code fragments and slot map are concrete. No TBD/TODO elsewhere.
- **Type consistency:** `reserve_spin(resv, delta, budget, spins)` and `reserve_release(resv, delta)` names/signatures match between Task 1 (definition), Task 1 test, and Task 2 race. `VRESV_OFF=52`, `PEAK_CONC`, `NFV`/`VLEAN`/`Δ`, occ slot offsets consistent across Tasks 3-8.
