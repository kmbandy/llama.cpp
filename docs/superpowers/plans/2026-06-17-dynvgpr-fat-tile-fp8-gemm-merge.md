# Configurable dyn-VGPR fat-tile fp8 WMMA GEMM (the "merge") — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (inline — this work needs supervised GPU stops and tight iteration against live ISA, so subagent-driven is NOT appropriate). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the existing PM4 micro-batch kernel into one fully `-defsym`-configurable gfx1201 fp8 GEMM whose inner loop matches hipcc's WMMA scheduling, so a dyn-VGPR fat accumulator tile can cut feed-instruction density and climb past the 155-TF feed wall toward 250 on wgrad shapes.

**Architecture:** The kernel already has the fat tile (FM×FN), dyn-VGPR grow/shrink, the persistent atomic queue, and a double-buffered A-direct/B-`global_load_tr` feed — but a *coarse* `s_wait_loadcnt WAITN` and only b64 loads. We replace the wait scheme with hipcc's fine-grained release ladder, add the feed/tile levers (`LDW`, `AFEED`, `BFEED`, `PIPE`, `TWM`, `TWN`, `LEANV`), and drive everything from the existing `occ_dispatch.cpp` PM4 harness, oracle-gated bit-exact at every config.

**Tech Stack:** Hand-written gfx1201 assembly (LLVM-MC `.macro`/`.rept`/`.set`/`-defsym`), `clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201`, `llvm-objdump`, libhsakmt KFD PM4 dispatch (C++17), CPU e4m3 oracle.

## Global Constraints

- **Vehicle, not rewrite:** all kernel work is in `spike/dvgpr_occ/occ_kernel_mbgemm.s`; all host/sweep work in `spike/dvgpr_occ/occ_dispatch.cpp`; all build-matrix work in `spike/dvgpr_occ/build.sh`. No new kernel files.
- **USER_SGPR map (fixed):** `s0:s1=occ s2:s3=A s4:s5=Bshuf s6:s7=C s8=KT s9=K(bytes/A-row) s10=NT*256 s11=TOTAL_TILES s12=NTILES_N_MASK s13=NTILES_N_LOG2 s14=FN*256 s15=TGID_X`. New levers must not reassign these; use scratch ≥ s33 / vector temps already in use.
- **Register map (current):** accumulators `v[32 : 32+FM*FN*8)`; `ABASE=32+FM*FN*8`; `BBASE=ABASE+FM*4`; `FATREGS=((BBASE+FN*4+15)&~15)`. Any new operand storage extends *above* `BBASE` and must be folded into `FATREGS`.
- **Shape rules:** K a multiple of 32 (KT even, unroll-by-2; tail pair peeled). Tile-grid column count a power of 2 (`ti→row=ti>>LOG2, col=ti&MASK`, no ISA divide). M,N divisible by the block tile.
- **dyn-VGPR pool cap (deadlock guard, KG `40cd2823`):** the host must size the persistent pool so `occ × FATREGS ≤ 1152` VGPR/SIMD; oversubscription lockstep-deadlocks and wedges the box.
- **Correctness gate:** every built config is bit-exact vs the chained-`wmma_ref` oracle before any perf number is trusted. `NOFEED=1` configs are perf-only (oracle expected to fail — skip the gate for them).
- **SUPERVISED rule:** every GPU run (oracle or perf) is raw PM4 on the gfx12 node and MUST be user-supervised; never auto-tear-down a hung queue. Tasks that launch the GPU are marked **[SUPERVISED]**.
- **Build RAM cap:** the C++ harness compile runs under the existing `run_capped` (`MemoryMax=4G`) in `build.sh`; never an uncapped host build.
- **Assemble command (per config):** `/opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,<LEVER>=<v>... -c occ_kernel_mbgemm.s -o <out>.o` then `llvm-objdump -d <out>.o`.

## File Structure

| File | Responsibility |
|---|---|
| `spike/dvgpr_occ/occ_kernel_mbgemm.s` | The kernel + all `-defsym` levers and macros (`LOADBUF`, `WMMABUF`, new `FEED`/`COMPUTE` wait scheme, LDS paths). |
| `spike/dvgpr_occ/occ_dispatch.cpp` | PM4 host harness: per-config FATREGS/pool/LDS sizing, oracle gate, `--merge` sweep + G2/G3 table, `--merge-dry` config print. |
| `spike/dvgpr_occ/build.sh` | RAM-capped `-defsym` build matrix `[1e]` for the merge lever set. |
| `spike/dvgpr_occ/RESULT_MERGE.md` | Outcome dossier (final task). |
| `/tmp/cg/gemm_fp8_levers-hip-amdgcn-amd-amdhsa-gfx1201.s` | **Reference only** (not edited): hipcc's winner inner loop, the transcription source for the wait ladder. |

> **Note on asm bodies:** assembly steps below give the exact instruction-level transformation, register map, and the `llvm-objdump` structural assertion that gates each change. The final instruction sequences are iterated against `llvm-mc` + the hipcc reference ISA during execution — that's the contract, not fabricated full listings.

---

### Task 1: Fine-grained `s_wait_loadcnt` release ladder (the codegen fix)

**Files:**
- Modify: `spike/dvgpr_occ/occ_kernel_mbgemm.s` (the `.Lkloop`/`.Ltail` body, lines ~225–242; `WMMABUF` macro lines ~71–81)

**Interfaces:**
- Consumes: existing `LOADBUF b` / `WMMABUF b` macros, `WAITN=FM+FN`, accumulators at `v[32+(mi*FN+ni)*8 …]`.
- Produces: a `WMMABUF_WAIT b` macro that interleaves each WMMA with the precise `s_wait_loadcnt <n>` that releases exactly that fragment's load, mirroring hipcc's `0x6→0x0` decrement; the hot loop stays unroll-by-2.

- [ ] **Step 1: Read hipcc's reference ladder.** `grep -n -B1 v_wmma /tmp/cg/gemm_fp8_levers-hip-amdgcn-amd-amdhsa-gfx1201.s | sed -n '1,60p'` — confirm the pattern: WMMAs back-to-back, each B-dependent one preceded by a decrementing `s_wait_loadcnt`, A-from-LDS WMMAs ungated. This is the target shape.

- [ ] **Step 2: Add `WMMABUF_WAIT b` macro.** In `occ_kernel_mbgemm.s` after `WMMABUF` (line 81), add a macro that emits the FM×FN WMMAs in the same (mi,ni) order but, for each `ni` fragment, precedes its first use with `s_wait_loadcnt (FN-1-ni)` so B-fragment `ni` is awaited exactly when needed (A frags already resident via the prior buffer). Outstanding-count math: with both buffers' B loads in flight, the ni-th of the current buffer lands at count `WAITN - 1 - ni` down to the A floor.

- [ ] **Step 3: Swap the hot loop to the ladder.** Replace the coarse `s_wait_loadcnt WAITN; WMMABUF 0` / `… WMMABUF 1` pairs in `.Lkloop` (lines ~227–231) and `.Ltail` (lines ~238–241) with `WMMABUF_WAIT 0` / `WMMABUF_WAIT 1` (the macro now carries its own per-fragment waits). Keep `LOADBUF 1` / `LOADBUF 0` prefetch placement unchanged.

- [ ] **Step 4: Assemble FM=4/FN=4 and FM=2/FN=2.** Run:
  `for s in "FM=4 FN=4" "FM=2 FN=2"; do /opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 $(printf -- '-Wa,-defsym,%s ' $s) -Wa,-defsym,DYNVGPR=0 -c occ_kernel_mbgemm.s -o /tmp/k.o && echo "$s OK"; done`
  Expected: both print `OK` (clean assemble).

- [ ] **Step 5: Structural assertion (the test).** `llvm-objdump -d /tmp/k.o | awk '/<occ_kernel>:/,/s_endpgm/' > /tmp/k.dis`; assert (a) a *descending* run of `s_wait_loadcnt` immediates appears interleaved in the WMMA region, and (b) for FM=4/FN=4 there is a back-to-back run of ≥16 `v_wmma` with no `s_branch` between them:
  `grep -c v_wmma /tmp/k.dis` (expect 32+ in the unrolled body) and `grep -n 's_wait_loadcnt\|v_wmma\|s_branch' /tmp/k.dis | sed -n '1,40p'` (expect the ladder, no branch inside the WMMA run).
  Expected: ladder present, no intra-run branch. If absent, the macro isn't emitting per-fragment waits — fix and re-run Step 4.

- [ ] **Step 6: Commit.**
  `git add spike/dvgpr_occ/occ_kernel_mbgemm.s && git commit -m "MAD-305 merge T1: fine-grained s_wait_loadcnt release ladder (hipcc-transcribed)"`

---

### Task 2 [SUPERVISED]: G2 go/no-go — FM=4/FN=4 oracle + hipcc-parity perf

**Files:**
- Modify: `spike/dvgpr_occ/build.sh` (add the merge config to the build matrix)
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (ensure `run_mbgemm` reports TF, %ceil, WMMA/cyc for the 4×4 config; no new lever yet)

**Interfaces:**
- Consumes: `WMMABUF_WAIT` from Task 1; existing `run_mbgemm()` oracle + timing path.
- Produces: a printed G2 line `merge 4x4 static  err=… TF=… (…%) … WMMA/cyc` used as the go/no-go.

- [ ] **Step 1: Add the build config.** In `build.sh`, add a `[1e] merge` block assembling `occ_mbgemm_merge_4x4_d{0,1}.bin` via `-Wa,-defsym,FM=4 -Wa,-defsym,FN=4 -Wa,-defsym,DYNVGPR={0,1}` (static fits: FATREGS=192 ≤ 256). Mirror the `[1d]` objcopy-to-`.bin` pattern.

- [ ] **Step 2: Build.** `cd spike/dvgpr_occ && MEMMAX=4G ./build.sh 2>&1 | tail -20`. Expected: the two `occ_mbgemm_merge_4x4_*.bin` byte sizes print; `test_oracle` runs green; `occ_dispatch` links.

- [ ] **Step 3 [SUPERVISED GPU]: oracle gate.** With the user present: `timeout 60 ./occ_dispatch --mbgemm` on the 4×4 merge config at the correctness shape. Expected: `okTiles N/N` (bit-exact, err < gate). If it fails, the ladder mis-counts a wait — return to Task 1 Step 2 (wrong `s_wait_loadcnt` index), do not proceed.

- [ ] **Step 4 [SUPERVISED GPU]: G2 perf.** Run the 4×4 merge perf at the wgrad shape (`4096²×K16384`). Expected: **≥ ~147 TF (≥ ~50%, within ~5% of hipcc's 155)**. This proves the hand-asm inner loop matches hipcc *before* the fat tile.

- [ ] **Step 5: G2 verdict.** If ≥147 TF: G2 PASSED — proceed. If short: the gap is pure scheduling — iterate the ladder/issue-order in Task 1 (waitcnt counts, mi/ni order) and re-run Steps 3–4. After 3 failed macro-tunes, STOP and escalate (the transcription premise may be wrong); do not bolt on the fat tile over a broken loop.

- [ ] **Step 6: Commit.**
  `git add spike/dvgpr_occ/build.sh spike/dvgpr_occ/occ_dispatch.cpp && git commit -m "MAD-305 merge T2: G2 gate — hand-asm 4x4 reaches hipcc parity (NNN TF), oracle bit-exact"`

---

### Task 3: `LDW` wide-load lever (b64 / b128)

**Files:**
- Modify: `spike/dvgpr_occ/occ_kernel_mbgemm.s` (`LOADBUF` macro, `ABASE`/`BBASE` strides, `WAITN`)
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (`LDW` defsym plumbing)

**Interfaces:**
- Consumes: `LOADBUF b`, `WMMABUF_WAIT b`.
- Produces: `LDW ∈ {64,128}`; at `LDW=128`, one `global_load_b128`/`global_load_tr_b128` fetches **two K-steps** of a fragment, halving the load-instruction count; the unroll consumes both halves before reloading.

- [ ] **Step 1: Add the lever.** In `occ_kernel_mbgemm.s` add `.ifndef LDW / .set LDW,64 / .endif`. Gate `LOADBUF` on `LDW`: `LDW=64` keeps the current `global_load_b64`/`global_load_tr_b64`; `LDW=128` emits `_b128` variants writing a 4-VGPR fragment pair (two K-steps), and the fragment-VGPR stride and `ABASE/BBASE` widen accordingly (fold into `FATREGS`). Halve the saddr advance frequency (one advance per two K-steps).

- [ ] **Step 2: Adjust the wait ladder for the wider grain.** With `LDW=128`, each load covers two K-steps, so `WAITN` and the `WMMABUF_WAIT` decrement counts halve. Recompute and emit the `LDW`-correct ladder.

- [ ] **Step 3: Assemble both widths.** `for w in 64 128; do /opt/rocm/llvm/bin/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -Wa,-defsym,FM=4 -Wa,-defsym,FN=4 -Wa,-defsym,LDW=$w -Wa,-defsym,DYNVGPR=0 -c occ_kernel_mbgemm.s -o /tmp/k_$w.o && echo "LDW=$w OK"; done`. Expected: both `OK`.

- [ ] **Step 4: Structural assertion (the test).** `llvm-objdump -d /tmp/k_128.o | grep -c 'global_load.*b128'` > 0 and the total `global_load` count at `LDW=128` is ≈ half of `LDW=64`:
  `for w in 64 128; do echo -n "LDW=$w loads: "; llvm-objdump -d /tmp/k_$w.o | grep -c global_load; done`. Expected: the 128 count ≈ ½ the 64 count.

- [ ] **Step 5: Commit.**
  `git add spike/dvgpr_occ/occ_kernel_mbgemm.s spike/dvgpr_occ/occ_dispatch.cpp && git commit -m "MAD-305 merge T3: LDW b128 wide-load lever (halves load-instruction count)"`

---

### Task 4 [SUPERVISED]: fat-tile + wide-load climb (FM×FN × LDW dyn-VGPR sweep)

**Files:**
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (`--merge` sweep over FM×FN × LDW, per-config FATREGS + pool cap + oracle gate + TF/%/WMMA-cyc row)
- Modify: `spike/dvgpr_occ/build.sh` (build FM×FN ∈ {4×4,6×6,8×8} × LDW ∈ {64,128} × DYNVGPR=1)

**Interfaces:**
- Consumes: Task 1–3 kernel; `FATREGS` formula; existing pool-cap logic `pool = (1152*?)/FATREGS`.
- Produces: a `--merge` results table; the best (tile,LDW) config recorded for Task 8.

- [ ] **Step 1: Wire the sweep.** In `occ_dispatch.cpp` `--merge`, loop `(FM,FN) ∈ {(4,4),(6,6),(8,8)}` × `LDW ∈ {64,128}`, compute `FATREGS` per the kernel formula, set pool `occ` so `occ*FATREGS ≤ 1152` (deadlock guard), oracle-gate, then time at the wgrad shape; print `merge FMxFN LDWw  FATREGS=… occ=…  err=…  TF=… (…%) … WMMA/cyc`.

- [ ] **Step 2: Build the matrix.** `cd spike/dvgpr_occ && MEMMAX=4G ./build.sh 2>&1 | tail -30`. Expected: all `occ_mbgemm_merge_*` bins emit; `occ_dispatch` links.

- [ ] **Step 3 [SUPERVISED GPU]: oracle-gate every config.** Run `--merge`; confirm each non-NOFEED config prints bit-exact `okTiles N/N`. A config that fails the oracle is dropped from the perf read (note it; likely a FATREGS/stride bug for that tile).

- [ ] **Step 4 [SUPERVISED GPU]: read the climb.** Record TF/%/WMMA-cyc per config. Expected signal: 8×8 (64 chains, 8× reuse) + LDW=128 beats the 4×4 baseline and clears 155 TF; how far toward 250 is the result. Watch for any pool-cap config that hangs — if so, the occ×FATREGS cap is too loose; tighten and STOP.

- [ ] **Step 5: Commit.**
  `git add spike/dvgpr_occ/occ_dispatch.cpp spike/dvgpr_occ/build.sh && git commit -m "MAD-305 merge T4: FMxFN x LDW dyn-VGPR sweep — fat-tile climb past 155 (best NNN TF)"`

---

### Task 5: `AFEED` LDS-A staging + `TWM`/`TWN` cooperating waves

**Files:**
- Modify: `spike/dvgpr_occ/occ_kernel_mbgemm.s` (LDS reservation, cooperative A fill, `ds_read` A path under `AFEED=0`; `TWM`/`TWN` wave-grid indexing)
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (LDS bytes per block, block = `TWM*TWN*32` threads)

**Interfaces:**
- Consumes: Task 1–3 kernel; the queue claims one *block* tile, `TWM*TWN` waves cooperate on it.
- Produces: `AFEED ∈ {0=LDS,1=direct}`, `TWM`,`TWN`; with `AFEED=0` the A tile is filled once into LDS and `ds_read` by all `TWM*TWN` waves (cross-wave reuse — the HIP winner's path).

- [ ] **Step 1: Add levers + LDS.** Add `.ifndef AFEED/TWM/TWN` defaults (`AFEED=1,TWM=1,TWN=1` = current single-wave behavior, no LDS). Under `AFEED=0`, reserve `TBM*TBK` LDS bytes, add a coalesced cooperative fill keyed on `TGID`/wave id, and switch the A operand of `WMMABUF_WAIT` to `ds_read_b64` from the staged tile.

- [ ] **Step 2: Default-path regression assert.** Assemble `AFEED=1,TWM=1,TWN=1` and `llvm-objdump -d` — assert byte-identical text to the Task 3 kernel (the new levers must be inert at defaults):
  `cmp <(llvm-objdump -d /tmp/k_64.o) <(llvm-objdump -d /tmp/k_afeed_default.o)`. Expected: identical.

- [ ] **Step 3: Assemble the LDS path.** Assemble `AFEED=0,TWM=2,TWN=2,FM=4,FN=4`. Expected: clean assemble; `llvm-objdump -d` shows `ds_read` for A and a `ds_store`/cooperative fill in the setup region.

- [ ] **Step 4: Commit.**
  `git add spike/dvgpr_occ/occ_kernel_mbgemm.s spike/dvgpr_occ/occ_dispatch.cpp && git commit -m "MAD-305 merge T5: AFEED LDS-A staging + TWM/TWN cooperating waves (default-inert)"`

---

### Task 6: `BFEED` LDS-B reuse lever

**Files:**
- Modify: `spike/dvgpr_occ/occ_kernel_mbgemm.s` (`BFEED=1` stages B in LDS, reused across the wave grid instead of per-fragment `global_load_tr`)
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (extra LDS bytes for B under `BFEED=1`)

**Interfaces:**
- Consumes: Task 5 LDS/wave-grid machinery.
- Produces: `BFEED ∈ {0=global_load_tr,1=LDS}`; default `0` = current behavior.

- [ ] **Step 1: Add the lever.** Add `.ifndef BFEED/.set BFEED,0`. Under `BFEED=1`, cooperatively stage the B col-tile into LDS once per K-tile and `ds_read` fragments in `WMMABUF_WAIT` (B then has no `s_wait_loadcnt`; gate on `ds`/`s_wait_dscnt` instead).

- [ ] **Step 2: Default-inert assert.** Assemble `BFEED=0` and assert text-identical to Task 5's `BFEED`-absent build. Expected: identical.

- [ ] **Step 3: Assemble + structural check.** Assemble `BFEED=1,AFEED=0,TWM=2,TWN=2`; assert B operands are `ds_read` (no `global_load_tr` in the K-loop):
  `llvm-objdump -d /tmp/k_bfeed.o | awk '/Lkloop/,/Lbatch_end/' | grep -c global_load_tr`. Expected: 0.

- [ ] **Step 4: Commit.**
  `git add spike/dvgpr_occ/occ_kernel_mbgemm.s spike/dvgpr_occ/occ_dispatch.cpp && git commit -m "MAD-305 merge T6: BFEED LDS-B reuse lever (default-inert)"`

---

### Task 7: `PIPE` prefetch depth + `LEANV` explicit lean reserve

**Files:**
- Modify: `spike/dvgpr_occ/occ_kernel_mbgemm.s` (generalize the fixed double-buffer to `PIPE ∈ {0,1,2}` stages; `LEANV` sets the shrink target)
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (`PIPE`/`LEANV` plumbing, operand-buffer count = `PIPE+1` into `FATREGS`)

**Interfaces:**
- Consumes: Task 1 ladder, Task 3 widths.
- Produces: `PIPE ∈ {0,1,2}` (0 = no prefetch, 1 = current double-buffer, 2 = triple), `LEANV` (shrink target VGPR, default 32).

- [ ] **Step 1: Parameterize buffers.** Replace the hard `2`-buffer indexing (`\b*FM*2`, the `LOADBUF 1`/`LOADBUF 0` alternation, the `s_alloc_vgpr 32` shrink) with `PIPE+1` rotating buffers and `s_alloc_vgpr LEANV`. `PIPE=0` collapses to load-then-compute (no overlap); `PIPE=1` reproduces today's double-buffer (regression anchor); `PIPE=2` adds a third stage.

- [ ] **Step 2: Regression assert at PIPE=1, LEANV=32.** Assemble and assert text-identical to the Task 6 default build. Expected: identical (today's behavior is exactly `PIPE=1,LEANV=32`).

- [ ] **Step 3: Assemble PIPE∈{0,2}.** Assemble both; `llvm-objdump` buffer-count sanity: `PIPE=0` has one operand buffer region, `PIPE=2` three. Expected: clean assemble, buffer regions match.

- [ ] **Step 4: Commit.**
  `git add spike/dvgpr_occ/occ_kernel_mbgemm.s spike/dvgpr_occ/occ_dispatch.cpp && git commit -m "MAD-305 merge T7: PIPE prefetch-depth + LEANV lean-reserve levers (PIPE=1 == prior double-buffer)"`

---

### Task 8 [SUPERVISED]: G3 — full lever sweep on wgrad, pick the winner

**Files:**
- Modify: `spike/dvgpr_occ/occ_dispatch.cpp` (`--merge` extended to the full lever surface + both wgrad shapes; emit a ranked table)
- Modify: `spike/dvgpr_occ/build.sh` (build the chosen sweep cells)

**Interfaces:**
- Consumes: all levers (Tasks 1–7).
- Produces: the winning config (FM×FN, LDW, AFEED, BFEED, PIPE, TWM/TWN) and its %-of-peak; the G3 verdict vs 155 / 250.

- [ ] **Step 1: Define the sweep.** In `--merge`, sweep a YAGNI-bounded grid centered on the Task 4 best tile: `LDW∈{64,128} × {AFEED0/BFEED0, AFEED0/BFEED1} × PIPE∈{1,2}` on shapes `4096²×K16384` and `4096×14336×K16384`. Oracle-gate each; rank by TF.

- [ ] **Step 2: Build.** `MEMMAX=4G ./build.sh 2>&1 | tail -40`. Expected: cells build; harness links.

- [ ] **Step 3 [SUPERVISED GPU]: run + rank.** Run `--merge`; confirm bit-exact per config; capture the ranked TF/% table for both shapes. Expected: a clear winner; record whether it clears 155 and how close to 250.

- [ ] **Step 4: Verdict.** State the outcome plainly: winning config + TF + %-of-peak per shape, vs the 155 baseline and 250 bar. If it cleared 155, the merge thesis is confirmed; if it stalled, attribute via a `PROFILE=1` run of the winner (feed vs compute vs grow/shrink) and record the bound.

- [ ] **Step 5: Commit.**
  `git add spike/dvgpr_occ/occ_dispatch.cpp spike/dvgpr_occ/build.sh && git commit -m "MAD-305 merge T8: G3 full lever sweep on wgrad — winner <cfg> at NNN TF (NN%)"`

---

### Task 9: Record outcome

**Files:**
- Create: `spike/dvgpr_occ/RESULT_MERGE.md`

- [ ] **Step 1: Write the dossier.** `RESULT_MERGE.md`: question → vehicle (the merge) → the G2 parity result → the G3 climb table (both wgrad shapes) → winning config + %-of-peak → verdict vs 155/250 → (if stalled) the `PROFILE` attribution and what the remaining gap is (silicon vs hand-asm vs feed). Mirror `RESULT_MBGEMM.md`'s structure.

- [ ] **Step 2: Update KG + Jira + task board.** `mneme_write` (type=project) the merge outcome linking `2601d691`, `f50237e2`; comment MAD-305 with the G2/G3 numbers; mark the merge build task completed.

- [ ] **Step 3: Commit.**
  `git add spike/dvgpr_occ/RESULT_MERGE.md && git commit -m "MAD-305 merge T9: record outcome (RESULT_MERGE + KG + Jira)"`

---

## Self-Review

**Spec coverage:** vehicle (T1–T8 all in `occ_kernel_mbgemm.s`/`occ_dispatch.cpp`/`build.sh` ✓); hipcc-transcribed inner loop (T1 ✓); full lever set — FM/FN/TWM/TWN/TBK (T2/T4/T5; TBK already a kernel param), AFEED/BFEED/LDW/PIPE (T5/T6/T3/T7), DYNVGPR/LEANV/BATCH (existing + T7; BATCH already a kernel param), NOFEED/PROFILE (existing) ✓; G1 oracle gate (every SUPERVISED task ✓); G2 (T2 ✓); G3 (T8 ✓); wgrad shapes (T4/T8 ✓); pool-cap deadlock guard (Global + T4 ✓); configurability↔codegen tension (T2 Step 5 macro-tune loop ✓). No spec section unmapped.

**Placeholder scan:** no TBD/TODO; every asm task carries an exact assemble command + `llvm-objdump` assertion; the one explicit deferral (final instruction listings iterated against `llvm-mc`) is called out in the File Structure note as the structural-test contract, not a hidden gap.

**Type/name consistency:** `FATREGS`, `ABASE`, `BBASE`, `WAITN`, `WMMABUF_WAIT`, `LDW`, `AFEED`, `BFEED`, `PIPE`, `LEANV`, `TWM`, `TWN` used identically across tasks; USER_SGPR map fixed in Global Constraints and unviolated; pool cap `occ*FATREGS ≤ 1152` stated once and referenced in T4.
