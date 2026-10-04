# RDNA4 dyn-VGPR GEMM occupancy de-risk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (INLINE — not subagent-driven, per the user). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cheaply decide go/no-go on the Phase-4 hand-written ASM fp8 GEMM by measuring, on gfx1201 silicon, whether the dyn-VGPR occupancy lever converts to WMMA throughput (Prong 1) and survives a GEMM-shaped long fat phase (Prong 2), plus a real-kernel peak-vs-steady cross-check.

**Architecture:** Extend the proven Phase-2 spike (`spike/dvgpr_occ/`): turn the occupancy-counter kernel into a host-timed WMMA-throughput chain (parameterized accumulator count `NACC` + runtime loop depth `KDEPTH`), dispatched via the same raw-PM4 vehicle. A separate normal-launch HIP file does the unsupervised real-kernel cross-check.

**Tech Stack:** hand-written gfx1201 wave32 ISA (clang integrated assembler), libhsakmt raw-PM4 KFD dispatch, HIP/rocWMMA (cross-check only), CPU fp8 e4m3 oracle.

**Working dir (ALWAYS cd here with absolute path — shell cwd drifts):**
`/home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ`

**Standing constraints:** RAM-capped builds only (`systemd-run --user --scope -p MemoryMax`). Per-task commits authorized. Supervised GPU runs (Tasks 6–7) = raw PM4 on the gfx12 node → **STOP before each for explicit go.** Lift every new encoding from a compiler seed + verify vs `llvm-objdump` — never guess (wrong bytes = hang).

---

## File Structure

| File | Responsibility |
|---|---|
| `accum_seed.hip` (new, throwaway) | Source of the two new gfx12 encodings: accumulating fp8 WMMA (`srcC`=reg) + scalar `s_load_b32`/`s_wait_kmcnt`. |
| `occ_kernel.s` (rewrite) | Timed WMMA-throughput chain. `-defsym NACC={8,16} DYNVGPR={0,1}`; KDEPTH read at runtime from `occ[8]`. Lean counter (v0–v4) → `s_alloc` → load A/B once → zero accumulators → KDEPTH×NACC accumulating WMMAs → store acc0 tile → shrink. |
| `occ_dispatch.cpp` (extend) | Host-timed throughput driver. `--prong1` (static occupancy sweep), `--prong2` (dyn-vs-static heavy KDEPTH sweep), default = Phase-2 correctness A/B. Sets `occ[8]=KDEPTH`, times submit→fence, computes TFLOPS, oracle-checks the KDEPTH=1 pass. |
| `build.sh` (extend) | Assemble the 4 variants (NACC×DYNVGPR) + RAM-capped harness build. |
| `gemm_occ_pad.hip` (new) | Approach-2 cross-check (normal HIP launch, unsupervised): `gemm_fp8_tiled` occupancy-down sweep → real-kernel TFLOPS vs occupancy. |
| `RESULT_P3.md` (new) | Curves, ratios, projected GEMM win, GREEN-1/GREEN-2 verdict, reproduce. |

---

## Task 1: Lift the two new gfx12 encodings from a compiler seed

**Files:** Create `accum_seed.hip` (throwaway, gitignored via `wmma_seed-*`/`*.hip` pattern — verify).

- [ ] **Step 1: Write the seed**

```cpp
// accum_seed.hip — lift accumulating fp8 WMMA (srcC=reg) + scalar load encodings.
#include <hip/hip_runtime.h>
typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));
extern "C" __global__ void aseed(const int* __restrict__ ab, float* __restrict__ out, const unsigned* __restrict__ kp) {
    v2i32 A = {ab[0], ab[1]}, B = {ab[2], ab[3]};
    v8f32 acc = {0,0,0,0,0,0,0,0};
    unsigned k = kp[2];                       // scalar load from a pointer + offset (KDEPTH proxy)
    for (unsigned i = 0; i < k; ++i)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A, B, acc);  // srcC = acc (accumulate)
    for (int j = 0; j < 8; ++j) out[threadIdx.x*8 + j] = acc[j];
}
```

- [ ] **Step 2: Compile to device asm**

Run: `/opt/rocm/llvm/bin/clang++ --offload-arch=gfx1201 --save-temps -S -c accum_seed.hip -o /dev/null 2>/dev/null; ls accum_seed-hip-amdgcn-amd-amdhsa-gfx1201.s`
Expected: the `.s` file exists.

- [ ] **Step 3: Extract the encodings**

Run: `grep -nE 'v_wmma_f32_16x16x16_fp8_fp8|s_load_b32|s_wait_kmcnt|s_load_b' accum_seed-hip-amdgcn-amd-amdhsa-gfx1201.s | head`
Expected: an accumulating WMMA line of the form `v_wmma_f32_16x16x16_fp8_fp8 v[A:A+7], v[..], v[..], v[A:A+7]` (dst == srcC), a `s_load_b32 sN, s[..], 0x..` and a `s_wait_kmcnt 0x0`. **Record the exact mnemonics/operand order** for Task 2.

- [ ] **Step 4: Commit the recorded encodings (as a comment block in the seed)**

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
git add accum_seed.hip
git commit -m "dvgpr_occ P3-T1: lift accumulating WMMA + scalar-load gfx12 encodings"
```

---

## Task 2: Rewrite `occ_kernel.s` as the timed throughput chain

**Files:** Modify `occ_kernel.s` (full rewrite, keeping the proven lane-0 atomic counter + dyn/static envelope).

- [ ] **Step 1: Write the kernel** (register map: v0=lane, v2/v3=atomic, v4=0, v6=lane*8, v7=lane*32; A=v8:v9, B=v10:v11; acc_k = v[16+8k:23+8k]; s8=exec save, s9=loop counter; KDEPTH from `occ[8]`). Accumulating WMMA + scalar-load mnemonics from Task 1. Accumulators and WMMA list written explicitly for 16, upper 8 guarded by `.if NACC > 8`.

Key structure (full file written at execution time):
```asm
.text
.globl occ_kernel
.p2align 8
.type occ_kernel,@function
occ_kernel:
    v_mov_b32 v4, 0
    // lane-0 admission counter (occupancy label) — unchanged from Phase 2
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
    s_wait_loadcnt 0x0
    v_add_nc_u32 v3, v3, 1
    global_atomic_max_u32 v4, v3, s[0:1] offset:4 scope:SCOPE_DEV
.Lafter_inc:
    s_mov_b32 exec_lo, s8
    // KDEPTH <- occ[8]  (scalar load; encoding from Task 1)
    s_load_b32 s9, s[0:1], 0x8
    s_wait_kmcnt 0x0
.if DYNVGPR
    s_alloc_vgpr FATREGS          // FATREGS = .if NACC>8 ? 144 : 80
.endif
    v_lshlrev_b32 v6, 3, v0
    global_load_b64 v[8:9],  v6, s[2:3]
    global_load_b64 v[10:11], v6, s[2:3] offset:256
    s_wait_loadcnt 0x0
    // zero accumulators acc0..acc{NACC-1}  (v_mov_b32 each of NACC*8 regs)  -- explicit, .if-guarded
    // ... v_mov_b32 v16,0 ... v79,0 ; .if NACC>8 ... v80,0 ... v143,0 ; .endif
.Lkloop:
    v_wmma_f32_16x16x16_fp8_fp8 v[16:23], v[8:9], v[10:11], v[16:23]
    v_wmma_f32_16x16x16_fp8_fp8 v[24:31], v[8:9], v[10:11], v[24:31]
    // ... through acc7 (v[72:79]) ...
.if NACC > 8
    v_wmma_f32_16x16x16_fp8_fp8 v[80:87], v[8:9], v[10:11], v[80:87]
    // ... through acc15 (v[136:143]) ...
.endif
    s_sub_u32 s9, s9, 1
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lkloop
    // store acc0 tile for the oracle (one 16x16 tile = v[16:23])
    v_lshlrev_b32 v7, 5, v0
    global_store_b128 v7, v[16:19], s[4:5]
    global_store_b128 v7, v[20:23], s[4:5] offset:16
    s_wait_storecnt 0x0
.if DYNVGPR
    s_alloc_vgpr 32
.endif
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v4, v2, s[0:1] scope:SCOPE_DEV
.Ldone:
    s_mov_b32 exec_lo, s8
    s_endpgm
    .size occ_kernel, .-occ_kernel
```
With `.ifndef NACC / .set NACC,16 / .endif`, `.ifndef DYNVGPR / .set DYNVGPR,0 / .endif`, and `.if NACC > 8 / .set FATREGS,144 / .else / .set FATREGS,80 / .endif`.

- [ ] **Step 2: Assemble all four variants**

Run (from working dir):
```bash
L=/opt/rocm/llvm/bin
for nacc in 8 16; do for dv in 0 1; do
  $L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 \
    -Wa,-defsym,DYNVGPR=$dv -Wa,-defsym,NACC=$nacc -c occ_kernel.s -o occ_n${nacc}_d${dv}.o &&
  $L/llvm-objcopy -O binary --only-section=.text occ_n${nacc}_d${dv}.o occ_n${nacc}_d${dv}.bin; done; done
ls -l occ_n*_d*.bin
```
Expected: 4 non-empty `.bin` files, no assembler errors.

- [ ] **Step 3: Disassemble one variant to verify the hot loop is real WMMAs (not garbage)**

Run: `$L/llvm-objdump -d --mcpu=gfx1201 occ_n16_d1.o | grep -cE 'v_wmma_f32_16x16x16_fp8_fp8'`
Expected: `16` (16 accumulating WMMAs in the loop body for NACC=16).

- [ ] **Step 4: Commit**

```bash
git add occ_kernel.s
git commit -m "dvgpr_occ P3-T2: occ_kernel.s -> timed WMMA-throughput chain (NACC/DYNVGPR, runtime KDEPTH)"
```

---

## Task 3: Extend `occ_dispatch.cpp` — timing, FLOPS, prong drivers

**Files:** Modify `occ_dispatch.cpp`.

- [ ] **Step 1:** Add to `RunResult`: `double secs;`. In `run_variant`, add params `uint32_t kdepth, uint32_t nacc, const char* isaPath` (caller picks the binary). Before placing packets, set `occW[2] = kdepth;` (the `occ[8]` word). Bracket submit→fence with `clock_gettime(CLOCK_MONOTONIC)` and store `res.secs = elapsed`. Keep the existing maxlive + acc0 oracle unpack (one tile).

- [ ] **Step 2:** Add a `tflops` helper:
```cpp
static double tflops(uint32_t nWG, uint32_t kdepth, uint32_t nacc, double secs) {
    // 1 wave32 / WG; each wave does kdepth*nacc WMMAs of 16x16x16 (=2*16^3 flop)
    double w = (double)nWG * (double)kdepth * (double)nacc;
    return w * (2.0*16*16*16) / secs / 1e12;
}
```

- [ ] **Step 3:** Add `--prong1` driver: light kernel (`occ_n8_d0.bin`, NACC=8), fixed large `nWG` and `KDEPTH`, sweep `staticVgprs` over `{80,96,128,160,192,256}`. For each: run 3×, keep min secs, print `reserve, measured-maxlive, TFLOPS`. First do a KDEPTH=1 oracle-check run; abort the prong if MISMATCH.

- [ ] **Step 4:** Add `--prong2` driver: heavy kernel (NACC=16). For `KDEPTH ∈ {256,1024,4096}`: run `occ_n16_d0.bin` (static, reserve=144) and `occ_n16_d1.bin` (dyn, reserve field=4/launch 32), 3× each keep-min, print `KDEPTH, static-TFLOPS, dyn-TFLOPS, ratio`. KDEPTH=1 oracle-check both first; abort on MISMATCH.

- [ ] **Step 5:** Keep default (no flag) = the Phase-2 correctness A/B (so we never lose the regression check). Parse `argv` for `--prong1`/`--prong2`.

- [ ] **Step 6: Commit**
```bash
git add occ_dispatch.cpp
git commit -m "dvgpr_occ P3-T3: throughput driver (--prong1/--prong2), host timing + TFLOPS + KDEPTH via occ[8]"
```

---

## Task 4: `build.sh` + CPU-side green (no GPU)

**Files:** Modify `build.sh`.

- [ ] **Step 1:** Replace the single dyn/static assemble block with the 4-variant loop from Task 2 Step 2. Keep the oracle self-test (`test_fp8_oracle.cpp`) and the RAM-capped harness link. Add the four `occ_n*_d*.bin` to the build output listing.

- [ ] **Step 2: Run the build**

Run: `cd <workdir> && bash build.sh`
Expected: 4 `.bin` files reported, `test_oracle` prints all-PASS, `occ_dispatch` links clean. **No GPU touched.**

- [ ] **Step 3: Commit**
```bash
git add build.sh
git commit -m "dvgpr_occ P3-T4: build.sh assembles the NACC x DYNVGPR matrix + harness (RAM-capped)"
```

---

## Task 5: Approach-2 cross-check (UNSUPERVISED — normal HIP launch)

**Files:** Create `gemm_occ_pad.hip`.

- [ ] **Step 1:** Write a self-contained HIP bench: copy the `gemm_fp8_tiled` raw-intrinsic kernel (from `../../gemm_wmma.hip` / `../gemm_wmma_raw_intrinsic_verified.hip`), add a compile-time occupancy cap via `__attribute__((amdgpu_waves_per_eu(CAP,CAP)))` and an optional `volatile` dead-VGPR pad. `main` allocates a fixed compute-bound GEMM (M=N=4096, K=4096), warms up, times 30 iters with `hipEvent`, prints `CAP, achieved-occupancy-proxy(VGPR from --save-temps), TFLOPS`.

- [ ] **Step 2: Build (RAM-capped) across caps and run**

Run:
```bash
for cap in 8 7 6 5 4; do
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
    /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -DCAP=$cap gemm_occ_pad.hip -o gemm_occ_pad_$cap 2>/dev/null
  ./gemm_occ_pad_$cap; done
```
Expected: a TFLOPS-vs-occupancy table on the real kernel (no hang risk — normal launch).

- [ ] **Step 3: Measure the peak-vs-steady VGPR gap**

Run: `/opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 --save-temps -S -c gemm_occ_pad.hip -o /dev/null 2>/dev/null; grep -E '\.vgpr_count|\.vgpr_spill' gemm_occ_pad-hip-*.s | head`
Record total VGPR (peak). Estimate steady = NACC_real*8 + A/B/addr; gap = peak − steady. Note both in RESULT_P3.

- [ ] **Step 4: Commit**
```bash
git add gemm_occ_pad.hip
git commit -m "dvgpr_occ P3-T5: real-kernel occupancy-down cross-check + peak-vs-steady VGPR gap"
```

---

## Task 6: [SUPERVISED — STOP FOR GO] Prong 1 — occupancy→throughput curve

- [ ] **Step 1:** Tell the user: code built, oracle green, Approach-2 data in hand; the only thing left is the supervised PM4 run. **Wait for explicit go.**
- [ ] **Step 2:** `cd <workdir> && timeout 40 ./occ_dispatch --prong1`
  Expected: KDEPTH=1 oracle OK, then a `reserve / maxlive / TFLOPS` table across occ ~16→5. No hang.
- [ ] **Step 3:** Record the curve. **GREEN-1** = TFLOPS rises materially toward higher occupancy; **NO-GO** = flat by occ 8.

---

## Task 7: [SUPERVISED — STOP FOR GO] Prong 2 — dyn vs static heavy

- [ ] **Step 1:** Wait for explicit go (separate from Task 6).
- [ ] **Step 2:** `cd <workdir> && timeout 60 ./occ_dispatch --prong2`
  Expected: oracle OK both variants, then `KDEPTH / static-TF / dyn-TF / ratio` across KDEPTH. No hang.
- [ ] **Step 3:** Record. **GREEN-2** = dyn ≥ static throughput at realistic KDEPTH (no serialization penalty).

---

## Task 8: Record outcome

- [ ] **Step 1:** Write `RESULT_P3.md`: Prong-1 curve, Prong-2 ratios, Approach-2 peak-vs-steady gap, the projected GEMM win (= gap × Prong-1 slope), GREEN-1/GREEN-2 verdict + greenlight decision, reproduce block.
- [ ] **Step 2:** Commit RESULT_P3.md.
- [ ] **Step 3:** `mneme_write` the silicon result (share=true) + update Jira MAD-305 with the verdict and the Phase-4 go/no-go.

---

## Self-Review

- **Spec coverage:** §3 Prong 1 → Task 6; §3 Prong 2 → Task 7; §3 cross-check → Task 5; §4 timing/FLOPS/oracle → Task 3; §5 files → Tasks 2–5; §6 gates → Tasks 6–8; §7 supervision → Tasks 6–7 STOP gates. Covered.
- **Encoding risk:** the two new encodings isolated in Task 1 (lift+verify) before any GPU dispatch. The RSRC1.VGPRS-below-usage hang trap is fenced by flooring the Prong-1 ladder at the kernel footprint (80 for NACC=8).
- **Type consistency:** binary names `occ_n{8,16}_d{0,1}.bin` used identically in Tasks 2/3/4; `occ[8]`=KDEPTH word, accumulators at v16+, A/B at v8:v11 consistent across kernel/dispatch.
