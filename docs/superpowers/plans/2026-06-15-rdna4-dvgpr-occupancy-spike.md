# RDNA4 dynamic-VGPR occupancy spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task (NOT subagent-driven — Tasks 5–7 are SUPERVISED on-silicon GPU runs
> that can hang the compute queue; a cold subagent must not fire them). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Prove the dynamic-VGPR occupancy lever on gfx1201: a kernel that launches at a 32-VGPR
block, `s_alloc_vgpr`s up to 128, runs a real fp8 WMMA (verified vs a CPU oracle), shrinks back,
and shows ≥2× more resident waves than a static-128-VGPR twin.

**Architecture:** Reuse the proven `spike/dvgpr_pm4/` raw-PM4 dispatch substrate (vendored kfdtest
encoder + hand-rolled ring/alloc). A CPU fp8 e4m3 oracle (TDD) verifies WMMA correctness. One
hand-written gfx1201 kernel (`occ_kernel.s`), assembled into a dyn and a static variant, does an
atomic max-resident-waves probe + a real `v_wmma_f32_16x16x16_fp8_fp8`. The harness packs/unpacks
fragments on the CPU using the proven §7.12 maps, so the kernel stays layout-agnostic.

**Tech Stack:** gfx1201 raw ISA (`/opt/rocm/llvm/bin/clang` assembler + `llvm-objdump`), HIP
(`hipcc`, for the WMMA seed only), C++17 harness linking `/opt/rocm/lib/libhsakmt.a`, the
`spike/dvgpr_pm4/` PM4 layer. Spec: `docs/superpowers/specs/2026-06-15-rdna4-dvgpr-occupancy-spike-design.md`.

**Reference (read before starting):**
- `spike/dvgpr_pm4/pm4_defs.h` — register offsets, `BuildPgmRsrc1/2`, the proven dispatch values.
- `spike/dvgpr_pm4/pm4_dispatch.cpp` — the harness to extend (alloc/queue/ring/fence).
- `spike/gemm_wmma_raw_intrinsic_verified.hip:111-182` — the proven §7.12 A/B/C-D fragment maps.
- `spike/dvgpr_pm4/RESULT.md` — the arming result this builds on.

**Convention used throughout:** all builds RAM-capped via `spike/dvgpr_pm4/build.sh`'s
`run_capped` pattern (`systemd-run --user --scope -p MemoryMax=4G`). GPU runs are SUPERVISED;
each is wrapped in `timeout 30` and the harness has a 2 s internal fence timeout.

---

## File Structure

All new files in `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`:

| File | Responsibility |
|---|---|
| `fp8_oracle.h` / `fp8_oracle.cpp` | CPU e4m3 decode + 16×16×16 reference matmul. Pure CPU, TDD. |
| `test_fp8_oracle.cpp` | Unit tests for the oracle. |
| `wmma_seed.hip` | One `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`; source of the real instruction encoding. |
| `frag_layout.h` | The §7.12 lane→element pack/unpack maps (lifted from the verified kernel), used CPU-side. |
| `occ_kernel.s` | Hand-written gfx1201 kernel: occupancy probe + `s_alloc_vgpr` envelope + WMMA. Two variants via `-defsym DYNVGPR=0/1`. |
| `occ_dispatch.cpp` | Extends the PM4 harness: buffers + large grid + A/B run + oracle compare + occupancy table. |
| `build.sh` | RAM-capped: assemble both kernel variants, compile seed, build harness + oracle. |
| `RESULT.md` | Findings, the A/B table, verdict. |

---

## Task 1: CPU fp8 e4m3 oracle (TDD)

**Files:**
- Create: `spike/dvgpr_occ/fp8_oracle.h`, `spike/dvgpr_occ/fp8_oracle.cpp`
- Test: `spike/dvgpr_occ/test_fp8_oracle.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
// test_fp8_oracle.cpp
#include "fp8_oracle.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

static int fails = 0;
#define CHECK(cond) do { if(!(cond)){ printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#cond); fails++; } } while(0)

int main() {
    // e4m3: 0x00=0.0, 0x38=1.0, 0x40=2.0, 0xB8=-1.0, 0x3C=1.5
    CHECK(fp8_e4m3_to_float(0x00) == 0.0f);
    CHECK(fp8_e4m3_to_float(0x38) == 1.0f);
    CHECK(fp8_e4m3_to_float(0x40) == 2.0f);
    CHECK(fp8_e4m3_to_float(0xB8) == -1.0f);
    CHECK(fp8_e4m3_to_float(0x3C) == 1.5f);

    // 16x16x16 reference: A = identity-ish (all 1.0 in row 0), B all 1.0, C=0
    // D[i][j] = sum_k A[i][k]*B[k][j]. A all-1, B all-1 => D[i][j] = 16.0
    uint8_t A[256], B[256]; float C[256] = {0}, D[256];
    for (int i = 0; i < 256; ++i) { A[i] = 0x38; B[i] = 0x38; }  // all 1.0
    wmma_ref_16x16x16(A, B, C, D);
    CHECK(std::fabs(D[0] - 16.0f) < 1e-6);
    CHECK(std::fabs(D[16*16-1] - 16.0f) < 1e-6);

    printf(fails ? "FAILED (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle
```
Expected: FAIL to compile — `fp8_oracle.h` / `.cpp` don't exist yet.

- [ ] **Step 3: Implement the oracle**

```cpp
// fp8_oracle.h
#pragma once
#include <cstdint>
// Decode one OCP e4m3 byte (1 sign, 4 exp bias-7, 3 mantissa; no inf; 0xFF/0x7F = NaN).
float fp8_e4m3_to_float(uint8_t b);
// Reference D = A*B + C. A,B are 16x16 row-major e4m3 bytes; C,D are 16x16 row-major f32.
void wmma_ref_16x16x16(const uint8_t* A, const uint8_t* B, const float* C, float* D);
```

```cpp
// fp8_oracle.cpp
#include "fp8_oracle.h"

float fp8_e4m3_to_float(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp  = (b >> 3) & 0xF;
    const int man  = b & 0x7;
    float v;
    if (exp == 0) {                       // subnormal (or zero)
        v = man / 8.0f * (1.0f / 64.0f);  // 2^(1-7) = 2^-6 = 1/64
    } else if (exp == 0xF && man == 0x7) {
        v = 0.0f;                          // NaN: not used by our test inputs; map to 0
    } else {
        v = (1.0f + man / 8.0f);
        int e = exp - 7;
        // scale by 2^e
        if (e >= 0) v *= (float)(1u << e);
        else        v /= (float)(1u << (-e));
    }
    return sign ? -v : v;
}

void wmma_ref_16x16x16(const uint8_t* A, const uint8_t* B, const float* C, float* D) {
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float acc = C[i * 16 + j];
            for (int k = 0; k < 16; ++k)
                acc += fp8_e4m3_to_float(A[i * 16 + k]) * fp8_e4m3_to_float(B[k * 16 + j]);
            D[i * 16 + j] = acc;
        }
}
```

- [ ] **Step 4: Run to verify it passes**

```bash
clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle && ./test_oracle
```
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add spike/dvgpr_occ/fp8_oracle.h spike/dvgpr_occ/fp8_oracle.cpp spike/dvgpr_occ/test_fp8_oracle.cpp
git commit -m "dvgpr_occ T1: CPU fp8 e4m3 oracle + tests"
```

---

## Task 2: WMMA seed + fragment layout header

Confirms the real WMMA instruction encoding and captures the proven §7.12 pack/unpack maps for
CPU-side fragment marshalling.

**Files:**
- Create: `spike/dvgpr_occ/wmma_seed.hip`, `spike/dvgpr_occ/frag_layout.h`

- [ ] **Step 1: Write the WMMA seed kernel**

```cpp
// wmma_seed.hip — emit exactly one fp8 WMMA so we can read its encoding.
typedef int   v2i32  __attribute__((ext_vector_type(2)));
typedef float v8f32  __attribute__((ext_vector_type(8)));
extern "C" __global__ void seed(const v2i32* a, const v2i32* b, v8f32* d) {
    int l = threadIdx.x;
    v8f32 acc = {0,0,0,0,0,0,0,0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[l], b[l], acc);
    d[l] = acc;
}
```

- [ ] **Step 2: Compile + disassemble; confirm the instruction exists**

```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -c wmma_seed.hip -o wmma_seed.o
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx1201 wmma_seed.o | grep -i wmma
```
Expected: a line containing `v_wmma_f32_16x16x16_fp8_fp8` (this is the exact mnemonic + operand
order to hand-write in Task 3). Record the mnemonic/operand form in a comment in `occ_kernel.s`.

- [ ] **Step 3: Write the fragment-layout header (lifted from the verified kernel §7.12)**

```cpp
// frag_layout.h — the PROVEN gfx12 fp8 WMMA lane maps, from
// spike/gemm_wmma_raw_intrinsic_verified.hip:111-182. Used CPU-side by the harness to pack
// inputs into per-lane fragments and unpack the per-lane v8f32 result into a 16x16 D matrix.
// The KERNEL never does layout math: lane L loads in_A[L*2..], in_B[L*2..], stores out[L*8..].
#pragma once
#include <cstdint>

// A (row-major 16x16 e4m3): lane L holds row=L&0xF, the 8 contiguous K-bytes [colhi*8..+7]
//   where colhi=(L>>4)&1. Pack into 32 lanes x 2 int32 (= 8 bytes) each.
static inline void pack_A(const uint8_t* A /*256*/, uint32_t* fragA /*64 = 32*2*/) {
    for (int L = 0; L < 32; ++L) {
        int row = L & 0xF, colhi = (L >> 4) & 1, kbase = colhi * 8;
        uint32_t lo = 0, hi = 0;
        for (int p = 0; p < 4; ++p) {
            lo |= (uint32_t)A[row * 16 + kbase + p]     << (p * 8);
            hi |= (uint32_t)A[row * 16 + kbase + 4 + p] << (p * 8);
        }
        fragA[L * 2 + 0] = lo; fragA[L * 2 + 1] = hi;
    }
}
// B (row-major 16x16 e4m3): lane L holds col=L&0xF, K-bytes [rowhi*8..+7], rowhi=(L>>4)&1.
static inline void pack_B(const uint8_t* B /*256*/, uint32_t* fragB /*64*/) {
    for (int L = 0; L < 32; ++L) {
        int col = L & 0xF, rowhi = (L >> 4) & 1, kbase = rowhi * 8;
        uint32_t lo = 0, hi = 0;
        for (int p = 0; p < 4; ++p) {
            lo |= (uint32_t)B[(kbase + p)     * 16 + col] << (p * 8);
            hi |= (uint32_t)B[(kbase + 4 + p) * 16 + col] << (p * 8);
        }
        fragB[L * 2 + 0] = lo; fragB[L * 2 + 1] = hi;
    }
}
// D/C (v8f32 per lane): lane L holds col=L&0xF, rows ((L>>4)&1)*8 + slot, slot in 0..7.
static inline void unpack_D(const float* fragD /*256 = 32*8*/, float* D /*256*/) {
    for (int L = 0; L < 32; ++L) {
        int col = L & 0xF, rowbase = ((L >> 4) & 1) * 8;
        for (int s = 0; s < 8; ++s)
            D[(rowbase + s) * 16 + col] = fragD[L * 8 + s];
    }
}
```

- [ ] **Step 4: Round-trip test the layout against the oracle (CPU-only, no GPU)**

Add to `test_fp8_oracle.cpp` a second check: this is a pure consistency test that `pack_*`/`unpack_D`
index math is self-consistent for a non-trivial matrix (catches transposition bugs before the GPU run).

```cpp
// appended to main() in test_fp8_oracle.cpp, before the final print:
{
    uint8_t A[256], B[256]; float C[256] = {0}, Dref[256];
    for (int i = 0; i < 256; ++i) { A[i] = (uint8_t)(0x38 + (i % 3)); B[i] = (uint8_t)(0x38 + (i % 2)); }
    wmma_ref_16x16x16(A, B, C, Dref);
    uint32_t fa[64], fb[64];
    pack_A(A, fa); pack_B(B, fb);
    // Emulate the WMMA on the packed fragments by decoding them back and re-doing the matmul,
    // proving pack_A/pack_B preserve the (row,k)/(k,col) values the hardware will see.
    uint8_t A2[256] = {0}, B2[256] = {0};
    for (int L = 0; L < 32; ++L) {
        int row = L & 0xF, colhi = (L>>4)&1, cb = colhi*8;
        for (int p = 0; p < 4; ++p) { A2[row*16+cb+p]   = (fa[L*2]   >> (p*8)) & 0xFF;
                                      A2[row*16+cb+4+p] = (fa[L*2+1] >> (p*8)) & 0xFF; }
        int col = L & 0xF, rowhi = (L>>4)&1, rb = rowhi*8;
        for (int p = 0; p < 4; ++p) { B2[(rb+p)*16+col]   = (fb[L*2]   >> (p*8)) & 0xFF;
                                      B2[(rb+4+p)*16+col] = (fb[L*2+1] >> (p*8)) & 0xFF; }
    }
    float Drt[256]; wmma_ref_16x16x16(A2, B2, C, Drt);
    for (int i = 0; i < 256; ++i) CHECK(A2[i] == A[i] && B2[i] == B[i]);
}
```
Run: `clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle && ./test_oracle`
Expected: `PASS` (pack maps reproduce A and B exactly).

- [ ] **Step 5: Commit**

```bash
git add spike/dvgpr_occ/wmma_seed.hip spike/dvgpr_occ/frag_layout.h spike/dvgpr_occ/test_fp8_oracle.cpp
git commit -m "dvgpr_occ T2: WMMA seed (instruction encoding) + proven frag-layout maps + round-trip test"
```

---

## Task 3: the hand-written gfx1201 kernel (`occ_kernel.s`)

Occupancy probe (lane-0-only atomics so we count WAVES) + `s_alloc_vgpr` envelope + the real WMMA.
Two variants via `-defsym`. Thread id in v0 (TIDIG_COMP_CNT=1, set by the harness, as in the
MAD-304 probe). User SGPRs: s0:s1=occ, s2:s3=fragIn (A then B), s4:s5=fragOut.

**Why a 4-accumulator tile, not one WMMA:** a single 16×16×16 fp8 WMMA needs only ~28 VGPRs,
which already fits the 32-VGPR small block — so a lone WMMA would make `s_alloc_vgpr` gratuitous
and the spike would NOT prove dyn-VGPR enables a register-hungry hot region. The kernel therefore
holds **4 WMMA accumulators** in `v[32:63]` (32 acc VGPRs on top of inputs/temps → >32 used), so
the grow is genuinely required. All 4 reuse the same A/B fragments and start at 0, so each computes
the same `D` — the oracle stays a single 16×16 matrix and the harness checks all 4 output tiles
equal it. The static twin declares 128 VGPRs (`v[32:63]` valid from launch); the dyn twin launches
at 32 and must `s_alloc_vgpr 128` before touching `v32`.

**Files:**
- Create: `spike/dvgpr_occ/occ_kernel.s`

- [ ] **Step 1: Write the kernel**

```asm
// occ_kernel.s  (gfx1201, wave32). Assemble twice:
//   dyn:    clang ... -defsym DYNVGPR=1   (harness launches 32-VGPR block + RSRC2 bit6)
//   static: clang ... -defsym DYNVGPR=0   (harness launches 128-VGPR static block)
// v_wmma mnemonic/operand order: see comment recorded from Task 2 disasm.
// User data (USER_SGPR=6): s0:s1=occ[live,maxlive]  s2:s3=fragIn  s4:s5=fragOut
// v0 = thread id x (lane 0..31) via TIDIG_COMP_CNT.
    .text
    .globl occ_kernel
    .p2align 8
    .type occ_kernel,@function
occ_kernel:
    // ---- lane-0-only: bump live, sample maxlive ----
    v_cmp_eq_u32 vcc_lo, 0, v0          // lane 0?
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo  // exec = {lane0}
    s_cbranch_execz .Lafter_inc
    v_mov_b32 v2, 1
    global_atomic_add_u32 v3, v4, v2, s[0:1] offset:0 th:TH_ATOMIC_RETURN   // v3=old live
    v_add_nc_u32 v3, v3, 1                                                  // new live
    global_atomic_max_u32 v5, v4, v3, s[0:1] offset:4                       // maxlive = max(.,new)
.Lafter_inc:
    s_mov_b32 exec_lo, s8               // restore full wave
    // ---- long busy-wait at the SMALL block (where occupancy is measured) ----
    s_movk_i32 s9, 0x4000
.Lspin:
    s_sub_u32 s9, s9, 1
    s_cmp_lg_u32 s9, 0
    s_cbranch_scc1 .Lspin
.if DYNVGPR
    s_alloc_vgpr 128                    // grow to 128 VGPRs for the WMMA
.endif
    // ---- per-lane fragment loads: fa=fragIn[lane*2..], fb=fragIn[64*4 + lane*2..] ----
    v_lshlrev_b32 v6, 3, v0             // lane*8 bytes (2 i32)
    global_load_b64 v[16:17], v6, s[2:3]            // A frag (2 i32)
    global_load_b64 v[18:19], v6, s[2:3] offset:256 // B frag (A block is 32*8=256 bytes)
    s_waitcnt vmcnt(0)
    // ---- zero the 4 accumulators v[32:63] (32x v_mov_b32 vX, 0 for X in 32..63) ----
    // (write all 32 explicitly; shown abbreviated — the engineer emits v_mov_b32 v32..v63, 0)
    v_mov_b32 v32, 0
    // ... v33..v62 = 0 ...
    v_mov_b32 v63, 0
    // ---- 4 WMMA accumulators, all reusing A=v[16:17], B=v[18:19] (each computes the same D) ----
    // Operand order per the Task 2 seed disasm.
    v_wmma_f32_16x16x16_fp8_fp8 v[32:39], v[16:17], v[18:19], v[32:39]
    v_wmma_f32_16x16x16_fp8_fp8 v[40:47], v[16:17], v[18:19], v[40:47]
    v_wmma_f32_16x16x16_fp8_fp8 v[48:55], v[16:17], v[18:19], v[48:55]
    v_wmma_f32_16x16x16_fp8_fp8 v[56:63], v[16:17], v[18:19], v[56:63]
    // ---- store the 4 tiles: fragOut + tile*1024 + lane*32 bytes (256 f32 per tile) ----
    v_lshlrev_b32 v7, 5, v0           // lane*32 bytes
    global_store_b128 v7, v[32:35], s[4:5]                 // tile0
    global_store_b128 v7, v[36:39], s[4:5] offset:16
    global_store_b128 v7, v[40:43], s[4:5] offset:1024     // tile1
    global_store_b128 v7, v[44:47], s[4:5] offset:1040
    global_store_b128 v7, v[48:51], s[4:5] offset:2048     // tile2
    global_store_b128 v7, v[52:55], s[4:5] offset:2064
    global_store_b128 v7, v[56:59], s[4:5] offset:3072     // tile3
    global_store_b128 v7, v[60:63], s[4:5] offset:3088
    s_waitcnt vmcnt(0)
.if DYNVGPR
    s_alloc_vgpr 32                    // shrink back to the small block
.endif
    // ---- lane-0-only: dec live ----
    v_cmp_eq_u32 vcc_lo, 0, v0
    s_mov_b32 s8, exec_lo
    s_and_b32 exec_lo, exec_lo, vcc_lo
    s_cbranch_execz .Ldone
    v_mov_b32 v2, -1
    global_atomic_add_u32 v3, v4, v2, s[0:1] offset:0
.Ldone:
    s_mov_b32 exec_lo, s8
    s_endpgm
    .size occ_kernel, .-occ_kernel
```

- [ ] **Step 2: Assemble both variants; correct any rejected mnemonic against the assembler**

```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
L=/opt/rocm/llvm/bin
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -defsym DYNVGPR=1 -c occ_kernel.s -o occ_dyn.o
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -defsym DYNVGPR=0 -c occ_kernel.s -o occ_static.o
```
Expected: both assemble. If a mnemonic/modifier is rejected (e.g. `th:TH_ATOMIC_RETURN`,
`global_atomic_add_u32` operand form, or the `v_wmma` operand order), use the EXACT spelling the
Task-2 seed disasm emitted (`llvm-objdump -d wmma_seed.o`) and re-assemble. This is the gfx1201
mnemonic-confirmation gate — fix until both `.o` build.

- [ ] **Step 3: Extract raw `.text` to flat binaries + disassemble to verify structure**

```bash
$L/llvm-objcopy -O binary --only-section=.text occ_dyn.o    occ_dyn.bin
$L/llvm-objcopy -O binary --only-section=.text occ_static.o occ_static.bin
$L/llvm-objdump -d --mcpu=gfx1201 occ_dyn.o | grep -iE 's_alloc_vgpr|v_wmma|global_atomic'
$L/llvm-objdump -d --mcpu=gfx1201 occ_static.o | grep -iE 's_alloc_vgpr|v_wmma|global_atomic'
```
Expected: `occ_dyn` shows two `s_alloc_vgpr` (128 then 32) around one `v_wmma…fp8_fp8` plus the
atomics; `occ_static` shows the same WMMA + atomics but **no** `s_alloc_vgpr`.

- [ ] **Step 4: Commit**

```bash
git add spike/dvgpr_occ/occ_kernel.s
git commit -m "dvgpr_occ T3: hand-written gfx1201 occupancy+WMMA kernel (dyn/static via -defsym)"
```

---

## Task 4: the dispatch harness (`occ_dispatch.cpp`)

Extends the MAD-304 harness. Allocates buffers, builds A/B test matrices, packs fragments,
launches a large grid, runs static then dyn, reads `maxlive` + WMMA result, compares to the oracle,
prints the A/B table. Reuses `pm4_defs.h` and the vendored encoder via include paths.

**Files:**
- Create: `spike/dvgpr_occ/occ_dispatch.cpp`

- [ ] **Step 1: Write the harness**

Key differences from `pm4_dispatch.cpp` (reuse its `AllocGpu`, `Ring`, `RingPlace`, `RingSubmit`,
`FindGfx1201Node`, `hsakmt_is_dgpu`, `CHECK` verbatim — copy them in or `#include` the file's bodies):

```cpp
// occ_dispatch.cpp  (sketch of the NEW logic; reuse the MAD-304 plumbing for KFD/alloc/ring/fence)
#include "fp8_oracle.h"
#include "frag_layout.h"
#include "pm4_defs.h"           // -I ../dvgpr_pm4
// ... (same hsakmt includes + AllocGpu/Ring/RingPlace/RingSubmit/FindGfx1201Node as pm4_dispatch.cpp)

struct RunResult { uint32_t maxlive; float Dtile[4][256]; };  // 4 WMMA tiles, each unpacked to 16x16

// Dispatch occ_kernel with a given variant. grid = nWG single-wave workgroups.
static RunResult run_variant(uint32_t node, const char* isaPath, bool dynvgpr,
                             const uint32_t* fragIn /*128 u32: 64 A then 64 B*/, uint32_t nWG) {
    // buffers: ISA, occ[2], fragIn[128], fragOut[1024 = 4 tiles x 256 f32], fence (host-visible GTT)
    // after the run: for t in 0..3  unpack_D(fragOut + t*256, result.Dtile[t]);
    // occ[0]=live=0, occ[1]=maxlive=0; copy fragIn in.
    // PGM_RSRC1: dyn -> BuildPgmRsrc1(false) (VGPRS field 0x4 = 32). static -> same but VGPRS
    //   field = 0x10 (128). NOTE: confirm the 128 encoding at disasm (Task 5) — MAD-304 proved
    //   0x4=32, i.e. field*8; 128 -> 0x10 expected but verify.
    // PGM_RSRC2: start from BuildPgmRsrc2(dynvgpr) (keeps TGID_X_EN|TIDIG_COMP_CNT|EXCP_EN_MSB and
    //   bit6 for dyn) and FORCE USER_SGPR field 4->6 (3 pointers): rsrc2 = (rsrc2 & ~0x3e) | (6<<1).
    //   TIDIG_COMP_CNT MUST remain set so v0 = lane id (0..31) at wave entry.
    // USER_DATA_0..5 = occVA lo/hi, fragInVA lo/hi, fragOutVA lo/hi.
    // dims block: NUM_THREAD_X = 32 (wave32, 1 wave/WG).  DISPATCH_DIRECT dimX = nWG.
    // Place ACQUIRE_MEM -> SET_SH_REGs -> DISPATCH_DIRECT(nWG,1,1) -> RELEASE_MEM(fence). Submit.
    // Poll fence; read occ[1]=maxlive and fragOut[256]; unpack_D(fragOut, result.D).
}
```

Full success-print logic in `main()`:

```cpp
int main(int argc, char** argv) {
    uint32_t nWG = (argc > 1) ? atoi(argv[1]) : 2048;
    // Build test matrices A,B (16x16 e4m3) — use a non-trivial pattern, and the CPU oracle D.
    uint8_t A[256], B[256]; float C[256] = {0}, Dref[256];
    for (int i = 0; i < 256; ++i) { A[i] = (uint8_t)(0x38 + (i % 3)); B[i] = (uint8_t)(0x38 + (i % 2)); }
    wmma_ref_16x16x16(A, B, C, Dref);
    uint32_t fragIn[128]; pack_A(A, fragIn); pack_B(B, fragIn + 64);

    CHECK(hsaKmtOpenKFD());
    uint32_t node = FindGfx1201Node();

    RunResult st = run_variant(node, "occ_static.bin", false, fragIn, nWG);
    RunResult dy = run_variant(node, "occ_dyn.bin",    true,  fragIn, nWG);

    auto wmma_ok = [&](const RunResult& r) {
        for (int t = 0; t < 4; ++t)
            for (int i = 0; i < 256; ++i)
                if (std::fabs(r.Dtile[t][i] - Dref[i]) > 1e-3f * std::fabs(Dref[i]) + 1e-3f) return false;
        return true;
    };
    bool gate_func = wmma_ok(dy);
    bool gate_occ  = (dy.maxlive >= 2 * st.maxlive) && st.maxlive > 0;
    printf("\n=== dyn-VGPR occupancy A/B (grid=%u WGs) ===\n", nWG);
    printf("  static: maxlive=%u  WMMA %s\n", st.maxlive, wmma_ok(st) ? "OK" : "MISMATCH");
    printf("  dyn   : maxlive=%u  WMMA %s\n", dy.maxlive, gate_func ? "OK" : "MISMATCH");
    printf("  gates : functional(dyn WMMA==oracle)=%d  occupancy(dyn>=2x static)=%d\n",
           gate_func, gate_occ);
    printf("  VERDICT: %s\n", (gate_func && gate_occ) ? "DYN-VGPR OCCUPANCY LEVER PROVEN"
                                                      : "see table / iterate");
    hsaKmtCloseKFD();
    return (gate_func && gate_occ) ? 0 : 5;
}
```

(Implement `run_variant` fully using the MAD-304 register-write sequence; the only additions are
the 3-pointer USER_DATA, the per-variant RSRC1 VGPRS field + RSRC2 bit6/USER_SGPR, the larger
`DISPATCH_DIRECT` dimX, and reading two output buffers.)

- [ ] **Step 2: Commit (after it builds in Task 5)** — the build is Task 5; commit there with build.sh.

---

## Task 5: build script + SUPERVISED baseline smoke

**Files:**
- Create: `spike/dvgpr_occ/build.sh`

- [ ] **Step 1: Write build.sh (RAM-capped)**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
L=/opt/rocm/llvm/bin; ROCM=/opt/rocm; MEMMAX="${MEMMAX:-4G}"
run_capped(){ systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 "$@"; }
echo "[1/3] assemble kernel variants"
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -defsym DYNVGPR=1 -c occ_kernel.s -o occ_dyn.o
$L/clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -defsym DYNVGPR=0 -c occ_kernel.s -o occ_static.o
$L/llvm-objcopy -O binary --only-section=.text occ_dyn.o    occ_dyn.bin
$L/llvm-objcopy -O binary --only-section=.text occ_static.o occ_static.bin
echo "[2/3] oracle self-test"
clang++ -std=c++17 test_fp8_oracle.cpp fp8_oracle.cpp -o test_oracle && ./test_oracle
echo "[3/3] build harness (MemoryMax=$MEMMAX)"
run_capped clang++ -std=c++17 -O2 -I ../dvgpr_pm4/compat -I ../dvgpr_pm4 -I ../dvgpr_pm4/vendor \
    -I "$ROCM/include" \
    occ_dispatch.cpp fp8_oracle.cpp ../dvgpr_pm4/vendor/PM4Packet.cpp ../dvgpr_pm4/vendor/BasePacket.cpp \
    "$ROCM/lib/libhsakmt.a" -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch
echo "OK -> ./occ_dispatch [nWG]"
```
(Note: `pm4_defs.h` lives in `../dvgpr_pm4`; ensure `occ_dispatch.cpp` includes the compat shim
path so the vendored encoder resolves, exactly as MAD-304's build.)

- [ ] **Step 2: Build**

```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
chmod +x build.sh && ./build.sh
```
Expected: oracle `PASS`, both `.bin` produced, `occ_dispatch` links. Fix compile errors (most
likely: USER_SGPR field, RSRC1 VGPRS field for the static 128 variant) until clean.

- [ ] **Step 3: SUPERVISED baseline smoke — tiny grid, prove no hang + WMMA correctness first**

```bash
# Small grid first (occupancy not the point yet — just prove the new dispatch path is sound).
timeout 30 ./occ_dispatch 64 ; echo "exit=$?"
```
Expected: fence signals cleanly (no hang), and BOTH variants print `WMMA OK` (the WMMA result
matches the oracle). `maxlive` numbers may be small at grid=64; that's fine here. If WMMA is
`MISMATCH`, debug the fragment marshalling / store before scaling the grid (Task 6). If it hangs,
STOP — check `dmesg` for a ring reset; do not re-fire.

- [ ] **Step 4: Commit**

```bash
git add spike/dvgpr_occ/occ_dispatch.cpp spike/dvgpr_occ/build.sh
git commit -m "dvgpr_occ T4+T5: PM4 dispatch harness (A/B + oracle compare) + build; baseline smoke clean"
```

---

## Task 6: SUPERVISED A/B occupancy experiment + RESULT.md

**Files:**
- Create: `spike/dvgpr_occ/RESULT.md`

- [ ] **Step 1: Run the full A/B at occupancy-saturating grid**

```bash
cd ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ
timeout 30 ./occ_dispatch 2048 ; echo "exit=$?"
# repeat x3 for determinism
for i in 1 2 3; do timeout 30 ./occ_dispatch 2048 | grep -E 'static:|dyn   :|gates|VERDICT'; done
```
Expected: `dyn maxlive` ≥ 2× `static maxlive`, both WMMA `OK`, deterministic across runs. Record
the actual numbers. (gfx1201 has 128 FCompute SIMDs; a 32-VGPR block should pack many more
waves/SIMD than a 128-VGPR static block — interpret `maxlive` against the SIMD count.)

- [ ] **Step 2: If occupancy did NOT rise ≥2×, diagnose (do not fake the gate)**

Honest-result branch (per spec): if `dyn maxlive ≈ static maxlive`, the concurrent `s_alloc_vgpr`
is serializing (FWD_PROGRESS), i.e. the steady-state benefit is limited. Capture this as the
finding and record what to vary in Phase 3 (stagger the alloc, shorter hot region, measure at the
small block only by moving the WMMA before the inc/maxlive sample). This is a valid spike outcome.

- [ ] **Step 3: Write RESULT.md**

Document: the A/B table (static vs dyn `maxlive`, WMMA OK), the grid size + SIMD-count
interpretation, determinism, no-hang, and the verdict against the three gates. Mirror the structure
of `spike/dvgpr_pm4/RESULT.md`. State plainly what is proven (occupancy lever delivers / does not)
and the precise Phase-3 implication.

- [ ] **Step 4: Commit**

```bash
git add spike/dvgpr_occ/RESULT.md
git commit -m "dvgpr_occ T6: A/B occupancy experiment result + verdict"
```

---

## Task 7: record the outcome (KG + Jira)

- [ ] **Step 1:** Write a `mneme_write` (type=project, share=true) capturing the spike verdict
  (occupancy numbers, WMMA-correct, gates pass/fail) and link it to the MAD-304 unlock memory.
- [ ] **Step 2:** Comment the result on the MAD-293 epic (and open/close a Phase-2 story if one
  is tracked), with the A/B table and the Phase-3 implication.
- [ ] **Step 3:** Mark the local Phase-2 tasks complete.

---

## Notes for the executor

- **Supervised GPU (Tasks 5–6):** the gfx12 node is the headless R9700, but graphics clients are
  attached → a hang could blip the desktop. Always `timeout 30`; the harness has a 2 s fence
  timeout. On timeout, STOP and check `dmesg`; do not re-fire blindly.
- **The one genuine hardware unknown** is whether `dyn maxlive` actually clears 2× — that is the
  experiment. Everything upstream (oracle, fragment maps, WMMA encoding, dispatch vehicle) is
  verified before the GPU is asked the real question.
- **DRY:** reuse `../dvgpr_pm4` plumbing; do not re-implement KFD alloc/queue/ring/fence.
- **YAGNI:** one WMMA tile, one fp8 dtype (e4m3), single workgroup size. No GEMM, no cs_chain.
