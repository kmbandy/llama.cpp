# RDNA4 fp8 Pipeline — Phase 1: Wide Feed (preshuffle + global_load_tr) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the B byte-gather with the Phase-0-verified hardware wide feed — pre-shuffle B once into the transpose-load layout, then load each B fragment with one `global_load_tr_b64` — and measure whether it converts to throughput (vs the ~69–90 TF byte-gather baseline, the 143 TF hipBLASLt bar, and the 307 TF ceiling) at a compute-bound 4096³.

**Architecture:** A bench (`bench/gemm_trfeed_bench.hip`) runs the SAME 128×128/BK=32/4-wave tiling as the production `gemm_wmma.hip`, two ways: the byte-gather baseline vs a `gemm_fp8_trfeed` kernel that keeps A in LDS (wide read, unchanged) but feeds B directly from a pre-shuffled global buffer via `global_load_tr_b64` (no LDS staging for B). Correctness is gated by the fp8 e4m3 oracle at 256³ + baseline/trfeed agreement at 4096³. The shared preshuffle + addressing live in `bench/trfeed_common.h`. If direct-from-global walls on B-reuse, a larger-M-tile fallback amortizes B re-fetch. The winning feed is then folded into `gemm_wmma.hip`'s fp8 path behind a new preshuffled-B entry point.

**Tech Stack:** HIP/hipcc, gfx1201 wave32, `__builtin_amdgcn_global_load_tr_b64_v2i32`, `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. RAM-capped builds. **Normal HIP — unsupervised, no PM4, no hang risk.**

**Spec:** `docs/superpowers/specs/2026-06-16-rdna4-cdna4-transpose-fed-fp8-wmma-pipeline-design.md`
**Phase 0 contract (verified bit-exact):** `bench/global_load_tr_contract.md`

All paths below are relative to `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`. Always `cd` to that absolute dir before building/running (shell cwd drifts after `git commit`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `bench/trfeed_common.h` | Create | `trperm`, `preshuffle_B` (full B[K][N]→tile-major Bshuf), `b_tile_offset` — shared by bench + (later) `gemm_wmma.hip` |
| `bench/gemm_trfeed_bench.hip` | Create | byte-gather baseline + `gemm_fp8_trfeed` kernels, fp8 e4m3 oracle, 256³ gate + 4096³ perf/agreement |
| `build.sh` | Modify | add RAM-capped build target for the bench |
| `gemm_wmma.hip` | Modify (Task 5) | fold the trfeed B-feed into the fp8 path behind a preshuffled-B entry point |
| `RESULT.md` | Modify | fill the Phase 1 row |

---

## Task 1: Shared preshuffle + addressing (`trfeed_common.h`) + CPU round-trip test

**Files:**
- Create: `bench/trfeed_common.h`
- Create (temporary test): `bench/trfeed_common_test.cpp`

- [ ] **Step 1: Write `bench/trfeed_common.h`**

```cpp
// bench/trfeed_common.h — Phase-1 shared preshuffle + transpose-load addressing.
// Verified contract: bench/global_load_tr_contract.md (Phase 0, bit-exact).
#pragma once
#include <cstdint>
#include <cstddef>

// Closed-form byte permutation of global_load_tr_b64 (lane L passes Bshuf + L*8;
// output (lane L, slot s) receives Bshuf[trperm(L,s)]).  L in 0..31, s in 0..7.
__host__ __device__ inline int trperm(int L, int s) {
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}

// Byte offset of the 16x16 (K x N) tile (kt, nt) within the tile-major Bshuf buffer.
__host__ __device__ inline size_t b_tile_offset(int kt, int nt, int NT) {
    return (size_t)(kt * NT + nt) * 256;  // 256 bytes per 16x16 fp8 tile
}

// Pre-shuffle row-major fp8 B[K][N] into tile-major Bshuf (16x16 tiles, trperm order).
// K, N multiples of 16. One-time repack (B is static weights -> free at runtime).
inline void preshuffle_B(const uint8_t* B, uint8_t* Bshuf, int K, int N) {
    int KT = K / 16, NT = N / 16;
    for (int kt = 0; kt < KT; ++kt)
        for (int nt = 0; nt < NT; ++nt) {
            uint8_t* tile = Bshuf + b_tile_offset(kt, nt, NT);
            for (int L = 0; L < 32; ++L)
                for (int s = 0; s < 8; ++s) {
                    int kl = ((L >> 4) & 1) * 8 + s, nl = L & 15;
                    tile[trperm(L, s)] = B[(size_t)(kt * 16 + kl) * N + (nt * 16 + nl)];
                }
        }
}
```

- [ ] **Step 2: Write the failing CPU test** `bench/trfeed_common_test.cpp`

```cpp
// bench/trfeed_common_test.cpp — preshuffle is a within-tile bijection (no byte lost/dup).
#include "trfeed_common.h"
#include <vector>
#include <cstdio>
int main() {
    // trperm must be a bijection over 0..255 across (L,s).
    std::vector<int> seen(256, 0);
    for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) seen[trperm(L, s)]++;
    int bad = 0; for (int i = 0; i < 256; ++i) if (seen[i] != 1) bad++;
    if (bad) { printf("trperm NOT a bijection: %d offsets wrong\n", bad); return 1; }

    // preshuffle then read-back-by-contract reconstructs B exactly.
    const int K = 32, N = 48;              // multi-tile, not square
    std::vector<uint8_t> B(K * N), Bshuf(K * N, 0), Brec(K * N, 0);
    for (int i = 0; i < K * N; ++i) B[i] = (uint8_t)(i * 31 + 7);
    preshuffle_B(B.data(), Bshuf.data(), K, N);
    int NT = N / 16;
    for (int kt = 0; kt < K / 16; ++kt) for (int nt = 0; nt < NT; ++nt) {
        const uint8_t* tile = Bshuf.data() + b_tile_offset(kt, nt, NT);
        for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) {
            int kl = ((L >> 4) & 1) * 8 + s, nl = L & 15;
            Brec[(size_t)(kt * 16 + kl) * N + (nt * 16 + nl)] = tile[trperm(L, s)];
        }
    }
    for (int i = 0; i < K * N; ++i) if (B[i] != Brec[i]) { printf("recon mismatch @%d\n", i); return 1; }
    printf("trfeed_common: PASS (bijection + multi-tile round-trip)\n");
    return 0;
}
```

- [ ] **Step 3: Run the test to verify it builds and passes**

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
g++ -O2 -std=c++17 -x c++ bench/trfeed_common_test.cpp -o /tmp/trfeed_test && /tmp/trfeed_test
```
Expected: `trfeed_common: PASS (bijection + multi-tile round-trip)`. (Pure host C++ — `__host__ __device__` are no-ops under g++. If g++ rejects them, add `-D__host__= -D__device__=`.)

- [ ] **Step 4: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/trfeed_common.h \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/trfeed_common_test.cpp
git commit -m "feat(MAD-305 P1): preshuffle + transpose-load addressing (trfeed_common.h) + CPU test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Bench with byte-gather baseline + `gemm_fp8_trfeed` + 256³ oracle gate

**Files:**
- Create: `bench/gemm_trfeed_bench.hip`
- Modify: `build.sh`

- [ ] **Step 1: Write `bench/gemm_trfeed_bench.hip`.** Reuse the proven structure of `bench/feedwidth_proto.hip` (read it first) — identical tile constants, the byte-gather baseline kernel, the `enc`/`dec`/`patA`/`patB` host helpers, the e4m3 oracle, and the hipEvent timing. Add the trfeed kernel and a preshuffled-B buffer. The trfeed kernel keeps the A path identical (LDS wide read) and replaces ONLY the B fragment load:

```cpp
#include "trfeed_common.h"
typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}

// Same tiling as gemm_wmma.hip: BM=BN=128, BK=32, 4 waves (WAVES_M=WAVES_N=2),
// FRAGS_M=FRAGS_N=2, KSTEPS=2, WAVE_SIZE=32, BLOCK_THREADS=128.
__global__ void __launch_bounds__(BLOCK_THREADS)
gemm_fp8_trfeed(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                __hip_bfloat16* __restrict__ C,
                const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                int M, int N, int K) {
    const int tm = blockIdx.y * BM, tn = blockIdx.x * BN;
    __shared__ float8_t As[BM * BK];                 // A still staged in LDS (wide read)
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;
    v8f32 acc[FRAGS_M][FRAGS_N];
    for (int mi=0;mi<FRAGS_M;++mi) for (int ni=0;ni<FRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};
    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = BM*BK/16, BKv = BK/16;
        for (int e=tid;e<AVEC;e+=BLOCK_THREADS){ int r=e/BKv,c=e%BKv; int gr=tm+r,gk=k0+c*16; Asv[e]=Av[(gr*K+gk)/16]; }
        __syncthreads();
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[FRAGS_M], fb[FRAGS_N];
            for (int mi=0;mi<FRAGS_M;++mi){ int lds_row=(wave_m*FRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As + lds_row*BK + kbase + colhi*8); }
            for (int ni=0;ni<FRAGS_N;++ni){
                int nt = (tn + (wave_n*FRAGS_N+ni)*16) / 16;
                fb[ni] = tr_load8(Bshuf + b_tile_offset(kt, nt, NT) + lane*8);   // one global_load_tr
            }
            for (int mi=0;mi<FRAGS_M;++mi) for (int ni=0;ni<FRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi],fb[ni],acc[mi][ni]);
        }
        __syncthreads();                              // A LDS reuse barrier
    }
    // epilogue: identical to feedwidth_proto.hip (scratch -> *a_scale*b_scale -> bf16).
    // <copy the epilogue block from feedwidth_proto.hip verbatim>
}
```

Host: build `Bshuf` with `preshuffle_B(Bbytes, Bshuf, K, N)` and upload it; the baseline kernel still takes plain row-major B. Correctness at M=N=K=256: run baseline + trfeed, compare both to the CPU oracle (max-rel-err < 0.03). Perf + agreement at 4096³ (Task 3).

- [ ] **Step 2: Add the build target to `build.sh`** (after the probe block):

```bash
echo "== build gemm_trfeed_bench =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" \
  "$HERE/bench/gemm_trfeed_bench.hip" -o "$HERE/out/gemm_trfeed_bench"
```

- [ ] **Step 3: Build + run the 256³ oracle gate**

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include bench/gemm_trfeed_bench.hip -o out/gemm_trfeed_bench
timeout 90 ./out/gemm_trfeed_bench
```
Expected: `baseline ... PASS` and `trfeed ... PASS` (both max-rel-err < 0.03). **This is the Phase-1 correctness gate.** If trfeed FAILS, the tile addressing (`b_tile_offset`/`kt`/`nt`) is wrong — re-derive against `global_load_tr_contract.md`; do not proceed to perf.

- [ ] **Step 4: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/gemm_trfeed_bench.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh
git commit -m "feat(MAD-305 P1): trfeed bench (global_load_tr B-feed) + 256^3 oracle gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 4096³ measurement + the L2-reuse evaluation (the Phase-1 gate)

**Files:**
- Reference: `bench/gemm_trfeed_bench.hip`

- [ ] **Step 1: Run the 4096³ perf + agreement** (the bench's perf path prints both kernels):

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
timeout 120 ./out/gemm_trfeed_bench | tee /tmp/trfeed_perf.txt
```
Expected: a table with `baseline (byte-gather)` and `trfeed (global_load_tr)` TFLOPS, `% of 307`, `× 143`, and `trfeed vs baseline max|Δ|` (must be ~0 — same math).

- [ ] **Step 2: Evaluate the crux.** Record in `RESULT.md` (Task in this plan's Task 6). Decision:
  - trfeed **> baseline and ≥ ~143** → wide feed converts; L2 reuse holds. Phase 1 PASS → Task 5 (fold into `gemm_wmma.hip`), then Phase 2.
  - trfeed **> baseline but < 143** → partial; B re-fetch is limiting. Go to Task 4 (larger-M-tile fallback), re-measure.
  - trfeed **≤ baseline** → STOP and report; the direct-global feed is L2-bound at this tiling. Reassess (the bench isolates this safely; no production code touched yet).

---

## Task 4: (conditional) Larger-M-tile fallback to amortize B re-fetch

Only if Task 3 shows trfeed B-reuse-limited.

**Files:**
- Modify: `bench/gemm_trfeed_bench.hip`

- [ ] **Step 1:** Add a `gemm_fp8_trfeed_bm256` variant: `BM=256` (keep `BN=128`, `BK=32`), `WAVES_M=4` (8 waves/block, `BLOCK_THREADS=256`), `FRAGS_M=(256/4)/16=4`. Each B fragment (`tr_load8`) is now reused across 4 (was 2) M-fragments per wave-row, halving B global traffic per MAC. Reuse the same A-LDS + trfeed B path; only the M-tiling constants and `acc[FRAGS_M][FRAGS_N]` grow. Keep the 256³ oracle gate green.

- [ ] **Step 2:** Build + run; compare `bm256` TFLOPS vs the `bm128` trfeed and the 143 bar. Record both rows.

- [ ] **Step 3: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/gemm_trfeed_bench.hip
git commit -m "feat(MAD-305 P1): larger-M-tile trfeed variant (amortize B re-fetch)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Fold the winning trfeed B-feed into `gemm_wmma.hip`

**Files:**
- Modify: `gemm_wmma.hip`
- Modify: `gemm_capi.h`

- [ ] **Step 1:** Add `#include "bench/trfeed_common.h"` to `gemm_wmma.hip`. Add a new kernel `gemm_fp8_trfeed_tiled` = the current `gemm_fp8_tiled` with the fp8 B path swapped to the winning trfeed approach from Task 3/4 (drop the `Bs` LDS staging + `load_matrix_sync(fb...)`; load `fb[ni]` via `tr_load8` from a preshuffled-B argument with the winning M-tiling). Keep the A path, accumulator, epilogue, and the `__launch_bounds__` intact.

- [ ] **Step 2:** Add a C-API entry `void rdna4_gemm_fp8_trfeed_forward(const void* A, const void* Bshuf, void* C, const float* a_scale, const float* b_scale, int M, int N, int K, hipStream_t stream);` to `gemm_capi.h` (B is **pre-shuffled** — caller repacks weights once at load via `preshuffle_B`). Leave the existing `rdna4_gemm_fp8_forward` (byte-gather) in place.

- [ ] **Step 3: Build the shared lib + smoke** the new entry against the oracle (reuse the bench's 256³ oracle calling `rdna4_gemm_fp8_trfeed_forward`). Expected: PASS.

```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
ARCH=gfx1201 MEM_MAX=6G bash build.sh 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_capi.h
git commit -m "feat(MAD-305 P1): fold trfeed B-feed into gemm_wmma.hip (preshuffled-B entry)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Record the Phase 1 outcome

**Files:**
- Modify: `RESULT.md`

- [ ] **Step 1:** Fill the Phase 1 row(s) of the `RESULT.md` ladder table with measured TFLOPS / % of 307 / × 143 / oracle status, and append a short prose block: the measured number, the L2-reuse verdict (held / needed larger-M fallback), and whether Phase 1 cleared the 143 bar. Note the next phase (async double-buffer).

- [ ] **Step 2: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/RESULT.md
git commit -m "docs(MAD-305 P1): record wide-feed measurement + L2-reuse verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** Implements Spec Phase 1 (replace B byte-gather with `global_load_tr` wide feed; oracle gate; measure vs 307/143; L2-reuse crux + larger-M fallback; fold into `gemm_wmma.hip`). Phases 2–5 remain gated follow-ons.
- **Placeholder scan:** The one `<copy the epilogue block from feedwidth_proto.hip verbatim>` marker (Task 2 Step 1) points at concrete existing code in a named file — Task 2 Step 1 instructs copying it. All new logic (trperm, preshuffle, b_tile_offset, trfeed B path, fallback constants) is shown in full.
- **Type consistency:** `trperm(L,s)`, `b_tile_offset(kt,nt,NT)`, `preshuffle_B(B,Bshuf,K,N)` defined in Task 1 and used unchanged in Tasks 2/4/5. `tr_load8`, tile constants, and the WMMA builtin name match `feedwidth_proto.hip` / the Phase-0 probe. New C-API `rdna4_gemm_fp8_trfeed_forward` (preshuffled B) is distinct from the existing `rdna4_gemm_fp8_forward`.

---

## Remaining ladder (after the Phase 1 gate)

- **Phase 2 — Async double-buffer:** software-pipeline the K-loop (prefetch A via `global_load_lds` + next-tile B transpose-loads), ping-pong à la ck_tile `comp_v6`.
- **Phase 3 — Big tiles + scheduler + wave32 occupancy retune** (AITER `ck_gemm_a8w8_blockscale` config).
- **Phase 4 — ml8 4-bit LUT front-end** on the optimized core (unpack-to-fp8-then-feed).
- **Phase 5 — Production integration + PPL-neutral** through the llama.cpp ml8 path.
