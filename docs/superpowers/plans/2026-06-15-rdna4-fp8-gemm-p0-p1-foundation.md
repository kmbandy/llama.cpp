# RDNA4 fp8 GEMM — P0 Foundation + P1 Occupancy Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a hand-written gfx1201 fp8 forward GEMM (rocWMMA/HIP) and drive it from naive-correct to **≥143 TF (hipBLASLt parity)** via occupancy-aware tiling — the unified-substrate foundation — plus a fail-fast Phase-2 dynamic-VGPR spike.

**Architecture:** One templated WMMA compute core fed by an fp8 front-end (and later an ml8 LUT-dequant front-end), called through a C API. Correctness is gated by a Python oracle (kernel ≡ `torch._scaled_mm` / the ml8 dequant reference, bit-exact within fp8); perf is ratcheted by a bench reporting **% of the measured 307 TF ceiling** and **× hipBLASLt**.

**Tech Stack:** HIP + rocWMMA (ROCm 7.2, gfx1201/R9700), `hipcc`/`amdclang++`, PyTorch 2.13 (ROCm) for the oracle/bench via `ctypes`, Python 3 (system interpreter resolves torch + the GPU). Spec: `docs/superpowers/specs/2026-06-15-rdna4-fp8-gemm-occupancy-unlock-design.md`.

**Standing constraints:**
- **RAM safety (15 GB host):** every C++/HIP build runs through the RAM-capped `build.sh` (systemd scope, MemoryMax). Never an uncapped LLVM build.
- **fp8 type compatibility (verified):** `torch.float8_e4m3fn` == rocWMMA `float8_t` (OCP e4m3) byte-for-byte; `torch.bfloat16` == `__hip_bfloat16`. Pass torch GPU tensor `.data_ptr()` straight through ctypes.
- **rocprof:** plain `--stats` does not run on gfx1201; use `rocprof --hip-trace` for kernel-time confirmation. The median-wall-time bench is the primary signal.
- **The oracle is the arbiter** on any rocWMMA fragment-layout ambiguity (row_major vs col_major / leading dimension): if a layout choice fails the oracle, flip it — do not loosen tolerance.

---

## File Structure

All under a new `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`:

| File | Responsibility |
|---|---|
| `bench/wmma_peak.hip` | the 307 TF raw-WMMA ceiling microbench (promoted from `/tmp/wmma_peak.hip`) — ceiling oracle + regression guard |
| `gemm_capi.h` | stable C API the consumers + the Python harness call |
| `gemm_wmma.hip` | the templated WMMA compute core + fp8 front-end + host launch implementing `gemm_capi.h` |
| `build.sh` | RAM-capped build → `librdna4_gemm.so` + the bench binaries |
| `test/oracle_harness.py` | correctness oracle: ctypes-load the `.so`, run kernel, compare to `torch._scaled_mm` (fp8) and the ml8 dequant reference |
| `bench/gemm_bench.py` | perf ratchet: % of 307, × hipBLASLt, at real model shapes |
| `spike/dyn_vgpr_spike.hip` | Phase-2 de-risk: launch DYN_VGPR_EN + emit `s_alloc_vgpr` + measure occupancy |

ml8 front-end (Task 7) is added into `gemm_wmma.hip` as a second front-end sharing the compute core.

---

## Task 1: Promote the WMMA ceiling microbench + RAM-capped build

**Files:**
- Create: `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/wmma_peak.hip` (copy of `/tmp/wmma_peak.hip`)
- Create: `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh`

- [ ] **Step 1: Copy the proven microbench into the repo**

```bash
mkdir -p ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/{bench,test,spike}
cp /tmp/wmma_peak.hip ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/wmma_peak.hip
```
(If `/tmp/wmma_peak.hip` is gone, it is the rocWMMA back-to-back `mma_sync` throughput probe — fp8 + f16, NACC ILP, grid-saturated — that measured fp8=307 TF / f16=147 TF. Reconstruct from the spec §1 description.)

- [ ] **Step 2: Write the RAM-capped build script**

Create `build.sh`:
```bash
#!/usr/bin/env bash
# RAM-capped HIP build for the RDNA4 fp8 GEMM (15 GB host — never uncapped).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-gfx1201}"
MEM_MAX="${MEM_MAX:-6G}"
ROCM_INC="${ROCM_INC:-/opt/rocm/include}"
cap() { systemd-run --user --scope -p MemoryMax="$MEM_MAX" -p MemoryHigh=5G "$@"; }

echo "== free RAM =="; free -h | awk 'NR<=2'
avail="$(free -m | awk '/^Mem:/{print $7}')"
[ "${avail:-0}" -ge 4000 ] || { echo "ABORT: <4GB available"; exit 1; }

mkdir -p "$HERE/out"
echo "== build ceiling microbench =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" "$HERE/bench/wmma_peak.hip" -o "$HERE/out/wmma_peak"

if [ -f "$HERE/gemm_wmma.hip" ]; then
  echo "== build librdna4_gemm.so =="
  cap hipcc --offload-arch="$ARCH" -O3 -fPIC --shared -I"$ROCM_INC" \
    "$HERE/gemm_wmma.hip" -o "$HERE/out/librdna4_gemm.so"
fi
echo "== DONE =="
```
```bash
chmod +x ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh
```

- [ ] **Step 3: Build + run the ceiling microbench**

Run: `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh && ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/out/wmma_peak`
Expected: `fp8 ... ~30x.x TF` peaking ≥ 290 TF and `f16 ... ~147 TF`, fp8/f16 ≈ 2.0×. This is the ceiling oracle. If fp8 < 290 TF, STOP and report (the toolchain/clock regressed vs the measured 307).

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/wmma_peak.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh
git commit -m "rdna4-gemm P0: promote 307 TF WMMA ceiling microbench + RAM-capped build"
```

---

## Task 2: Naive-but-correct rocWMMA fp8 forward GEMM + C API + oracle

**Files:**
- Create: `rdna4_fp8_gemm/gemm_capi.h`
- Create: `rdna4_fp8_gemm/gemm_wmma.hip`
- Create: `rdna4_fp8_gemm/test/oracle_harness.py`

- [ ] **Step 1: Write the C API header**

Create `gemm_capi.h`:
```c
#pragma once
#include <hip/hip_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// C = (A_fp8[M,K] @ B_fp8[K,N]) * a_scale[M] (per-row) * b_scale[N] (per-col), out bf16[M,N].
// A,B are float8_e4m3 (OCP). All device pointers. Row-major. M,N multiples of 16; K multiple of 16.
void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                            const float* a_scale, const float* b_scale,
                            int M, int N, int K, hipStream_t stream);
#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the failing oracle test**

Create `test/oracle_harness.py`:
```python
#!/usr/bin/env python3
"""Correctness oracle for the RDNA4 fp8 GEMM: kernel == torch._scaled_mm within fp8."""
import ctypes, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent.parent
LIB = HERE / "out" / "librdna4_gemm.so"


def _lib():
    lib = ctypes.CDLL(str(LIB))
    lib.rdna4_gemm_fp8_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.rdna4_gemm_fp8_forward.restype = None
    return lib


def gemm_fp8(a_fp8, b_fp8, a_scale, b_scale):
    M, K = a_fp8.shape; N = b_fp8.shape[1]
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    _lib().rdna4_gemm_fp8_forward(
        a_fp8.data_ptr(), b_fp8.data_ptr(), c.data_ptr(),
        a_scale.data_ptr(), b_scale.data_ptr(), M, N, K, None)
    torch.cuda.synchronize()
    return c


def _case(M, N, K, seed, tol):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    a = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    b = (torch.randn(K, N, device=dev) * 0.3).to(torch.float8_e4m3fn)
    a_scale = (torch.rand(M, device=dev) * 0.1 + 0.01)
    b_scale = (torch.rand(N, device=dev) * 0.1 + 0.01)
    # reference: torch._scaled_mm wants x row-major, w col-major; scales [M,1],[1,N]
    ref = torch._scaled_mm(a, b.t().contiguous().t(),
                           scale_a=a_scale[:, None].float(), scale_b=b_scale[None, :].float(),
                           out_dtype=torch.bfloat16).to(torch.float32)
    out = gemm_fp8(a, b, a_scale, b_scale).to(torch.float32)
    max_err = (out - ref).abs().max().item()
    assert max_err < tol, f"M={M} N={N} K={K}: max_err {max_err:.4g} >= {tol}"


def test_single_tile():   _case(16, 16, 16, 1, 5e-2)
def test_square():        _case(256, 256, 256, 2, 5e-2)
def test_real_shape():    _case(2048, 2560, 9216, 3, 5e-2)  # down-proj-ish

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no GPU"); sys.exit(1)
    for t in (test_single_tile, test_square, test_real_shape):
        t(); print(f"  ✓ {t.__name__}")
    print("PASS")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python3 ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/test/oracle_harness.py`
Expected: FAIL — `OSError`/`librdna4_gemm.so` not found (kernel not built yet).

- [ ] **Step 4: Write the naive kernel + host launch**

Create `gemm_wmma.hip`:
```cpp
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "gemm_capi.h"
using namespace rocwmma;

// Naive: one wave (32 lanes) computes one 16x16 C tile, looping K in steps of 16.
// Correctness-first; occupancy tiling comes in P1. Epilogue scales via an LDS f32 scratch.
__global__ void gemm_fp8_naive(const float8_t* A, const float8_t* B, __hip_bfloat16* C,
                               const float* a_scale, const float* b_scale,
                               int M, int N, int K) {
    int ntiles_n = N / 16;
    int tm = (blockIdx.x / ntiles_n) * 16;   // C tile row origin
    int tn = (blockIdx.x % ntiles_n) * 16;   // C tile col origin
    fragment<accumulator, 16, 16, 16, float> acc;
    fragment<matrix_a, 16, 16, 16, float8_t, row_major> fa;   // A[M,K] row-major, ld=K
    fragment<matrix_b, 16, 16, 16, float8_t, row_major> fb;   // B[K,N] row-major, ld=N
    fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(fa, A + tm * K + k, K);
        load_matrix_sync(fb, B + k * N + tn, N);
        mma_sync(acc, fa, fb, acc);
    }
    __shared__ float scratch[16 * 16];
    store_matrix_sync(scratch, acc, 16, mem_row_major);
    __syncthreads();
    // 32 lanes cooperatively scale + cast 256 elements -> bf16
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int r = tm + idx / 16, c = tn + idx % 16;
        float v = scratch[idx] * a_scale[r] * b_scale[c];
        C[r * N + c] = (__hip_bfloat16)v;
    }
}

extern "C" void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                                       const float* a_scale, const float* b_scale,
                                       int M, int N, int K, hipStream_t stream) {
    dim3 grid((M / 16) * (N / 16));
    gemm_fp8_naive<<<grid, 32, 0, stream>>>(
        (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, a_scale, b_scale, M, N, K);
}
```

- [ ] **Step 5: Build + run the oracle**

Run: `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh && python3 ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/test/oracle_harness.py`
Expected: `PASS` (all 3 cases). If `test_single_tile` fails on a transpose-looking error (output is the transpose of ref, or wrong by a permutation), flip `fb` to `col_major` with `load_matrix_sync(fb, B + k*N + tn, N)` — the oracle is the arbiter. If it fails numerically large everywhere, check the scale indexing (`a_scale[r]`, `b_scale[c]`).

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_capi.h \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/test/oracle_harness.py
git commit -m "rdna4-gemm P0: naive-correct rocWMMA fp8 forward GEMM + C API + oracle (green)"
```

---

## Task 3: Perf bench (% of 307, × hipBLASLt) + naive baseline

**Files:**
- Create: `rdna4_fp8_gemm/bench/gemm_bench.py`

- [ ] **Step 1: Write the bench**

Create `bench/gemm_bench.py`:
```python
#!/usr/bin/env python3
"""Perf ratchet for the RDNA4 fp8 GEMM: TFLOPS, % of measured 307 TF ceiling, x hipBLASLt."""
import statistics, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from oracle_harness import gemm_fp8  # noqa: E402

CEILING_TF = 307.0          # measured raw fp8 WMMA peak (wmma_peak.hip)


def tflops(M, N, K, sec):
    return 2.0 * M * N * K / sec / 1e12


def _median(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def shapes():
    return [("gate", 9216, 2560), ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def main():
    dev = torch.device("cuda"); M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    print(f"M={M}  ceiling={CEILING_TF} TF")
    print(f"{'shape':7} {'ours TF':>8} {'%307':>6} {'hipBLASLt TF':>12} {'x hbl':>6}")
    for name, N, K in shapes():
        a = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
        b = (torch.randn(K, N, device=dev) * 0.3).to(torch.float8_e4m3fn)
        asc = torch.ones(M, device=dev); bsc = torch.ones(N, device=dev)
        ours = tflops(M, N, K, _median(lambda: gemm_fp8(a, b, asc, bsc)))
        bt = b.t().contiguous().t()
        sa = torch.ones((M, 1), device=dev); sb = torch.ones((1, N), device=dev)
        hbl = tflops(M, N, K, _median(lambda: torch._scaled_mm(a, bt, scale_a=sa, scale_b=sb,
                                                               out_dtype=torch.bfloat16)))
        print(f"{name:7} {ours:8.1f} {ours/CEILING_TF*100:5.1f}% {hbl:12.1f} {ours/hbl:5.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the naive baseline**

Run: `python3 ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/gemm_bench.py 8192`
Expected: prints our TFLOPS (naive will be low — single-tile-per-wave, no reuse, likely ~10–40 TF), the % of 307, and hipBLASLt (~140–156). Record the table — this is the P1 starting line.

- [ ] **Step 3: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/gemm_bench.py
git commit -m "rdna4-gemm P0: perf bench (% of 307, x hipBLASLt) + naive baseline recorded"
```

---

## Task 4: Tiled core with LDS staging (multi-WMMA-per-wave reuse)

**Files:**
- Modify: `rdna4_fp8_gemm/gemm_wmma.hip` (replace the naive kernel's inner structure)

**Goal:** stop computing one 16×16 tile per wave. Each wave/block computes a larger `BLOCK_M×BLOCK_N` output tile from multiple WMMA fragments, staging A/B `BLOCK_*×BLOCK_K` tiles through LDS so each loaded element feeds many MACs. This is where reuse (and TFLOPS) jumps. **Oracle stays green; bench ratchets up.**

- [ ] **Step 1: gitnexus impact (repo rule) + confirm oracle green**

Run `gitnexus_impact({target: "rdna4_gemm_fp8_forward", direction: "upstream"})` (new symbol → low). Then `python3 .../test/oracle_harness.py` → PASS (pre-edit anchor).

- [ ] **Step 2: Implement the LDS-tiled kernel**

Replace `gemm_fp8_naive` with a tiled kernel. Structure (fill in against rocWMMA + ISA §11.6.2; the oracle + bench are the arbiters):
```cpp
// Tunables (start here; Task 6 sweeps them):
constexpr int BM = 64, BN = 64, BK = 32;   // output tile + K-step
constexpr int WM = BM / 16, WN = BN / 16;  // WMMA frags per tile = 4x4
// blockDim = WM*WN waves * 32 lanes? -> choose a warp layout; start blockDim=256 (8 waves).
// __shared__ float8_t As[BM*BK], Bs[BK*BN];  // double-buffer in Task 5
// Each iter: cooperatively global->LDS load A,B tiles; __syncthreads();
//   for each (wm,wn) this wave owns: load_matrix_sync from LDS, mma into acc[wm][wn];
// advance K. Epilogue: scale + bf16 store all WM*WN frags.
```
Provide the full kernel: global→LDS coalesced loads (each thread loads `BM*BK/blockDim` A elems + `BK*BN/blockDim` B elems), `load_matrix_sync` from the LDS tiles into fragments, accumulate `acc[WM][WN]`, then the scaled bf16 epilogue (LDS f32 scratch per fragment as in Task 2). Keep `BK=32` (two 16-wide WMMA K-steps per LDS load) to start.

- [ ] **Step 3: Oracle must stay green**

Run: `python3 .../test/oracle_harness.py` → PASS (same tolerances). On failure, use superpowers:systematic-debugging — suspects: LDS tile indexing, the global→LDS load mapping, fragment ld from LDS, or accumulator→C mapping in the epilogue. One hypothesis at a time. Do NOT loosen tolerance.

- [ ] **Step 4: Bench must improve**

Run: `python3 .../bench/gemm_bench.py 8192`. Expected: a large jump over the naive baseline (reuse engaged). Record.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip
git commit -m "rdna4-gemm P1: LDS-tiled WMMA core (multi-frag reuse, oracle green, bench up)"
```

---

## Task 5: Double-buffered K-loop + load-transpose feeding (ISA §11.6.2)

**Files:**
- Modify: `rdna4_fp8_gemm/gemm_wmma.hip`

**Goal:** overlap the next K-tile's global→LDS load with the current tile's WMMA compute (double-buffer LDS), and feed fragments via the WMMA **load-transpose** instructions (ISA §11.6.2) to avoid the swizzle/transpose penalty. This is the latency-hiding that lifts utilization toward parity.

- [ ] **Step 1: Confirm oracle green (anchor)** — `python3 .../test/oracle_harness.py` → PASS.

- [ ] **Step 2: Implement double-buffering + load-transpose**

In `gemm_wmma.hip`: allocate two LDS buffers `As[2][...]`, `Bs[2][...]`; prologue-load buffer 0; in the K-loop, issue the global→LDS load for buffer `(i+1)%2` *before* the WMMA on buffer `i%2`, with `s_waitcnt` ordering managed so compute overlaps load. Replace the plain LDS→fragment loads with the load-transpose path per ISA §11.6.2 (read the section; cite the exact `ds`-transpose op used). Keep multiple accumulator fragments live (ILP) so WMMA latency is hidden — the microbench proved NACC≥8 saturates.

- [ ] **Step 3: Oracle green** — `python3 .../test/oracle_harness.py` → PASS. (Suspect on failure: double-buffer index parity, missing `__syncthreads()` between load and consume, load-transpose layout.)

- [ ] **Step 4: Bench up + rocprof confirm** — `python3 .../bench/gemm_bench.py 8192` (record), then optionally `rocprof --hip-trace ...` to confirm kernel GPU time dropped. Expected: utilization climbing toward ~100+ TF.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip
git commit -m "rdna4-gemm P1: double-buffered K-loop + load-transpose feeding (oracle green, bench up)"
```

---

## Task 6: Occupancy tuning → hipBLASLt parity (≥143 TF)

**Files:**
- Modify: `rdna4_fp8_gemm/gemm_wmma.hip` (tile/warp/ILP constants)

**Goal:** find the `BM×BN×BK` + warps + accumulator-ILP point that holds reuse **without** crushing occupancy below the wave count needed to hide memory latency. The diagnostic (HIP equivalent of Triton's `n_regs`): compile with `-Rpass-analysis=kernel-resource-usage` to print VGPR/SGPR/LDS/occupancy per kernel. Target: **bench ≥ 143 TF (parity)**, stretch ~180.

- [ ] **Step 1: Add the VGPR/occupancy readout to the build**

Append to `build.sh` the `.so` compile line (one-off, manual): add `-Rpass-analysis=kernel-resource-usage` and capture stderr, e.g.
`ARCH=gfx1201 hipcc --offload-arch=gfx1201 -O3 -fPIC --shared -Rpass-analysis=kernel-resource-usage -I/opt/rocm/include gemm_wmma.hip -o out/librdna4_gemm.so 2>&1 | grep -iE 'VGPR|occupancy'`
Note the kernel's VGPRs and occupancy (waves/SIMD). RDNA4: ≤96 VGPRs = full 16 waves; 1536 VGPR-slots/SIMD.

- [ ] **Step 2: Sweep the config (oracle-gated each pick)**

For `BM,BN ∈ {64,128}`, `BK ∈ {32,64}`, warps ∈ {4,8}, accumulator-ILP per the WM×WN: rebuild, run the oracle (must PASS), run the bench, record VGPR/occupancy + TFLOPS. Keep the config that maximizes bench TFLOPS while the oracle is green. (Avoid spills — `n_spills`/scratch in the resource readout = disqualify.)

- [ ] **Step 3: Lock the winning config + confirm parity**

Set the winning constants in `gemm_wmma.hip`. Run: `python3 .../bench/gemm_bench.py 8192` → **our TF ≥ 143 (× hipBLASLt ≥ 1.0)**. If the best honest result is below 143, that is a real finding — record it with the occupancy data and the wall; do not fake it.

- [ ] **Step 4: Oracle final green** — `python3 .../test/oracle_harness.py` → PASS at the locked config.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh
git commit -m "rdna4-gemm P1: occupancy-tuned config -> hipBLASLt parity (>=143 TF, oracle green)"
```

---

## Task 7: ml8 front-end (LUT dequant → fp8 fragments, shared core)

**Files:**
- Modify: `rdna4_fp8_gemm/gemm_capi.h` (add `rdna4_gemm_ml8_forward`)
- Modify: `rdna4_fp8_gemm/gemm_wmma.hip` (ml8 front-end)
- Modify: `rdna4_fp8_gemm/test/oracle_harness.py` (ml8 oracle case)

**Goal:** feed the *same* tiled core from 4-bit ml8 weights — dequant LUT indices → fp8 in the global→LDS load (prologue), then identical WMMA. The dequant is cheap (proven); it rides for free on the occupancy-fixed core.

- [ ] **Step 1: Add the ml8 C API**

In `gemm_capi.h`:
```c
// B is ml8: packed 4-bit indices [K/2, N] uint8 (lo-first) + per-K-group fp8 centroid LUT
// [n_groups_k,16] + per-(group,N) fp32 scale [n_groups_k, N]. group_size = K / n_groups_k.
void rdna4_gemm_ml8_forward(const void* A, const void* B_idx, void* C,
                            const float* a_scale, const void* centroids_fp8,
                            const float* b_group_scale,
                            int M, int N, int K, int group_size, hipStream_t stream);
```

- [ ] **Step 2: Write the failing ml8 oracle case**

In `test/oracle_harness.py`, add an ml8 case that reuses the existing dequant reference. Import the proven reference + packing from the prior oracle:
```python
def test_ml8_real_shape():
    sys.path.insert(0, str(HERE.parent.parent.parent.parent.parent / "tests"))
    from test_ml8_kernel_stage1_dequant import reference_dequant_gemm  # bit-exact ml8 ref
    # build random ml8 layer (centroids[G,16] fp8, idx[K,N], group=64), call rdna4_gemm_ml8_forward,
    # compare to reference_dequant_gemm within 5e-2. (Mirror the packing in
    # tests/test_ml8_gemm_optimization.py::_pack_kn.)
    ...
```
(Write the full case: random centroids→fp8, random indices [0,15], pack lo-first to [K/2,N], per-group scales; call the ctypes wrapper; compare to `reference_dequant_gemm`.)

- [ ] **Step 3: Run it to verify it fails** — `python3 .../test/oracle_harness.py` → the ml8 case errors (no `rdna4_gemm_ml8_forward`).

- [ ] **Step 4: Implement the ml8 front-end**

In `gemm_wmma.hip`: a second host entry + kernel (or a templated front-end flag) that, in the global→LDS load, reads the packed nibble, looks up the per-K-group fp8 centroid (LDS-resident 16-entry LUT per group), writes fp8 into the same `Bs` LDS tile — then the identical tiled WMMA core runs. Apply the per-group scale in the epilogue.

- [ ] **Step 5: Oracle green** — `python3 .../test/oracle_harness.py` → PASS (fp8 cases + ml8 case).

- [ ] **Step 6: Bench ml8 + commit**

Add an ml8 row to `gemm_bench.py` (or a `--ml8` flag) and record ml8-forward TFLOPS (should track the fp8 core minus a small dequant cost). Then:
```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_capi.h \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/gemm_wmma.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/test/oracle_harness.py \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/gemm_bench.py
git commit -m "rdna4-gemm P1: ml8 LUT-dequant front-end on the shared tiled core (oracle green)"
```

---

## Task 8: Phase-2 dynamic-VGPR de-risk spike

**Files:**
- Create: `rdna4_fp8_gemm/spike/dyn_vgpr_spike.hip`
- Create: `rdna4_fp8_gemm/spike/FINDINGS.md`

**Goal:** answer the two campaign-gating unknowns (spec §7) before any P2 kernel work, and produce the "what's missing in the toolchain" report (spec §11). **This is a research spike, not TDD** — success = occupancy provably rises; failure = a documented wall.

- [ ] **Step 1: Check whether LLVM 22 already exposes the intrinsic**

Run probes (cheap, no GPU): grep clang/LLVM for an `s.alloc.vgpr` intrinsic/builtin and try to `-fsyntax-only` compile a call:
```bash
echo '__global__ void k(){ __builtin_amdgcn_s_alloc_vgpr(64); }' | \
  hipcc --cuda-device-only -fsyntax-only --offload-arch=gfx1201 -xhip - 2>&1 | head -3
llvm-mc -triple=amdgcn -mcpu=gfx1201 <<<'s_alloc_vgpr 64' 2>&1 | head -3   # assembler-level
```
Record in `FINDINGS.md`: does the builtin exist? does the assembler accept `s_alloc_vgpr` (opcode 83, ISA §3.3.3 / line ~14348)? → defines whether the upstream contribution is an LLVM intrinsic, HIP runtime support, or both.

- [ ] **Step 2: Determine how to launch in DYN_VGPR_EN mode**

Investigate (ISA §3.3.3: dynamic-VGPR mode is a dispatch/launch property, takes over the whole WGP). Check whether HIP/ROCr exposes it (kernel attribute, `hipModuleLaunchKernel` config, code-object metadata, or none). Document the mechanism — or that it requires a hand-built kernel descriptor / HSA path. Record in `FINDINGS.md`.

- [ ] **Step 3: Write the spike kernel + measure occupancy**

Create `spike/dyn_vgpr_spike.hip`: a kernel that (per Steps 1–2) enters dynamic-VGPR mode, starts lean, calls `s_alloc_vgpr` (builtin if it exists, else inline asm against the §3.3.3 encoding), checks SCC for success, then does dummy WMMA work needing the big allocation. Launch enough blocks to fill the GPU and measure resident waves/SIMD (via the resource-usage readout under dynamic mode, or by occupancy inference from timing vs a static-VGPR control). Build via `build.sh` (RAM-capped).

- [ ] **Step 4: Verdict**

Run the spike. Success criterion: a big-tile-class kernel that runs at **more waves/SIMD than the static-VGPR equivalent** (e.g. 12–16 vs 6–8). Write the verdict to `FINDINGS.md`: GO (mechanism works, occupancy rises → Phase 2 viable) or NO-GO (documented wall → ship Phase 1 parity as the win, per spec §2/§7). Include the toolchain-gap list = the upstream PR scope (spec §11).

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/
git commit -m "rdna4-gemm P2 spike: s_alloc_vgpr emit + DYN_VGPR launch + occupancy verdict (FINDINGS.md)"
```

---

## After this plan

P2 dynamic-VGPR implementation (gated on Task 8 GO), the backward pass (dgrad/wgrad — spec §4.1), and consumer integration (llama.cpp `ggml-cuda` op + trainer ctypes/pybind) each get their own plan. Dispatch a final code review over the whole P0/P1 change set, then `superpowers:finishing-a-development-branch`.

---

## Self-Review (planner)

**Spec coverage:** §2 P1 target (≥143) → Tasks 4–6; §3 architecture (core + fp8 + ml8 front-ends) → Tasks 2,4–7; §5 components (microbench, harness, bench, spike) → Tasks 1,2,3,8; §6 ISA grounding (§11.6.2 load-transpose, §3.3.3 dyn-VGPR) → Tasks 5,8; §7 spike → Task 8; §8 gates (oracle + ratchet) → every task; §11 upstream report → Task 8.4. P2 impl / backward / integration explicitly deferred to later plans (spec §4.1, §2) — in-scope-of-spec, out-of-scope-of-this-plan, stated.

**Placeholder scan:** Tasks 1–3 carry complete code. Tasks 4–7 carry concrete skeletons + exact APIs/ISA refs + the oracle/bench/diagnostic loop — the honest form for measure-driven kernel optimization (a pre-written final optimized kernel would be fiction). Task 7 Step 2/4 and Task 4 Step 2 say "provide the full kernel/case" as the implementer's deliverable with the structure given — acceptable because the exact tiling is empirically tuned, but flagged so the implementer knows to write real code, not stubs.

**Type/identifier consistency:** `rdna4_gemm_fp8_forward` (capi/host/ctypes) and `rdna4_gemm_ml8_forward` consistent across gemm_capi.h, gemm_wmma.hip, oracle_harness.py. `gemm_fp8` / `CEILING_TF=307` consistent across harness + bench. `float8_t`/`__hip_bfloat16` match the torch dtypes per the verified compatibility note. Build artifacts land in `out/`; harness loads `out/librdna4_gemm.so`.
