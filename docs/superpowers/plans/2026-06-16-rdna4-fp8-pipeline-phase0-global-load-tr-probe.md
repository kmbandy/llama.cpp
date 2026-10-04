# RDNA4 fp8 Pipeline — Phase 0: `global_load_tr_b64` Layout Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically nail the exact gfx1201 `global_load_tr_b64` fp8 addressing/layout contract — what global pointer each lane passes and how the returned `v2i32` maps onto the §7.12 WMMA B-fragment — so Phase 1 can wire the wide feed with zero guessing.

**Architecture:** A standalone HIP microprobe loads a uniquely-byte-tagged 16×16 fp8 tile through the hardware transpose-load and dumps each lane's 8 returned bytes; the byte values reveal the (lane, slot)→(k, n) mapping. We then prove the mapping by feeding a single 16×16×16 fp8 WMMA from it and comparing against a CPU e4m3 oracle. Output: a documented, oracle-verified `tr_load_b_fragment()` contract helper.

**Tech Stack:** HIP / hipcc, gfx1201 (RDNA4 wave32), rocWMMA `float8_t`, `__builtin_amdgcn_global_load_tr_b64_v2i32`, `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`. Builds RAM-capped via `systemd-run`. **All normal HIP — unsupervised, no PM4, no hang risk.**

**Spec:** `docs/superpowers/specs/2026-06-16-rdna4-cdna4-transpose-fed-fp8-wmma-pipeline-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/global_load_tr_probe.hip` | The probe + single-tile WMMA validation + CPU oracle (all in one self-contained file) |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/global_load_tr_contract.md` | The documented, verified mapping contract Phase 1 consumes |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh` | Add a RAM-capped build target for the probe |
| `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/RESULT.md` | Append the Phase 0 outcome row |

All paths below are relative to `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`. Always `cd` to that absolute directory before building/running (shell cwd drifts after `git commit`).

---

## Task 1: Probe harness — dump the transpose-load byte distribution

**Files:**
- Create: `bench/global_load_tr_probe.hip`

- [ ] **Step 1: Write the probe kernel + host driver**

```cpp
// bench/global_load_tr_probe.hip — Phase 0: discover gfx1201 global_load_tr_b64 fp8 layout.
// Unsupervised normal-HIP launch (no PM4). Mirrors ck/utility/amd_transpose_load.hpp fp8 path.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

typedef int v2i32 __attribute__((ext_vector_type(2)));

// gfx12 global transpose-load, 8-bit element / b64 form (one wave32 lane gets 8 bytes).
__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto gptr = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                    reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(gptr);
}

// One wave32. Lane L passes the "natural contiguous" base (L*8 bytes into the tile).
// The hardware transpose redistributes; we record exactly what each lane receives.
extern "C" __global__ void probe(const uint8_t* __restrict__ Bsrc, uint8_t* __restrict__ out) {
    int lane = threadIdx.x;                 // 0..31
    v2i32 r = tr_load8(Bsrc + lane * 8);
    uint32_t lo = (uint32_t)r.x, hi = (uint32_t)r.y;
    #pragma unroll
    for (int s = 0; s < 4; s++) out[lane * 8 + s]     = (lo >> (s * 8)) & 0xFF;
    #pragma unroll
    for (int s = 0; s < 4; s++) out[lane * 8 + 4 + s] = (hi >> (s * 8)) & 0xFF;
}

int main() {
    const int NB = 256;                     // 16(K) x 16(N) byte tile
    uint8_t hsrc[NB];
    for (int i = 0; i < NB; i++) hsrc[i] = (uint8_t)i;   // byte at [k][n] == k*16 + n (all distinct)
    uint8_t hout[256] = {0};
    uint8_t *dsrc, *dout;
    if (hipMalloc(&dsrc, NB) || hipMalloc(&dout, 256)) { printf("hipMalloc fail\n"); return 3; }
    hipMemcpy(dsrc, hsrc, NB, hipMemcpyHostToDevice);
    hipMemset(dout, 0, 256);
    probe<<<1, 32>>>(dsrc, dout);
    if (hipDeviceSynchronize() != hipSuccess) { printf("launch fail\n"); return 3; }
    hipMemcpy(hout, dout, 256, hipMemcpyDeviceToHost);
    printf("# byte value v at [k][n] == k*16+n, so v -> (k=v/16, n=v%%16)\n");
    printf("lane :  s0  s1  s2  s3  s4  s5  s6  s7   ->  (k,n) per slot\n");
    for (int L = 0; L < 32; L++) {
        printf("%4d :", L);
        for (int s = 0; s < 8; s++) printf(" %3d", hout[L * 8 + s]);
        printf("   ->");
        for (int s = 0; s < 8; s++) printf(" (%d,%d)", hout[L * 8 + s] / 16, hout[L * 8 + s] % 16);
        printf("\n");
    }
    hipFree(dsrc); hipFree(dout);
    return 0;
}
```

- [ ] **Step 2: Add a RAM-capped build target to `build.sh`**

Insert after the `wmma_peak` build block (around `build.sh:16`):

```bash
echo "== build global_load_tr_probe =="
cap hipcc --offload-arch="$ARCH" -O3 -I"$ROCM_INC" \
  "$HERE/bench/global_load_tr_probe.hip" -o "$HERE/out/global_load_tr_probe"
```

- [ ] **Step 3: Build it**

Run (from the rdna4_fp8_gemm dir):
```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
free -m | awk '/^Mem:/{print "avail_MB="$7}'   # must be >= 4000
systemd-run --user --scope -q -p MemoryMax=6G -p MemoryHigh=5G -p MemorySwapMax=0 \
  hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include bench/global_load_tr_probe.hip -o out/global_load_tr_probe
```
Expected: compiles clean (only `-Wnodiscard` warnings acceptable). If `__builtin_amdgcn_global_load_tr_b64_v2i32` is rejected, STOP — the toolchain lacks the builtin (re-check `BuiltinsAMDGPU.def` for `gfx12-insts,wavefrontsize32`).

- [ ] **Step 4: Run it and capture the table**

Run:
```bash
timeout 60 ./out/global_load_tr_probe | tee /tmp/tr_probe_table.txt
```
Expected: a 32-row table, each row 8 distinct byte values in 0..255. **Analyze:** for each lane L, the 8 `(k,n)` pairs reveal what the lane holds. A clean fp8 WMMA B-feed should show each lane holding **8 contiguous K-values of a single N-column** (i.e., the 8 slots share one `n`, and their `k` run consecutively) — matching the §7.12 B-map (lane → col = lane&0xF, K-bytes [rowhi*8 .. +7]). Record whether the observed pattern matches this.

- [ ] **Step 5: If the pattern is NOT a clean (lane→column, slots→contiguous-K) bijection**, the per-lane base pointer convention differs. Adjust `Bsrc + lane*8` in the kernel to the candidate suggested by the observed table (e.g., `Bsrc + (lane & 15) * 16` for per-row addressing, or a 16-lane-group base), rebuild (Step 3), rerun (Step 4). Iterate until the table is a clean bijection. This is compiler/hardware-as-oracle — the table is ground truth, never guess past it.

- [ ] **Step 6: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/global_load_tr_probe.hip \
        ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/build.sh
git commit -m "feat(MAD-305 P0): global_load_tr_b64 layout probe harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Confirm the instruction actually lowers to `global_load_tr_b64`

**Files:**
- Reference only: `bench/global_load_tr_probe.hip`

- [ ] **Step 1: Emit device assembly**

Run (RAM-capped):
```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include -S \
  bench/global_load_tr_probe.hip -o /tmp/tr_probe.s
```

- [ ] **Step 2: Verify the opcode is present**

Run:
```bash
grep -nE 'global_load_tr_b64|global_load_tr' /tmp/tr_probe.s
```
Expected: at least one `global_load_tr_b64` instruction in the `probe` kernel body. If absent (e.g., the builtin was optimized away or lowered to scalar loads), STOP and investigate — the probe is not exercising the hardware path.

- [ ] **Step 3: Commit** (no code change — record the verification in the contract doc instead; see Task 3.)

---

## Task 3: Document the verified mapping contract as a reusable helper

**Files:**
- Create: `bench/global_load_tr_contract.md`

- [ ] **Step 1: Write the contract doc** capturing (a) the observed probe table from `/tmp/tr_probe_table.txt`, (b) the confirmed per-lane base-pointer convention, and (c) the resulting `tr_load_b_fragment` helper that Phase 1 will paste into `gemm_wmma.hip`. Use the convention Task 1 confirmed. Template:

````markdown
# gfx1201 `global_load_tr_b64` fp8 B-fragment contract (MAD-305 Phase 0, verified <DATE>)

Verified by `bench/global_load_tr_probe.hip` + oracle (Task 4). Lowers to `global_load_tr_b64`
(Task 2). Source tile B is row-major `[K][N]` in global; element `B[k][n]`.

## Observed mapping
<paste the relevant rows of /tmp/tr_probe_table.txt here>

## Convention (confirmed)
Lane L, to obtain the WMMA B-fragment for N-column `n0 + (L & 15)` and K-half `(L >> 4) & 1`:
  base pointer = &B[ k0 + <confirmed expr> ][ n0 + <confirmed expr> ]
The returned v2i32 IS the WMMA B operand for that lane (feeds the intrinsic directly, no remap).

## Helper (paste into gemm_wmma.hip in Phase 1)
```cpp
typedef int v2i32 __attribute__((ext_vector_type(2)));
__device__ inline v2i32 tr_load_b_fragment(const uint8_t* B, int N, int k0, int n0, int lane) {
    // <confirmed addressing from the probe table>
    const uint8_t* p = B + /* confirmed byte offset(k0, n0, lane, N) */;
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}
```
````

Replace every `<...>` with the concrete values from the probe before committing — the doc must contain a compilable helper, no placeholders.

- [ ] **Step 2: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/global_load_tr_contract.md
git commit -m "docs(MAD-305 P0): verified global_load_tr_b64 fp8 fragment contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Prove the contract with a single 16×16×16 WMMA vs CPU oracle

**Files:**
- Modify: `bench/global_load_tr_probe.hip` (add a `--validate` path)

- [ ] **Step 1: Add a CPU e4m3 oracle + single-tile WMMA validation** to the probe file. Append a second kernel that loads A wide (the proven A-map from `spike/gemm_wmma_raw_intrinsic_verified.hip`) and B via the `tr_load_b_fragment` helper from Task 3, runs one `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, and writes the 16×16 accumulator. Host fills A and B with distinct e4m3 patterns (`patA(i,k)`, `patB(k,n)` from `bench/feedwidth_proto.hip`), computes the CPU reference `C[i][n] = Σ_k dec(A)·dec(B)`, and asserts max-rel-err < 0.03. Gate `main()` on `argc>1 && argv[1]=="--validate"`.

```cpp
// (append to global_load_tr_probe.hip)
#include <hip/hip_fp8.h>
#include <vector>
#include <cmath>
typedef float v8f32 __attribute__((ext_vector_type(8)));
static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int n){ return (float)(((k*5 + n*2) % 9) - 4) * 0.5f; }

// <Paste tr_load_b_fragment from the Task-3 contract here.>

__global__ void onetile(const uint8_t* A, const uint8_t* B, float* C) {
    int lane = threadIdx.x;
    // A-map (§7.12): lane->row=lane&0xF, colhi=(lane>>4)&1; 8 contiguous K-bytes of that row.
    int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
    auto ap = reinterpret_cast<const v2i32*>(A + row_a * 16 + colhi * 8);
    v2i32 fa = *ap;
    v2i32 fb = tr_load_b_fragment(B, 16, 0, 0, lane);
    v8f32 acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa, fb, v8f32{0,0,0,0,0,0,0,0});
    // C/D map: lane holds col=lane&0xF, rows ((lane>>4)&1)*8 + s
    int col = lane & 0xF, row0 = ((lane >> 4) & 1) * 8;
    #pragma unroll
    for (int s = 0; s < 8; s++) C[(row0 + s) * 16 + col] = acc[s];
}

static int validate() {
    std::vector<uint8_t> Ah(256), Bh(256);
    for (int i=0;i<16;i++) for (int k=0;k<16;k++) Ah[i*16+k]=enc(patA(i,k));
    for (int k=0;k<16;k++) for (int n=0;n<16;n++) Bh[k*16+n]=enc(patB(k,n));
    std::vector<float> ref(256,0.f);
    for (int i=0;i<16;i++) for (int n=0;n<16;n++){ float a=0;
        for (int k=0;k<16;k++) a += dec(Ah[i*16+k])*dec(Bh[k*16+n]); ref[i*16+n]=a; }
    uint8_t *dA,*dB; float* dC;
    hipMalloc(&dA,256); hipMalloc(&dB,256); hipMalloc(&dC,256*4);
    hipMemcpy(dA,Ah.data(),256,hipMemcpyHostToDevice);
    hipMemcpy(dB,Bh.data(),256,hipMemcpyHostToDevice);
    onetile<<<1,32>>>(dA,dB,dC); hipDeviceSynchronize();
    std::vector<float> Ch(256); hipMemcpy(Ch.data(),dC,256*4,hipMemcpyDeviceToHost);
    float mx=0; for (int t=0;t<256;t++){ float e=fabsf(Ch[t]-ref[t])/(fabsf(ref[t])+1e-3f); if(e>mx)mx=e; }
    printf("onetile max_rel_err=%.4f  %s\n", mx, mx<0.03f?"PASS":"*** FAIL ***");
    hipFree(dA);hipFree(dB);hipFree(dC);
    return mx<0.03f?0:1;
}
```

In `main()`, add at the top: `if (argc>1 && std::string(argv[1])=="--validate") return validate();` (add `#include <string>`).

- [ ] **Step 2: Build (Task 1 Step 3 command) and run the validation**

Run:
```bash
cd /home/kmbandy/GitHub/llama.cpp/ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include bench/global_load_tr_probe.hip -o out/global_load_tr_probe
timeout 60 ./out/global_load_tr_probe --validate
```
Expected: `onetile max_rel_err=... PASS` (< 0.03). **This is the Phase-0 gate.** If FAIL, the addressing convention in `tr_load_b_fragment` is wrong — return to Task 1 Step 5, refine from the probe table, update the Task-3 helper, rerun.

- [ ] **Step 3: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/bench/global_load_tr_probe.hip
git commit -m "test(MAD-305 P0): single-tile WMMA validates global_load_tr contract vs oracle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Record the Phase 0 outcome

**Files:**
- Create or append: `RESULT.md`

- [ ] **Step 1: Append the Phase 0 row** to `RESULT.md`:

```markdown
## Phase 0 — global_load_tr_b64 layout probe (MAD-305)
- Instruction confirmed lowering to `global_load_tr_b64` (asm check, Task 2).
- Per-lane fragment contract verified vs CPU e4m3 oracle (single-tile WMMA, max_rel_err < 0.03).
- Contract documented in `bench/global_load_tr_contract.md` — consumed by Phase 1.
- Status: GATE PASSED — wide-feed addressing is known, no guessing required for Phase 1.
```

- [ ] **Step 2: Commit**

```bash
cd /home/kmbandy/GitHub/llama.cpp
git add ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/RESULT.md
git commit -m "docs(MAD-305 P0): record layout-probe gate outcome

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Remaining ladder (gated — each gets its own detailed plan AFTER its predecessor's gate)

Per the spec, the full ladder is staged with evaluation gates. The next phases are NOT detailed here
because each one's exact code depends on the prior phase's measured result (e.g., Phase 1's fragment
addressing depends on the Phase 0 contract just derived; Phase 1's L2-reuse outcome decides whether
Phase 2/3 tiling needs the larger-M fallback). After Phase 0's gate passes, write the Phase 1 plan.

- **Phase 1 — Wide feed:** replace the B byte-gather in `gemm_wmma.hip` with `tr_load_b_fragment`; oracle gate + measure vs 307/143. Crux: does direct-from-global feed retain enough B-reuse via L2?
- **Phase 2 — Async double-buffer:** software-pipeline the K-loop (`global_load_lds` prefetch + ping-pong), mirroring ck_tile `comp_v6`; oracle gate + measure.
- **Phase 3 — Big tiles + scheduler + wave32 occupancy retune:** mirror AITER `ck_gemm_a8w8_blockscale` tile config; `s_setprio`/`sched_barrier`; measure vs ~245 target.
- **Phase 4 — ml8 LUT front-end:** unpack-to-fp8-then-feed onto the optimized core; ml8 oracle gate.
- **Phase 5 — Production integration + PPL-neutral:** wire through the llama.cpp ml8 path; graph correctness + PPL gate.

---

## Self-Review

- **Spec coverage:** This plan implements Spec Phase 0 in full (probe → asm-confirm → contract → oracle-validation → record). Spec Phases 1–5 are explicitly deferred to per-phase plans (staged-evaluation requirement + genuine data-dependency), each named with its spec gate. No Phase-0 requirement is unaddressed.
- **Placeholder scan:** The only `<...>` markers are in the Task-3 contract *template*, which Task-3 Step-1 explicitly requires filling with the concrete probe-derived values before commit. All executable code (probe kernel, oracle, validation) is complete.
- **Type consistency:** `tr_load_b_fragment(B, N, k0, n0, lane) -> v2i32` is defined in Task 3 and consumed unchanged in Task 4. `enc`/`dec`/`patA`/`patB` match `bench/feedwidth_proto.hip`. WMMA builtin name matches the verified kernel.
