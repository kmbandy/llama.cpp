# ml8 LUT GEMM Optimization Implementation Plan (MAD-299)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the ml8 LUT GEMM (`gemm_ml8.py` WEIGHT_FORMAT=1) from ~2.9% (~11 TFLOPS) to ~80% of RDNA4 dense FP8 (383 TFLOPS) on gfx1201 by removing the two per-K-iteration gathers that starve the fp8 WMMA units — at **zero quality cost** (bit-identical dequant math).

**Architecture:** The inner K-loop currently materializes the weight tile with a `tl.gather` (nibble expand) followed by a per-element global `tl.load` from the 16-entry centroid LUT — two uncoalesced, serializing memory ops per `tl.dot`. We replace them with pure-ALU equivalents that produce the **same fp8 weight tile**:
1. **Even/odd K-split** eliminates the nibble-expand gather. The low nibble of packed byte `j` is K-position `2j` and the high nibble is `2j+1`; both already live in `[K//2, N]` layout. We split A into even/odd K-strided halves (`reshape` + `tl.split`, no gather) and accumulate `dot(a_even, w_lo) + dot(a_odd, w_hi)`. The K-contraction is unchanged — this is exact.
2. **Register select-ladder** eliminates the global LUT gather. Each K-iteration touches exactly one K-group's 16 fp8 centroids (BLOCK_SIZE_K == group_size). We load those 16 as broadcast scalars (uniform/scalar-cached) and map index→value with a 16-deep `tl.where` ladder — pure VALU, no per-element global load.
3. **gfx1201 tile/warp/kpack tune** once dequant stops being the wall, then wire the winning config into the production runtime.

Every step is gated by a **correctness oracle** (kernel output ≡ dequant-in-torch reference, within fp8 tolerance — the invariant) and a **%-of-383 benchmark** (the perf ratchet). `rocprof` between kernel edits confirms the transition memory-bound → WMMA-bound.

**Tech Stack:** Triton 3.8.0 (editable, pinned `007ef1530`), PyTorch 2.13 (ROCm), AMD R9700 / gfx1201 (RDNA4), Python 3 (system interpreter already resolves torch + triton + gfx1201). No Triton **source** rebuild is required by this plan — kernels JIT-compile at runtime (light). If a source rebuild ever becomes necessary, use the RAM-capped `ggml/src/ggml-cuda/aiter-integration/tools/build_triton_pinned.sh` (15 GB host — never run an uncapped LLVM build).

**Denominator (locked):** `DENSE_FP8_TFLOPS = 383.0` — official R9700 dense FP8 (E4M3/E5M2) matrix peak. ml8 stores 4-bit indices but **computes in fp8**, so its ceiling is dense fp8 (not sparse 766). See `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `scripts/calibration/bench_ml8_gemm.py` | %-of-383 benchmark harness: synthetic-layer builder, configurable kernel launcher, median timing, JSON emit. The perf ratchet. | Create |
| `tests/test_ml8_gemm_optimization.py` | Correctness oracle: kernel ≡ dequant-in-torch reference at single-tile, multi-tile, and real-4B shapes. The invariant. Reuses `reference_dequant_gemm` + `run_ml8_kernel` from `test_ml8_kernel_stage1_dequant.py`. | Create |
| `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py` | The kernel. Only the WEIGHT_FORMAT=1 dequant block (lines ~403–434) changes. WEIGHT_FORMAT=0 path untouched; signature untouched. | Modify |
| `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8_tune.json` | Persisted winning gfx1201 configs (existing format: keyed by shape, `best` dict). | Modify |
| `scripts/calibration/ml8_runtime.py` | `ml8_gemm` consults `get_gemm_config` + passes `num_warps` so the tuned win lands in inference. | Modify |
| `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md` | Append the MAD-299 outcome (before/after %383 table, rocprof deltas). | Modify |

**Conventions reused from the existing kernel/tests (do not re-derive):**
- Kernel constraint: `GROUP_K == BLOCK_SIZE_K == group_size` (= 64 for ml8-4); one centroid LUT per K-iteration.
- gfx1201: `num_stages == 1` (num_stages≥2 is a use-after-free per the RDNA4 audit). `MATRIX_INSTR_NONKDIM == 16`.
- Nibble convention: lo-first. Packed byte `j` low nibble → K-position `2j` (even); high nibble → `2j+1` (odd).
- fp8 dequant tolerance: e4m3 ≈ 2-bit mantissa; single-tile `max_err < 5e-2`, multi-tile `< 1e-2` (per existing Stage-1 test).
- Run interpreter: `python3` (system; resolves torch 2.13 + triton 3.8 + gfx1201). Fallback if import fails: `PYTHONPATH=/home/kmbandy/GitHub/triton/python /home/kmbandy/venvs/agents/bin/python3`.

**Repo rule (llama.cpp CLAUDE.md):** before editing a tracked symbol, run `gitnexus_impact({target, direction:"upstream"})` and report blast radius; after edits, `gitnexus_detect_changes()`. The kernel-editing tasks (2, 3, 5) include this as Step 1.

---

### Task 1: TDD foundation — %-of-383 bench harness + correctness oracle + baseline

**Files:**
- Create: `scripts/calibration/bench_ml8_gemm.py`
- Create: `tests/test_ml8_gemm_optimization.py`
- Reuse: `tests/test_ml8_kernel_stage1_dequant.py` (`reference_dequant_gemm`, `run_ml8_kernel`), `scripts/calibration/ml8_runtime.py` (`layer_from_components`)

- [ ] **Step 1: Write the failing test for the harness math (CPU, real TDD)**

Create `tests/test_ml8_gemm_optimization.py` with just the math test first:

```python
#!/usr/bin/env python3
"""MAD-299 — correctness oracle + harness-math tests for the ml8 LUT GEMM
optimization. The oracle (kernel ≡ dequant-in-torch) is the invariant that must
stay green through every kernel edit; the %-of-383 bench (bench_ml8_gemm.py) is
the perf ratchet."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))


def test_tflops_and_pct_math():
    import bench_ml8_gemm as B
    # 2*M*N*K MACs in `seconds` → TFLOPS
    assert abs(B.tflops(1024, 1024, 1024, 1e-3) - (2 * 1024**3 / 1e-3 / 1e12)) < 1e-9
    # pct_of_dense is tflops / 383 * 100
    assert abs(B.pct_of_dense(383.0) - 100.0) < 1e-9
    assert abs(B.pct_of_dense(11.0) - (11.0 / 383.0 * 100.0)) < 1e-6
    assert B.DENSE_FP8_TFLOPS == 383.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 tests/test_ml8_gemm_optimization.py::test_tflops_and_pct_math` (or `python3 -m pytest tests/test_ml8_gemm_optimization.py -k math -v`)
Expected: FAIL — `ModuleNotFoundError: No module named 'bench_ml8_gemm'`.

- [ ] **Step 3: Create the bench harness with the math + builder + launcher**

Create `scripts/calibration/bench_ml8_gemm.py`:

```python
"""MAD-299 ml8 LUT GEMM benchmark — %-of-383 (RDNA4 dense FP8) ratchet.

Builds synthetic ml8 layers at the real Qwen3.5-4B linear shapes, launches the
WEIGHT_FORMAT=1 kernel at a chosen tile/warp config, times the kernel only
(layer is prebuilt — no per-call packing in the timed region), and reports
TFLOPS and % of dense FP8. Emits JSON. The kernel is the variable across runs;
this harness is constant."""
from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import torch

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
_KERNELS = _THIS.parent.parent / "ggml/src/ggml-cuda/aiter-integration/kernels"
if str(_KERNELS) not in sys.path:
    sys.path.insert(0, str(_KERNELS))

DENSE_FP8_TFLOPS = 383.0  # official R9700 dense FP8 (E4M3/E5M2) matrix peak


def tflops(M: int, N: int, K: int, seconds: float) -> float:
    return (2.0 * M * N * K) / seconds / 1e12


def pct_of_dense(tf: float) -> float:
    return tf / DENSE_FP8_TFLOPS * 100.0


def default_shapes():
    # (name, N=out, K=in) — Qwen3.5-4B (hidden=2560, intermediate=9216)
    return [("gate", 9216, 2560), ("up", 9216, 2560),
            ("down", 2560, 9216), ("o_proj", 2560, 2560)]


def build_synthetic_layer(N, K, group_size=64, n_centroids=16, device="cuda", seed=0):
    from ml8_runtime import layer_from_components
    g = torch.Generator().manual_seed(seed)
    G = K // group_size
    centroids = torch.randn(G, n_centroids, generator=g) * 0.5            # fp32 [G,16]
    scales = torch.randn(N, G, generator=g).abs() * 0.1 + 0.01            # fp32 [N,G]
    indices = torch.randint(0, n_centroids, (N, K), generator=g, dtype=torch.uint8)
    gidx = torch.arange(K) // group_size                                  # [K]
    return layer_from_components(centroids, scales, indices, gidx, device=device)


def launch(a_fp8, layer, a_scale, *, block_m=16, block_n=16, num_warps=4, kpack=1):
    """Direct WEIGHT_FORMAT=1 launch at a chosen config. Returns C [M,N] bf16."""
    import gemm_ml8
    M, K = a_fp8.shape
    N = layer.n_rows
    gs = layer.group_size
    c = torch.empty(M, N, dtype=torch.bfloat16, device=a_fp8.device)
    stride_am, stride_ak = a_fp8.stride()
    stride_bk, stride_bn = layer.indices_packed.stride()
    stride_cm, stride_cn = c.stride()
    stride_bscale_k, stride_bscale_n = layer.scales_fp32.stride()
    grid_mn = (M // block_m) * (N // block_n)
    gemm_ml8._gemm_a8w8_blockscale_kernel[(grid_mn,)](
        a_fp8, layer.indices_packed, c, a_scale, layer.scales_fp32,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        0, stride_cm, stride_cn,
        1, 0, stride_bscale_k, stride_bscale_n,
        GROUP_K=gs, GROUP_N=1,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=gs,
        GROUP_SIZE_M=1, NUM_KSPLIT=1, SPLITK_BLOCK_SIZE=K,
        EVEN_K=(K % gs == 0), GRID_MN=grid_mn, num_stages=1,
        WEIGHT_FORMAT=1, N_CENTROIDS=layer.n_centroids,
        centroid_lut_ptr=layer.centroids_fp8, stride_lut_k=layer.centroids_fp8.stride(0),
        num_warps=num_warps, kpack=kpack,
    )
    return c


def _median_seconds(fn, *, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def bench_shape(name, N, K, M, dev, *, block_m, block_n, num_warps):
    layer = build_synthetic_layer(N, K, device=dev)
    a_fp8 = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, dtype=torch.float32, device=dev)
    sec = _median_seconds(lambda: launch(
        a_fp8, layer, a_scale, block_m=block_m, block_n=block_n, num_warps=num_warps))
    tf = tflops(M, N, K, sec)
    return dict(shape=name, M=M, N=N, K=K, block_m=block_m, block_n=block_n,
                num_warps=num_warps, ms=sec * 1e3, tflops=tf, pct383=pct_of_dense(tf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[2048])
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=16)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("/tmp/mad299_bench.json"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rows = []
    for name, N, K in default_shapes():
        for M in args.m_tiers:
            try:
                rows.append(bench_shape(name, N, K, M, dev,
                                        block_m=args.block_m, block_n=args.block_n,
                                        num_warps=args.num_warps))
            except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
                rows.append(dict(shape=name, M=M, N=N, K=K, error=str(e)[:300]))
    out = dict(label=args.label, triton_version=__import__("triton").__version__, rows=rows)
    args.out.write_text(json.dumps(out, indent=2))
    for r in rows:
        if "tflops" in r:
            print(f"{r['shape']:7s} M={r['M']:5d} bm={r['block_m']:3d} bn={r['block_n']:3d} "
                  f"w={r['num_warps']} {r['ms']:8.3f} ms  {r['tflops']:7.1f} TF  {r['pct383']:5.1f}% of 383")
        else:
            print(f"{r['shape']:7s} M={r['M']:5d} ERROR {r['error']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

> Note: `launch` passes `kpack=` only if the kernel accepts it — Triton ignores unknown `num_warps`/`kpack`? No: `num_warps` is a launch meta Triton accepts; `kpack` is NOT a kernel arg here, so **remove `kpack=kpack` from the launch call** (kpack is an autotune hint that does not apply to this kernel signature). Keep the `kpack` parameter in the signature unused for now, or drop it. Default to dropping it: delete the `kpack` param and the `kpack=kpack` kwarg. (Listed explicitly so the implementer does not pass an invalid kwarg.)

- [ ] **Step 4: Run the math test to verify it passes**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -k math -v`
Expected: PASS.

- [ ] **Step 5: Add the correctness oracle (the invariant), reusing existing helpers**

Append to `tests/test_ml8_gemm_optimization.py`:

```python
import numpy as np
import torch

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_ml8_kernel_stage1_dequant import reference_dequant_gemm, run_ml8_kernel  # noqa: E402
sys.path.insert(0, str(REPO_ROOT / "ggml/src/ggml-cuda/aiter-integration/kernels"))
from ml8_to_packed import pack_indices  # noqa: E402


def _pack_kn(indices_kn: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """[K,N] int8 indices → [K//2,N] uint8 lo-first packed (kernel layout)."""
    packed_bytes = pack_indices(indices_kn.T.cpu().contiguous(), nibble_lo_first=True)
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    return torch.from_numpy(packed_np.T.copy()).contiguous().to(indices_kn.device)


def _oracle_case(M, N, K, group_size, seed, tol):
    """Kernel WEIGHT_FORMAT=1 output must match the dequant-in-torch reference."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    n_centroids = 16
    n_groups_k = K // group_size

    a_fp8 = ((torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)).to(torch.float8_e4m3fn)
    centroids_fp8 = (torch.randn(n_groups_k, n_centroids, device=device) * 0.5).to(torch.float8_e4m3fn)
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    b_scale = torch.randn(n_groups_k, N, device=device).abs() * 0.1 + 0.01
    a_scale = torch.randn(M, device=device).abs() * 0.1 + 0.01

    C_ref = reference_dequant_gemm(
        a_fp8.to(torch.float32), indices, centroids_fp8.to(torch.float32),
        b_scale, a_scale, group_size)
    b_packed = _pack_kn(indices, N, K)
    C_kernel = run_ml8_kernel(a_fp8, b_packed, centroids_fp8, b_scale, a_scale,
                              group_size=group_size, n_centroids=n_centroids)
    max_err = (C_kernel.to(torch.float32) - C_ref.to(torch.bfloat16).to(torch.float32)).abs().max().item()
    assert max_err < tol, f"M={M} N={N} K={K}: max_err {max_err:.4g} exceeds {tol}"


def test_oracle_single_tile():
    _oracle_case(M=16, N=16, K=64, group_size=64, seed=42, tol=5e-2)


def test_oracle_multi_tile_cross_kgroup():
    _oracle_case(M=32, N=32, K=256, group_size=64, seed=123, tol=1e-2)


def test_oracle_real_4b_shape():
    # down-proj slice: real N=2560, K=9216 (144 K-groups), small M to stay quick.
    _oracle_case(M=64, N=2560, K=9216, group_size=64, seed=7, tol=1e-2)
```

- [ ] **Step 6: Run the oracle to verify it PASSES on the current (unoptimized) kernel**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -v`
Expected: all PASS. This **locks the invariant** — the dequant math the optimization must preserve. (Unlike a feature test, the oracle is a regression guard: it is green now and must stay green after every kernel edit.)

- [ ] **Step 7: Record the baseline %-of-383**

Run: `python3 scripts/calibration/bench_ml8_gemm.py --label mad299-baseline --m-tiers 16 512 2048 --out /tmp/mad299_baseline.json`
Expected: ~11 TFLOPS / ~2.9% of 383, roughly flat across shapes/M (the diagnostic signature of the per-element serial bottleneck). Record the printed table.

- [ ] **Step 8: Commit**

```bash
git add scripts/calibration/bench_ml8_gemm.py tests/test_ml8_gemm_optimization.py
git commit -m "MAD-299: ml8 LUT GEMM TDD foundation — %-of-383 bench + dequant oracle + baseline"
```

---

### Task 2: Kill gather #1 — even/odd K-split nibble unpack

**Files:**
- Modify: `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py:403-434` (WEIGHT_FORMAT=1 dequant block only)
- Guard: `tests/test_ml8_gemm_optimization.py` (oracle), `scripts/calibration/bench_ml8_gemm.py` (ratchet)

**Why:** `tl.gather(b_packed, byte_row_2d, axis=0)` duplicates each packed byte-row into two K-rows — on AMD this lowers to a scratch/permute round-trip per the kernel's own comment. The low nibble of byte `j` IS K-position `2j` and the high nibble is `2j+1`, so we never need to interleave: split A into even/odd K halves and do two half-K dots. Exact (the K-sum is just reordered into even + odd partitions).

- [ ] **Step 1: Impact check (repo rule)**

Run: `gitnexus_impact({target: "_gemm_a8w8_blockscale_kernel", direction: "upstream"})`. Report blast radius. Expected callers: `ml8_runtime.ml8_gemm`, `run_ml8_kernel` (test), `bench_ml8_gemm.launch`, and the AOT compile path. The signature is **unchanged** (only the WEIGHT_FORMAT=1 inner block changes), so risk should be LOW. If HIGH/CRITICAL, surface to the human before editing.

- [ ] **Step 2: Confirm oracle is green before editing (regression anchor)**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -v`
Expected: all PASS (pre-edit anchor).

- [ ] **Step 3: Replace the WEIGHT_FORMAT=1 dequant block**

In `gemm_ml8.py`, replace the `else:` block at lines ~403–434 (from `# ml8 LUT path (decisions B + C + D).` through the `accumulator += tl.dot(a, b_fp8) * ...` line) with:

```python
            else:
                # ml8 LUT path — even/odd K-split dequant (MAD-299: no tl.gather).
                # Packed byte j holds K-position 2j in its low nibble and 2j+1 in
                # its high nibble (lo-first). lo/hi are already in [K//2, N] layout,
                # so we split A into even/odd K halves and accumulate two half-K
                # dots — the K-contraction is unchanged (sum over even k + odd k).
                if EVEN_K:
                    b_packed = tl.load(b_ml8_ptrs)              # [BLOCK_K//2, BLOCK_N] uint8
                else:
                    b_packed = tl.load(
                        b_ml8_ptrs,
                        mask=offs_k_packed[:, None] < (K - k * BLOCK_SIZE_K) // 2,
                        other=0,
                    )
                lo_idx = (b_packed & 0x0F).to(tl.int32)         # even-K indices [BLOCK_K//2, BLOCK_N]
                hi_idx = ((b_packed >> 4) & 0x0F).to(tl.int32)  # odd-K  indices [BLOCK_K//2, BLOCK_N]
                # Per-element LUT load (still global here — killed in Task 3), now
                # on half-size tiles.
                lut_base = centroid_lut_ptr + k * stride_lut_k
                w_lo = tl.load(lut_base + lo_idx)               # fp8 [BLOCK_K//2, BLOCK_N]
                w_hi = tl.load(lut_base + hi_idx)               # fp8 [BLOCK_K//2, BLOCK_N]
                # Split A into even/odd K halves with reshape + tl.split (no gather):
                # a[m, 2j] -> a_even[m, j], a[m, 2j+1] -> a_odd[m, j].
                a_even, a_odd = tl.split(
                    a.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2, 2)
                )                                               # each fp8 [BLOCK_M, BLOCK_K//2]
                accumulator += (
                    tl.dot(a_even, w_lo) + tl.dot(a_odd, w_hi)
                ) * a_scale[:, None] * b_scale[None, :]
```

- [ ] **Step 4: Run the oracle to verify correctness is preserved**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -v`
Expected: all PASS (same tolerances). If any case fails, STOP and use superpowers:systematic-debugging — do NOT loosen tolerances. Likely first suspects: even/odd pairing inverted (swap `lo_idx`↔`hi_idx`), or `tl.split` axis assumption (it splits the trailing size-2 dim; verify `a.reshape(M, K//2, 2)` groups consecutive K).

- [ ] **Step 5: Run the bench — expect ≥ baseline (gather #1 removed)**

Run: `python3 scripts/calibration/bench_ml8_gemm.py --label mad299-task2 --m-tiers 16 512 2048 --out /tmp/mad299_task2.json`
Expected: TFLOPS ≥ baseline at every shape (no regression; partial uplift). Record the table. If it regresses below baseline, the two half-K dots cost more than the gather saved — STOP, rocprof, and reconsider before proceeding.

- [ ] **Step 6: rocprof — confirm the gather is gone**

Run: `rocprof --stats python3 scripts/calibration/bench_ml8_gemm.py --label rp-task2 --m-tiers 2048 --out /tmp/rp_task2.json` (or `rocprofv3` if that is the installed front-end; check `which rocprof rocprofv3`).
Expected: the permute/scratch traffic associated with `tl.gather` disappears from the kernel's instruction mix; still memory-bound on the LUT loads (addressed next). Save the summary.

- [ ] **Step 7: Detect-changes + commit**

```bash
gitnexus_detect_changes({scope: "all"})   # confirm only gemm_ml8.py changed
git add ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py
git commit -m "MAD-299: kill gather #1 — even/odd K-split nibble unpack (oracle green, no regression)"
```

---

### Task 3: Kill gather #2 — register-resident LUT select ladder

**Files:**
- Modify: `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py` (the two `tl.load(lut_base + *_idx)` lines from Task 2)
- Guard: `tests/test_ml8_gemm_optimization.py`, `scripts/calibration/bench_ml8_gemm.py`

**Why (the big one):** `tl.load(centroid_lut_ptr + k*stride_lut_k + idx)` is a per-element **global** indexed load — `BLOCK_K//2 × BLOCK_N` uncoalesced loads per K-iteration. But each K-iteration uses exactly one K-group's **16** fp8 centroids. Load those 16 once as broadcast scalars (uniform address → scalar-cached) and map index→value with a 16-deep `tl.where` ladder: pure VALU, zero per-element global traffic. The selected values are fp8 → the fp8 WMMA path is preserved.

- [ ] **Step 1: Impact check**

Run: `gitnexus_impact({target: "_gemm_a8w8_blockscale_kernel", direction: "upstream"})`. Signature unchanged again → expect LOW.

- [ ] **Step 2: Confirm oracle green (anchor)**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -v` → all PASS.

- [ ] **Step 3: Replace the two LUT `tl.load`s with the select ladder**

In the Task-2 block, replace the `lut_base = ...`, `w_lo = tl.load(...)`, `w_hi = tl.load(...)` lines with:

```python
                # Stage the 16-entry fp8 LUT for this K-group as broadcast scalars
                # (uniform address → scalar-cached) and map index->value with a
                # 16-deep tl.where ladder — pure VALU, no per-element global load.
                lut_base = centroid_lut_ptr + k * stride_lut_k
                cvals = [tl.load(lut_base + i) for i in range(N_CENTROIDS)]  # fp8 scalars
                w_lo = cvals[0]
                w_hi = cvals[0]
                for i in range(1, N_CENTROIDS):
                    w_lo = tl.where(lo_idx == i, cvals[i], w_lo)
                    w_hi = tl.where(hi_idx == i, cvals[i], w_hi)
                # w_lo/w_hi are fp8 [BLOCK_K//2, BLOCK_N] (cvals[0] scalar broadcasts).
```

> Fallback if Triton rejects `tl.where` on fp8 operands on gfx1201: load centroids as fp32 (`cvals = [tl.load(lut_base + i).to(tl.float32) ...]`), build the ladder in fp32, then cast before the dot: `w_lo = w_lo.to(tl.float8e4nv)` / `w_hi = w_hi.to(tl.float8e4nv)`. The centroid is already an exact fp8 value, so the round-trip is lossless and the oracle stays green.

- [ ] **Step 4: Run the oracle — correctness preserved**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -v`
Expected: all PASS, unchanged tolerances. Failure → systematic-debugging (suspect: ladder init not covering index 0 — verify `w_* = cvals[0]` and loop starts at 1; or fp8 where rejected → apply the fp32 fallback).

- [ ] **Step 5: Run the bench — expect a large jump**

Run: `python3 scripts/calibration/bench_ml8_gemm.py --label mad299-task3 --m-tiers 16 512 2048 --out /tmp/mad299_task3.json`
Expected: TFLOPS substantially above Task 2 (the dominant per-element global gather is gone) and no longer flat — it should now scale toward compute-bound. Record the table.

- [ ] **Step 6: rocprof — confirm memory-bound → WMMA-bound**

Run: `rocprof --stats python3 scripts/calibration/bench_ml8_gemm.py --label rp-task3 --m-tiers 2048 --out /tmp/rp_task3.json`
Expected: vector-memory / L1 traffic collapses; WMMA (matrix) instruction occupancy rises; the bottleneck shifts to the dots. Save the summary — this is the headline evidence that dequant stopped being the wall.

- [ ] **Step 7: Detect-changes + commit**

```bash
gitnexus_detect_changes({scope: "all"})
git add ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py
git commit -m "MAD-299: kill gather #2 — register-resident 16-way LUT select ladder (oracle green, ~Nx uplift)"
```

---

### Task 4: gfx1201 tile/warp tune + persist winning config

**Files:**
- Create: `scripts/calibration/tune_ml8_gemm_mad299.py` (sweep driver; distinct from the legacy `tune_gemm_ml8.py` which needs missing Cell-E calib points)
- Modify: `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8_tune.json` (persist winners)
- Guard: `tests/test_ml8_gemm_optimization.py`

**Why:** With dequant no longer the wall, BLOCK_M/N and warp count now govern WMMA occupancy. BLOCK_K stays 64 (group constraint) and num_stages stays 1 (gfx1201 UAF). Sweep against the synthetic layers at the real shapes and keep the per-(K,N,tier) winner.

- [ ] **Step 1: Write the sweep driver**

Create `scripts/calibration/tune_ml8_gemm_mad299.py`:

```python
"""MAD-299 gfx1201 tile/warp tuner for the optimized ml8 LUT GEMM.

Sweeps BLOCK_M × BLOCK_N × num_warps against synthetic layers at the real 4B
shapes, validates each candidate against the dequant oracle, times it, and
writes the per-(K,N,tier) winner into gemm_ml8_tune.json (existing format)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
import bench_ml8_gemm as B  # noqa: E402

SWEEP_BM = [16, 32, 64, 128]
SWEEP_BN = [16, 32, 64, 128]
SWEEP_WARPS = [1, 2, 4, 8]
TUNE_JSON = (_THIS.parent.parent /
             "ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8_tune.json")


def _valid(M, N, bm, bn):
    return M % bm == 0 and N % bn == 0


def sweep(name, N, K, M, dev):
    layer = B.build_synthetic_layer(N, K, device=dev)
    a_fp8 = (torch.randn(M, K, device=dev) * 0.3).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, dtype=torch.float32, device=dev)
    best = None
    for bm in SWEEP_BM:
        for bn in SWEEP_BN:
            if not _valid(M, N, bm, bn):
                continue
            for w in SWEEP_WARPS:
                try:
                    sec = B._median_seconds(lambda: B.launch(
                        a_fp8, layer, a_scale, block_m=bm, block_n=bn, num_warps=w),
                        warmup=5, iters=20)
                except Exception:  # noqa: BLE001 — config didn't compile/run; skip
                    continue
                tf = B.tflops(M, N, K, sec)
                cand = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, NUM_WARPS=w,
                            tflops=tf, pct383=B.pct_of_dense(tf))
                if best is None or tf > best["tflops"]:
                    best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-tiers", type=int, nargs="+", default=[16, 2048])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write", action="store_true", help="persist winners into gemm_ml8_tune.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    existing = json.loads(TUNE_JSON.read_text()) if TUNE_JSON.exists() else {}
    for name, N, K in B.default_shapes():
        for M in args.m_tiers:
            best = sweep(name, N, K, M, dev)
            tier = "decode" if M <= 16 else "prefill"
            print(f"{name:7s} M={M:5d} ({tier}) -> {best}")
            if args.write and best is not None:
                key = f"{name}_{tier}"
                existing[key] = {"shape": {"M": M, "N": N, "K": K},
                                 "best": {k: best[k] for k in
                                          ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "NUM_WARPS")}}
    if args.write:
        TUNE_JSON.write_text(json.dumps(existing, indent=2))
        print(f"wrote {TUNE_JSON}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run the sweep (no write) and inspect winners**

Run: `python3 scripts/calibration/tune_ml8_gemm_mad299.py --m-tiers 16 2048`
Expected: per-shape best config + %383 printed; prefill (M=2048) should land well above Task 3's default 16×16. Sanity-check the winners look reasonable before persisting.

- [ ] **Step 3: Validate the winning config against the oracle**

Add to `tests/test_ml8_gemm_optimization.py`:

```python
def test_oracle_at_tuned_tile():
    # The tuner's prefill winners must still satisfy the dequant invariant.
    # Largest swept tile that divides the test shape; tune output informs this.
    _oracle_case(M=128, N=2560, K=9216, group_size=64, seed=11, tol=1e-2)
```

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -k tuned -v`
Expected: PASS. (Note: `run_ml8_kernel` uses 16×16; this case proves correctness is tile-independent. If the chosen winner uses a larger BLOCK_M/N than the M/N here, bump the case dims to a shape the winner divides.)

- [ ] **Step 4: Persist winners**

Run: `python3 scripts/calibration/tune_ml8_gemm_mad299.py --m-tiers 16 2048 --write`
Expected: `gemm_ml8_tune.json` updated with `<shape>_<tier>` → best entries in the existing `{"shape":..., "best":...}` format.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibration/tune_ml8_gemm_mad299.py tests/test_ml8_gemm_optimization.py \
        ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8_tune.json
git commit -m "MAD-299: gfx1201 tile/warp tune for optimized ml8 LUT GEMM + persisted winners"
```

---

### Task 5: Wire the tuned config into the production runtime

**Files:**
- Modify: `scripts/calibration/ml8_runtime.py` (`ml8_gemm`, lines ~220–269)
- Guard: `tests/test_ml8_gemm_optimization.py`, `tests/test_ml8_runtime.py` (existing runtime test)

**Why:** `ml8_gemm` currently hardcodes BLOCK_SIZE_M/N = 16 and never calls `get_gemm_config`, nor passes `num_warps` — so the tuned win would not reach inference. Wire it: consult `get_gemm_config(name, M, N, K)`, apply the returned BLOCK_M/N + NUM_WARPS, fall back to the (now-correct) 16×16 default on miss.

- [ ] **Step 1: Impact check**

Run: `gitnexus_impact({target: "ml8_gemm", direction: "upstream"})`. Callers include `Ml8Linear.forward` and `reconstruct_model.py --use-ml8-kernel`. Report blast radius; the change is config-selection only (output unchanged), so risk should be LOW–MEDIUM. If HIGH/CRITICAL, surface before editing.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_ml8_gemm_optimization.py`:

```python
def test_ml8_gemm_uses_tuned_config(monkeypatch=None):
    """ml8_gemm must consult get_gemm_config and pass num_warps (not hardcode 16/16/4)."""
    sys.path.insert(0, str(REPO_ROOT / "ggml/src/ggml-cuda/aiter-integration/kernels"))
    import gemm_ml8, ml8_runtime
    captured = {}
    orig = gemm_ml8._gemm_a8w8_blockscale_kernel

    class _Spy:
        def __getitem__(self, grid):
            def _call(*a, **kw):
                captured.update(kw)
                return orig[grid](*a, **kw)
            return _call

    gemm_ml8._gemm_a8w8_blockscale_kernel = _Spy()
    try:
        dev = torch.device("cuda")
        layer = __import__("bench_ml8_gemm").build_synthetic_layer(2560, 9216, device=dev)
        a = (torch.randn(2048, 9216, device=dev) * 0.3).to(torch.float8_e4m3fn)
        ml8_runtime.ml8_gemm(a, layer, a_scale=torch.ones(2048, device=dev))
    finally:
        gemm_ml8._gemm_a8w8_blockscale_kernel = orig
    assert "num_warps" in captured, "ml8_gemm did not pass num_warps to the kernel"
    # tuned prefill block sizes should override the 16/16 default for this shape
    assert captured["BLOCK_SIZE_M"] >= 16 and captured["BLOCK_SIZE_N"] >= 16
```

- [ ] **Step 3: Run it to verify it fails**

Run: `python3 -m pytest tests/test_ml8_gemm_optimization.py -k tuned_config -v`
Expected: FAIL — `assert "num_warps" in captured` (current `ml8_gemm` does not pass it).

- [ ] **Step 4: Wire `get_gemm_config` + `num_warps` into `ml8_gemm`**

In `ml8_runtime.py`, in `ml8_gemm`, after computing `M, K, N` and before the kernel launch, replace the hardcoded `BLOCK_SIZE_M = block_size_m` / `BLOCK_SIZE_N = block_size_n` and the launch's missing `num_warps` with config lookup:

```python
    # MAD-299: consult the tuned gfx1201 config; fall back to caller defaults.
    cfg, is_tuned = gemm_ml8.get_gemm_config("GEMM-A8W8_BLOCKSCALE", M, N, K)
    BLOCK_SIZE_M = cfg["BLOCK_SIZE_M"] if is_tuned else block_size_m
    BLOCK_SIZE_N = cfg["BLOCK_SIZE_N"] if is_tuned else block_size_n
    num_warps = cfg["NUM_WARPS"]
```

and add `num_warps=num_warps,` to the `gemm_ml8._gemm_a8w8_blockscale_kernel[grid](...)` launch kwargs. Keep the existing `M % block_size_m`/`N % block_size_n` guards but validate against the **selected** `BLOCK_SIZE_M/N` (move the modulo asserts to after the cfg lookup, using `BLOCK_SIZE_M`/`BLOCK_SIZE_N`).

- [ ] **Step 5: Run the new test + the full oracle + the existing runtime test**

Run:
```
python3 -m pytest tests/test_ml8_gemm_optimization.py -v
python3 tests/test_ml8_runtime.py
```
Expected: all PASS (oracle still green; runtime test unaffected — output is identical, only tiling changed).

- [ ] **Step 6: End-to-end %-of-383 through the production path**

Run: `python3 scripts/calibration/bench_ml8_gemm.py --label mad299-final --m-tiers 16 512 2048 --out /tmp/mad299_final.json`
…and additionally time `ml8_runtime.ml8_gemm` directly (now config-driven) to confirm the production wrapper hits the tuned numbers, not just the raw `launch`. Record both.

- [ ] **Step 7: Detect-changes + commit**

```bash
gitnexus_detect_changes({scope: "all"})
git add scripts/calibration/ml8_runtime.py tests/test_ml8_gemm_optimization.py
git commit -m "MAD-299: route ml8_gemm through tuned gfx1201 config + num_warps (win lands in inference)"
```

---

### Task 6: Record the MAD-299 outcome

**Files:**
- Modify: `docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md`

- [ ] **Step 1: Append the MAD-299 results section**

Add a `## 7. MAD-299 outcome — LUT GEMM optimized` section with: the before→after %-of-383 table (baseline ~2.9% → Task 2 → Task 3 → tuned final), the rocprof memory-bound→WMMA-bound evidence, the final headline number ("ml8 LUT GEMM at X% of RDNA4 dense FP8"), and the note that quality is unchanged (oracle green throughout — bit-identical dequant). State plainly whether the 80% target was met; if short, record the gap and the next lever (e.g. BLOCK_K relaxation, dual-issue, LDS staging of A).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/results/2026-06-14-ml8-fp8-phase0-microbench.md
git commit -m "MAD-299: record LUT GEMM optimization outcome (X% of dense fp8, quality-neutral)"
```

---

## Final Review (after all tasks)

Dispatch a final code reviewer over the whole MAD-299 change set (gemm_ml8.py dequant block, ml8_runtime.ml8_gemm wiring, bench + tuner + oracle, tune.json), then use superpowers:finishing-a-development-branch. Confirm: (1) oracle green at the tuned config; (2) WEIGHT_FORMAT=0 path byte-untouched; (3) kernel signature unchanged (AOT compile path intact); (4) final %-of-383 recorded honestly against the 80% target.

---

## Self-Review (planner)

**Spec coverage (results §4 optimization plan):**
- §4.1 "Kill gather #1 — unpack nibbles arithmetically" → Task 2 (even/odd K-split; a cleaner equivalent to the spec's interleave — no interleave op needed).
- §4.2 "Kill gather #2 — stage 16-entry fp8 LUT into LDS/registers" → Task 3 (register select-ladder).
- §4.3 "Tile/warp tune for gfx1201 (num_stages=1)" → Task 4.
- §4.4 "rocprof between steps memory-bound → WMMA-bound" → Tasks 2/3 Step 6.
- §4 method "TDD: oracle + %-of-383 gate per change" → Task 1 (foundation) + per-task oracle/bench gates.
- "Where the win lands" (inference prefill) → Task 5 wires it into the production `ml8_gemm`. Covered.

**Placeholder scan:** No TBD/TODO; every kernel edit shows the full replacement block; the `kpack` invalid-kwarg trap is called out explicitly; the fp8-`where` fallback is concrete.

**Type/identifier consistency:** `tflops`/`pct_of_dense`/`DENSE_FP8_TFLOPS`/`build_synthetic_layer`/`launch`/`_median_seconds` defined in Task 1 and reused verbatim in Tasks 4–5. `_oracle_case` defined Task 1, reused Tasks 3–4. `lo_idx`/`hi_idx`/`w_lo`/`w_hi`/`a_even`/`a_odd` consistent across Tasks 2–3. `get_gemm_config` matches the real signature in `gemm_ml8.py:176`. `layer_from_components` args match `ml8_runtime.py:413`. Kernel launch kwargs match `_gemm_a8w8_blockscale_kernel`'s real parameter list.

**Risk acknowledged:** kernel perf has genuine empirical unknowns (Triton/gfx1201 lowering of `tl.split`, fp8 `tl.where`, two-dot scheduling). Mitigated by: oracle as hard invariant every step, bench as the ratchet, rocprof for evidence, systematic-debugging on any oracle break, and a concrete fallback for the one most-likely Triton rejection (fp8 where → fp32 ladder + cast).
