# Handoff: turbo4_0 multi-warp WMMA prefill-tile corruption on RDNA4 (Qwen3.6-27B, hd256)

**Date:** 2026-07-11 · **Machine:** mad-lab-main, R9700 / gfx1201 (RDNA4), `build-hip`
**Repo:** master @ `90bdfa9c5` (gate just removed — the bug is now exposed on the ungated mw path)
**Status:** root-caused to a specific kernel; **not fixed**. This is a handoff for Codex.

---

## 0. Scope — what this IS and is NOT

**IS:** the **turbo4_0** (128-elem blocks, head_dim **256**) **multi-warp WMMA _prefill_ tile kernel** corrupting attention output on RDNA4/gfx1201. Repro model: **Qwen3.6-27B-Q8_0** (arch `qwen35`, hd256, n_head=24, **n_head_kv=4**, 65 blocks). Output = pure `////` garbage (HTTP 200, no crash).

**IS NOT:** the turbo4_**64** (head_dim-64, LFM2.5) flash-**decode** centroid-table bug. That was a *separate* bug (decode used the N128 table instead of N64) — **already fixed** by Codex in `e591a6747` (`mt_pagedattn_decode.cu:318`). It is head_dim-64-only and does **not** touch this hd256 turbo4_0 path. Do not conflate them.

---

## 1. Symptom & reliable isolation

Qwen3.6-27B + `--cache-type-k/v turbo4` + `--flash-attn on` + `--kv-tiered 90,10,0 --kv-tier-paged-blocks`, on a prompt long enough that the tile fires (`avg_q_len ≥ 16`) → reasoning content is a solid run of `/`.

**Runtime A/B (env toggles, one binary, no rebuild). The load-bearing signal is the MULTIWARP toggle:**

| Env | Prefill path | Result |
|---|---|---|
| default | multi-warp WMMA tile | **GARBAGE** |
| `GGML_PAGED_TILE_MULTIWARP=0` | **single-warp** WMMA tile | **coherent** |
| `GGML_PAGED_TILE=0` | scalar prefill | coherent |
| `GGML_PAGED_DECODE=0` (tile still on) | mw tile prefill | **GARBAGE** (⇒ prefill, not decode, is the corruptor) |

So: **only the multi-warp variant of the prefill tile corrupts.** Single-warp tile, scalar prefill, split-K decode, and the turbo4 scatter are all correct.

> ⚠️ Two confounds to ignore: (a) `--cache-type q8_0` "coherent" is meaningless — q8_0 isn't admitted to the tile gate at all. (b) Some of claude's 2026-07-11 *morning* A/B calls were later shown unreliable (e.g. it wrongly concluded the RDNA4 dispatch gate was a "no-op" — it was actually effective; see §5). Trust the MULTIWARP toggle above, which is consistent across sessions.

**Discriminators (signature):** breaks only for **turbo4_0 × head_dim 256 × paged × multi-warp**.
- hd128 (llama/orpheus) turbo4-paged: **coherent**.
- hd256 **MoE** (ornith, `qwen35moe`) turbo4-paged: **coherent** — *even though it also hits the mw tile.* ornith has **n_head_kv=2**; Qwen3.6-27B has **n_head_kv=4**. This n_head_kv=2-vs-4 difference is **unexplained and is a prime lead** — the cooperative tile likely has a per-(kv_head) grid/smem index that only mis-behaves at n_kv_heads=4.

---

## 2. It is a REGRESSION (07-06 upstream sync)

Bisected on the R9700, same repro:
- **GOOD** `66eec7f54` (2026-07-01, pre-sync) → coherent.
- **BAD** `676b87554` (2026-07-09) → garbage.
- Only commit in range touching the paged/turbo path = the **07-06 upstream sync merge `06a3da0e6`**.

**But the mw-kernel source barely changed:**
- The only sync change to `mt_pagedattn_tile.cu` is `9e8310ecb` ("size mw tile Q_TILES to device shared-mem limit"), and **on AMD it is a no-op** — AMD takes the `FULL` config (`Q_TILES=6`, 64 KiB), byte-identical to pre-change; only the NVIDIA `NARROW` variant and a `NARROW_SMEM=false` template arg were added.
- `mma.cuh` was **not** changed by the sync.

⇒ The trigger is a **changed shared dependency the mw TU `#include`s**, almost certainly **`turbo-quant.cuh` (+142)**: the sync added `TURBO_CENTROIDS_4BIT_N64` + `TURBO4_64_OL*` `__constant__` tables and a `__device__ int g_turbo4_64_ol_use_n64_table`. **Leading theory: those added `__constant__`/`__device__` symbols shifted the mw kernel's register/occupancy/codegen and tipped a _latent_ cooperative-kernel bug that was accidentally-correct before.** (turbo4_0's own `TURBO_CENTROIDS_4BIT` N128 table is unchanged, and `coop_stage_turbo4_tile` uses it correctly — so this is NOT a centroid-table mismatch like the turbo4_64 decode bug was.)

---

## 3. Static analysis (2 passes: Codex + claude) — no readable logic defect

`coop_stage_turbo4_tile` (`mt_pagedattn_tile.cu:215-278`, the mw-only turbo4_0 staging, the sole source-level delta vs the correct single-warp path) reads **correct**:
- warp→qblock round-robin `qb=warp_id; qb+=N_WARPS` over 32 qblocks / 6 warps covers every qblock exactly once (no gap/overlap).
- `blk` depends on `qb` (not `lane_id`), so all lanes in a warp agree; lane-0 norm broadcast via `__shfl_sync(...,0,WARP_SIZE)` is consistent.
- `blk->qs + 2*lane_id` for lanes 0-31 covers `qs[0..63]` exactly (in-bounds); smem write offsets in range.
- **RDNA4 is wave32**, so the wave64 shuffle-width hazard class (the old gfx803 bug) does **not** apply here.
- `__syncthreads()` present after each cooperative stage (Q/K/V) in both kernel bodies.

Kernel bodies: WMMA path `#if defined(AMD_WMMA_AVAILABLE)` (~817-1147) and FMA `#else` (~1148-1394); both call `coop_stage_turbo4_tile`. No incorrect index / missing barrier / OOB / race was found by reading. ⇒ most likely a **compiler/resource-sensitivity**, not a fixable-by-reading line.

**Strongest (unproven) suspect — register/occupancy pressure in the HS=256 mw specialization:** `__launch_bounds__((NARROW?..:6)*32, 2)` = **(192, 2)** forces ≥2 blocks/SM → tight VGPR cap → possible spill/miscompile on RDNA4. Live state is unusually heavy: 16 Q-frags + 16 output-frags + ping-pong K-frags + 2 score accumulators + 6 cooperative warps doing divergent constant-table gathers.

---

## 4. What claude already tried

- **`__launch_bounds__` relaxation** (dropped the `,2` min-blocks → let the compiler use more registers / less spill): **BUILT but NEVER TESTED** (interrupted, then reverted). **This is the #1 untested lead — Codex should test it first.** Line `mt_pagedattn_tile.cu:803-804`.
- **RDNA4→single-warp dispatch gate** (`mw_on = ... && !(IS_RDNA4(cc) && HS==256)`): worked (correct bypass, single-warp is still WMMA) but is a bypass, not a fix — **removed** in `90bdfa9c5` per kmbandy, who wants the real kernel fix.
- Differential smem dump: planned, not executed.

---

## 5. Corrected fact (don't repeat claude's error)

gfx1201 cc parses (via `ggml_cuda_parse_id`, `ggml-cuda.cu:156`) to `GGML_CUDA_CC_OFFSET_AMD + 0x1201`, so `GGML_CUDA_CC_IS_RDNA4(cc)` is **true**. The dispatch gate was therefore **effective** (claude's morning "no-op" claim was wrong). Relevant only because it explains why the R9700 was coherent while the gate was in — the mw bug was bypassed, not gone.

---

## 6. Suggested experiments (in priority order)

Brick note: gfx1201 drives the desktop. The bug is **wrong-result, not a fault**, so *running* the mw kernel is safe; a bad kernel *edit* could fault → use crash-survivable iteration, and prefer iterating on a non-display card if one is available.

1. **Test the launch_bounds relaxation** (`(192,2)` → drop the `,2`, or try `(192,1)`), rebuild, repro. If mw goes coherent → it's register/occupancy pressure; the fix is a proper launch-bounds / register-budget adjustment.
2. **Confirm the second-order-compile mechanism:** on GOOD commit `66eec7f54`, add ONLY the sync's new `__constant__` tables + `__device__` global to `turbo-quant.cuh` (no call sites), rebuild, repro. Garbage ⇒ mechanism confirmed (codegen shift tips a latent bug) → then hunt the latent bug with instrumentation.
3. **Differential dump** (env-gated): dump `smem_k`/`smem_v` after staging, then QK scores, softmax max/sum, and PV output for block(0,0,0) in BOTH the mw and single-warp kernels on a fixed tiny input; first divergence localizes it (staging vs QK vs softmax vs PV).
4. **Chase the n_head_kv=4 discriminator:** instrument the cooperative tile's per-(kv_head) grid/smem indexing; ornith (n_kv=2) clean vs Qwen (n_kv=4) garbage should point at the faulty index.
5. **Compiler resource report (RGA / `--save-temps`)** for `mt_paged_attention_tile_mw_kernel<256,16,TURBO4_0,false>` on gfx1201 at GOOD vs BAD (or ±the turbo-quant.cuh constants); a VGPR/spill/occupancy delta confirms §3's theory.

---

## 7. Key references

- **mw kernel:** `ggml/src/ggml-cuda/mt_pagedattn_tile.cu` — `mt_paged_attention_tile_mw_kernel` ~805; `__launch_bounds__` ~803-804; `coop_stage_turbo4_tile` 215-278; WMMA body ~817-1147; FMA `#else` ~1148-1394.
- **dispatch:** `ggml/src/ggml-cuda/mt_pagedattn.cu` — `mw_on` select ~1720 (gate removed); tile gate (`HS 128/256 & F16/TURBO4_0/TURBO3_0 & avg_q_len≥16`) ~1685; env toggles + `MAD_PAGEDATTN_PROBE` ~55-90 / ~1700.
- **suspected trigger:** `ggml/src/ggml-cuda/turbo-quant.cuh` — added `TURBO_CENTROIDS_4BIT_N64` etc. (~L352-500). turbo4_0's `TURBO_CENTROIDS_4BIT` (N128) unchanged.
- **cc parse:** `ggml/src/ggml-cuda/ggml-cuda.cu:156` `ggml_cuda_parse_id`.
- **older/companion writeup:** `docs/dev/2026-07-10-turbo4-paged-mw-wmma-rdna4-corruption.md` (has the full A/B matrix + bisect detail; note its §4.2 gate discussion is superseded by §5 above).

## 8. Repro command

```
<ENV> ./build-hip/bin/llama-server \
  --model ~/models/Qwen3.6-27B-Q8_0.gguf --n-gpu-layers 999 --ctx-size 131072 \
  --parallel 1 --cache-type-k turbo4 --cache-type-v turbo4 --flash-attn on \
  --kv-tiered 90,10,0 --kv-tier-ssd-path <dir> --kv-tier-paged-blocks \
  --no-mmap --no-warmup --jinja --host 127.0.0.1 --port 8091 --device ROCm0
```
Send a demanding, multi-paragraph reasoning prompt (short prompts take the scalar path and hide the bug). Env toggles: `GGML_PAGED_TILE_MULTIWARP`, `GGML_PAGED_TILE`, `GGML_PAGED_DECODE`, `MAD_PAGEDATTN_PROBE=verbose` (logs `[probe-tile] ... wmma=1` when the tile fires). The live R9700 service is `llama-server-qwen36-27b-swatm-r9700` (:8090); its binary is the old *gated* build (still coherent) until `build-hip` is rebuilt from `90bdfa9c5`.
```

---

## 9. Codex update (2026-07-11, before handing back to Opus)

Codex ran the investigation directly on the R9700. The systemd service on
`:8090` and the manual test server on `:8091` are both stopped at handoff.
The checkout is intentionally dirty with diagnostic edits described below.
Do not treat the current source as a proposed fix.

### 9.1 Compiler metadata findings

The gfx1201 code object for the failing specialization was extracted from
`build-hip/bin/libggml-hip.so.0.15.3` and inspected with `llvm-readelf`:

| Specialization | VGPR | VGPR spills | private scratch/thread |
|---|---:|---:|---:|
| HS256 TURBO4_0 FULL (6 warps) | 256 | 21 | 88 B |
| HS256 TURBO4_0 NARROW (3 warps) | 248 | 0 | 0 B |
| HS256 F16 FULL | 247 | 0 | 0 B |

This initially looked like a strong lead, but two follow-up compilations
disproved it as the regression mechanism:

1. Removing only the second `__launch_bounds__` argument (`..., 2`) produced
   exactly the same 256 VGPR / 21 spill / 88 B metadata.
2. Recompiling known-good commit `66eec7f54` with the current compiler also
   produced exactly 256 / 21 / 88 for HS256 TURBO4_0, with the same kernel
   code size (`0x3278`).

Therefore the spills are real but are not new in the bad source, and the
new turbo4_64 centroid constants did not shift this kernel's resource
allocation under the current compiler. The original incremental bisect
should eventually be revalidated with clean build directories in case a
stale HIP object affected it.

### 9.2 Runtime experiment matrix

All requests used the same 153-token planning prompt, temperature 0, and
`MAD_PAGEDATTN_PROBE=verbose`. Every bad run logged 16 calls like:

```
[probe-tile] avg_q_len=149 max_ctx=131072 wmma=1 total_q=149
```

The failure was deterministic: the response contained only `/` in
`reasoning_content` (128 or 256 tokens).

| Experiment | Result | Conclusion |
|---|---|---|
| Force existing NARROW kernel on RDNA4 HS256 TURBO4_0 (3 warps, 40 KiB LDS, zero spills) | GARBAGE | Not caused by 64 KiB LDS or reported spills |
| Replace `coop_stage_turbo4_tile` for both K and V with generic per-element `ops::k_load` / `ops::v_load` staging | GARBAGE | Not the cooperative turbo4 staging, centroid gather, or its n_kv-head indexing |
| Replace multi-warp N_ACC=2 QK accumulation and K/V ping-pong prefetch with the coherent single-warp kernel's direct load-then-WMMA loops | GARBAGE | Not the N_ACC or ping-pong optimization |
| Set HS256 NARROW `Q_TILES_PER_BLOCK=1` while still using the multi-warp kernel body | GARBAGE | Does not require multiple warps or inter-warp interference |
| Carry FP32 scores across the V-load barrier and pack to half2 after `__syncthreads()`, matching the single-warp kernel | GARBAGE | Not the packed `scores_h` lifetime across the barrier |
| Remove the complete `__launch_bounds__` attribute | GARBAGE | Not the launch-bounds attribute |

These results substantially narrow the bug: even with one warp, generic K/V
staging, direct WMMA loops, post-barrier packing, and no launch bounds, the
multi-warp kernel body still corrupts while dispatching to the separate
single-warp kernel remains coherent.

### 9.3 Current untested build and dirty source state

The final diagnostic change added the extra 1 KiB FP32 score scratch that
the coherent single-warp launcher allocates unconditionally:

```
smem_bytes = mw_tile_smem_bytes(...) + 16 * 16 * sizeof(float)
```

This build completed successfully but was **not runtime-tested** before the
handoff. Test this first. It is the last obvious structural difference in
the reduced one-warp experiment.

Current diagnostic edits in `mt_pagedattn_tile.cu` are cumulative:

- RDNA4 + HS256 + TURBO4_0 forces `NARROW_SMEM=true`.
- HS256 narrow Q tiles changed from 3 to 1.
- TURBO4_0 uses generic per-element K/V staging in the multi-warp body.
- N_ACC=2 and K/V ping-pong prefetch were replaced by direct WMMA loops.
- FP32 scores, rather than packed half2 scores, cross the V-load barrier.
- The multi-warp kernel has no `__launch_bounds__` attribute.
- The launcher adds 1 KiB unused dynamic-smem padding (built, untested).

Because the edits are cumulative diagnostics, revert them selectively or
restore the file from `90bdfa9c5` before constructing a final fix.

### 9.4 Recommended next steps

1. Runtime-test the already-built 1 KiB smem-padding candidate first.
2. If it still fails, stop source-toggle experiments and add a differential
   numeric dump for block `(head=0, seq=0, q-group=0)` in both kernel entry
   points. Capture, in order: staged Q, staged K/V, first QK accumulator,
   masked/scaled scores, softmax max/sum, first PV accumulator, final output.
3. Because the reduced multi-warp body is now one warp and nearly mirrors the
   coherent kernel, the first differing checkpoint should expose either a
   launch/grid/shared-memory-layout difference or a compiler codegen issue.
4. Re-run GOOD/BAD with fresh build directories before relying further on
   the July 6 source bisect.

---

## 10. Grok resolution (2026-07-11 evening)

### 10.1 Root-cause reframe (supersedes §0–§3 “mw-only” framing)

Codex’s reductions were valuable but aimed at the wrong layer. Runtime
isolation on clean `90bdfa9c5` shows:

| Path | Simple math (`17+25`, ~20 tok) | Plan-style prompt (~38–105 tok) |
|---|---|---|
| Scalar prefill (`GGML_PAGED_TILE=0`) | coherent | coherent |
| **WMMA** single-warp tile | often coherent | **pure `/` garbage** |
| **WMMA** multi-warp tile | **usually garbage** | **pure `/` garbage** |
| **FMA** single-warp tile | coherent | coherent |
| **FMA** multi-warp tile | coherent | coherent |

So:

1. **Not multi-warp-specific.** Cooperative staging, N_ACC, launch_bounds,
   smem padding, and Q_TILES are all red herrings once WMMA is removed.
2. **WMMA tile path is numerically wrong on RDNA4.** Multi-warp looked like
   “the” bug because it fails even easy prompts; single-warp *masks* the
   bug on easy prompts (template math) and fails on demanding ones.
3. Prior “MULTIWARP=0 coherent” A/Bs were confounded by using easy math
   canaries (or short prompts). The handoff’s own plan prompt **fails on
   single-warp WMMA** too.
4. FMA tile multi-warp working proves the cooperative algorithm, GQA
   indexing, turbo4 staging, and softmax/mask logic are fine.

### 10.2 1 KiB smem-padding candidate

Tested (Codex’s untested build): still garbage under WMMA. Closed.

### 10.3 Layout probe result (WMMA matmul is fine)

Standalone HIP probes on gfx1201 (`/tmp/rdna4_wmma_{tile,pv,ninner}_probe.hip`):

- QK `mma(D,K,Q)` + `get_i`/`get_j` matches FMA host ref (max |diff| ~1e-4)
- PV `load_ldmatrix_trans` + `mma(D,V,scores_h)` matches FMA host ref (exact)
- HS=256 multi-chunk `N_INNER=16` preload-all-Q matches FMA host ref

**Operand order / C-lane map are not the bug.**

### 10.4 Actual root cause + fix

`mt_pagedattn_tile.cu` is built with **`-ffast-math`** (`-ffinite-math-only`).
Under that, IEEE **`-INFINITY` is UB**. The compiler may delete inf
materialization and `x == -INFINITY` tests → causal mask / softmax breaks
in the WMMA specialization (FMA path was less sensitive / “accidentally
OK” on easy prompts).

**Fix (working tree):** replace all `-INFINITY` in this TU with a finite
sentinel:

```cpp
static constexpr float SOFTMAX_MASK_VAL = -1.0e30f;
```

Used for mask fill, `running_max` init, comparisons, and masked-exp zeroing.
No FMA force, no multiwarp gate. WMMA multi-warp stays on.

### 10.5 Verified (R9700, default multiwarp ON, turbo4+paged)

| Check | Result |
|---|---|
| math | coherent |
| full plan handoff prompt | coherent (real plan text, not `/`) |
| PP ~200 tok | ~539 t/s |
| PP ~800 tok | ~895 t/s |
| PP ~2k tok | ~988 t/s |
| PP ~4k tok | **~1015 t/s** (matches pre-fix WMMA ~1048; FMA was ~667) |

Perf hit ≈ **0** vs broken WMMA baseline; **not** the FMA crutch.

### 10.6 Follow-ups

- Same `-INFINITY` pattern still exists in `mt_pagedattn_decode.cu` and
  scalar `mt_pagedattn.cu` — latent under `-ffast-math`. Apply the same
  sentinel when those paths misbehave.
- Prefer a TU-wide ban on IEEE inf in HIP sources built with fast-math,
  or drop `-ffinite-math-only` for attention TUs.
