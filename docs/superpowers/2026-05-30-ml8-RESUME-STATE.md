# ml8 Full-Model Coverage — Resume State (2026-05-30 night)

Post-compaction handoff. Plan: `docs/superpowers/plans/2026-05-30-ml8-full-model-coverage.md` (18 tasks).
Spec: `docs/superpowers/specs/2026-05-30-ml8-full-model-coverage-design.md`.
Branch: `feat/ml8-full-model-coverage` (off `feat/upstream-merge-2026-05-27`).
Execution: subagent-driven-development, inline controller. Tracking tasks #133–150.

## DONE + committed (9 commits, T1–T10 + build fix) — all tested/device-verified

| commit | task | what |
|--------|------|------|
| a67d5b0b2 | T1 | `role_targets.py` — `classify_role(hf)->(gguf_name, role, Tier{ML8,FP8,NATIVE})` |
| 366f142e2 | T2 | `scaled_fp8.py` — `quantize/dequantize_scaled_fp8` (per-group scale + e4m3, group 32) |
| 2a0cded53 | T3 | `ssm_sensitivity.py` — kurtosis + fp8 SNR + `report()` |
| 6db1b6f17 | T4 | `calibrate_ml8_paged.py --dense-coverage full` (default `ffn`, opt-in; FFN path untouched). `find_dense_full_targets` enumerates ML8 roles (attn q/k/v/o, qkv, attn_gate, ssm_out, eh_proj, lm_head) + FP8 roles (token_embd, ssm_alpha, ssm_beta). ML8 → existing GPTQ loop; FP8 → separate pass (no Hessian, group_size FIXED 32), saves `*.fp8.pt`. MoE path untouched. |
| 5fd132bfb | T5 | gguf-py `ML8_FP8 = 51`, `GGML_QUANT_SIZES (32, 34)` |
| 45925db35 | T6+T7 | `ml8_to_gguf.py`: `_build_blob_map` discovers new ML8 roles via `classify_role` fallback (excludes `*.fp8.pt`); `pack_scaled_fp8_blocks` + `_build_fp8_blob_map` write FP8 as `ML8_FP8`; `evaluate_coverage(params_ml8, params_fp8, params_passthrough_weight, min_coverage) -> (cov, below, breakdown)`; FFN-only still refuses |
| — | T8 | CLOSED: `ml8_to_packed.py` (native `.ml8`) is name-agnostic → ML8 roles already pack; native-FP8 deferred (not pod-critical) |
| 9bfee4509 | T9 | ggml `ML8_FP8` type: `ggml.h` enum 51 / `GGML_TYPE_COUNT` 52; `ggml-common.h` `block_ml8_fp8 { ggml_half scale; uint8_t qs[32]; }` 34B static_assert; `ggml.c` traits + `dequantize_row_ml8_fp8` in `ggml-turbo-quant.c` (reuses `g_fp8_e4m3_lut`) |
| a6cf88d95 | T10 | `ML8_FP8` get_rows — `getrows.cu` `get_rows_cuda_q<32,1,dequantize_ml8_fp8>`, `from_float` (`quantize_row_ml8_fp8_ref` + `ggml_quantize_chunk` case), `supports_op` GET_ROWS+ML8_FP8 (`ggml-cuda.cu:5567`), `ggml-cpu/ops.cpp` CPU case. **DEVICE-VERIFIED** test-backend-ops (ROCm0/1/CPU OK) |
| 2466fe133 | build | aiter-link CMake fix (permanent) |

## Critical non-obvious facts (do NOT re-derive)

- **GPU FP8 decode bug caught in T10:** use `ggml_cuda_e4m3fn_to_fp32` (signed OCP e4m3fn, bias 7, max 448), **NOT** `ggml_cuda_ue4m3_to_fp32` (that has a built-in `/2` — it's for NVFP4 block *scales*, not e4m3 *values*).
- **ML8_FP8 on-disk block (34 B / 32 elems, row-major along K):** `[fp16 scale (2B)][32 × e4m3 byte]`. `scale = amax(|x|)/448`. Encode = `torch.float8_e4m3fn` (Python) / `quantize_row_f8_e4m3_ref` (C). CPU decode = `g_fp8_e4m3_lut`; GPU decode = `ggml_cuda_e4m3fn_to_fp32`.
- **Blob contract:** ML8 = `*.pt` (existing schema, discovered via `classify_role` fallback in `_build_blob_map`); FP8 = `*.fp8.pt` `{name, kind:"scaled_fp8", e4m3 f32, scale fp16 [N,K/32], group_size, shape}`.
- **9B model:** 33-layer hybrid, vocab 248320, `token_embd`+`output` untied (~1B params each).

## BUILD ENVIRONMENT (permanently fixed tonight)

- **Triton** 3.7.0 built from source at `~/GitHub/triton`; was orphaned (lost its editable registration). Restored via `.pth`: `~/.local/lib/python3.14/site-packages/triton-editable.pth` → `~/GitHub/triton/python`. Survives clean env. If it vanishes again, recreate that one-line `.pth`.
- **aiter link:** `ggml-hip/CMakeLists.txt` now guards `if(TARGET aiter_triton_aot)` + pins `target_link_directories($<TARGET_FILE_DIR:...>)` + `FATAL_ERROR` if Triton missing. No more `LIBRARY_PATH` workaround.
- **HIP build:** `cmake --build build-hip --target ggml-hip -j3` (arches `gfx1201;gfx1030`). Host: `build-ml8fp8-host` (target `ggml-base`). **ml8 mul_mat ops are `#ifdef GGML_HIP_AITER`** → require Triton/aiter present.
- Test: `./build-hip/bin/test-backend-ops test -o GET_ROWS` (look for `ml8_fp8 ... OK`).

## DONE 2026-05-31 night (T11–T14, 4 more commits) — C++ inference wiring COMPLETE

| commit | task | what |
|--------|------|------|
| e6212075a | T11 | no-LUT FP8-WMMA `mul_mat` for ML8_FP8 weights. KEY: did NOT write a new kernel — the AITER Triton `_gemm_a8w8_blockscale_kernel` already has `WEIGHT_FORMAT=0` (raw fp8 B + per-group scale = the ML8_FP8 contract). Exposed it via `mt_ml8_gemm` (new `weight_format` field on the shape struct + ShapeKey discriminator; b_ptr dtype `*fp8e4nv` vs `*i8`). `ggml_cuda_op_ml8_fp8_mul_mat` in ml8.cu (src[0]=w, src[1]=x, NO centroid; repack on-disk `[N,K]` 34B blocks → raw e4m3 `[K,N]` + fp32 `[n_groups_k,N]` scale). `ggml_cuda_mul_mat` routes `src0->type==ML8_FP8` early (NO op-swap — stays plain MUL_MAT). +CPU vec_dot traits for the test reference. Device-verified test-backend-ops `-o MUL_MAT`. |
| f69566870 | T11-fix | **Perf bug found+fixed in review:** the FP8 repack cached by `w->data` but content-checksummed (256 D2H + `cudaStreamSynchronize`) on EVERY call to defend against test-backend-ops buffer recycling → a per-token stream stall on the α/β path. Replaced with pointer-only keying (like ml8-4) + wired the (previously-dead) `ggml_cuda_ml8_clear_cache()` into `ggml_backend_cuda_buffer_free_buffer` (correctness invariant: freeing a buffer invalidates pointer keys into it; also fixes a latent repack-buffer leak). |
| ab59cde0a | T12 | `src/llama-ml8-registry.{h,cpp}`: `ml8_sidecars{centroids,rotation_h_a,awq_scale}` + `ml8_registry` (pointer-keyed map, `register_weight`/`find`) + `build_ml8_or_mul_mat(ctx,reg,w,x)` — ML8_4+centroids → AWQ+rotation then `ggml_ml8_mul_mat` (`GGML_OP_ML8_MUL_MAT`); ML8_FP8/else → plain `ggml_mul_mat` (zero-impact fallback); ML8_4-without-centroids → loud GGML_ASSERT. Unit test `test-ml8-registry`. |
| 2cc59047e | T13 | Routed ALL qwen35 attn/ssm/lm_head GEMMs by making `build_lora_mm` ml8-aware (they all funnel through it — call sites unchanged; LoRA + `w_s` output-scale preserved; AWQ is input-side, composes correctly). `ml8_registry ml8_reg` member on `llama_model`, plumbed via `llm_graph_params.ml8_reg`→ctor→`build_lora_mm` exactly like `loras` (set at `llama-context.cpp:2335`). qwen35 load registers sidecars (guarded on `GGML_TYPE_ML8_4`) for wqkv/wq/wk/wv/wo/wqkv_gate/ssm_out/eh_proj/output — NO new layer fields (registry holds the pointers). **Regression guard (Case D): empty-registry fallback byte-identical op/src/shape to `ggml_mul_mat` — PASSES.** Also caught: test-ml8-registry built `-DNDEBUG` so original `assert()`s were vacuous → converted to always-on `CHECK()` (proved it fires). |
| — | T14 | **CODE COMPLETE BY CONSTRUCTION, no commit.** token_embd + α/β accept ML8_FP8 with zero new code: `create_tensor` reads GGUF type ungated; `build_inp_embd`→`ggml_get_rows` (T10 ML8_FP8 case); α/β→`build_lora_mm`→`ggml_mul_mat`→CUDA no-LUT FP8 (T11/T13). Synthetic-GGUF TEST folded into T15 (kmbandy approved 2026-05-31). |

## ⚠️ BLOCKER — DEBUG IN PROGRESS (2026-05-31 morning, Phase 1 partial)

**Symptom (last night):** `mt_ml8_gemm dispatch failed` — `llama-cli -ngl 99` on the FFN-only 9B `/home/kmbandy/models/Qwen3.5-9B-ml8.gguf` loaded + built graph + passed bf16 attn, then aborted in the pre-existing FFN ml8 CUDA path (`ggml_cuda_op_ml8_mul_mat` → `mt_ml8_gemm`, **ml8.cu:989** `GGML_ASSERT(gemm_rc==hipSuccess)`). MAD-223 code, NOT T11–T14. bf16 9B runs clean.

**Phase-1 findings so far (do NOT re-derive):**
- `mt_ml8_gemm` returns non-success from 3 places, each with an existing `fprintf` diagnostic: **compile fail** (mt_ml8_gemm.cpp:209 "kernel compile failed for shape"), **validation fail** (232/239/245: M%BLOCK / N%BLOCK / K%group_size), **launch fail** (349). → a clean repro with full stderr will say WHICH.
- **Runtime JIT CAN compile**, hypothesis "python3 lacks Triton" RULED OUT: `python3` = `/usr/bin/python3` = Python **3.14.4**, `import triton` = **3.7.0 OK** there. `AITER_PYTHON` default `python3`; no `AITER_*`/`TRITON_*` env set.
- aiter cache `~/.cache/llama.cpp/aiter/` = **139 entries**, mostly `_gemm_a8w8_blockscale_kernel__hip-gfx1201-32__W4S1__<hash>` (May 26 bulk + a few May 30 from last night's test-backend-ops). W4S1 = the WF=1 LUT path.
- **Device naming on this build:** `--device cuda:0` is REJECTED. Use **`--device ROCm0`** (= R9700 gfx1201 32GB, the ml8 target) or `ROCm1` (= RX 6900 XT gfx1030 16GB). `--list-devices` to confirm.
- **The abort did NOT reproduce this morning.** With `--device ROCm0 -p Hello -n 4 -no-cnv`, the 9B ml8 GGUF loaded and went **INTERACTIVE** (`> ` prompts) until my 300s timeout (exit 124). `-no-cnv` did NOT suppress interactive mode → I got neither a clean abort NOR a clean token. So the blocker may have been transient (e.g. last night's rebuild JIT-compiled the needed kernels into cache), or it's intermittent.

**NEXT STEP (clean repro, deterministic, non-interactive):** use **`llama-perplexity`** instead of llama-cli — it runs a fixed forward pass through the FFN ml8 path and exits (no interactivity), so it either prints a PPL (blocker gone) or emits the real `fprintf` error. Command shape:
`./build-hip/bin/llama-perplexity --no-mmap -m /home/kmbandy/models/Qwen3.5-9B-ml8.gguf -ngl 99 --device ROCm0 -f <small.txt> 2>&1 | tee /tmp/ml8_repro.log` (need a small wikitext/calibration .txt; --chunks 1-2 is enough). If it ABORTS → read the fprintf (compile/validation/launch). If it PPLs → blocker was transient, unblock T15.

**T15 + T16 9B gate cannot produce a token until this is confirmed clear.** Use systematic-debugging — get the actual error/return path before any fix (no Microsoft-pattern workarounds).

## REMAINING

- **T15** (also covers T14) — `check_bit_equivalence.py` (Python `Ml8Linear` ref vs C++ graph PPL to ≥4 decimals; ml8 kernel deterministic at 8.2990 per prior fact; start on 4B). **Blocked on the dispatch failure above.**
- **T16 / T18** — multi-hour GPU calibration runs. **FLAG BEFORE KICKING OFF** (user wants explicit go). T16 dense 9B: `calibrate_ml8_paged.py --dense-coverage full` (resident, local). Gate: ΔPPL ≤ +0.08–0.10 vs bf16 9B, size < UD-Q4 9B (~4.7 bpv / ~5.4 GB), coverage clears, long-ctx probe (T17, SSM-gate compounding). T18 35B MoE both-axes vs UD-Q4_K_XL (5.7507). NOTE: the pod does CALIBRATION (Track A, fully done) — these gates are LOCAL post-calib validation and depend on the dispatch failure being fixed.

## POD GAUNTLET MATRIX — now durable in git (2026-05-31)

`scripts/calibration/mi300x/` (was untracked, now committed). The MI300X pod runs **calibration only** (blobs + Y_SNR); PPL runs later on R9700.
- `92a73fd24` — tier-1 harness + `gauntlet_tier1.json` (12 cells: 27B dense + 35B MoE × group_size grid gu{128,64}×d{16,32,64}, bpv 4.25, rotation=kronecker fixed).
- `4a2435b34` — `TIERS.md`: tier-1→Y_SNR-rank→tier-2(top-3 × levers)→R9700-PPL flow. Tier-2 levers: mag_weighted-down, heavy++, corpus-scale, rotation. Rotation menu = `identity | kronecker | hadamard_full | quarot-R1/R_resid` (only none|kronecker live; hadamard_full needs wiring; quarot-R1 blocked on γ-absorption bug).
- `6fd251c50` — added AWQ (`--awq none|mean`+α) as 5th tier-2 lever + **the unifying root cause**: this Qwen family has per-TOKEN (not per-channel) outliers (down_proj SwiGLU median kurtosis 121–183), so per-channel levers (AWQ + quarot-R1/residual-Hadamard) are both low-prior; budget goes to mag_weighted-down + group_size-down.
- TODO when tier-1 runs: write `gen_gauntlet_tier2.py` (input = top-3 tier-1 cell IDs → emit lever cross). Tier-2 is data-dependent so it's generated AFTER tier-1.

## Uncommitted WIP — LEAVE ALONE (not part of tonight's plan)

`ggml/src/ggml-cuda/ml8.cu`, `fattn-mma-f16.cuh` (earlier-session audit fixes: AITER gating, 640/512 + fail-loud), various `scripts/calibration/*.py`, and the untracked `docs/superpowers/` spec+plan. The plan's task commits are surgical (`git add <specific files>`) and don't touch these.
