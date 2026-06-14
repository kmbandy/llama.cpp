# Recon: aiter RDNA4 (gfx1201) FP8 substrate — shape coverage, backward GEMM, reusable kernels, Triton coupling

Date: 2026-06-14
Scope: ROCm/aiter Triton kernels for our vendored fp8/ml8 quant substrate in the llama.cpp fork.
Hardware in scope: **gfx1201** (R9700 / RDNA4, training), **gfx942** (MI300X / CDNA3, datacenter).
Out of scope: **gfx1250** (newer arch we do NOT have — flagged inline below).

Method note: this was read-only research. Git/`gh`/GitHub-MCP were sandbox-denied in this
environment, so remote git state (branches/PRs/releases) was gathered via web fetch of GitHub
pages and search; everything about kernel/config *contents* was read directly from the local
clone at `/home/kmbandy/GitHub/aiter`. Unknowns are flagged explicitly.

## Provenance / versions

- Vendored at `9c79a5b59` (2026-05-25), roughly the v0.1.14 era.
- Local working clone is newer than vendored (per task context, clone @ `69cbe3ff8`). The file
  layout in the clone is the **reorganized** tree (`aiter/ops/triton/gemm/basic/...`,
  `_triton_kernels/...`, `gluon/...`), i.e. newer than the vendored snapshot. Treat the contents
  below as "latest aiter," not as a byte-for-byte description of `9c79a5b59`.
- Latest aiter releases (GitHub): **v0.1.15.post1** (Jun 8 2026), v0.1.15 (Jun 6), v0.1.14.post1
  (Jun 5), v0.1.14 (May 23). So re-vendoring to v0.1.15.post1 is the current released ceiling.

---

## 1. Exact shape coverage for OUR shapes (tuned file vs generic fallback)

Our real GEMMs: in/out features (N, K) drawn from **{2560, 4096, 8192, 9216}**; M (tokens) ≈ 800–7000.

### How selection works (confirmed in code)
`aiter/ops/triton/utils/gemm_config_utils.py :: get_gemm_config()` (called via `_get_config` in
`_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`):
1. Always loads the **default** (generic) file `gfx1201-GEMM-A8W8_BLOCKSCALE.json` (must exist).
2. If a per-shape file `gfx1201-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json` exists, it is used and the
   function returns `is_tuned=True`. If it does **not** exist, it silently uses the default file
   and returns `is_tuned=False`.
3. Within the chosen file it picks an M bucket: `M_LEQ_x` (x in 4,8,16,32,64,128,256,512,1024,
   2048,4096,8192), then `M_GEQ_x`, else `"any"`.

The generic `gfx1201-GEMM-A8W8_BLOCKSCALE.json` only defines buckets **M_LEQ_8 … M_LEQ_512** plus
**`"any"`**. It has no `M_GEQ_*` and no `M_LEQ_>512`. Therefore for our token counts (M ≈ 800–7000)
the fallback path always lands on the single generic **`"any"`** config
(`BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, GROUP_M=4, num_warps=8, num_stages=2, waves_per_eu=2,
NUM_KSPLIT=1`). That is the same low-effort config for every large-M shape that misses a tuned file.

### Coverage table (non-preshuffled A8W8_BLOCKSCALE, gfx1201)

Existing per-shape files that touch any of our N/K values:
`N=8192-K=8192`, `N=8192-K=1024`, `N=8192-K=32768`, `N=4096-K=512`, `N=4096-K=7168`,
`N=1024-K=8192`, `N=32768-K=8192`. There are **zero** files with `N=2560`, `N=9216`, `K=2560`,
or `K=9216`.

| (N, K)        | Tuned file present? | Result |
|---------------|---------------------|--------|
| 8192, 8192    | **YES** (`...-N=8192-K=8192.json`) | tuned (`is_tuned=True`) |
| 8192, 4096    | no | generic `"any"` fallback |
| 8192, 2560    | no | generic `"any"` fallback |
| 8192, 9216    | no | generic `"any"` fallback |
| 4096, 4096    | no | generic `"any"` fallback |
| 4096, 8192    | no | generic `"any"` fallback |
| 4096, 2560    | no | generic `"any"` fallback |
| 4096, 9216    | no | generic `"any"` fallback |
| 2560, *any of 2560/4096/8192/9216* | no | generic `"any"` fallback |
| 9216, *any of 2560/4096/8192/9216* | no | generic `"any"` fallback |

**Bottom line on coverage: of our 16 (N,K) combinations, exactly ONE — (8192, 8192) — hits a
specialized tuned file. The other 15 fall back to the generic M-bucketed config, and because our M
is large, they all collapse onto the single generic `"any"` entry.** Note `N=4096-K=512` and
`N=4096-K=7168` exist but do NOT help us — their K (512 / 7168) is not in our set, so our (4096, ·)
shapes still miss.

If you use the **preshuffled** variant (`A8W8_BLOCKSCALE_PRESHUFFLED`), tuned files are:
`N=4096-K=4096`, `N=4096-K=12288`, `N=6144-K=4096`, `N=24576-K=4096`. Of our shapes only
**(4096, 4096)** is covered there. (Preshuffle path is `gemm_a8w8_blockscale_preshuffle()` +
PR #3611 "padded-K a8w8 bpreshuffle", merged Jun 14 2026.)

Practical implication: our most common shapes (anything with 2560 or 9216, plus 4096×* and
8192×{2560,4096,9216}) run on the generic config. If perf matters, we should tune-and-vendor
per-shape JSONs for our exact (N,K) set — this is data-only, no kernel changes, no Triton bump.

---

## 2. Backward GEMM (dgrad / wgrad) availability — DEFINITIVE

**There is NO fp8 (or any) dgrad/wgrad backward GEMM in aiter.** aiter's a8w8/blockscale GEMMs are
**forward-only inference kernels**. Confirmed by:

- No `dgrad`/`wgrad`/`grad_output` symbols anywhere under `aiter/ops/triton/gemm/` or
  `_triton_kernels/gemm/`.
- No `torch.autograd.Function`, no `def backward`, no `ctx.save_*` in the GEMM tree.
- The only "backward" string in the GEMM tree is a docstring "backward-compatible API … deprecated"
  in `gemm_afp4wfp4.py` (API-compat, not a gradient kernel).
- The only backward kernels in the whole Triton tree are **attention** (`attention/mha_fused_bwd.py`)
  and **gated_delta_rule** — neither is an fp8 GEMM.
- Remote: searching ROCm/aiter PRs for `dgrad/wgrad/backward fp8` surfaces no fp8-GEMM-backward PR;
  matches are tuning/config/communication PRs only.

Caveat / red herring to ignore: a web search surfaced "cast_transpose / cast_transpose_bgrad … fp8
training." Those belong to **ROCm/TransformerEngine**, not aiter. Grep for `cast_transpose`/`bgrad`
across the aiter clone returns **nothing**. So if we need fp8 dgrad/wgrad for QAT we must either
(a) write our own backward GEMMs (can reuse the forward a8w8 blockscale kernel body with
transposed/relayout operands), or (b) pull from TransformerEngine, not aiter.

---

## 3. Reusable fused quant / rotation / LUT-weight kernels for gfx1201

### Fused FP8 (e4m3) quant prologues — YES, several (reusable)
`aiter/ops/triton/quant/fused_fp8_quant.py` (+ `_triton_kernels/quant/fused_fp8_quant.py`) exposes
fused **norm/activation + e4m3 group/per-tensor quant** kernels:
- `_fused_rms_fp8_per_tensor_static_quant_kernel` — RMSNorm + fp8 per-tensor static quant
- `_fused_rms_fp8_group_quant_kernel` — RMSNorm + fp8 per-group quant
- `_fused_rms_gated_fp8_group_quant_kernel` — gated RMSNorm + fp8 group quant
- `_fused_flatten_fp8_group_quant_kernel`
- `_fused_reduce_act_mul_fp8_group_quant` / `_fused_reduce_rms_fp8_group_quant_kernel`
- `_fused_silu_mul_fp8_per_tensor_static_quant_kernel`

Output dtype is e4m3 (`get_fp8_e4m3_dtype`). These are exactly the "fused prologue that lands fp8
activations" shape we want, and they are arch-generic Triton (work on gfx1201 since
`is_fp8_avail()` includes gfx1201). Plain (non-fused) quant lives in `quant/quant.py`; MoE-side
fused quant in `moe/quant_moe.py`, `moe/moe_op_*_fused.py`.

### Fused **rotation** (Hadamard / QuaRot-style) + quant — NO
There is **no Hadamard / online-rotation kernel** in aiter. Every "rotat*/rotary*" hit is **RoPE**
(rotary *position* embedding): `aiter/ops/triton/rope/rope.py`,
`fused_qkv_split_qk_norm_rope_cache.py`, `fusions/fused_*rope*`, `rotary_embedding.py`. None of
these perform a quantization rotation. **Our "fused rotation + e4m3-activation-quant prologue" does
not exist in aiter and must be authored.** Closest reusable building block: the
`_fused_rms_*_fp8_group_quant` kernels above — graft a rotation/Hadamard matmul stage in front of
the existing quant epilogue rather than starting from scratch.

### LUT-weight fp8 kernels — NO (none found)
No lookup-table / codebook weight GEMM for fp8 on gfx1201 (or any arch) in the clone. Closest
adjacent low-bit weight kernels are MoE `moe_op_gemm_a16w4 / a4w4 / a8w4` and `gemm_afp4wfp4`
(mxfp4), but these are not LUT/codebook designs. **LUT-weight is not available; author it.**

### Gluon kernels are NOT for us
`aiter/ops/triton/gluon/*` and `_gluon_kernels/gfx1250/*` exist, but `arch_info.is_gluon_avail()`
returns true only for **gfx950 and gfx1250**. There is a `gluon/gemm_a8w8_blockscale.py`, but it
will not be selected on gfx1201. Likewise `_gluon_kernels/gfx1250/...` and the `flydsl`
`fmha_gfx1250` kernels are **gfx1250-only → NOT-FOR-US**. Stick to the standard
`gemm/basic/gemm_a8w8_blockscale.py` path on gfx1201.

Minor arch note: `arch_info._LDS_CAP_BYTES` has entries only for gfx950 (160 KiB) and gfx942
(64 KiB) — **gfx1201 is absent**, so `pick_gemm_num_stages()` hits its `cap is None` branch and
returns `num_stages=2` unconditionally for gfx1201. Not a correctness bug, but it means the
LDS-aware stage heuristic is effectively disabled for us; tuned JSON configs are how stages get set
on gfx1201.

---

## 4. aiter ↔ Triton version coupling

**Yes, newer aiter pins a newer Triton, and it is a hard requirement in this clone:**

- `.github/scripts/install_triton.sh` ends with an explicit guard:
  `if Version(triton.__version__) < Version("3.6.0"): raise … "triton>=3.6.0 is required"`.
  → **aiter (current) requires Triton >= 3.6.0.**
- That script installs Triton **and a separate `triton-kernels` package** from AMD's index
  `https://pypi.amd.com/triton/release_/rocm-<major.minor>.0/simple/` (defaults to rocm-7.0.0).
  So the expected Triton is **AMD's ROCm-7.0 Triton build**, not upstream PyPI Triton.
- `setup.py` will, by default, **uninstall** any existing triton/pytorch-triton-rocm and reinstall
  the aiter-compatible one, unless `AITER_USE_SYSTEM_TRITON=1`. It also warns it wants **torch >=
  2.9.1** (below that, the triton reinstall is skipped for compat).
- Gluon (`@gluon` kernels) and fp4/mxfp8 paths are the features driving the new-Triton dependency;
  they are gated to gfx950/gfx1250 (not us), but the **3.6.0 floor + `triton-kernels` package +
  ROCm-7.0 wheel** apply to the whole package install regardless of arch.
- No explicit `triton==` pin in `requirements.txt`/`pyproject.toml` (Triton is installed by the
  script, not declared as a normal dep). The version constraint lives in `install_triton.sh`.

Implication for our AOT vendoring: since we AOT-compile the vendored kernels into the inference
binary rather than `pip install aiter`, we don't have to run `install_triton.sh` — **but the kernel
source assumes Triton-3.6.0-class language features** (the reorganized tree, gluon imports guarded
by arch, newer `triton.language` usage). If our build currently compiles against an older Triton,
re-vendoring the latest tree likely forces a Triton bump to >= 3.6.0 (ROCm 7.0 build). If we stay on
the older vendored `9c79a5b59` substrate we may avoid the bump — verify our current Triton version
against 3.6.0 before deciding. **UNKNOWN here: the exact Triton version our fork currently builds
against** (not in this repo's scope to read).

---

## 5. Beyond main — in-flight RDNA4 / gfx1201 / fp8 / backward work (released vs unreleased)

From GitHub PR pages (ROCm/aiter). "Merged" = in main/released line; "Open"/"Draft" = unreleased.

Merged (available, in or near our vendoring window):
- **#3484** "configs: tune gfx1201 Qwen3-8B-FP8 blockscale GEMMs" — merged Jun 2 2026. (Source of
  several of our gfx1201 tuned JSONs.)
- **#3611** "Support padded K for a8w8 bpreshuffle GEMM" — merged Jun 14 2026. Relevant to the
  preshuffle path / padded-K shapes.
- **#3696** "de-torch module_fused_qk_norm_mrope_cache_quant_shuffle" — merged Jun 13 2026
  (fused qk-norm + rope + quant + shuffle; RoPE-side, not a rotation-for-quant kernel).
- Known prior: #3332 (gfx1200/1201 in FP8 dtype map), #3568 (fused a8w8 blockscale tune) — predate
  our vendored SHA, should already be in.

Open / unreleased (do NOT assume present):
- **#3343** "gfx1201 gemm_a8w8: blockscale HIP→triton fallback + tuning configs" (open) — adds
  more gfx1201 tuning + the HIP→Triton fallback wiring. **Most relevant open PR for us.**
- **#2350** "[gfx1201] Added tuned gemm_a8w8 configs for gfx1201" (open).
- **#1829** "[TRITON] Support gfx1201 for triton gemm_a8w8_blockscale" (open, since Jan).
- **#3662** "add tuned files for minimax-m2.5 PTPC fp8 model" (open), **#3636** DSv4 a8w8_blockscale
  for gfx950 (open, not our arch), **#3629** fp8 blockwise batched_gemm dsv4 (open).
- Closed/superseded: #3234, #2242 (early gfx1201 tuning / RDNA4 FP8 attention).

**No open or merged PR adds an fp8 dgrad/wgrad backward GEMM.** The in-flight gfx1201 work is all
forward-inference tuning + fallback plumbing.

Context issues worth knowing (not aiter PRs): ROCm/TransformerEngine #359 / #520 (enable / fix
gfx1201 FP8), vLLM #28649 ("upstream this gfx1201/RDNA4 FP8 patch") — all about gfx1201 *inference*
FP8 enablement and the historical "gfx1201 missing from arch table → silent FP32 fallback" bug.
In the current clone `arch_info.is_fp8_avail()` **does** include `gfx1200`/`gfx1201`, so that bug is
fixed in our line.

---

## Bottom line — what to re-vendor for RDNA4 now

1. **Tuned configs are our biggest gap, and it's data-only.** Only (8192,8192) of our 16 shapes is
   tuned; everything else hits the generic `"any"` config at large M. Generate
   `gfx1201-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json` for our exact (N,K) set
   ({2560,4096,8192,9216}²) and vendor them. **No kernel change, no Triton bump.** Watch open
   **#3343 / #2350 / #1829** — if/when merged they add more gfx1201 tuning we can pull instead of
   self-tuning.
2. **Forward a8w8 blockscale kernel itself is fine to re-vendor** from latest (incl. merged #3611
   padded-K preshuffle if we use the preshuffle path). Use the standard
   `gemm/basic/gemm_a8w8_blockscale.py` — **avoid the `gluon/` variant** (gfx950/gfx1250 only).
3. **Reuse the fused fp8-e4m3 quant prologues** in `quant/fused_fp8_quant.py` for our quant-prologue
   needs. **Author ourselves:** the fused *rotation*(Hadamard)+quant prologue and any LUT-weight fp8
   kernel — neither exists in aiter. Graft rotation onto the existing `_fused_rms_*_fp8_group_quant`
   epilogue rather than greenfield.
4. **Backward GEMMs for QAT must be authored by us (or pulled from TransformerEngine).** aiter has
   no fp8 dgrad/wgrad, on main or in any PR. Reuse the forward kernel body with transposed operands.
5. **Triton bump risk:** the latest aiter tree requires **Triton >= 3.6.0 (AMD ROCm-7.0 build) +
   `triton-kernels` + torch >= 2.9.1**. Because we AOT-compile rather than pip-install, we skip the
   installer, but re-vendoring the *latest* kernel source likely forces a Triton bump to 3.6.0-class.
   **Verify our fork's current Triton version against 3.6.0 before re-vendoring latest.** If we stay
   on the `9c79a5b59`-era substrate we may dodge the bump.

### Stated unknowns
- Exact Triton version our llama.cpp fork currently builds the vendored kernels against (out of
  scope of the aiter repo; must be checked in the fork's build).
- Whether the vendored `9c79a5b59` snapshot already has the reorganized tree or the older flat
  layout (the local working clone is newer and reorganized; the vendored snapshot may differ).
- Remote branch list (`git branch -r`) could not be enumerated in this sandbox (git/gh/MCP denied);
  PR status above came from GitHub web pages and may lag real-time merge state by minutes/hours.
