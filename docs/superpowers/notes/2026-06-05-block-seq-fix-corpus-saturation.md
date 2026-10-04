# 2026-06-05 — Block-seq KV-cache fix + corpus-size sweep + domain-gap analysis (MAD-264)

Branch `calib-pipeline-speedup`. Model: Qwen3.5-0.8B dense-hybrid, mix corpus, pd0.01,
`--rotation kronecker --group-size 64 --n-centroids 16 --fit-loss mse`, full-coverage,
`ML8_DETERMINISTIC=1`, full `wiki.test` + held-out, all on the 9700 (gfx1201).

## 1. Block-seq KV-cache bug — fixed (commit `d5186596f`)

The block-sequential walk threaded a per-sample HF `DynamicCache` (`use_cache=True`)
through **every** block re-forward (collect-H per sub-group + the propagate pass).
`DynamicCache.update()` **concatenates** K/V, so re-forwarding a block appended state each
time. Two consequences:

- **Memory leak**: per-sample recurrent/KV states accumulated ~N_samples *per block* →
  peak VRAM ∝ tokens·blocks → **OOM above 256k even on the 32 GB R9700** (~30 GB at 512k).
- **Latent correctness bug on full-attention blocks**: the 2nd/3rd re-forward attended over
  **doubled/tripled KV** → `o_proj` and `mlp.gate/up` mis-calibrated. `q/k/v` (1st forward)
  were correct; SSM/delta-net blocks immune (recompute full seq from zero state, only *write*
  the final state). Smoking gun: blob-diff showed `q/k/v` bit-identical, `o_proj`+`gate/up`
  differing by ~2% scale — exactly the "later forwards polluted" signature.

**Fix**: capture block inputs with `use_cache=False` + `past_key_values=None` → stateless
per-block forwards. SSM blobs **bit-identical**; attention now **correct** (matches inference,
where each context window is a clean causal forward). Peak VRAM now flat (~6 GB @512k).
Added env-gated diagnostics: `BLOCKSEQ_MEMLOG=1` (per-block VRAM), `=2` (live-tensor census),
`BLOCKSEQ_KEEP_CACHE=1` (A/B toggle for the exactness gate). Dropped `--allow-partial` from
sweep scripts (it had silently shipped a mostly-bf16 1225 MB "result" when an OOM truncated calib).

**Implication**: the pre-fix "near-gold" block-seq 256k (wiki 19.6608) was slightly
mis-calibrated on the few attention blocks. The **fixed** numbers below are canonical.

## 2. Corpus-size curve (fixed block-seq)

| budget | wiki | held-out (quant_so) | size | calib |
|---|---|---|---|---|
| 256k | 19.6869 | 12.2839 | 498 MB | 15 m |
| 512k | 19.7129 | 12.2862 | 498 MB | 24 m |
| 1M | 19.4320 | 12.2508 | 498 MB | 41 m |
| 1.5M | 19.2516 | 12.2914 | 498 MB | 59 m |

bf16 baselines (same `-c512` full-file): **wiki 18.6071, quant-heldout 11.3487.**

## 3. The finding: token-budget is a DOMAIN lever, not a QUALITY lever

Gap-to-bf16:

| budget | wiki gap | quant-heldout gap |
|---|---|---|
| 256k | +1.08 | +0.94 |
| 512k | +1.11 | +0.94 |
| 1M | +0.82 | +0.90 |
| 1.5M | **+0.64** | +0.94 |

The **wiki** gap closes steadily with tokens (marching toward lossless); the **held-out**
gap is **dead flat at ~+0.90** across every budget. This refutes both naive reads:

- NOT a held-out ceiling — there is +0.90 of *real recoverable* headroom on held-out,
  as much as wiki ever had.
- NOT harmful overfit — held-out never worsens.

Mechanism: the mix is **28% wikipedia** (the wiki eval domain) → more tokens preserve
wiki-like activations → wiki improves. The held-out domain (`quant_so`) is only **2%** of
the mix → more tokens of other domains don't touch it.

**Strategic reframe**: stop chasing token budget for *generalization*. The +0.90 held-out
headroom is the real prize and needs **structure-matched** levers (#169: `mag_weighted` for
Qwen's per-token outliers, finer `group_size`, codebook fine-tune) on the now-fast+correct
Hessian — not more tokens.

## 4. In flight (2026-06-05) — `quant_so` is niche; re-eval on math_se + swe

`quant_so` (quant-finance SE, 2% of mix) is a narrow generalization probe. Re-evaluating the
4-budget curve on **math_se** and **swe** held-outs (carved leakage-free from mid-file windows;
expected calib overlap <0.4 docs) via cheap **re-convert of the saved calib blobs** (no
recalibration). `swe` directly tests the "model is near-ceiling on its coding strength"
hypothesis (small swe gap → confirmed). Results → `~/models/phase2/sizesweep/results_heldout_domains.tsv`.

## 5. Operational gotchas banked this session

- **GPU map** (reversed across tools): 9700 = torch `cuda:0` = `ROCm0` = rocm-smi device **1**;
  6900xt = `cuda:1` = `ROCm1` = rocm-smi device **0**. Verify placement via
  `torch.cuda.mem_get_info` (34 GB free = 9700), never the rocm-smi label.
- **ml8 PPL runs ONLY on the 9700** (gfx1201 FP8-WMMA). On the 6900xt/gfx1030 it falls into a
  CPU path and **hangs** (cost 112 min before caught). Calibrate on either card; PPL on the 9700.

## Next

1. Read the math/swe domain matrix (in flight) → settles quant-niche vs general flatness.
2. **#169 bit-free lever battery** on the fast+correct Hessian — the real attack on the
   +0.90 held-out gap (mag_weighted, group_size, percdamp re-test, codebook-FT).
3. Seed sweep to firm up single-draw ~1σ (the curve is non-monotonic — 512k flat, 1M dropped).
4. Then close MAD-264 + finish the branch.
