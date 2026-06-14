# Paged turbo4 @ head_dim 64 (LFM2.5) — handoff (2026-06-14)

## Goal
Run LiquidAI **LFM2.5-8B-A1B** (arch `lfm2moe`, hybrid: conv layers n_head_kv=0 +
6 attention layers n_head_kv=8, **head_dim 64**, 32 q-heads / 8 kv-heads GQA) on
mad-lab-2026 RX480 (gfx803, `rx480-army` container) with the FULL service config:
paged KV tiering + `--cache-type-k/v turbo4` at 512K ctx.

## DONE + on master (origin/master, pushed) — all verified correct
- `100bce65d` kv-cache: turbo K-alloc pad 64->128 (was 64; cpy_k/get_k/V-alloc use 128).
- `6653f6418` context: keep n_outputs==n_tokens after n_seqs rounding (mean/rank pooling) — fixed the Mneme embedder (llama-embed) crash.
- `1f671b8e0` graph: turbo V-unpad head count from cur->ne[0]/padded_v_head (GQA), not n_head_kv.
- `8d078eace` kv-paged: size paged K/V from first attention layer (hybrid; layer 0 is conv n_head_kv=0).
Result: **non-paged turbo4 LFM2.5 = CORRECT** (coherent, "Paris"). **paged+q8_0 = loads**.

## THIS BRANCH (wip/paged-turbo-headdim64) — scaffold, NOT correct yet
Graph-level 64->128 zero-padding for the paged turbo path:
- `src/llama-kv-cache-paged.cpp` ctor: pad head_dim->128 for turbo cache sizing.
- `src/llama-graph.cpp` build_attn is_paged branch: zero-pad q/k/v ne[0] to 128 BEFORE F16 cast (ggml_pad needs F32 -> pad F32 then cast), run HS=128 kernel, ggml_view_3d slice first 64 off the [128,n_head,n_tokens] output before the wo reshape.
- Gated on head_dim%128!=0 -> head_dim>=128 models (omnicoder HS=256) unaffected.

STATUS: **loads + runs (no crash) but OUTPUT IS DEGRADED** — paged+turbo4 gives
"the capital of the French... of the of the" vs non-paged turbo4 correct "Paris...".
So graph padding is necessary but NOT sufficient: a numerical bug in the paged
turbo scatter/decode/cache-layout for the padded head.

## Why it *should* work (so the bug is implementation, not math)
turbo4 dequant applies the inverse WHT, returning K to original space; padded
upper-64 are zero. <Q_pad, dequant(K_pad)> = <Q[:64],K[:64]>. V dequant -> padded
V, weighted-sum -> first 64 real + last 64 zero, sliced off. No Q pre-WHT needed
(dequant handles it; the working is_paged path passes raw Q for omnicoder HS=256).

## NEXT (isolated kernel harness — do NOT iterate via server)
Harness lives in **~/GitHub/llama-gpu/tests/test_turbo4_*.cu** (NOT this checkout).
Build: nvcc -O2 -arch=sm_61 -x cu <f> / hipcc -O2 --offload-arch=gfx803 -x hip <f>,
-I ggml/include -I ggml/src, link build-rocm-gfx803/bin/libggml-cuda.so.
1. FIRST: does HS=128 paged-turbo even work natively? All tested models were HS=256
   (omnicoder). Run test_turbo4_decode_full.cu / test_turbo4_scatter.cu with
   -DHEAD_SIZE=128 (FULL nonzero data). If HS=128 already fails -> bug is the
   HS=128 path itself, independent of padding.
2. THEN: HS=128 with K/V where only dims [0:64] are nonzero (simulates padded-64).
   Compare scatter round-trip + full decode vs CPU ref. Bisect scatter vs decode.
3. Kernel files: mt_pagedattn_decode.cu (decode_coop_stage_turbo4 @200, dequant
   variant @261, launch_paged_attn_decode @1119; static_assert HS%128==0 @213,
   Q_BLOCK==128 @214; cache index `ib` @232/296 uses QBLOCKS_PER_TOKEN=HS/128 &
   n_kv_heads), mt_pagedattn.cu (dispatch ~1080, HS from q->ne[0] @1057, HS
   instantiations 64/128/256 @1307-1316), scatter mt_scatter_kv_turbo4_0_kernel.

## Repro (degraded output)
docker exec -e HIP_VISIBLE_DEVICES=0 -i rx480-army /workspace/llama.cpp/build-rocm-gfx803/bin/llama-server \
  --model /home/kmbandy/models/LFM2.5-8B-A1B-Q5_K_M.gguf --n-gpu-layers 999 --ctx-size 32768 --parallel 1 \
  --cache-type-k turbo4 --cache-type-v turbo4 --flash-attn on --kv-tiered 25,25,50 \
  --kv-tier-ssd-path /home/kmbandy/llama/kv-cold/lfm25-test --kv-tier-paged-blocks \
  --kv-tier-semantic-index /home/kmbandy/models/granite-embedding-small-english-r2.Q8_0.gguf \
  --jinja --no-mmap --no-warmup --host 127.0.0.1 --port 8099
# then: curl :8099/completion -d {"prompt":"The capital of France is","n_predict":18,"temperature":0}

## Builds (both off master): build-army (CUDA/1070 sm_61), build-rocm-gfx803 (HIP/gfx803, in rx480-army; /home/kmbandy/GitHub/llama.cpp bind-mounted to /workspace/llama.cpp). UI offline-build fix: placeholder assets in tools/ui/dist/.
## Related: MAD-288 (paged-turbo4 CUDA-only token corruption @ HS>=128 — DIFFERENT bug, this is sub-128 head support).
