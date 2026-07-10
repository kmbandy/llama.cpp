# WP hetero device split + DFlash draft-oracle plan

**Date:** 2026-07-08  
**Branch:** `feat/wp-dflash-ds4`  
**Hardware:** mad-lab-main — R9700 (gfx1201) + 6900 XT eGPU (gfx1030, TB3)  
**Status:** design locked for implementation; draft-oracle v1 measured; device split not yet shipped on this branch

Related:

- [2026-07-08-ds4flash-decode-levers.md](./2026-07-08-ds4flash-decode-levers.md) — broader decode lever catalog  
- [weight-paging-batch-eval-continuation.md](./weight-paging-batch-eval-continuation.md) — WP_RESIDENT_DENSE history  
- `~/wp_logs/codex-draft-oracle-instrument-report.md` — oracle counter reference  

---

## 1. Goal

Maximize single-stream DS4 Flash decode under weight paging by:

1. **Never moving routed expert weights over TB3** — experts page NVMe → R9700 VRAM only.  
2. **Filling the 6900 XT** with always-resident dense + DFlash draft so R9700 VRAM is almost entirely expert pool.  
3. **Using DFlash as a paging oracle** (tid2eid hash experts) more than as multi-token free accepts (under WP, multi-row target thrash is worse than accept gains).  
4. Keeping hot-tier KV small (**~1–2 GB**): MLA + 4-bit turboquant + paged/tiered KV already cover the rest.

---

## 2. Hardware (constraints that shape placement)

| Device | Link | Role |
|--------|------|------|
| **R9700** | PCIe 4 x16 (~28 GB/s), SAM | Expert page pool (and optional always-resident shexp) |
| **6900 XT** | TB3 eGPU (~2.7 GB/s), **no inter-GPU P2P** | Dense attention island + draft + hot KV |
| **Host RAM** | — | `token_embd`; optional HostTier for demoted experts later |

**Do not** put expert weight traffic on TB3. Residual / activation crossings per layer are acceptable; **weight** crossings for 147 GB of experts are not.

**Flash Attention:** attention weights + KV + FA node must co-locate on the **resident** device (6900 XT under this plan). Routed experts override to R9700. Existing WP router C1 (`dev_layer` → resident) is the intended mechanism.

---

## 3. Model sizes (measured from GGUF)

### Target: DeepSeek V4 Flash UD-Q8_K_XL (~162 GB)

| Bucket | ~GB | Placement under this plan |
|--------|-----|---------------------------|
| Routed `ffn_{up,gate,down}_exps` | **147.2** | **Paged on R9700** |
| Attention / MLA core | **9.3** | **6900 XT resident** |
| Indexer + compressors | **1.0** | **6900 XT resident** |
| Shared expert `shexp` | **2.2** | **R9700 always-resident** (default) or paged on R9700 |
| `token_embd` | **1.1** | **CPU / host** |
| LM head / output (+ HC bits) | **~1.1** | **6900 XT resident** |
| Routers / norms / tid2eid | **~0.2** | **6900 XT resident** |

### Draft: DFlash speculator

| File | ~GB | Notes |
|------|-----|--------|
| `dflash-speculator-DS4.gguf` | 3.6 | bf16 source |
| **`dflash-speculator-DS4-Q8_0.gguf`** | **1.9** | **preferred** — Q8_0, ~53% of bf16 |

Path: `/home/kmbandy/models/dflash-speculator-DS4-Q8_0.gguf`

---

## 4. Target layout (locked)

```
6900 XT (TB3, 16 GB)
  attention / MLA / indexer / compress   ~10.2 GB
  lm_head / routers / norms / tid2eid    ~1.3 GB
  draft Q8_0                             ~1.9 GB
  hot-tier KV (MLA + TQ + paged tier)    ~1-2 GB
  ----------------------------------------
  total                                  ~14-15.5 GB  (fill eGPU; not a waste)

R9700 (x16, 32 GB)
  expert page pool                       most of card (grow past 8.5 GB slots)
  optional: always-resident shexp        +2.2 GB if not paged
  ----------------------------------------
  NO dense attention weights
  NO draft model (unless forced for debug)

Host RAM
  token_embd                             ~1.1 GB always
  HostTier demoted experts (later)      ~0-4 GB selective; NOT used for embd/shexp
```

### Placement rules

| Tensor | Device | Rationale |
|--------|--------|-----------|
| Routed experts | R9700 pool (NVMe) | Fast link; never TB3 |
| Attention + FA + KV hot | 6900 XT | Intra-device FA; fills eGPU |
| Draft Q8 | 6900 XT | Same card as residual/attention island |
| `token_embd` | **CPU** | Row gather only; free GPU VRAM |
| `shexp` | **R9700 resident** (default) | Every-layer use; not CPU; not TB3 |
| HostTier experts | Host RAM later | Slow drain for demoted *routed* pages only |

**Do not** put `shexp` on CPU (every-layer H2D or CPU FFN).  
**Do not** repurpose HostTier budget as embd+shexp storage.

### shexp alternatives (if default is wrong later)

| Choice | Pros | Cons |
|--------|------|------|
| R9700 always-resident (default) | No shexp page thrash; simple | −2.2 GB from expert pool |
| Paged on R9700 | Full 32 GB for routed pool | shexp page_ins every layer |
| On 6900 XT dense | Saves R9700 | Needs embd+something else off eGPU to fit; denser TB3 card |

---

## 5. DFlash under weight paging (current behavior)

### Target batch policy

Under WP, **strip drafts from the target batch** (`WP_SPEC_VERIFY_MAX` auto → 0 drafts):

- Target always **single-token** MoE (top-k=6, pi/route ~5.5).  
- Multi-row verify (`[sampled, draft, ...]`) unions experts → thrash (measured 15k vs 12.7k page_ins).  
- Accept rate under strip = 0 by design; draft is a **cache controller**, not free tokens.

Override: `WP_SPEC_VERIFY_MAX=N` to re-enable multi-row (expect thrash).

### Draft-as-paging-oracle

1. DFlash produces draft token ids.  
2. `llama_wp_on_draft_tokens` maps ids through host **tid2eid** (hash layers 0–2).  
3. Cold pages submitted in **waves** (default 1 wave ≈ QD=4); Done orphans reaped so free_q is not stuck at 0.  
4. Pin tid2eid pages across draft→target; clear after sample.  
5. Softmax MMID history is **metrics-only** (do not over-pin).

### Defaults that worked (measured)

| Env / setting | Default | Note |
|---------------|---------|------|
| `WP_DRAFT_PREFETCH` | on | 0 disables oracle |
| `WP_DRAFT_PREFETCH_WAVES` | **1** | multi-wave thrash when draft ≠ sample |
| `WP_DRAFT_ORACLE_MAX_TOK` | **1** | first draft token only under strip |
| `WP_DRAFT_ADAPTIVE` | on | hit-ratio / warm skip |
| `WP_DRAFT_ALWAYS_FIRST` | 4 | cold start |
| `WP_DRAFT_STATS` | off | set 1 for per-fire lines |

### Measured (single-card dense-resident, before hetero split)

| Config | n | t/s | page_ins | notes |
|--------|---|-----|----------|--------|
| nodraft | 16 | ~1.11 | 12735 | baseline shape |
| dflash oracle | 16 | ~1.22 | ~12781 | ~10% up |
| nodraft | 64 | ~1.11 | 28852 | |
| dflash oracle | 64 | ~1.22 | ~29046 | sub>0, blocked=0 after harvest fix |

Cold-submit bug (historical): scheduler **Done** slots unreaped after pool eviction → free_q stuck at 0. Fixed via harvest + `reap_finished()`.

---

## 6. Implementation checklist — hetero split (next to code)

Order matters; validate after each step.

### 6.1 Placement predicates

- [ ] Keep `WP_RESIDENT_DENSE=1`: page only routed `ffn_*_exps` by default.  
- [ ] **Page or rehome `shexp`:** default **always-resident on paging device (R9700)**, not in 6900 XT dense blob.  
- [ ] **`token_embd` on CPU** (or explicit non-resident override) — not on 6900 XT.  
- [ ] Confirm LM head stays on resident (6900 XT) unless measured otherwise.

### 6.2 Device router

- [ ] Paging device = R9700 (`--device` / main GPU for pool).  
- [ ] Resident device = 6900 XT (`--weight-paging-resident-device ROCm1` or correct name).  
- [ ] Draft model load on same device as resident / ROCm1.  
- [ ] Log: `WP_RESIDENT_DENSE router: paging=... resident=...`  
- [ ] Log: no `Flash Attention ... DISABLED` / cross-device FA.  
- [ ] Layer home + KV on resident (existing C1 path).

### 6.3 Expert pool on R9700

- [ ] Grow `--weight-paging-slots` / budget once dense leaves R9700 (8.5 GB was a single-card compromise).  
- [ ] If shexp resident on R9700, subtract ~2.2 GB from pool budget.  
- [ ] Confirm P2P dma_buf still on R9700 only.

### 6.4 Draft model

- [ ] Default draft path: **`dflash-speculator-DS4-Q8_0.gguf`**.  
- [ ] A/B vs bf16 only if quality/accept/oracle hits regress.

### 6.5 Validation gates

| Gate | Pass |
|------|------|
| Load | No OOM on either GPU |
| FA | Stays enabled; no full-matrix alloc on eGPU |
| Placement | Expert pages only on R9700; no expert file I/O device mismatch |
| page_ins shape | ~nodraft single-token (pi/route ~5.5) under strip |
| Decode | t/s ≥ single-card dense-resident baseline (~1.1–1.2) on same prompt; aim beat |
| Coherence | Same smoke prompt as WP validation |
| VRAM | 6900 XT ~13–15.5 GB weights + 1–2 GB hot KV; R9700 dominated by pool |

---

## 7. Next levers (after device split — do not lose track)

Full catalog (including TB3 residual transport T0-T5): [2026-07-08-ds4flash-decode-levers.md](./2026-07-08-ds4flash-decode-levers.md).

Priority after hetero layout is stable:

### P0 — Prefill / cold-start oracle — DONE

Fire tid2eid (last prompt token and/or one draft) **before** first decode so early tokens are not pure cold NVMe.

### P0b — Grow R9700 expert pool — DONE (2026-07-09)

Measured hetero nodraft n=32: **2000→6500 slots** = warm **1.02→1.53 t/s**, page_ins **34k→23k**.  
**Sweet spot: 6500** (~27.6 GiB pool). **7000 OOMs** (compute buffer). See levers doc table.

### P0c — Hetero nodraft A/B

Separate TB3 + eGPU-attn tax from draft GPU cost on ROCm1.

### P1 — Cut draft GPU cost under strip

Only first draft token is used for oracle -> consider `spec-draft-n-max=1` under WP, or skip DFlash when sticky set is hot.

### P2 — Sticky L2 expert set

Small pin set (e.g. 32-64 pages) of high `draft_tid2eid_hits_in_ensure` experts across adaptive skip steps.

### P3 — Draft-only QD bump

Optional higher queue depth for draft waves only (global QD>4 historically hung without demux — keep isolated).

### P4 — Conditional multi-row verify

Re-enable `WP_SPEC_VERIFY_MAX=1` only when hit_ratio high and free pool healthy; else strip. Goal: free tokens without permanent thrash.

### P5 — HostTier expert slow-drain (~4 GB)

Selective host cache for demoted **routed** experts (not embd/shexp). Lower priority until pool locality / oracle improve; earlier measurements showed weak cross-token locality.

### P6 — Softmax prior

Still no strong prior without multi-row thrash or learned router cache. Revisit only after hash oracle is maxed.

### P7 — TB3 residual transport (after pool + T0 measure)

Residuals are KB-scale; ~+380 ms/tok hetero tax is **not** pure TB3 bandwidth. See levers doc section **C. TB3 residual transport**: instrument (T0), sticky bounce buffer (T1), async stage (T2), activation-only pipeline not full PP (T3), fewer FFN boundaries (T4). Do not re-enable full pipeline_parallel without a residual-only plan (94 GiB path).

### Explicit non-goals (for now)

- Sequential single-token "speculation" as a speed project (same AR work + draft tax).  
- Full multi-wave cold submit of all draft tokens (measured thrash).  
- Expert weights on TB3.  
- Chunking residual into smaller pieces (already tiny; overhead rises).

---

## 8. Env / CLI cheat sheet (intended end state)

```bash
export WP_RESIDENT_DENSE=1
export WP_SIZE_CLASS_SLOTS=1
export WP_PAGED_BATCH=1
export WP_DENSE_PREFETCH_N=8
export LLAMA_WP_TRANSPORT=p2p
export LLAMA_WP_TRANSPORT_FORCE=1
export WP_DRAFT_PREFETCH=1
# WP_DRAFT_PREFETCH_WAVES=1
# WP_DRAFT_ORACLE_MAX_TOK=1
# WP_DRAFT_STATS=1   # debug

./build-hip/bin/llama-server \
  -m .../DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf \
  --model-draft /home/kmbandy/models/dflash-speculator-DS4-Q8_0.gguf \
  --spec-type draft-dflash --spec-draft-n-max 4 \
  --no-mmap --weight-paging --weight-paging-prefetch \
  --weight-paging-slots <larger once R9700 is pool-only> \
  --weight-paging-resident-device ROCm1 \
  --device ROCm0,ROCm1 \
  -ngl 99 -c 2048 --parallel 1 \
  --host 127.0.0.1 --port 8080
```

Exact ROCm names / main_gpu indexing: confirm with `rocminfo` / load log (`paging=` / `resident=`).

---

## 9. Decision log (this session)

| Decision | Choice |
|----------|--------|
| Target under WP | Single-token; strip multi-row drafts |
| Draft role | Paging oracle (tid2eid) first; accept second |
| Draft quant | **Q8_0** at `...-Q8_0.gguf` |
| eGPU fill | Full attention dense + draft + hot KV (~13–15 GB) |
| embd | **CPU** |
| shexp | **R9700 resident** (default), not CPU, not HostTier |
| HostTier | Expert demote only; ~4 GB later; not embd/shexp |
| Next implement | This hetero device split (section 6) |

---

## 10. Open questions at implement time

1. Confirm ROCm index: which of ROCm0/ROCm1 is R9700 vs 6900 XT on boot.  
2. shexp: resident on R9700 vs paged (A/B after split loads).  
3. How large can expert pool grow on R9700 with dense gone (slots / size-class budget).  
4. Draft device binding: does `--model-draft` follow resident device or need explicit flag.

---

## 11. Implementation progress (2026-07-08)

### Router (landed in tree)

`wp::build_router_overrides(paging, resident, cpu, user)` order:

1. `ffn_*_exps` → paging (paged pool)  
2. `ffn_*_shexp` → paging (always-resident on R9700)  
3. `token_embd` → CPU  
4. user overrides  
5. `.*` → resident (attention + lm_head + … on 6900 XT)

Caller: `llama-model.cpp` under `WP_RESIDENT_DENSE` + dual-device (or multi-device) with paging/resident bufts.

### Still TODO for full hetero bring-up

- [x] GPU validation load with `--weight-paging-resident-device ROCm1` + dual `--device`
- [x] Draft Q8 on resident via `--spec-draft-device ROCm1`  
- [x] Prefill oracle (P0) landed in server-context  
- [x] FA stays enabled; no FA device-mismatch under load  
- [x] 94 GiB PP reserve fixed (`has_tensor_overrides` + force PP off under WP)  
- [x] Decode t/s smoke vs single-card (`~/wp_logs/het-hdf16/64`, `het-scnd16`)  
- [ ] Grow expert pool slots on R9700 after dense leaves  
- [ ] Hetero nodraft A/B; then TB3 residual T0 instrument if still behind  

### Decode smoke (post-fix rebuild, 2000 slots)

| Config | n | t/s | page_ins |
|--------|---|-----|----------|
| hetero dflash Q8 | 16 | 0.78 | 12783 |
| hetero dflash Q8 | 64 | 0.81 | 29247 |
| single-card nodraft | 16 | 1.11 | 12738 |

Placement: ROCm0 paged exps + shexp; ROCm1 dense + draft; host embd. No 94 GiB / no PP retry.

### Earlier smoke notes (`~/wp_logs/hetero-smoke.log`, pre-fix binary)

Placement observed:

| Ctx | buft | Contents |
|-----|------|----------|
| 0 | ROCm0 | paged `*_exps` (data nil) + **shexp resident** |
| 1 | ROCm1 | dense attention / output (~1069 tensors) |
| 2 | ROCm_Host | **token_embd** |
| draft | ROCm1 | Q8 draft tensors |

`health_ok=1` after PP retry. Pre-fix binary hit ~94 GiB ROCm0 reserve then retried without PP — fixed in tree.

Example CLI:

```bash
WP_RESIDENT_DENSE=1 ... \
./build-hip/bin/llama-server \
  -m ...DeepSeek-V4-Flash...gguf \
  --model-draft .../dflash-speculator-DS4-Q8_0.gguf \
  --spec-type draft-dflash --spec-draft-n-max 4 \
  --spec-draft-device ROCm1 \
  --weight-paging --weight-paging-prefetch --weight-paging-slots 2000 \
  --weight-paging-resident-device ROCm1 \
  --device ROCm0,ROCm1 -ngl 99
```
