# Act-Replay fp8-Native KL Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train ml8-4 codebooks+scales (indices frozen) of cell_A0_anchor_A3.gguf against the bf16 teacher with KL loss through an fp8-faithful forward, re-emit a deployable GGUF.

**Architecture:** Rehydrate trainer state from the GGUF (no blob dirs survive); student = HF model with dequant-STE weight overrides + FaithfulActHook acts; `TeacherSource` interface with live/cache/device:N backends all yielding `(topk_idx, topk_logits, tail_logsumexp)`.

**Tech Stack:** PyTorch+ROCm, gguf-py, existing scripts/calibration modules (`ml8_io`, `ml8_e4m3_sim`, `codebook_finetune_rig.dequant`, `faithful_forward`, `calib_corpus`, `ml8_to_gguf`).

**Spec:** `docs/superpowers/specs/2026-06-10-act-replay-kl-trainer-design.md`
**All work in:** `scripts/calibration/` on branch `sync/upstream-2026-06-09`. Run tests from `scripts/calibration/` with `PYTHONPATH=../../gguf-py`. CPU tasks 1–6 need no GPU.

---

### Task 1: Reverse block unpackers (`gguf_state.py` part 1)

The exact inverses of `ml8_to_gguf.pack_ml8_blocks` / `pack_scaled_fp8_blocks` / `cast_centroids_to_fp8`. Roundtrip property tests against the existing packers.

**Files:** Create `scripts/calibration/gguf_state.py`, `scripts/calibration/test_gguf_state.py`

- [ ] **Step 1: failing roundtrip tests**

```python
# test_gguf_state.py
import numpy as np, torch
from ml8_to_gguf import pack_ml8_blocks, pack_scaled_fp8_blocks, cast_centroids_to_fp8
from gguf_state import unpack_ml8_blocks, unpack_scaled_fp8_blocks, decode_centroids_fp8

def test_ml8_roundtrip():
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 16, (8, 128), generator=g, dtype=torch.int8)
    scl = torch.rand(8, 2, generator=g) + 0.1
    packed = pack_ml8_blocks(idx, scl)
    idx2, scl2 = unpack_ml8_blocks(packed, N=8, K=128)
    assert torch.equal(idx2, idx.to(torch.long))
    assert torch.equal(scl2, scl)

def test_fp8_roundtrip():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(4, 64, generator=g)
    e4m3 = w.to(torch.float8_e4m3fn).to(torch.float32)
    scale = (torch.rand(4, 2, generator=g) + 0.5).to(torch.float16)
    packed = pack_scaled_fp8_blocks(e4m3, scale)
    e2, s2 = unpack_scaled_fp8_blocks(packed, N=4, K=64)
    assert torch.equal(e2, e4m3) and torch.equal(s2, scale)

def test_centroid_roundtrip():
    g = torch.Generator().manual_seed(2)
    c = torch.randn(2, 16, generator=g)
    on_lattice = c.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(decode_centroids_fp8(cast_centroids_to_fp8(c)), on_lattice)
```

- [ ] **Step 2:** `PYTHONPATH=../../gguf-py python3 -m pytest test_gguf_state.py -q` → FAIL (no module gguf_state)
- [ ] **Step 3: implement**

```python
# gguf_state.py
"""Rehydrate act-replay trainer state from an ml8 GGUF (exact inverse of ml8_to_gguf packing)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
from ml8_to_gguf import QK_ML8, ML8_BLOCK_BYTES, N_CENTROIDS, _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES

def unpack_ml8_blocks(packed: np.ndarray, N: int, K: int):
    """packed [N, n_g*36] uint8 -> (indices long [N,K], scales fp32 [N,K//64])."""
    n_g = K // QK_ML8
    blocks = np.ascontiguousarray(packed).reshape(N, n_g, ML8_BLOCK_BYTES)
    scales = blocks[:, :, :4].copy().view('<f4').reshape(N, n_g)
    qs = blocks[:, :, 4:]                       # [N, n_g, 32]
    idx = np.empty((N, n_g, QK_ML8), dtype=np.uint8)
    idx[:, :, 0::2] = qs & 0x0F
    idx[:, :, 1::2] = qs >> 4
    return (torch.from_numpy(idx.reshape(N, K).astype(np.int64)),
            torch.from_numpy(scales.astype(np.float32)))

def unpack_scaled_fp8_blocks(packed: np.ndarray, N: int, K: int):
    """packed [N, n_b*34] uint8 -> (e4m3 fp32 [N,K], scale fp16 [N,K//32])."""
    n_b = K // _FP8_GROUP_SIZE
    blocks = np.ascontiguousarray(packed).reshape(N, n_b, _FP8_BLOCK_BYTES)
    scale = torch.from_numpy(blocks[:, :, :2].copy()).view(torch.float16).reshape(N, n_b)
    qs = torch.from_numpy(blocks[:, :, 2:].copy()).view(torch.float8_e4m3fn)
    return qs.to(torch.float32).reshape(N, K), scale

def decode_centroids_fp8(cent_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(cent_u8)).view(torch.float8_e4m3fn).to(torch.float32)
```

- [ ] **Step 4:** rerun → 3 PASS
- [ ] **Step 5:** `git add gguf_state.py test_gguf_state.py && git commit -m "feat(act-replay): ml8/fp8 block unpackers (pack roundtrip)"`

---

### Task 2: GGUF rehydrator + bit-equality gate

Read a real ml8 GGUF into `{gguf_name: target}` state. ML8_4 → `{indices, centroids, scales, rotation(h_a,a,b)|None}`; ML8_FP8 → frozen fp32 weight; BF16/F32 pass-through → fp32 weight. Dequant for gates: `centroids[g, idx]*scale` reproduces GGUF bytes exactly (bit-equality gate, run as a script vs cell_A0_anchor_A3.gguf).

**Files:** Modify `gguf_state.py`, `test_gguf_state.py` (synthetic mini-GGUF fixture)

- [ ] **Step 1: failing test** — fixture writes a 2-tensor GGUF via gguf.GGUFWriter (one ML8_4 + centroids sidecar, packed from random idx/scl/cent on the e4m3 lattice; one F32 pass-through 1D), then asserts `load_ml8_gguf(path)` returns identical idx/scales/centroids and that `dequant_ml8(t)` matches CPU `W[r,c]=cent[g,idx]*scl[r,g]`.

```python
def test_rehydrate_synthetic(tmp_path):
    import gguf
    from gguf import GGMLQuantizationType
    from ml8_to_gguf import pack_ml8_blocks, cast_centroids_to_fp8
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 16, (8, 128), generator=g, dtype=torch.int8)
    scl = torch.rand(8, 2, generator=g) + 0.1
    cent = torch.randn(2, 16, generator=g).to(torch.float8_e4m3fn).to(torch.float32)
    p = tmp_path / "mini.gguf"
    w = gguf.GGUFWriter(str(p), arch="qwen35")
    w.add_key_value("qwen35.block_count", 1, gguf.GGUFValueType.UINT32)
    w.add_tensor("blk.0.ffn_up.weight", pack_ml8_blocks(idx, scl), raw_dtype=GGMLQuantizationType.ML8_4)
    w.add_tensor("blk.0.ffn_up.centroids", cast_centroids_to_fp8(cent), raw_dtype=GGMLQuantizationType.F8_E4M3)
    w.add_tensor("blk.0.ffn_norm.weight", np.ones(128, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    from gguf_state import load_ml8_gguf, dequant_ml8_state
    st = load_ml8_gguf(p)
    t = st.ml8["blk.0.ffn_up.weight"]
    assert torch.equal(t["indices"], idx.long()) and torch.equal(t["scales"], scl) and torch.equal(t["centroids"], cent)
    gidx = torch.arange(128) // 64
    W = cent[gidx, t["indices"]] * scl[:, gidx]
    assert torch.equal(dequant_ml8_state(t), W)
    assert "blk.0.ffn_norm.weight" in st.frozen
```

- [ ] **Step 2:** run → FAIL (`load_ml8_gguf` missing)
- [ ] **Step 3: implement** — `class Ml8State: ml8: dict; frozen: dict; meta: dict`. `load_ml8_gguf(path)`: GGUFReader; per tensor by `tensor_type`: ML8_4 → reader shape gives (K,N) reversed → N,K = shape[::-1] → unpack; F8_E4M3 sidecar `.centroids` → attach decoded; F32 `.rotation_h_a`/I32 `.rotation_meta` → attach rotation; ML8_FP8 → `unpack_scaled_fp8_blocks`, frozen W = e4m3*scale-expanded; BF16/F16/F32 → frozen fp32. `dequant_ml8_state(t)` per ml8_io formula. Skip `.awq_scale` (none in A3).
- [ ] **Step 4:** run → PASS
- [ ] **Step 5: real-GGUF gate script** — `python3 gguf_state.py --gguf ~/models/mi300x-ggufs/cell_A0_anchor_A3.gguf --bitcheck`: for every ML8_4 tensor re-pack `pack_ml8_blocks(idx, scl)` and byte-compare vs GGUF; e4m3-roundtrip centroids must be idempotent. Expected: `bitcheck OK on <n> ml8 tensors, <m> fp8 frozen, <k> passthrough`.
- [ ] **Step 6:** commit `feat(act-replay): GGUF rehydrator + bit-equality gate`

---

### Task 3: KL loss (top-K + tail bucket) — CPU TDD

**Files:** Create `scripts/calibration/kl_loss.py`; extend `test_gguf_state.py`-style `test_kl_loss.py`

- [ ] **Step 1: failing tests** — `topk_teacher(logits, K)` → (idx, vals, tail_logsumexp); `kl_topk(student_logits, idx, vals, tail)`: student logits gathered at idx + bucket = rest; teacher probs softmax([vals, tail]); KL exact. Tests: (a) K=V (full vocab) kl_topk == F.kl_div full; (b) student==teacher → 0; (c) mask drops padded tokens.

```python
def test_kl_full_equals_topk():
    g = torch.Generator().manual_seed(0)
    t, s = torch.randn(5, 32, generator=g), torch.randn(5, 32, generator=g)
    idx, vals, tail = topk_teacher(t, 31)
    full = torch.nn.functional.kl_div(torch.log_softmax(s, -1), torch.log_softmax(t, -1),
                                      log_target=True, reduction="batchmean")
    assert abs(kl_topk(s, idx, vals, tail) - full) < 1e-4
```

- [ ] **Step 2:** FAIL → **Step 3:** implement (fp32, logsumexp over remainder = `logsumexp(all) ⊖ logsumexp(topk)` via stable log-sub-exp; student tail bucket same way; loss = sum p_i (log p_i − log q_i) over K+1 buckets, token-mean over mask). **Step 4:** PASS. **Step 5:** commit `feat(act-replay): exact top-K+tail KL loss`.

---

### Task 4: TeacherSource (live / cache / device:N) — CPU TDD with stub model

**Files:** Create `scripts/calibration/teacher_source.py`, `test_teacher_source.py`

- [ ] **Step 1: failing tests** — stub `nn.Module` returning fixed logits. (a) `LiveTeacher(model, K).get(ids)` == topk_teacher of model logits; (b) `CachedTeacher.build(model, batches, dir, key)` writes shards; `.get(i)` equals live; reuse hits cache (model not called — count calls); (c) `DeviceTeacher(model, "cpu", K)` equals live (device move is the only difference).
- [ ] **Step 3: implement** — common ABC `TeacherSource.get(batch_idx, ids) -> (idx,vals,tail)`, K ctor arg; cache = one `.pt` shard per batch + `meta.json` key (gguf-hash, corpus key, K), to `--teacher-cache-dir` (NOT /tmp). Factory `make_teacher(spec, model_loader, K, cache_dir)` parsing `live|cache|device:N`.
- [ ] **Step 5:** commit `feat(act-replay): TeacherSource live/cache/deviceN`.

---

### Task 5: Student wrapper (dequant-STE overrides + faithful acts)

**Files:** Create `scripts/calibration/act_replay_student.py`, `test_act_replay_student.py`

- [ ] **Step 1: failing tests** on a stub `nn.Linear(128,8)`. (a) install ml8 target: master `centroids` [G,16] + `scales` [N,G] fp32 leaves; forward W = `cent_ste[gidx, idx]*scl[:,gidx]` with `cent_ste = c + (snap_to_e4m3(c)−c).detach()`; bit-equals GGUF dequant at step 0 (centroids on lattice ⇒ no-op). (b) backward fills grads on cent/scl only; module.weight stays untouched. (c) acts: `--faithful-acts` pre-hook reproduces `quantize_act_per_row(x@Q)@Q.T` (Hessian off). (d) tensor filter `select_targets(state, train="ml8", skip="ffn_down*")` glob/role semantics.
- [ ] **Step 3: implement** — `AttachedTarget`: hooks via fn override `module.forward = lambda x: F.linear(x, dequant_live())` (no copy); token_embd = embedding-weight override + tied head shares the same dequant tensor. Rotation hooks per GGUF sidecar (h_a + sylvester(b)). Targets selected by tier + name filters; non-targets frozen.
- [ ] **Step 5:** commit `feat(act-replay): student dequant-STE wrapper + faithful acts`.

---

### Task 6: Trainer CLI + holdout/ckpt + GGUF round-trip

**Files:** Create `scripts/calibration/act_replay.py`, `test_act_replay_cli.py` (HF model loading mocked with the stub)

- [ ] **Step 1:** tests for config plumbing (corpus/budget/tensors/lr/steps/grad-accum/seed), 90/10 split deterministic by seed, train step on stub decreases loss, ckpt save/resume reproduces step-state, export: tuned cent/scl → blob dicts (ml8_io schema: HF names via reverse GGUF→HF mapping, snap e4m3 final; fp8 + frozen tensors back as `.fp8.pt`/passthrough) → `ml8_to_gguf` builds GGUF → `load_ml8_gguf` round-trips. **Step 3:** implement (corpus via `collect_calibration(... composition=args.corpus, token_budget=...)`, chat-formatted; loss masked to response tokens; logs train/holdout KL every eval-interval). **Step 5:** commit.

---

### Task 7 [GPU]: step-0 sanity + 1-layer overfit smoke (R9700)

- [ ] HIP_VISIBLE_DEVICES=0; bitcheck on cell_A0_anchor_A3.gguf (Task 2 script) → OK.
- [ ] step-0 KL (live teacher, 8 batches mix) within ~A3 pilot magnitude (~0.1–0.3); fla-RDNA arch shim active.
- [ ] `--tensors-train 'blk.0.ffn_up*' --overfit-one-batch --steps 300` → KL → ≪ step0/10.
- [ ] Commit fixes if shaken loose.

### Task 8 [GPU]: loss-down@200 real config + VRAM gate

- [ ] Full config (mix 512k, teacher live, grad-accum 8, ckpt grad) → 200 steps: holdout KL ↓, peak VRAM ≤28GB (`torch.cuda.max_memory_allocated`), throughput logged. Tune micro-batch if needed.

### Task 9 [GPU]: overnight + morning re-score

- [ ] Launch overnight under nohup, time-based monitor, out `~/models/act_replay/A3/`.
- [ ] Morning: export GGUF; KL pipeline (wiki+agentic, 150×c512 vs bf16 base) + wiki PPL. Compare to A3 0.115 / UD 0.052. Update MAD256 doc + Jira.

## Self-review (done)
Spec covered: rehydrate(T2), STE+acts(T5), teacher×3(T4), config(T6), gates(T7–9), no /tmp.
