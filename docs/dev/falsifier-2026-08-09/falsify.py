#!/usr/bin/env python3
"""THE PRE-REGISTERED FALSIFIER (2026-08-07 analysis §4, run 2026-08-09).

Metric: recall of the first-non-resident expert set (the miss stream) against
a simulated pool fed by this run's own reference stream, at M in {8,16} hints
per layer. Bar: the router-k=2 first-non-resident baseline (28.3% on the 08-07
code capture; recomputed here on the same miss stream). NOT rank-1 precision,
NOT union recall.

Pools (config-of-record topology, DSPARK_HOST=CPU):
  R9700  2200 slots, pages (layer 0..45, expert 85..255)   LRU
  2026   1100 slots (550+550 merged), pages (layer 0..42, expert 0..84) LRU
  CPU    layers 43..45 x experts 0..84: fully resident, never scored
Scored miss stream: layers 0..42 on test-task steps, after warmup exclusion.
Draft-pass records (43..45) feed the R9700 pool but are not scored.

Methods at the same M/layer budget:
  probe   : trained DSpark-embedding probes (train tasks only), max-pooled over
            the draft block's embeddings. Width-1 steps have no draft -> empty.
  router2 : router_L(h_{L-2}) within the step, layers 2..42 (the 08-07 method).
  ngram   : token-conditional expert counts from train tasks, scored with the
            step's input ids (known pre-dispatch), smoothed toward popularity.
  pop     : train-task layer popularity top-M (the floor).
"""
import os
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from collections import OrderedDict
import numpy as np
import torch
torch.set_num_threads(24)

SCRATCH = os.path.dirname(os.path.abspath(__file__))
DS = np.load(os.path.join(SCRATCH, 'falsifier_dataset.npz'))
META = np.load(os.path.join(SCRATCH, 'probe_meta.npz'))
IX = np.load(os.path.join(SCRATCH, 'capture_index.npz'))
W = np.load(os.path.join(SCRATCH, 'probe_W.npy'))
PRED = '/home/kmbandy/ds4-runs/2026-08-08-serve/predcap.bin.3995715.bin'
DRAFT = '/home/kmbandy/ds4-runs/2026-08-08-serve/draftcap.bin'
GGUF = '/home/kmbandy/models/DS4-Flash-dense/ds4-dense.gguf'

N_EMBD_D = 16384
N_EMBD_H = 4096
N_EXP = 256
N_EU = 6
N_LAYERS = 43
MS = [8, 16]
MMAX = max(MS)
WARMUP_STEPS = 300

r_layer = DS['r_layer']; r_ntok = DS['r_ntok']; r_step = DS['r_step']
rec_offs = DS['rec_offs']
sel_flat = DS['sel_flat']; sel_off = DS['sel_off']
s_width = DS['s_width']; s_task = DS['s_task']
s_tokens_flat = DS['s_tokens_flat']; s_tokens_off = DS['s_tokens_off']
s_draft_block = DS['s_draft_block']
n_tasks = int(DS['n_tasks'][0])
n_steps = len(s_width)
n_wpc = len(r_layer)

samp_block = META['samp_block']; samp_pos = META['samp_pos']; samp_step = META['samp_step']
d_offs = IX['d_offs']

step_first_rec = np.full(n_steps, -1, np.int64)
for i in range(n_wpc):
    if r_layer[i] == 0:
        step_first_rec[r_step[i]] = i

def sel_of_rec(rec):
    nt = int(r_ntok[rec])
    return sel_flat[sel_off[rec]:sel_off[rec + 1]].reshape(nt, N_EU)

# ---- splits ----
is_test_task = np.array([(t % 5) == 4 for t in range(n_tasks)])
is_train_task = np.array([(t % 5) not in (3, 4) for t in range(n_tasks)])
tail_test_task = np.zeros(n_tasks, bool); tail_test_task[n_tasks - 15:] = True
step_is_scored = (np.arange(n_steps) >= WARMUP_STEPS) & (
    is_test_task[s_task] | tail_test_task[s_task])
scored_sids = np.where(step_is_scored)[0]
sid_slot = {int(s): i for i, s in enumerate(scored_sids)}
n_sc = len(scored_sids)
print(f'tasks={n_tasks} scored steps={n_sc} '
      f'(interleaved tasks={is_test_task.sum()}, tail=15, overlap ok)')

# ---- popularity + n-gram tables from TRAIN tasks ----
pop = np.zeros((N_LAYERS, N_EXP), np.int64)
ngram = {}
for sid in range(n_steps):
    if not is_train_task[s_task[sid]]:
        continue
    toks = s_tokens_flat[s_tokens_off[sid]:s_tokens_off[sid + 1]]
    base = step_first_rec[sid]
    for L in range(N_LAYERS):
        sel = sel_of_rec(base + L)
        for p in range(sel.shape[0]):
            e = sel[p].astype(np.int64)
            pop[L][e] += 1
            if p < len(toks):
                key = (int(toks[p]), L)
                arr = ngram.get(key)
                if arr is None:
                    arr = np.zeros(N_EXP, np.int32)
                    ngram[key] = arr
                arr[e] += 1
print(f'ngram table: {len(ngram)} (token,layer) keys')
pop_top16 = np.argsort(-pop, axis=1)[:, :MMAX].astype(np.int16)   # (43,16)
pop_norm = (pop / np.maximum(pop.sum(1, keepdims=True), 1)).astype(np.float32)

# ---- probe hints: one batched GEMM over scored-step draft embeddings ----
print('probe hints...')
mask = step_is_scored[samp_step]
p_idx = np.where(mask)[0]
E = np.empty((len(p_idx), N_EMBD_D), np.float32)
f = open(DRAFT, 'rb')
for j, i in enumerate(p_idx):
    f.seek(int(d_offs[samp_block[i]]) + int(samp_pos[i]) * N_EMBD_D * 4)
    E[j] = np.frombuffer(f.read(N_EMBD_D * 4), np.float32)
f.close()
S = (torch.from_numpy(E) @ torch.from_numpy(W[:N_EMBD_D]) +
     torch.from_numpy(W[N_EMBD_D])).numpy()          # (n, 43*256)
del E
probe_scores = np.full((n_sc, N_LAYERS, N_EXP), -np.inf, np.float32)
for j, i in enumerate(p_idx):
    slot = sid_slot[int(samp_step[i])]
    np.maximum(probe_scores[slot], S[j].reshape(N_LAYERS, N_EXP),
               out=probe_scores[slot])
del S
has_probe = np.isfinite(probe_scores[:, 0, 0])
probe_scores[~has_probe] = 0.0
probe_top16 = torch.topk(torch.from_numpy(probe_scores), MMAX, dim=2
                         ).indices.numpy().astype(np.int16)
del probe_scores
print(f'probe hints for {int(has_probe.sum())}/{n_sc} scored steps')

# ---- router-k=2 hints: chunked batched GEMMs per layer ----
print('loading routers from gguf...')
sys.path.insert(0, '/home/kmbandy/GitHub/llama.cpp/gguf-py')
from gguf import GGUFReader
rd = GGUFReader(GGUF)
tensors = {t.name: t for t in rd.tensors}
def bf16(t):
    raw = np.asarray(t.data)
    if raw.dtype == np.float32:
        return raw
    u = raw.view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)
routers = {}
for L in range(N_LAYERS):
    wn = f'blk.{L}.ffn_gate_inp.weight'; bn = f'blk.{L}.exp_probs_b.bias'
    if wn in tensors and bn in tensors:
        routers[L] = (torch.from_numpy(bf16(tensors[wn]).astype(np.float32).copy()),
                      torch.from_numpy(np.asarray(tensors[bn].data, np.float32).copy()))
print(f'routers loaded: {len(routers)}/{N_LAYERS}')

router_scores = np.full((n_sc, N_LAYERS, N_EXP), -np.inf, np.float32)
fpred = open(PRED, 'rb')
CH = 2000
for lo in range(0, n_sc, CH):
    sids = scored_sids[lo:lo + CH]
    for L in range(2, N_LAYERS):
        if L not in routers:
            continue
        rows, seg = [], []
        for k, sid in enumerate(sids):
            rec = step_first_rec[sid] + (L - 2)
            nt = int(r_ntok[rec])
            fpred.seek(int(rec_offs[rec]))
            rows.append(np.frombuffer(fpred.read(nt * N_EMBD_H * 4), np.float32
                                      ).reshape(nt, N_EMBD_H))
            seg.extend([k] * nt)
        H = torch.from_numpy(np.concatenate(rows))
        Wr, br = routers[L]
        logits = H @ Wr.T
        probs = torch.sqrt(torch.nn.functional.softplus(-logits.abs()).log1p() * 0
                           + torch.log1p(torch.exp(-logits.abs())) + logits.clamp(min=0))
        sc = (probs + br).numpy()
        seg = np.array(seg)
        for k in range(len(sids)):
            m = sc[seg == k]
            if len(m):
                router_scores[lo + k, L] = m.max(0)
fpred.close()
router_scores[:, :2, :] = 0.0
router_has = np.isfinite(router_scores).all(2)     # (n_sc, 43)
router_scores[~router_has] = 0.0
router_top16 = torch.topk(torch.from_numpy(router_scores), MMAX, dim=2
                          ).indices.numpy().astype(np.int16)
del router_scores
print('router hints done')

# ---- n-gram hints ----
print('ngram hints...')
ngram_top16 = np.empty((n_sc, N_LAYERS, MMAX), np.int16)
for j, sid in enumerate(scored_sids):
    toks = s_tokens_flat[s_tokens_off[sid]:s_tokens_off[sid + 1]]
    for L in range(N_LAYERS):
        sc = pop_norm[L] * 1e-3
        for t in toks.tolist():
            arr = ngram.get((t, L))
            if arr is not None:
                sc = sc + arr / max(arr.sum(), 1)
        ngram_top16[j, L] = np.argpartition(-sc, MMAX)[:MMAX][
            np.argsort(-sc[np.argpartition(-sc, MMAX)[:MMAX]])].astype(np.int16)
print('ngram hints done')

# ---- pool replay + scoring ----
class LRU:
    __slots__ = ('cap', 'd', 'hits', 'touches')
    def __init__(self, cap):
        self.cap = cap; self.d = OrderedDict(); self.hits = 0; self.touches = 0
    def touch(self, key):
        self.touches += 1
        if key in self.d:
            self.d.move_to_end(key)
            self.hits += 1
            return True
        self.d[key] = True
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return False

r9700 = LRU(2200)
g2026 = LRU(1100)

score_splits = {
    'test-interleaved': is_test_task,
    'test-tail15':      tail_test_task,
}
methods = ['probe', 'router2', 'ngram', 'pop']
top_arrays = {'probe': probe_top16, 'router2': router_top16,
              'ngram': ngram_top16}
stats = {(sp, m, M): [0, 0] for sp in score_splits for m in methods for M in MS}
depth_hist = []       # first-miss layer per scored step
miss_per_step = []

print('replaying dispatch stream...')
cur_sets = None
cur_slot = -1
cur_splits = None
first_miss_L = None
nmiss_step = 0
for rec in range(n_wpc):
    L = int(r_layer[rec])
    if L >= 43:
        sel = sel_of_rec(rec)
        for p in range(sel.shape[0]):
            for e in set(sel[p].tolist()):
                if e >= 85:
                    r9700.touch((L, e))
        continue
    sid = int(r_step[rec])
    if L == 0:
        if cur_sets is not None:
            depth_hist.append(first_miss_L)
            miss_per_step.append(nmiss_step)
        cur_sets = None
        first_miss_L = None; nmiss_step = 0
        if step_is_scored[sid]:
            cur_slot = sid_slot[sid]
            t = int(s_task[sid])
            cur_splits = [sp for sp, arr in score_splits.items() if arr[t]]
            cur_sets = {}
            for m in methods:
                for M in MS:
                    if m == 'pop':
                        tops = pop_top16
                        cur_sets[(m, M)] = [set(tops[Lx, :M].tolist())
                                            for Lx in range(N_LAYERS)]
                    else:
                        ta = top_arrays[m]
                        if m == 'probe' and not has_probe[cur_slot]:
                            cur_sets[(m, M)] = [set()] * N_LAYERS
                        else:
                            cur_sets[(m, M)] = [set(ta[cur_slot, Lx, :M].tolist())
                                                for Lx in range(N_LAYERS)]
            # router layers 0,1 have no lead
            for M in MS:
                cur_sets[('router2', M)][0] = set()
                cur_sets[('router2', M)][1] = set()
    sel = sel_of_rec(rec)
    uniq = set()
    for p in range(sel.shape[0]):
        uniq.update(sel[p].tolist())
    missed = []
    for e in uniq:
        if e >= 85:
            if (L, e) not in r9700.d:
                missed.append(e)
            r9700.touch((L, e))
        else:
            if (L, e) not in g2026.d:
                missed.append(e)
            g2026.touch((L, e))
    if cur_sets is None or not missed:
        continue
    if first_miss_L is None:
        first_miss_L = L
    nmiss_step += len(missed)
    for sp in cur_splits:
        for M in MS:
            for m in methods:
                hs = cur_sets[(m, M)][L]
                st = stats[(sp, m, M)]
                st[0] += sum(1 for e in missed if e in hs)
                st[1] += len(missed)

print(f'\npool residency: R9700 {100*r9700.hits/max(r9700.touches,1):.1f}%  '
      f'2026 {100*g2026.hits/max(g2026.touches,1):.1f}%')
dh = np.array([d for d in depth_hist if d is not None])
mp = np.array(miss_per_step)
print(f'scored steps with >=1 miss: {len(dh)}/{len(miss_per_step)}  '
      f'first-miss depth p50={np.median(dh) if len(dh) else -1:.0f}  '
      f'misses/step p50={np.median(mp):.0f} mean={mp.mean():.1f}')
print('\n== FALSIFIER: recall of the first-non-resident set (miss stream) ==')
print(f'{"split":<18} {"method":<8} ' + '  '.join(f'  R@{M}' for M in MS))
for sp in score_splits:
    for m in methods:
        cells = []
        for M in MS:
            h, t = stats[(sp, m, M)]
            cells.append(f'{100*h/t:5.1f}%' if t else '  n/a')
        h, t = stats[(sp, m, MS[0])]
        extra = f'   (n_miss={t})' if m == 'probe' else ''
        print(f'{sp:<18} {m:<8} ' + '  '.join(cells) + extra)
print('\nBAR: router-k=2 08-07 baseline = 28.3%. Beat it clearly on the miss '
      'stream at M in {8,16}, or the direction closes.')
