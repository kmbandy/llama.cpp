#!/usr/bin/env python3
"""Train per-layer linear probes: DSpark nextn embedding (16384) -> 256 experts.

Ridge closed form with a shared Gram matrix: XtX built once over train samples,
one Cholesky per lambda, all 43 layer heads solved together (Y = multi-hot).
Split is BY TASK: task % 5 == 4 -> test, else train (interleaved), plus the
tail split (last 15 tasks) evaluated separately in the falsifier.
Lambda picked on a held-out slice of TRAIN tasks (task % 5 == 3 -> val).
"""
import os
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['ROCR_VISIBLE_DEVICES'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
torch.set_num_threads(24)

SCRATCH = os.path.dirname(os.path.abspath(__file__))
IX = np.load(os.path.join(SCRATCH, 'capture_index.npz'))
DS = np.load(os.path.join(SCRATCH, 'falsifier_dataset.npz'))
DRAFT = '/home/kmbandy/ds4-runs/2026-08-08-serve/draftcap.bin'

N_EMBD = 16384
N_EXP = 256
N_EU = 6
N_LAYERS = 43

d_nd = IX['d_nd']; d_offs = IX['d_offs']
d_step = DS['d_step']
s_task = DS['s_task']; s_width = DS['s_width']
r_layer = DS['r_layer']; r_ntok = DS['r_ntok']; r_step = DS['r_step']
sel_flat = DS['sel_flat']; sel_off = DS['sel_off']

# ---- build per-sample table: (block, pos_in_block) -> step, verify position ----
blocks = np.where(d_step >= 0)[0]
samp_block, samp_pos, samp_step = [], [], []
for b in blocks:
    for p in range(int(d_nd[b])):
        samp_block.append(b); samp_pos.append(p); samp_step.append(int(d_step[b]))
samp_block = np.array(samp_block); samp_pos = np.array(samp_pos)
samp_step = np.array(samp_step)
n_samp = len(samp_block)
print(f'samples (drafted tokens with a verified step): {n_samp}')

# ---- labels: sel[(step, layer, verify_pos=pos+1)] for layers 0..42 ----
# Build a (step, layer) -> record row lookup. Records for a step are its 43
# verify layers in order; find each step's first record index.
n_wpc = len(r_layer)
step_first_rec = np.full(int(s_task.shape[0]), -1, np.int64)
for i in range(n_wpc):
    if r_layer[i] == 0:
        step_first_rec[r_step[i]] = i
# Vectorized labels: lab[i, L, :] = the 6 experts of sample i at layer L.
# rec(i, L) = step_first_rec[samp_step[i]] + L; flat index = sel_off[rec] + (pos+1)*6 + k
rec_mat = step_first_rec[samp_step][:, None] + np.arange(N_LAYERS)[None, :]
base_idx = sel_off[rec_mat] + ((samp_pos + 1) * N_EU)[:, None]
lab = sel_flat[base_idx[:, :, None] + np.arange(N_EU)[None, None, :]].astype(np.int64)
del rec_mat, base_idx
print('labels:', lab.shape)

# ---- task split ----
task_of_samp = s_task[samp_step]
is_test = (task_of_samp % 5) == 4
is_val  = (task_of_samp % 5) == 3
is_train = ~(is_test | is_val)
print(f'train={is_train.sum()} val={is_val.sum()} test={is_test.sum()} (by task%5)')

# ---- load embeddings (all samples; 34k x 16384 f32 = 2.2 GB) ----
X = np.empty((n_samp, N_EMBD), np.float32)
f = open(DRAFT, 'rb')
for i in range(n_samp):
    b = samp_block[i]; p = samp_pos[i]
    f.seek(int(d_offs[b]) + p * N_EMBD * 4)
    X[i] = np.frombuffer(f.read(N_EMBD * 4), np.float32)
f.close()
print('X loaded:', X.shape, f'{X.nbytes/1e9:.2f} GB  norm p50 =',
      float(np.median(np.linalg.norm(X[:200], axis=1))))

# bias column via augmented feature
ones = np.ones((n_samp, 1), np.float32)
D = N_EMBD + 1

Xtr = torch.from_numpy(np.ascontiguousarray(X[is_train]))
XtX = torch.zeros((D, D), dtype=torch.float64)
XtX[:N_EMBD, :N_EMBD] = (Xtr.T @ Xtr).double()
XtX[:N_EMBD, N_EMBD] = Xtr.sum(0).double()
XtX[N_EMBD, :N_EMBD] = XtX[:N_EMBD, N_EMBD]
XtX[N_EMBD, N_EMBD] = Xtr.shape[0]
del Xtr
print('XtX done')

# XtY per layer, all layers stacked: (D, 43*256)
XtY = torch.zeros((D, N_LAYERS * N_EXP), dtype=torch.float64)
tr_idx = np.where(is_train)[0]
# build Y row-block per layer in chunks to bound memory
CH = 8192
lay_off = (np.arange(N_LAYERS) * N_EXP)[None, :, None]
for lo in range(0, len(tr_idx), CH):
    idx = tr_idx[lo:lo + CH]
    Yc = np.zeros((len(idx), N_LAYERS * N_EXP), np.float32)
    cols = (lab[idx] + lay_off).reshape(len(idx), -1)
    Yc[np.repeat(np.arange(len(idx)), N_LAYERS * N_EU), cols.ravel()] = 1.0
    Xc = torch.from_numpy(np.ascontiguousarray(X[idx]))
    XtY[:N_EMBD] += (Xc.T @ torch.from_numpy(Yc)).double()
    XtY[N_EMBD] += torch.from_numpy(Yc.sum(0)).double()
print('XtY done')

# ---- validation labels for lambda pick (top-6 hit rate on val) ----
val_idx = np.where(is_val)[0]
def topk_hit_rate(W):
    """mean over val samples+layers of |top6(pred) & actual| / 6"""
    hits = tot = 0
    for lo in range(0, len(val_idx), 4096):
        idx = val_idx[lo:lo + 4096]
        Xc = torch.from_numpy(np.ascontiguousarray(X[idx]))
        S = (Xc @ torch.from_numpy(W[:N_EMBD]) + torch.from_numpy(W[N_EMBD])).numpy()
        S = S.reshape(len(idx), N_LAYERS, N_EXP)
        top6 = np.argpartition(-S, 6, axis=2)[:, :, :6]
        e = lab[idx]                              # (n, 43, 6)
        hits += (top6[:, :, :, None] == e[:, :, None, :]).any(2).sum()
        tot += e.size
    return hits / tot

best = None
for lam in [3e2, 3e3, 3e4]:
    A = XtX.clone()
    A.diagonal().add_(lam)
    A[N_EMBD, N_EMBD] -= lam  # do not penalize the bias
    L = torch.linalg.cholesky(A)
    del A
    W = torch.cholesky_solve(XtY, L).float().numpy()
    del L
    hr = topk_hit_rate(W)
    print(f'lambda={lam:g}  val top-6 hit-rate={hr:.4f}')
    if best is None or hr > best[1]:
        best = (lam, hr, W)
lam, hr, W = best
print(f'CHOSEN lambda={lam:g} val hit-rate={hr:.4f}')
np.save(os.path.join(SCRATCH, 'probe_W.npy'), W)
np.savez(os.path.join(SCRATCH, 'probe_meta.npz'),
         lam=np.array([lam]), split_mod=np.array([5]),
         samp_block=samp_block, samp_pos=samp_pos, samp_step=samp_step)
print('saved probe_W.npy (D x 43*256)')
