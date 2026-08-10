#!/usr/bin/env python3
"""Pass 2: build the falsifier dataset from the index.

Outputs falsifier_dataset.npz:
  Per WPC1 record (file order, the dispatch order):
    r_layer, r_ntok, r_step (verify-step id or -1 for draft-pass records),
    sel_flat (concatenated n_tok*6 int16 per record), sel_off
  Per verify step:
    s_width, s_task, s_tokens_flat/off (the WTB1 ids), s_draft_block (or -1)
  Per draft block: d_step (the verify step it drafted, or -1)
  Task table: t_start_step
"""
import struct, os
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
PRED = '/home/kmbandy/ds4-runs/2026-08-08-serve/predcap.bin.3995715.bin'
IX = np.load(os.path.join(SCRATCH, 'capture_index.npz'))

kinds = IX['kinds']; layers = IX['layers']; ntoks = IX['ntoks']; offs = IX['offs']
wtb_flat = IX['wtb_flat']; wtb_off = IX['wtb_off']
d_ids = IX['d_ids']; d_nd = IX['d_nd']; d_tok = IX['d_tok']; d_tokoff = IX['d_tokoff']

n_rec = len(kinds)
N_EU = 6

# ---- read all WPC1 sel payloads (skip h) ----
f = open(PRED, 'rb')
sel_chunks = []
r_layer, r_ntok = [], []
wpc_rec_ids = []
for i in range(n_rec):
    if kinds[i] != 0:
        continue
    nt = int(ntoks[i]); ly = int(layers[i])
    # payload starts at offs[i]: h (nt*n_embd*4) then sel (nt*6*4). n_embd=4096.
    f.seek(offs[i] + nt * 4096 * 4)
    sel = np.frombuffer(f.read(nt * N_EU * 4), dtype=np.int32)
    sel_chunks.append(sel.astype(np.int16))
    r_layer.append(ly); r_ntok.append(nt); wpc_rec_ids.append(i)
f.close()
r_layer = np.array(r_layer, np.int16)
r_ntok = np.array(r_ntok, np.int32)
wpc_rec_ids = np.array(wpc_rec_ids, np.int64)
sel_lens = r_ntok * N_EU
sel_off = np.concatenate([[0], np.cumsum(sel_lens)])
sel_flat = np.concatenate(sel_chunks)
del sel_chunks
print(f'sel_flat: {sel_flat.nbytes/1e6:.0f} MB over {len(r_layer)} records')

# ---- step segmentation on WPC1 stream ----
# A verify step starts at a layer-0 record whose n_tokens <= 8 that follows a
# WTB1 of the same count (verified 27446/27446). Draft-pass records are layers
# 43..45. Records with layer==0 always start a step (first-layer hist: all 0).
n_wpc = len(r_layer)
is_start = r_layer == 0
step_of_rec = np.cumsum(is_start) - 1          # verify-step id for W0..42
step_starts_w = np.where(is_start)[0]
n_steps = len(step_starts_w)
s_width = r_ntok[step_starts_w]
# draft-pass records (43..45) belong to the PRECEDING verify step
is_draft_rec = r_layer >= 43
r_step = np.where(is_draft_rec, -1, step_of_rec).astype(np.int64)

# ---- WTB1 bookkeeping: task boundaries + verify token ids ----
# Walk records in file order; a WTB1 with count>8 is a prefill chunk -> a task
# boundary before the NEXT verify step. WTB1 with count<=8 immediately before a
# layer-0 WPC1 is that step's token list.
wtb_rec_ids = np.where(kinds == 1)[0]
wtb_slot_of_rec = {int(r): j for j, r in enumerate(wtb_rec_ids)}
s_tokens = [None] * n_steps
s_task = np.zeros(n_steps, np.int32)
task_id = -1
pending_new_task = True
last_small_wtb = None
step_cursor = 0
wpc_pos = 0  # index into wpc arrays
for i in range(n_rec):
    if kinds[i] == 1:
        j = wtb_slot_of_rec[i]
        toks = wtb_flat[wtb_off[j]:wtb_off[j + 1]]
        if ntoks[i] > 8:
            pending_new_task = True
        else:
            last_small_wtb = toks
    else:
        if r_layer[wpc_pos] == 0:
            sid = step_of_rec[wpc_pos]
            if pending_new_task:
                task_id += 1
                pending_new_task = False
            s_task[sid] = task_id
            s_tokens[sid] = last_small_wtb
        wpc_pos += 1
n_tasks = task_id + 1
print(f'steps={n_steps} tasks={n_tasks}')
steps_per_task = np.bincount(s_task)
print('steps/task: min/p50/max =', steps_per_task.min(),
      int(np.median(steps_per_task)), steps_per_task.max())

# ---- draft-block -> verify-step join ----
# Draft pass for block b runs after verify step s-1; block b's tokens should be
# the NEXT verify step's tokens[1:]. Walk steps in order keeping a cursor into
# draft blocks; match on width AND token ids.
d_step = np.full(len(d_ids), -1, np.int64)
s_draft_block = np.full(n_steps, -1, np.int64)
cur = 0
matched = mismatched = 0
for sid in range(n_steps):
    w = int(s_width[sid])
    if w < 2 or cur >= len(d_ids):
        continue
    nd = w - 1
    if int(d_nd[cur]) != nd:
        # widths must agree; try skipping stale draft blocks (draft made but
        # block never verified, e.g. task end)
        probe = cur
        while probe < min(cur + 3, len(d_ids)) and int(d_nd[probe]) != nd:
            probe += 1
        if probe >= len(d_ids) or int(d_nd[probe]) != nd:
            mismatched += 1
            continue
        cur = probe
    btoks = d_tok[d_tokoff[cur]:d_tokoff[cur + 1]]
    stoks = s_tokens[sid]
    if stoks is not None and len(stoks) == nd + 1 and np.array_equal(stoks[1:], btoks):
        d_step[cur] = sid
        s_draft_block[sid] = cur
        matched += 1
        cur += 1
    else:
        # token mismatch: the draft block may have been rejected; advance draft
        # cursor only when its tokens appear nowhere in this step
        mismatched += 1
print(f'draft-block join: matched={matched} mismatched={mismatched} '
      f'(blocks={len(d_ids)}, eligible steps={(s_width >= 2).sum()})')

s_tokens_flat = np.concatenate([t if t is not None else np.zeros(0, np.int32)
                                for t in s_tokens])
s_tokens_len = np.array([0 if t is None else len(t) for t in s_tokens], np.int64)
s_tokens_off = np.concatenate([[0], np.cumsum(s_tokens_len)])

np.savez_compressed(os.path.join(SCRATCH, 'falsifier_dataset.npz'),
                    r_layer=r_layer, r_ntok=r_ntok, r_step=r_step,
                    wpc_rec_ids=wpc_rec_ids, rec_offs=offs[wpc_rec_ids],
                    sel_flat=sel_flat, sel_off=sel_off,
                    s_width=s_width, s_task=s_task,
                    s_tokens_flat=s_tokens_flat, s_tokens_off=s_tokens_off,
                    s_draft_block=s_draft_block, d_step=d_step,
                    n_tasks=np.array([n_tasks]))
print('saved falsifier_dataset.npz')
