#!/usr/bin/env python3
"""Pass 1: index the capture files (metadata + token ids, skip embeddings).

Emits an npz with:
  predcap record table: kind (0=WPC1, 1=WTB1), layer, n_tokens, file offset of payload
  WTB1 token id lists (ragged, stored flat + offsets)
  WPC1 selection arrays are NOT read here (payload skipped) except offsets kept.
  draftcap block table: block_id, n_drafted, token ids, offset of embedding payload
Then prints alignment/task-boundary diagnostics.
"""
import struct, os
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
PRED = '/home/kmbandy/ds4-runs/2026-08-08-serve/predcap.bin.3995715.bin'
DRAFT = '/home/kmbandy/ds4-runs/2026-08-08-serve/draftcap.bin'

def index_predcap(path):
    kinds, layers, ntoks, offs = [], [], [], []
    wtb_flat, wtb_off = [], [0]
    f = open(path, 'rb')
    while True:
        m = f.read(4)
        if len(m) < 4:
            break
        (magic,) = struct.unpack('<I', m)
        if magic == 0x31435057:
            layer, n_tokens, n_embd, n_eu = struct.unpack('<iiii', f.read(16))
            kinds.append(0); layers.append(layer); ntoks.append(n_tokens); offs.append(f.tell())
            f.seek(n_tokens * n_embd * 4 + n_tokens * n_eu * 4, 1)
        elif magic == 0x31425457:
            (count,) = struct.unpack('<I', f.read(4))
            toks = np.frombuffer(f.read(count * 4), dtype=np.int32)
            kinds.append(1); layers.append(-1); ntoks.append(count); offs.append(f.tell())
            wtb_flat.append(toks)
            wtb_off.append(wtb_off[-1] + count)
        else:
            print(f'BAD magic {hex(magic)} at {f.tell()-4}')
            break
    f.close()
    return (np.array(kinds, np.int8), np.array(layers, np.int16),
            np.array(ntoks, np.int32), np.array(offs, np.int64),
            np.concatenate(wtb_flat) if wtb_flat else np.zeros(0, np.int32),
            np.array(wtb_off, np.int64))

def index_draftcap(path):
    f = open(path, 'rb')
    magic, ver, n_embd, dtype = struct.unpack('<IIII', f.read(16))
    ids, nds, offs = [], [], []
    tok_flat, tok_off = [], [0]
    while True:
        hdr = f.read(16)
        if len(hdr) < 16:
            break
        marker, blk_id, nd = struct.unpack('<IQI', hdr)
        assert marker == 0x31445257
        toks = np.frombuffer(f.read(nd * 4), dtype=np.int32)
        ids.append(blk_id); nds.append(nd); offs.append(f.tell())
        tok_flat.append(toks)
        tok_off.append(tok_off[-1] + nd)
        f.seek(nd * n_embd * 4, 1)
    f.close()
    return (np.array(ids, np.int64), np.array(nds, np.int32), np.array(offs, np.int64),
            np.concatenate(tok_flat), np.array(tok_off, np.int64), n_embd)

kinds, layers, ntoks, offs, wtb_flat, wtb_off = index_predcap(PRED)
d_ids, d_nd, d_offs, d_tok, d_tokoff, d_nembd = index_draftcap(DRAFT)

np.savez(os.path.join(SCRATCH, 'capture_index.npz'),
         kinds=kinds, layers=layers, ntoks=ntoks, offs=offs,
         wtb_flat=wtb_flat, wtb_off=wtb_off,
         d_ids=d_ids, d_nd=d_nd, d_offs=d_offs, d_tok=d_tok, d_tokoff=d_tokoff,
         d_nembd=np.array([d_nembd]))

# ---- diagnostics ----
n_rec = len(kinds)
wpc = kinds == 0
wtb = kinds == 1
print(f'records={n_rec}  WPC1={wpc.sum()}  WTB1={wtb.sum()}')

# WTB1 count histogram
wtb_counts = ntoks[wtb]
import collections
hist = collections.Counter(wtb_counts.tolist())
print('WTB1 count hist (top12):', sorted(hist.items(), key=lambda kv: -kv[1])[:12])
print('WTB1 counts >8:', sum(c for n, c in hist.items() if n > 8), 'records,',
      sum(n*c for n, c in hist.items() if n > 8), 'tokens')

# Step segmentation over WPC1 stream: new step when layer <= prev layer
wpc_idx = np.where(wpc)[0]
wl = layers[wpc_idx]
step_starts = np.where(np.diff(wl, prepend=99) <= 0)[0]  # indices into wpc_idx
# fix: first record starts step 0
step_starts = np.where(np.concatenate([[True], np.diff(wl.astype(int)) <= 0]))[0]
print(f'steps={len(step_starts)}')
step_len = np.diff(np.concatenate([step_starts, [len(wl)]]))
print('records/step hist:', collections.Counter(step_len.tolist()).most_common(6))
step_first_layer = wl[step_starts]
print('first layer of step hist:', collections.Counter(step_first_layer.tolist()).most_common(6))
step_ntok = ntoks[wpc_idx][step_starts]
print('step n_tokens hist:', collections.Counter(step_ntok.tolist()).most_common(10))

# Interleaving: where do WTB1 records sit relative to steps? Look at the pattern
# of kinds around the first few steps.
print('\nfirst 60 records (kind,layer/count):')
print(' '.join((f'W{layers[i]}' if kinds[i]==0 else f'T{ntoks[i]}') for i in range(60)))
print('\nrecords 300000..300060:')
print(' '.join((f'W{layers[i]}' if kinds[i]==0 else f'T{ntoks[i]}') for i in range(300000, 300060)))

# Draft block sizes
print('\ndraft blocks:', len(d_ids), 'sizes:', collections.Counter(d_nd.tolist()))
print('block_id contiguous:', bool(np.all(np.diff(d_ids) == 1)), ' first/last:', d_ids[0], d_ids[-1])

# Try alignment: for each step with n_tokens in {2,3}, tokens[1:] should equal some
# draft block's tokens. Compare sequence of step widths vs draft nd+1 stream.
# Also check WTB1 immediately preceding each step start matches step n_tokens.
pre_wtb_match = 0
checked = 0
rec_is_step_start = np.zeros(n_rec, bool)
rec_is_step_start[wpc_idx[step_starts]] = True
last_wtb_count = -1
last_wtb_slot = -1
step_pre_wtb = []  # count of the WTB1 record most recently seen before each step start
for i in range(n_rec):
    if kinds[i] == 1:
        last_wtb_count = ntoks[i]
        last_wtb_slot = int(np.searchsorted(np.where(wtb)[0], i))
    elif rec_is_step_start[i]:
        step_pre_wtb.append(last_wtb_count)
step_pre_wtb = np.array(step_pre_wtb)
w = step_ntok[:len(step_pre_wtb)]
print('\nWTB1-before-step count == step n_tokens:',
      int((step_pre_wtb[:len(w)] == w).sum()), '/', len(w))
