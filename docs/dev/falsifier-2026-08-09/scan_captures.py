#!/usr/bin/env python3
"""Scan WPD1 draft-capture and WPC1/WTB1 predict-capture files: counts only, seek past payloads."""
import struct, sys, os
from collections import Counter

def scan_draftcap(path):
    size = os.path.getsize(path)
    f = open(path, 'rb')
    magic, ver, n_embd, dtype = struct.unpack('<IIII', f.read(16))
    assert magic == 0x31445057, hex(magic)
    n_blocks = 0
    total_drafted = 0
    dist = Counter()
    while True:
        hdr = f.read(16)  # marker u32 + id u64 + n_drafted u32
        if len(hdr) < 16:
            break
        marker, blk_id, n_drafted = struct.unpack('<IQI', hdr)
        if marker != 0x31445257:
            print(f'  BAD marker {hex(marker)} at offset {f.tell()-16}, stopping')
            break
        f.seek(n_drafted * 4 + n_drafted * n_embd * 4, 1)
        n_blocks += 1
        total_drafted += n_drafted
        dist[n_drafted] += 1
    f.close()
    print(f'DRAFTCAP {path}')
    print(f'  size={size/1e9:.2f} GB  n_embd={n_embd} ver={ver} dtype={dtype}')
    print(f'  blocks={n_blocks}  drafted_tokens={total_drafted}  mean_block={total_drafted/max(1,n_blocks):.2f}')
    print(f'  block-size dist: {dict(sorted(dist.items()))}')

def scan_predcap(path):
    size = os.path.getsize(path)
    f = open(path, 'rb')
    n_wpc = 0
    n_wtb = 0
    wtb_tokens = 0
    layers = Counter()          # layer -> record count
    tok_positions = 0           # sum of n_tokens over WPC1 records
    ntok_dist = Counter()       # n_tokens histogram (decode=small, prefill=big)
    decode_recs = 0             # n_tokens <= 8 (spec block width)
    decode_positions = 0
    n_embd_seen = set()
    neu_seen = set()
    bad = 0
    while True:
        m = f.read(4)
        if len(m) < 4:
            break
        (magic,) = struct.unpack('<I', m)
        if magic == 0x31435057:  # WPC1
            hdr = f.read(16)
            if len(hdr) < 16: break
            layer, n_tokens, n_embd, n_eu = struct.unpack('<iiii', hdr)
            f.seek(n_tokens * n_embd * 4 + n_tokens * n_eu * 4, 1)
            n_wpc += 1
            layers[layer] += 1
            tok_positions += n_tokens
            ntok_dist[n_tokens] += 1
            if n_tokens <= 8:
                decode_recs += 1
                decode_positions += n_tokens
            n_embd_seen.add(n_embd)
            neu_seen.add(n_eu)
        elif magic == 0x31425457:  # WTB1
            c = f.read(4)
            if len(c) < 4: break
            (count,) = struct.unpack('<I', c)
            f.seek(count * 4, 1)
            n_wtb += 1
            wtb_tokens += count
        else:
            bad += 1
            print(f'  BAD magic {hex(magic)} at {f.tell()-4}, stopping')
            break
    f.close()
    print(f'PREDCAP {path}')
    print(f'  size={size/1e9:.2f} GB  n_embd={sorted(n_embd_seen)}  n_expert_used={sorted(neu_seen)}')
    print(f'  WPC1 records={n_wpc}  token-positions={tok_positions}')
    print(f'  layers covered={len(layers)} ({min(layers)}..{max(layers)})  per-layer min/max recs={min(layers.values())}/{max(layers.values())}')
    print(f'  decode-ish (n_tokens<=8): {decode_recs} recs / {decode_positions} positions')
    top = sorted(ntok_dist.items(), key=lambda kv: -kv[1])[:8]
    print(f'  n_tokens top: {top}')
    big = sum(c for nt, c in ntok_dist.items() if nt > 8)
    bigpos = sum(nt*c for nt, c in ntok_dist.items() if nt > 8)
    print(f'  prefill-ish (n_tokens>8): {big} recs / {bigpos} positions')
    print(f'  WTB1 batch records={n_wtb}  batch tokens={wtb_tokens}')

if __name__ == '__main__':
    scan_draftcap('/home/kmbandy/ds4-runs/2026-08-08-serve/draftcap.bin')
    print()
    scan_predcap('/home/kmbandy/ds4-runs/2026-08-08-serve/predcap.bin.3995715.bin')
