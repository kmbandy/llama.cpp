#!/usr/bin/env python3
"""Bit-reproducibility check for two ml8 calibration runs. For every matching blob
in dirA and dirB, compare the quantization-defining tensors (indices, centroids,
scales) for EXACT equality. Prints per-blob match and an overall verdict.

Usage: diff_calib_blobs.py <dirA> <dirB>
"""
import sys
import pathlib
import torch

dirA, dirB = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
KEYS = ("indices", "centroids_per_group", "scale_per_group")

blobsA = sorted(p.name for p in dirA.glob("*.pt"))
blobsB = set(p.name for p in dirB.glob("*.pt"))

n_match = n_diff = n_missing = 0
diffs = []
for name in blobsA:
    if name not in blobsB:
        n_missing += 1
        diffs.append(f"  MISSING in B: {name}")
        continue
    a = torch.load(dirA / name, map_location="cpu", weights_only=False)
    b = torch.load(dirB / name, map_location="cpu", weights_only=False)
    blob_ok = True
    for k in KEYS:
        if k in a or k in b:
            ta, tb = a.get(k), b.get(k)
            if ta is None or tb is None or ta.shape != tb.shape or not torch.equal(ta, tb):
                blob_ok = False
                diffs.append(f"  DIFF {name}: tensor '{k}' differs")
    if blob_ok:
        n_match += 1
    else:
        n_diff += 1

print(f"blobs compared: {len(blobsA)}  identical: {n_match}  differing: {n_diff}  missing: {n_missing}")
for d in diffs[:20]:
    print(d)
if n_diff == 0 and n_missing == 0:
    print("\n✅ REPRODUCIBLE — every blob's indices/centroids/scales are bit-identical across runs.")
    sys.exit(0)
else:
    print("\n❌ NOT reproducible — see diffs above.")
    sys.exit(1)
