# gguf_state.py
"""Rehydrate act-replay trainer state from an ml8 GGUF (exact inverse of ml8_to_gguf packing)."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))
from ml8_to_gguf import QK_ML8, ML8_BLOCK_BYTES, _FP8_GROUP_SIZE, _FP8_BLOCK_BYTES

def unpack_ml8_blocks(packed, N, K):
    """packed [N, n_g*36] uint8 -> (indices long [N,K], scales fp32 [N,K//64])."""
    n_g = K // QK_ML8
    blocks = np.ascontiguousarray(packed).reshape(N, n_g, ML8_BLOCK_BYTES)
    scales = blocks[:, :, :4].copy().view('<f4').reshape(N, n_g)
    qs = blocks[:, :, 4:]
    idx = np.empty((N, n_g, QK_ML8), dtype=np.uint8)
    idx[:, :, 0::2] = qs & 0x0F
    idx[:, :, 1::2] = qs >> 4
    return (torch.from_numpy(idx.reshape(N, K).astype(np.int64)),
            torch.from_numpy(scales.astype(np.float32)))

def unpack_scaled_fp8_blocks(packed, N, K):
    """packed [N, n_b*34] uint8 -> (e4m3 fp32 [N,K], scale fp16 [N,K//32])."""
    n_b = K // _FP8_GROUP_SIZE
    blocks = np.ascontiguousarray(packed).reshape(N, n_b, _FP8_BLOCK_BYTES)
    scale = torch.from_numpy(blocks[:, :, :2].copy()).view(torch.float16).reshape(N, n_b)
    qs = torch.from_numpy(blocks[:, :, 2:].copy()).view(torch.float8_e4m3fn)
    return qs.to(torch.float32).reshape(N, K), scale

def decode_centroids_fp8(cent_u8):
    return torch.from_numpy(np.ascontiguousarray(cent_u8)).view(torch.float8_e4m3fn).to(torch.float32)
