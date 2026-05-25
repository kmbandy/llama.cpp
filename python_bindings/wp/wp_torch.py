"""wp_torch — PyTorch integration on top of wp_native.

`ensure_as_torch(pager, page_idx, shape, dtype)` is the safe copy-out path: it
ensures the page is resident, allocates a fresh torch CUDA tensor, and memcpys
the slot bytes into it. Returns a fully-owned torch tensor; the pager's slot
is free to evict immediately. This is the right path until MAD-231's slot-pin
support lands; then a zero-copy view variant can replace it.
"""
from __future__ import annotations

from typing import Sequence

import torch

import wp_native


# GGUF/numpy → torch dtype map. Extend as needed when new ggml types appear.
_GGML_TYPE_TO_TORCH = {
    "F32":  torch.float32,
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "I8":   torch.int8,
    "I16":  torch.int16,
    "I32":  torch.int32,
    "I64":  torch.int64,
}


def gguf_type_to_torch(gguf_type) -> torch.dtype:
    """Map a gguf-py tensor_type to a torch dtype. Raises for unsupported types."""
    name = gguf_type.name if hasattr(gguf_type, "name") else str(gguf_type)
    if name not in _GGML_TYPE_TO_TORCH:
        raise NotImplementedError(
            f"gguf_type_to_torch: no torch mapping for ggml type {name!r}. "
            f"Likely a quantized type (Q4_K / Q6_K / etc) — those need to be "
            f"dequantized on the C++ side before exposure to torch."
        )
    return _GGML_TYPE_TO_TORCH[name]


def ensure_as_torch(
    pager: "wp_native.WeightPager",
    page_idx: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device_idx: int = 0,
) -> torch.Tensor:
    """Page the tensor into VRAM and return a torch CUDA tensor of (shape, dtype).

    Allocates a fresh CUDA tensor and memcpys from the pager slot. The slot is
    free to be LRU-evicted after this returns; the returned tensor owns its data.

    Raises if ensure() returns null (page out of range or pager error).
    """
    src_ptr = pager.ensure(page_idx)
    if src_ptr == 0:
        meta_name = pager.page_meta(page_idx).tensor_name if 0 <= page_idx < pager.n_pages() else "<oob>"
        raise RuntimeError(f"WeightPager.ensure({page_idx}={meta_name!r}) returned null")

    out = torch.empty(tuple(shape), dtype=dtype, device=f"cuda:{device_idx}")
    nbytes = out.element_size() * out.numel()
    wp_native.device_memcpy(out.data_ptr(), src_ptr, nbytes)
    return out


__all__ = ["ensure_as_torch", "gguf_type_to_torch"]
