#!/usr/bin/env python3
"""Tests for wp_native pybind11 bindings (MAD-238 iteration 1).

Iteration 1 scope: catalog-only operations. No GPU init, no ensure().
Iteration 2 will add init/ensure + a GGUF-based catalog populator + a
PyTorch wrapper for the VRAM pointer.
"""

import sys
from pathlib import Path

# Make sure the freshly-built .so is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp_native


def test_module_imports():
    """wp_native imports without DSO link errors against libllama.so."""
    assert hasattr(wp_native, "WeightPager"), "WeightPager not exported"
    assert hasattr(wp_native, "Config"), "Config not exported"
    assert hasattr(wp_native, "PageMeta"), "PageMeta not exported"
    print("  PASS test_module_imports")


def test_empty_pager():
    """Fresh WeightPager has zero pages and is not initialized."""
    p = wp_native.WeightPager()
    assert p.n_pages() == 0
    assert p.max_page_size() == 0
    assert p.is_initialized() is False
    assert p.find_page("anything") == -1
    print("  PASS test_empty_pager")


def test_add_page_returns_sequential_indices():
    p = wp_native.WeightPager()
    idx0 = p.add_page("blk.0.attn_q.weight", 0, 1024, 65536)
    idx1 = p.add_page("blk.0.attn_k.weight", 0, 66560, 65536)
    idx2 = p.add_page("blk.5.ffn_down.weight", 0, 200000, 131072)
    assert idx0 == 0 and idx1 == 1 and idx2 == 2
    assert p.n_pages() == 3
    assert p.max_page_size() == 131072
    print("  PASS test_add_page_returns_sequential_indices")


def test_find_page_round_trip():
    p = wp_native.WeightPager()
    p.add_page("blk.0.attn_q.weight", 0, 0, 1024)
    p.add_page("blk.0.attn_k.weight", 0, 1024, 1024)
    assert p.find_page("blk.0.attn_q.weight") == 0
    assert p.find_page("blk.0.attn_k.weight") == 1
    assert p.find_page("does.not.exist") == -1
    print("  PASS test_find_page_round_trip")


def test_page_meta_parses_block_idx():
    """The C++ PageCatalog::add() parses block_idx from 'blk.N.' prefix —
    we verify the parse is reachable from Python via page_meta()."""
    p = wp_native.WeightPager()
    p.add_page("token_embd.weight", 0, 0, 4096)         # non-block
    p.add_page("blk.3.attn_q.weight", 0, 0, 1024)        # block 3
    p.add_page("blk.17.ffn_down.weight", 0, 0, 2048)     # block 17

    m_emb = p.page_meta(0)
    assert m_emb.tensor_name == "token_embd.weight"
    assert m_emb.block_idx == -1   # non-block tensor

    m_3 = p.page_meta(1)
    assert m_3.tensor_name == "blk.3.attn_q.weight"
    assert m_3.block_idx == 3

    m_17 = p.page_meta(2)
    assert m_17.block_idx == 17
    print("  PASS test_page_meta_parses_block_idx")


def test_config_struct():
    """Config struct is readable + writable from Python."""
    cfg = wp_native.Config()
    assert cfg.n_slots == 0
    assert cfg.prefetch_depth == 4
    assert cfg.prefer_async_io is True

    cfg.n_slots = 200
    cfg.prefetch_depth = 8
    cfg.prefer_async_io = False
    assert cfg.n_slots == 200 and cfg.prefetch_depth == 8 and cfg.prefer_async_io is False
    print("  PASS test_config_struct")


def test_consolidated_moe_expert_subpages():
    """When n_experts > 1, the catalog creates a parent + N sub-pages."""
    p = wp_native.WeightPager()
    # 8 experts, total 1 MB → each expert sub-page = 128 KB at offsets 0, 128k, 256k, ...
    p.add_page("blk.0.ffn_down_exps.weight", 0, 0, 1024 * 1024, n_experts=8)
    # Expect: 1 parent + 8 sub-pages = 9 total
    assert p.n_pages() == 9, f"got {p.n_pages()}"
    # Parent at idx 0
    parent = p.page_meta(0)
    assert parent.is_consolidated, "parent should be is_consolidated"
    assert parent.is_sub_expert is False
    # First sub-expert at idx 1
    sub = p.page_meta(1)
    assert sub.is_sub_expert
    assert sub.parent_page_idx == 0
    assert sub.expert_idx == 0
    print("  PASS test_consolidated_moe_expert_subpages")


def _populate_catalog_from_gguf(pager, gguf_path: str) -> int:
    """Helper: read GGUF metadata and add each tensor to the pager catalog.

    Returns the page index for blk.0.ffn_gate.weight (a convenient handle for
    init+ensure tests). Returns -1 if not found.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
    import gguf as gguf_lib  # type: ignore
    reader = gguf_lib.GGUFReader(gguf_path)
    target_idx = -1
    for t in reader.tensors:
        page_idx = pager.add_page(
            name=t.name,
            file_idx=0,
            file_offset=int(t.data_offset),
            size=int(t.n_bytes),
            n_experts=1,  # we'd parse this from GGUF metadata for MoE
        )
        if t.name == "blk.0.ffn_gate.weight":
            target_idx = page_idx
    return target_idx


def test_init_and_ensure_on_real_gguf():
    """End-to-end: init pager on the Qwen3.5-4B f16 GGUF, ensure() a tensor, get non-zero VRAM ptr.

    GATED: requires Qwen3.5-4B-f16.gguf at ~/models/ AND a free HIP GPU.
    Skips silently if either is missing.
    """
    import os
    gguf_path = os.path.expanduser("~/models/Qwen3.5-4B-f16.gguf")
    if not os.path.exists(gguf_path):
        print("  SKIP test_init_and_ensure_on_real_gguf (no Qwen3.5-4B-f16.gguf)")
        return

    p = wp_native.WeightPager()
    target_idx = _populate_catalog_from_gguf(p, gguf_path)
    assert p.n_pages() == 441, f"expected 441 tensors in Qwen3.5-4B, got {p.n_pages()}"
    assert target_idx >= 0, "blk.0.ffn_gate.weight not found in catalog"

    cfg = wp_native.Config()
    cfg.n_slots = 4         # tiny pool — pool = 4 × max_page_size (~1.27 GB for token_embd) = 5 GB
    cfg.prefetch_depth = 2
    cfg.prefer_async_io = False  # SyncPread path is simpler for first smoke

    ok = p.init_for_device(cfg, 0, [gguf_path])  # device 0 = ROCm0
    assert ok, "init_for_device returned False"
    assert p.is_initialized()

    # Page in blk.0.ffn_gate.weight
    ptr = p.ensure(target_idx)
    assert ptr != 0, f"ensure({target_idx}) returned null"
    print(f"  PASS test_init_and_ensure_on_real_gguf (ptr=0x{ptr:x})")

    p.shutdown()
    assert not p.is_initialized()


if __name__ == "__main__":
    test_module_imports()
    test_empty_pager()
    test_add_page_returns_sequential_indices()
    test_find_page_round_trip()
    test_page_meta_parses_block_idx()
    test_config_struct()
    test_consolidated_moe_expert_subpages()
    # Iteration 2: init + ensure. Requires GPU — skip if busy.
    import os as _os
    if _os.environ.get("WP_TEST_SKIP_GPU"):
        print("  SKIP test_init_and_ensure_on_real_gguf (WP_TEST_SKIP_GPU set)")
    else:
        test_init_and_ensure_on_real_gguf()
    print("\nALL TESTS PASSED")
