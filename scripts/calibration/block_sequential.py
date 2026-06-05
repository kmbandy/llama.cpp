"""Block-sequential GPTQ: AutoGPTQ-style catcher/replay walk for the dense ml8 path."""
import os
import torch

class _StopForward(Exception):
    pass


def _memlog(tag):
    """Env-gated GPU memory trace (BLOCKSEQ_MEMLOG=1) for diagnosing the walk's peak VRAM.
    BLOCKSEQ_MEMLOG=2 also dumps a live-CUDA-tensor census (aggregate bytes by shape)."""
    lvl = os.environ.get("BLOCKSEQ_MEMLOG")
    if not lvl or not torch.cuda.is_available():
        return
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    m = torch.cuda.max_memory_allocated() / 1e9
    print(f"[blockseq-mem] {tag}: allocated={a:.2f}GB reserved={r:.2f}GB max={m:.2f}GB", flush=True)
    if lvl == "2":
        import gc, collections
        agg = collections.defaultdict(lambda: [0, 0])  # (shape,dtype) -> [count, bytes]
        for o in gc.get_objects():
            try:
                if torch.is_tensor(o) and o.is_cuda:
                    key = (tuple(o.shape), str(o.dtype))
                    agg[key][0] += 1
                    agg[key][1] += o.element_size() * o.nelement()
            except Exception:
                pass
        top = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)[:8]
        for (shape, dt), (cnt, byt) in top:
            print(f"    [census] {byt/1e9:6.3f}GB  x{cnt:<5d} {dt:14s} {shape}", flush=True)

@torch.no_grad()
def capture_block_inputs(model, block0, calib, device):
    """Run each calib sample up to block0, capturing (args, kwargs) into block0,
    then abort the rest of the forward (sentinel). Returns list[(args, kwargs)]."""
    inps = []
    def hook(module, args, kwargs):
        if os.environ.get("BLOCKSEQ_MEMLOG") and not inps:
            print(f"[blockseq-kwargs] args={[type(a).__name__ + (str(tuple(a.shape)) if torch.is_tensor(a) else '') for a in args]}", flush=True)
            for k, v in kwargs.items():
                desc = str(tuple(v.shape)) if torch.is_tensor(v) else type(v).__name__
                print(f"[blockseq-kwargs]   {k} = {desc}", flush=True)
        kw = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        # Stateless walk: each block is re-forwarded over the FULL captured sequence, so a
        # saved cache is never reused — but a retained DynamicCache accumulates one recurrent
        # state per (sample, block) across the walk → peak VRAM ∝ tokens·blocks (OOM at ≥512k).
        # Dropping it is numerically identical for a full-sequence forward (the cache only
        # carries state to FUTURE incremental tokens, which we never decode). Gated bit-identical.
        if not os.environ.get("BLOCKSEQ_KEEP_CACHE"):  # toggle off only for the A/B exactness gate
            if "use_cache" in kw:
                kw["use_cache"] = False
            if "past_key_values" in kw:
                kw["past_key_values"] = None
        inps.append((tuple(a.detach() if torch.is_tensor(a) else a for a in args), kw))
        raise _StopForward
    h = block0.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for ids in calib:
            try:
                model(ids.to(device) if torch.is_tensor(ids) else ids)
            except _StopForward:
                pass
    finally:
        h.remove()
    return inps


@torch.no_grad()
def run_walk(model, adapter, calib, device, is_ml8, quantize_fn, hook_factory):
    """Catcher/replay block-sequential walk.

    is_ml8(full_dotted_name)->bool   : tier map from the driver.
    quantize_fn(name, layer, idx, H, n_tok, sum_abs, rotation_hook) : per-target quantize
                                       (driver passes a quantize_one_target closure).
    hook_factory()                   : builds a per-linear Hessian hook exposing
                                       set_hessian_target/reset_hessian/.H/.n_tokens,
                                       .install(block, leaf_name), .remove(), .rotation.
    Returns the number of ML8 targets quantized.
    """
    from faithful_forward import collect_block_hessians
    blocks = adapter.iter_blocks(model)
    inps = capture_block_inputs(model, blocks[0], calib, device)
    _memlog(f"after capture (n_inps={len(inps)})")
    global_idx = 0
    for b_idx, block in enumerate(blocks):
        _memlog(f"block {b_idx} start")
        groups = adapter.ml8_targets(block, b_idx, is_ml8)
        leaf_to_mod = dict(block.named_modules())
        prefix = _block_prefix(model, b_idx)   # e.g. "model.layers.3."
        for grp in groups:
            # (a) collect H for this sub-group against the CURRENT (quantized-upstream
            #     + quantized-earlier-subgroup) block state. Reuse Task 2's collector.
            hooks = {}
            for leaf in grp.names:
                hk = hook_factory(); hk.install(block, leaf); hooks[leaf] = hk
            Hs = collect_block_hessians(block, hooks, inps, adapter.run_block)
            _memlog(f"block {b_idx} after collect_H (grp={grp.names})")
            # (b) quantize each target in the sub-group (writeback => intra-block causal)
            for leaf in grp.names:
                H, n_tok = Hs[leaf]
                quantize_fn(prefix + leaf, leaf_to_mod[leaf], global_idx,
                            H, n_tok, None, hooks[leaf].rotation)
                global_idx += 1
                hooks[leaf].remove()
        # (c) propagate: re-forward the fully-quantized block to build next inputs
        nxt = []
        for args, kwargs in inps:
            out, nkw = adapter.run_block(block, args, kwargs)
            if not torch.isfinite(out).all():
                raise RuntimeError(f"block-sequential: non-finite activations after block {b_idx}")
            nxt.append(((out.detach(),), nkw))
        inps = nxt
        _memlog(f"block {b_idx} after propagate")
    return global_idx


def _block_prefix(model, b_idx):
    # qwen35 / standard HF decoder: decoder layers live at model.model.layers, so the
    # dotted name of a linear inside block b is "model.layers.{b}.{leaf}". This matches
    # the driver's target enumeration (find_dense_full_targets uses the same prefix).
    return f"model.layers.{b_idx}."
