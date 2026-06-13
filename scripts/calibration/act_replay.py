#!/usr/bin/env python3
"""act_replay.py — act-replay KL trainer CLI (Task 6).

Ties together the act-replay modules into a runnable trainer:

  * gguf_state.load_ml8_gguf      — rehydrate ml8 + frozen fp8 trainer state
  * act_replay_student            — attach ml8 dequant-STE targets to HF linears
  * teacher_source.make_teacher   — live / cache / device:N teacher top-K
  * kl_loss.kl_topk               — forward-KL on the top-K + tail partition
  * calib_corpus.collect_calibration — calibration draw (chat-formatted)
  * ml8_io (schema) + ml8_to_gguf.convert_to_ml8_gguf — blob export + re-emit

Only the codebook centroids/scales of the selected ml8 targets train; the host
nn.Linear weights stay frozen. Output dirs/GGUFs go under user-provided paths,
NEVER /tmp.

Run the CLI tests from scripts/calibration with PYTHONPATH=../../gguf-py.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

from act_replay_student import attach_to_linear, select_targets
from centroid_quantizer import snap_to_e4m3
from fp8_qat import Ml8Fp8Fn
from index_reassign import index_reassign
from kl_loss import kl_topk


# Optimizer steps between torch.cuda.empty_cache() drains in the train loop
# (0 disables). See the train-loop comment: caps the caching allocator's
# reserved-VRAM ratchet now that expandable_segments is banned on gfx1201.
_EMPTY_CACHE_EVERY = int(os.environ.get("ML8_EMPTY_CACHE_EVERY", "10"))


def _trim_host():
    """Return freed glibc arenas to the OS (gc first so refcounts drop).

    CPython/glibc retain the allocation high-water mark: streaming a 4B model's
    safetensors shards through host buffers leaves multi-GB of freed-but-held
    arena RSS (measured 2026-06-10: 9GB RSS over a ~3GB live working set during
    the teacher-load phase on the 15GB host). malloc_trim(0) actually gives the
    memory back; called at phase boundaries, never in the hot loop.
    """
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass  # non-glibc platform — trimming is best-effort


def lr_warmup_cosine(step, warmup, total):
    """LR multiplier in [0,1]: linear ramp to 1.0 over `warmup` steps, then
    cosine decay to 0.0 at `total`. step is 1-based."""
    if warmup > 0 and step <= warmup:
        return step / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _apply_lr_schedule(optimizer, base_lrs, step, warmup, total):
    """Set each param-group lr = base_lr * lr_warmup_cosine(step, warmup, total).
    Returns the multiplier. base_lrs is the per-group starting lrs (same order)."""
    m = lr_warmup_cosine(step, warmup, total)
    for g, blr in zip(optimizer.param_groups, base_lrs):
        g["lr"] = blr * m
    return m


def reassign_targets(targets, mode, frac=0.1):
    """Axis-B discrete index reassignment over AttachedTargets, in place.

    mode 'none' -> no-op (returns 0). 'mse' re-solves indices vs each target's
    W_orig anchor using current snapped centroids/scales. 'pv' uses the loss
    gradient dL/dW_raw and the diagonal curvature h stashed in
    Ml8Fp8Fn.last_dLdW / last_h (keyed by id(at.indices)) from the most recent
    backward; targets without a stashed grad+curvature are skipped.
    Returns the total number of elements changed (mse counts changed entries).
    """
    if mode == "none":
        return 0
    total = 0
    for at in targets:
        cent = snap_to_e4m3(at.centroids).detach()
        scl = at.scales.detach()
        dLdW = Ml8Fp8Fn.last_dLdW.get(id(at.indices))
        h = Ml8Fp8Fn.last_h.get(id(at.indices))
        if mode == "pv" and (dLdW is None or h is None):
            continue
        new_idx, n = index_reassign(at.indices, mode, at.W_orig, dLdW, h,
                                    cent, scl, at.gidx, frac=frac)
        changed = int((new_idx != at.indices).sum().item()) if n < 0 else int(n)
        at.indices.copy_(new_idx.to(at.indices.dtype))
        total += changed
    return total


def collect_target_hessians(targets, calib, model, dev):
    """Per-target static activation Hessian H = (1/N) sum Xrot^T Xrot over the
    calib windows, where Xrot = x @ Q is the ROTATED (faithful) activation the
    ml8 weights consume (NOT the raw linear input). Returns {name: H[K,K] fp32}.

    One forward pass over `calib`; all targets accumulate simultaneously.

    CRITICAL: the Hessian is in the ROTATED basis (x @ Q, post-rotation pre-quant)
    because the ml8 weight indices are solved in that rotated basis. Capturing raw
    x would produce a silently wrong-space H — the whole point of this function.

    Memory: each target holds a live [K, K] float32 accumulator (~K²·4 bytes) for
    the whole pass — ~196 MB at K=7168. For large K or many simultaneous targets,
    collect in target subgroups rather than all at once.
    """
    for at in targets.values():
        at.start_hessian_collection()
    model.eval()
    with torch.no_grad():
        for ids in calib:
            model(ids.to(dev))
    return {name: at.finalize_hessian() for name, at in targets.items()}


def gptq_reassign_targets(targets, H_by_name, *, percdamp=0.05, act_order=True):
    """Axis B (full-H GPTQ): re-solve each target's indices vs its CURRENT
    (Axis-A-tuned, e4m3-snapped) centroids using the per-target rotated Hessian.
    Builds E=1 stacks and calls batched_gptq_reassign. Targets without a stashed H
    are skipped. Returns the total number of index entries changed.

    W_orig, the rotated Hessian H, and the centroids all live in the SAME rotated
    basis — no rotation handling is needed here. Reassigns against the SNAPPED
    centroids (the actual e4m3 LUT values the forward/kernel uses), matching the
    convention in reassign_targets.
    """
    from batched_gptq import batched_gptq_reassign

    total = 0
    for name, at in targets.items():
        H = H_by_name.get(name)
        if H is None:
            continue
        K = at.indices.shape[1]
        n_groups = at.centroids.shape[0]
        group_size = K // n_groups
        W = at.W_orig.unsqueeze(0).float()                        # [1, N, K]
        Hs = H.to(W.device).unsqueeze(0).float()                  # [1, K, K]
        cents = snap_to_e4m3(at.centroids).detach().unsqueeze(0)  # [1, n_groups, NC]
        scl = at.scales.detach().unsqueeze(0)                     # [1, N, n_groups]
        new_idx = batched_gptq_reassign(W, Hs, cents, scl,
                                        group_size=group_size,
                                        percdamp=percdamp,
                                        act_order=act_order)[0]   # [N, K] int8
        new_idx = new_idx.to(at.indices.dtype)
        changed = int((new_idx != at.indices).sum().item())
        at.indices.copy_(new_idx)
        total += changed
    return total


# ─── env-gated host-RSS phase accounting ─────────────────────────────────────


def _rss_gb():
    """Resident set size in GB from /proc/self/status VmRSS (no psutil dep)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # "VmRSS:\t   12345 kB"
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("nan")


def _memlog(phase):
    """Env-gated host-RSS phase trace (ACT_REPLAY_MEMLOG=1) for localizing the
    act-replay trainer's anonymous-RAM growth. Mirrors block_sequential._memlog,
    but tracks HOST RSS (the leak we're chasing is on the 15GB host, not VRAM)."""
    if not os.environ.get("ACT_REPLAY_MEMLOG"):
        return
    vram = ""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        vram = (f" vram={torch.cuda.memory_allocated()/1e9:.2f}GB"
                f" peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"[memlog] {phase} rss={_rss_gb():.2f}GB{vram}", flush=True)


def _tensor_census(phase, top_n=10):
    """Live-tensor census (ACT_REPLAY_MEMLOG=2): aggregate every torch tensor
    reachable from the GC by (shape,dtype,device), report the top_n by bytes.
    Same pattern as block_sequential's BLOCKSEQ_MEMLOG=2 CUDA census, but covers
    ALL devices (the host leak is CPU tensors), so we can see retained dequants /
    fp32 KL chains hanging off the graph after a train step."""
    if os.environ.get("ACT_REPLAY_MEMLOG") != "2":
        return
    import gc
    import collections
    agg = collections.defaultdict(lambda: [0, 0])  # (shape,dtype,dev) -> [count, bytes]
    total = 0
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o):
                key = (tuple(o.shape), str(o.dtype), str(o.device))
                byt = o.element_size() * o.nelement()
                agg[key][0] += 1
                agg[key][1] += byt
                total += byt
        except Exception:
            pass
    top = sorted(agg.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]
    print(f"[memlog] {phase} tensor-census total={total/1e9:.3f}GB "
          f"across {len(agg)} (shape,dtype,device) groups", flush=True)
    for (shape, dt, dev), (cnt, byt) in top:
        print(f"    [census] {byt/1e9:7.3f}GB  x{cnt:<5d} {dt:14s} {dev:8s} {shape}",
              flush=True)


# ─── HF model loading + LM wrapper ───────────────────────────────────────────


class _LMWrap(torch.nn.Module):
    """Wrap an HF causal-LM so callers (teacher_source / kl) get a plain logits
    tensor. The forward moves ids onto the wrapped model's device first."""

    def __init__(self, m, device=None):
        super().__init__()
        self.m = m
        if device is None:
            try:
                device = next(m.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

    def forward(self, ids):
        return self.m(ids.to(self.device)).logits


def load_hf_model(path, device, *, grad_ckpt=False, freeze=False):
    """Load a fresh HF causal-LM from `path` onto `device`.

    A NEW from_pretrained instance every call — the student and the (frozen
    bf16 parent) teacher must be SEPARATE objects, never the same monkeypatched
    model. Applies the RDNA fp32 linear-attn scan shim (no-op off RDNA/CPU).

    grad_ckpt: enable non-reentrant gradient checkpointing (student path).
    freeze:    eval() + requires_grad_(False) — the frozen teacher parent.
    """
    from transformers import AutoModelForCausalLM
    import fla_compat

    device = torch.device(device)
    # device_map streams each bf16 weight from the checkpoint mmap STRAIGHT to
    # the target device — never assembling the full model in host RAM. The 15GB
    # host OOMs if the 4B bf16 parent (8.3GB) is staged on CPU before .to():
    # measured 11G cgroup kill with low_cpu_mem_usage + .to(device) alone.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": device}, attn_implementation="sdpa").eval()

    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    fla_compat.apply_fla_arch_shim(model, device)
    fla_compat.apply_fla_cpu_fallback(model, device)

    if freeze:
        model.requires_grad_(False)
    return model


# ─── arg parsing ─────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    """Parse trainer flags. argv=None reads sys.argv (CLI); pass a list in tests."""
    p = argparse.ArgumentParser(
        description="Act-replay KL trainer: fine-tune ml8 codebooks against a "
                    "live/cached teacher, then export blobs + re-emit a GGUF.")
    # required I/O
    p.add_argument("--gguf", required=True, help="ml8 GGUF to rehydrate trainer state from")
    p.add_argument("--base-gguf", required=True,
                   help="bf16/F16 base GGUF to re-emit FROM (NOT an ml8 GGUF); "
                        "convert_to_ml8_gguf streams its pass-through tensors and "
                        "overlays the trained ml8 blobs. Refused if it looks ml8.")
    p.add_argument("--model", required=True, help="HF model id/path for the student + live teacher")
    p.add_argument("--out-dir", required=True, help="output dir (blobs + GGUF re-emit); never /tmp")
    # corpus
    p.add_argument("--corpus", default="mix", help="calib_corpus composition name")
    p.add_argument("--token-budget", type=int, default=512000, help="total calibration token budget")
    p.add_argument("--seq-len", type=int, default=2048, help="per-sample sequence length")
    p.add_argument("--train-seq-len", type=int, default=None,
                   help="if set, TRAIN batches are split along the time axis into "
                        "windows of this length (response mask preserved per window) "
                        "to cap per-forward activation memory on the 4B host. The "
                        "corpus is still drawn at --seq-len and EVAL keeps the full "
                        "sequence. Default None = train at the full --seq-len.")
    # teacher
    p.add_argument("--teacher", default="live", help="teacher source spec: live | cache | device:N")
    p.add_argument("--teacher-cache-dir", default=None, help="dir for the cache teacher's shards (never /tmp)")
    p.add_argument("--topk", type=int, default=256, help="teacher top-K width for the KL partition")
    # optimization
    p.add_argument("--lr-cent", type=float, default=2e-4, help="lr for centroid params")
    p.add_argument("--lr-scale", type=float, default=2e-5, help="lr for per-row scale params")
    p.add_argument("--lr-warmup-steps", type=int, default=0,
                   help="linear lr warmup steps before cosine decay")
    p.add_argument("--grad-accum", type=int, default=8, help="grad-accumulation steps per optimizer step")
    p.add_argument("--micro-batch", type=int, default=1, help="micro-batch size (sequences per forward)")
    # fp8 engine + discrete index reassignment
    p.add_argument("--fp8", action="store_true",
                   help="train through the fp8 forward+backward engine (Ml8Fp8Fn)")
    p.add_argument("--loss-scale", type=float, default=1.0,
                   help="loss scale for fp8 backward dy (e5m2) dynamic range")
    p.add_argument("--reassign", default="none", choices=["none", "mse", "pv"],
                   help="discrete index reassignment mode (Axis B)")
    p.add_argument("--reassign-interval", type=int, default=50,
                   help="optimizer steps between index reassignments")
    p.add_argument("--reassign-frac", type=float, default=0.1,
                   help="pv-mode: fraction of elements to flip per reassign")
    # target selection
    p.add_argument("--tensors-train", default="ml8",
                   help="'ml8' (all ml8 tensors) or comma-separated fnmatch globs")
    p.add_argument("--tensors-skip", default="", help="comma-separated fnmatch globs to drop")
    # schedule
    p.add_argument("--steps", type=int, default=None, help="hard cap on optimizer steps (None = full epochs)")
    p.add_argument("--epochs", type=int, default=1, help="passes over the train split")
    p.add_argument("--eval-interval", type=int, default=200, help="optimizer steps between holdout evals")
    p.add_argument("--seed", type=int, default=0, help="seed for the holdout split + corpus draw")
    # ckpt / device
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--device", default="cuda:0", help="device for student + live teacher")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (default: enabled, non-reentrant)")
    return p.parse_args(argv)


# ─── deterministic holdout split ─────────────────────────────────────────────


def split_holdout(n: int, frac: float, seed: int):
    """Deterministic (train_idx, hold_idx) split of range(n).

    Uses a seeded torch.Generator permutation so the same (n, frac, seed) always
    yields the same partition; different seeds yield different partitions. The
    last floor(n*frac) of the permutation are held out.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_hold = int(n * frac)
    hold_idx = perm[n - n_hold:].clone() if n_hold > 0 else perm[n:].clone()
    train_idx = perm[: n - n_hold].clone()
    # sort each side so callers iterate in stable batch order
    return torch.sort(train_idx).values, torch.sort(hold_idx).values


def split_batches_seq(batches, train_idx, train_seq_len):
    """Split TRAIN batches along the time axis into windows of `train_seq_len`.

    The 4B-host memory knob: the corpus is drawn at the full --seq-len (eval needs
    it), but a full-length forward + KL retains too much activation memory at 4B.
    Splitting each TRAIN sequence into shorter windows caps the per-forward cost
    while keeping every token (and thus the response mask, which is recomputed from
    the windowed ids downstream) — windows just tile the original sequence.

    Args:
        batches: list of [1, T] (or [T]) LongTensors — the full corpus draw.
        train_idx: 1-D LongTensor of batch indices to train on.
        train_seq_len: window length along T. None / <= 0 / >= T disables splitting
                       for a given batch (that batch passes through whole).

    Returns:
        (win_batches, win_train_idx):
          win_batches  — new list of windowed [1, t] tensors (t <= train_seq_len),
                         a tail window shorter than train_seq_len is kept (not
                         dropped) so no tokens are lost.
          win_train_idx — LongTensor indexing win_batches, in train_idx order.

    Eval is unaffected: callers keep iterating the ORIGINAL `batches` for holdout.
    """
    if train_seq_len is None or train_seq_len <= 0:
        return list(batches), train_idx.clone()
    win_batches = []
    win_idx = []
    for i in train_idx.tolist():
        ids = batches[i]
        twod = ids.dim() == 2
        T = ids.shape[-1]
        if T <= train_seq_len:
            win_idx.append(len(win_batches))
            win_batches.append(ids)
            continue
        for start in range(0, T, train_seq_len):
            stop = min(start + train_seq_len, T)
            window = ids[:, start:stop] if twod else ids[start:stop]
            win_idx.append(len(win_batches))
            win_batches.append(window)
    return win_batches, torch.tensor(win_idx, dtype=torch.long)


# ─── response-token masking ──────────────────────────────────────────────────


def _find_subseq(hay, needle, start=0):
    """Index of the first occurrence of list `needle` in list `hay` at/after
    `start`, or -1. Plain O(n*m) scan — needles are short delimiter sequences."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    for i in range(start, n - m + 1):
        if hay[i:i + m] == needle:
            return i
    return -1


def build_response_mask(ids, start_seq, end_seq):
    """Per-token assistant-response mask for one sequence of token ids.

    Design ("Loss = KL over response tokens"): we only want the KL to count the
    model's *assistant* output, not the prompt/system/user scaffolding. The chat
    template renders assistant turns as ``<start_seq> … <end_seq>`` (e.g.
    ``<|im_start|>assistant`` … ``<|im_end|>``); `start_seq`/`end_seq` are those
    delimiters already tokenized to id lists.

    Marks tokens strictly INSIDE each assistant span = 1, else 0:
      * the start delimiter tokens themselves are 0 (mask begins right after them);
      * the end delimiter is exclusive — tokens up to (not including) `end_seq` are
        1, and the end delimiter tokens are 0.
    Spans that open but never close run to the end of the sequence. If no start
    delimiter occurs at all the mask is all-zeros — the caller decides the
    all-ones raw-text fallback (so this stays a pure, side-effect-free function).

    Pure tensor function: `ids` is a 1-D LongTensor (or list); returns a 1-D
    float mask of the same length. Fully CPU-testable with a stub tokenizer.
    """
    seq = ids.tolist() if torch.is_tensor(ids) else list(ids)
    n = len(seq)
    mask = torch.zeros(n, dtype=torch.float32)
    if not start_seq:
        return mask
    s = list(start_seq)
    e = list(end_seq) if end_seq else []
    pos = 0
    while pos < n:
        a = _find_subseq(seq, s, pos)
        if a < 0:
            break
        inner = a + len(s)                       # first token after the start delim
        if e:
            b = _find_subseq(seq, e, inner)
            if b < 0:
                mask[inner:] = 1.0               # unclosed span → to end of sequence
                break
            mask[inner:b] = 1.0                  # end delim exclusive
            pos = b + len(e)
        else:
            mask[inner:] = 1.0                   # no end delim → rest of sequence
            break
    return mask


def assistant_delimiters(tokenizer):
    """Tokenize the model's assistant-turn open/close delimiters from its chat
    template. Returns (start_ids, end_ids) as plain int lists.

    Renders a tiny two-turn conversation through the tokenizer's own chat template
    and reads back the literal text it inserts around the assistant content, so
    this needs ZERO per-model delimiter strings (Qwen ``<|im_start|>assistant`` /
    ``<|im_end|>``, Llama headers, Gemma ``<start_of_turn>model`` …). Falls back to
    the common ChatML pair if the template can't be introspected.
    """
    SENTINEL = "⁣RESP⁣"               # invisible separator, unlikely to tokenize-merge
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"},
             {"role": "assistant", "content": SENTINEL}],
            tokenize=False, add_generation_prompt=False)
        head, _, tail = rendered.partition(SENTINEL)
        # start delim = template text from the last newline before the sentinel
        # (the "<|im_start|>assistant\n" header); end delim = text right after it.
        start_txt = head[head.rfind("<") :] if "<" in head else head[-32:]
        end_txt = tail[: tail.find(">") + 1] if ">" in tail else tail[:32]
        start_ids = tokenizer(start_txt, add_special_tokens=False)["input_ids"]
        end_ids = tokenizer(end_txt, add_special_tokens=False)["input_ids"]
        if start_ids and end_ids:
            return list(start_ids), list(end_ids)
    except Exception:
        pass
    # ChatML fallback.
    return (tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"],
            tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"])


def batch_response_mask(ids, start_seq, end_seq, _warned=[]):
    """Build a [*ids.shape] float mask for one batch's ids via build_response_mask.

    ids: a [1, T] (or [T]) LongTensor. If the batch has no assistant span (e.g. a
    raw wiki-text record with no chat delimiters) the mask falls back to ALL-ONES
    so the KL still trains on it; that fallback is logged exactly once.
    """
    flat = ids.reshape(-1)
    m = build_response_mask(flat, start_seq, end_seq)
    if m.sum() == 0:
        if not _warned:
            print("[act-replay] no assistant span in a batch — falling back to "
                  "all-ones response mask (raw-text record). Logged once.", flush=True)
            _warned.append(True)
        m = torch.ones_like(m)
    return m.to(ids.device).reshape(ids.shape if ids.dim() > 1 else (-1,))


# ─── GGUF -> HF name mapping ─────────────────────────────────────────────────

# Dense / attention / hybrid-linear-attn GGUF tensor stems -> HF module path
# (sans trailing ".weight"). Unknown stems raise KeyError so a typo or an
# unsupported module fails loudly rather than silently skipping a target.
#
# The linear-attn (qwen35 gated-delta-net) entries below mirror role_targets'
# authoritative TensorNameMap resolution. Only the ML8-tier *2D matmul* linears
# of a linear-attn block are mapped — in_proj_qkv (attn_qkv), in_proj_z
# (attn_gate) and out_proj (ssm_out). The block's other tensors are intentionally
# omitted: in_proj_a/in_proj_b (ssm_alpha/ssm_beta) are FP8 (frozen, never trained
# as ml8 targets), and conv1d/dt/A_log/norm (ssm_conv1d/ssm_dt/ssm_a/ssm_norm) are
# NATIVE SSM-core tensors, not nn.Linear matmul weights. An unmapped linear-attn
# stem therefore raises KeyError rather than silently attaching the wrong module.
_GGUF_STEM_TO_HF = {
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    # hybrid (qwen35) linear-attn 2D matmul targets
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_out": "linear_attn.out_proj",
    # FP8-tier linear-attn projections (frozen, never trained as ml8 targets):
    # ssm_alpha/ssm_beta are the in_proj_a/in_proj_b gates. These never appear in
    # `targets` (only ml8 tiers attach), but they DO surface in the frozen-fp8
    # re-emit map, so map them here so export_blobs writes a `name` the converter's
    # classify_role resolves to Tier.FP8 (rather than a GGUF-name fallback the
    # converter rejects).
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
}


def map_gguf_to_hf(gguf_name: str) -> str:
    """Map a GGUF tensor name to its HF module path.

    blk.N.<stem>.weight -> model.layers.N.<hf_suffix>
    token_embd.weight    -> model.embed_tokens
    Unknown names raise KeyError.
    """
    name = gguf_name
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    if name == "token_embd":
        return "model.embed_tokens"
    parts = name.split(".")
    if len(parts) == 3 and parts[0] == "blk":
        layer = parts[1]
        stem = parts[2]
        if stem in _GGUF_STEM_TO_HF:
            return f"model.layers.{layer}.{_GGUF_STEM_TO_HF[stem]}"
    # MTP / NextN draft head: blk.{L}.nextn.eh_proj.weight -> model.layers.{L}.nextn.eh_proj
    # (4-part GGUF name; classify_role keys eh_proj off the HF leaf, not the GGUF one.)
    if len(parts) == 4 and parts[0] == "blk" and parts[2] == "nextn" and parts[3] == "eh_proj":
        return f"model.layers.{parts[1]}.nextn.eh_proj"
    raise KeyError(f"no HF mapping for GGUF tensor {gguf_name!r}")


# ─── GGUF -> HF linear-attn V-head reorder inversion ─────────────────────────
#
# convert_hf_to_gguf.py (_LinearAttentionVReorderBase in conversion/qwen.py)
# reorders the linear-attn V heads from HF's GROUPED layout
# ([G0_v0..v{r-1}, G1_v0..v{r-1}, ...]) to ggml's TILED layout
# ([G0_v0, G1_v0, ..., G0_v1, G1_v1, ...]) so ggml_repeat can broadcast K over V.
# When num_value_heads == num_key_heads (r == 1, e.g. the 0.8B model) the reorder
# is the identity; for r > 1 (4B/9B/27B) it is a real permutation.
#
# The act-replay trainer rehydrates GGUF-order weights and attaches them onto HF
# modules whose downstream consumers (SSM core, gates, out_proj) expect the
# GROUPED order. We must therefore apply the INVERSE reorder (tiled -> grouped)
# along the OUTPUT-channel axis (rows) for in_proj_qkv (V rows only), in_proj_z,
# in_proj_a, in_proj_b, and along the INPUT axis (columns) for out_proj.
#
# This is a pure index reorder of output (or input) channels — no arithmetic on
# the weight values — so it commutes with both the ml8 dequant (reorder the
# `indices`/`scales` rows together) and the fp8 dequant (reorder the rows of the
# materialized weight). The Kronecker rotation carried by the ml8 targets is an
# INPUT-space transform handled separately by AttachedTarget.apply_acts and is
# orthogonal to this output-channel reorder.

# GGUF stem -> (axis, head_dim_key) for the V-reorder inversion.
#   axis 0  -> reorder output rows; axis 1 -> reorder input columns.
#   head_dim_key picks which HF head dim sizes the reordered block:
#     "v"  -> linear_value_head_dim;  "1" -> scalar per-head (head_dim == 1).
#   "qkv" is special-cased: only the trailing V rows are reordered.
_GGUF_STEM_V_REORDER = {
    "attn_qkv": ("qkv", "v"),   # in_proj_qkv: reorder ONLY the V rows
    "attn_gate": (0, "v"),      # in_proj_z:   reorder all rows
    "ssm_alpha": (0, "1"),      # in_proj_a:   reorder rows (head_dim == 1)
    "ssm_beta": (0, "1"),       # in_proj_b:   reorder rows (head_dim == 1)
    "ssm_out": (1, "v"),        # out_proj:    reorder input columns
}


def _linear_attn_head_dims(model_config):
    """Pull (num_k_heads, num_v_heads, head_v_dim, head_k_dim) from an HF config.

    Qwen3.5 nests these under config.text_config; fall back to the top level for
    a flat text-only config. Returns None if the model is not a linear-attn
    variant (no linear_num_value_heads) — callers then treat every perm as
    identity.
    """
    cfg = getattr(model_config, "text_config", None) or model_config

    def _get(key):
        if isinstance(cfg, dict):
            return cfg.get(key)
        return getattr(cfg, key, None)

    num_v = _get("linear_num_value_heads")
    num_k = _get("linear_num_key_heads")
    if not num_v or not num_k:
        return None
    head_v = _get("linear_value_head_dim")
    head_k = _get("linear_key_head_dim")
    return int(num_k), int(num_v), int(head_v), int(head_k)


def _tiled_to_grouped_index(num_k_heads, num_v_per_k, head_dim):
    """Index tensor that maps a TILED-order axis back to GROUPED order.

    The forward (HF grouped -> GGUF tiled) reorder swaps the (num_k_heads,
    num_v_per_k) axes of a [num_k_heads, num_v_per_k, head_dim] view. Building
    the forward permutation on an arange and argsort-ing it yields the inverse
    (tiled -> grouped) index. `tiled[inv]` is grouped order.
    """
    n = num_k_heads * num_v_per_k * head_dim
    arange = torch.arange(n, dtype=torch.long).reshape(
        num_k_heads, num_v_per_k, head_dim)
    fwd = arange.permute(1, 0, 2).reshape(-1)  # grouped -> tiled mapping
    return torch.argsort(fwd)                  # tiled -> grouped


def gguf_to_hf_perm(gguf_name, shape, model_config):
    """Inverse V-head reorder for one linear-attn GGUF tensor, or None.

    Returns (axis, index) such that applying ``t.index_select(axis, index)`` to a
    GGUF-order tensor of the given ``shape`` yields the HF (grouped) order the
    student module expects. Returns None when no reorder is needed: the tensor is
    not a linear-attn projection, the model is not a linear-attn variant, or the
    reorder is the identity (num_value_heads == num_key_heads, e.g. the 0.8B
    model — so existing 0.8B tests stay green).

    The Kronecker rotation on ml8 targets is NOT handled here (it is an input-space
    transform applied by AttachedTarget.apply_acts); this is purely an
    output-/input-channel index reorder.
    """
    name = gguf_name
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    parts = name.split(".")
    if len(parts) != 3 or parts[0] != "blk":
        return None
    stem = parts[2]
    spec = _GGUF_STEM_V_REORDER.get(stem)
    if spec is None:
        return None

    dims = _linear_attn_head_dims(model_config)
    if dims is None:
        return None
    num_k_heads, num_v_heads, head_v_dim, head_k_dim = dims
    if num_v_heads == num_k_heads:
        return None  # identity reorder (num_v_per_k == 1)
    num_v_per_k = num_v_heads // num_k_heads

    axis, hd_key = spec
    head_dim = head_v_dim if hd_key == "v" else 1
    inv = _tiled_to_grouped_index(num_k_heads, num_v_per_k, head_dim)

    if axis == "qkv":
        # in_proj_qkv rows are [q (k_dim) | k (k_dim) | v (v_dim)]; only V reorders.
        k_dim = head_k_dim * num_k_heads
        v_dim = num_v_heads * head_v_dim
        n_rows = shape[0]
        if n_rows != 2 * k_dim + v_dim:
            raise ValueError(
                f"{gguf_name}: rows {n_rows} != 2*k_dim({k_dim}) + v_dim({v_dim})")
        idx = torch.arange(n_rows, dtype=torch.long)
        idx[2 * k_dim:] = (2 * k_dim) + inv
        return 0, idx

    return axis, inv


def _apply_perm_to_ml8_entry(entry, perm):
    """Reorder an ml8 state entry's rows in place per a (axis, index) perm.

    Only output-row reorders (axis 0) are valid for an ml8 target: the indices
    [N,K] and scales [N,G] are reordered along their row axis together (the
    per-group centroids are untouched — they index the input/group axis). Column
    reorders (axis 1) on an ml8 target are unsupported and raise; the only
    column-reordered linear-attn tensor (out_proj) is an fp8 frozen weight, never
    an ml8 target.
    """
    if perm is None:
        return entry
    axis, index = perm
    if axis != 0:
        raise ValueError(
            "ml8 target reorder must be along output rows (axis 0); got "
            f"axis={axis}")
    entry = dict(entry)
    entry["indices"] = entry["indices"].index_select(0, index.to(entry["indices"].device))
    entry["scales"] = entry["scales"].index_select(0, index.to(entry["scales"].device))
    return entry


# ─── attach targets to an HF model ───────────────────────────────────────────


def attach_targets(model_named_modules, state, train, skip, model_config=None,
                   fp8: bool = False):
    """Attach selected ml8 targets to their host linears in an HF model.

    model_named_modules: dict-like {module_path: nn.Module} (e.g. dict(model.named_modules())).
    state: an Ml8State (or anything with a `.ml8` dict of {gguf_name: target}).
    train/skip: forwarded to select_targets.
    model_config: HF model config used to invert the linear-attn V-head reorder
        (see gguf_to_hf_perm). None disables the reorder (identity) — safe for
        non-linear-attn models and the 0.8B case.
    fp8: when True, wire the fp8 forward+backward engine for each attached target.

    Returns {gguf_name: AttachedTarget} for every attached target. Raises KeyError
    if a selected target's mapped HF module is not present in the model.
    """
    modules = dict(model_named_modules)
    selected = select_targets(list(state.ml8.keys()), train=train, skip=skip)
    return {g: _attach_one(modules, g, state.ml8[g], model_config, fp8=fp8)
            for g in selected}


def _attach_one(modules, gguf_name, entry, model_config=None, fp8: bool = False):
    """Attach ONE ml8 entry to its host linear; returns the AttachedTarget.

    Applies the linear-attn V-head reorder inversion when model_config is given
    (see gguf_to_hf_perm). Raises KeyError if the mapped HF module is missing.
    The streaming rehydrate path calls this per tensor so the host never holds
    the full trainer state; attach_to_linear moves the entry to the module's
    device, freeing the host copy as each tensor lands.
    fp8: when True, wire the fp8 forward+backward engine (Ml8Fp8Fn) via
    attach_to_linear's fp8 flag.
    """
    hf_path = map_gguf_to_hf(gguf_name)
    if hf_path not in modules:
        raise KeyError(
            f"target {gguf_name!r} -> {hf_path!r} not found in model modules")
    if model_config is not None:
        shape = tuple(entry["indices"].shape)
        perm = gguf_to_hf_perm(gguf_name, shape, model_config)
        entry = _apply_perm_to_ml8_entry(entry, perm)
    return attach_to_linear(modules[hf_path], entry, fp8=fp8)


def install_frozen_fp8(model, frozen, map_fn, device, dtype, model_config=None):
    """Install dequantized frozen fp8 weights INTO the student in-place.

    The HF student is loaded with its bf16 parent weights; for tensors that were
    quantized to ML8_FP8 we want the student to carry the FP8-faithful dequant,
    not the bf16 parent (closing the faithfulness gap). For each {gguf_name:
    weight} in `frozen`, map gguf_name -> HF module path via `map_fn` and copy the
    weight into that module's `.weight` in-place under no_grad. Unmapped names
    (KeyError from map_fn) are skipped with a single warning. Each frozen tensor
    is popped from `frozen` as it is consumed so its RAM is freed immediately
    (host has only 15GB). Returns the number of weights actually installed.

    model_config: when given, the linear-attn V-head reorder is inverted
    (see gguf_to_hf_perm) before the weight is copied in — out_proj (in_proj_a/b)
    columns/rows are reordered from GGUF (tiled) to HF (grouped) order so the
    installed weight matches the HF module's channel layout. None -> identity.
    """
    modules = dict(model.named_modules())
    n_installed = 0
    state = {"warned": False}
    for gguf_name in list(frozen.keys()):
        w = frozen.pop(gguf_name)  # free as we go
        n_installed += _install_one(modules, gguf_name, w, map_fn, device,
                                    dtype, model_config, state)
    return n_installed


def _install_one(modules, gguf_name, w, map_fn, device, dtype,
                 model_config=None, warn_state=None):
    """Install ONE frozen fp8 dequant into the student; returns 1 if installed.

    Used by both install_frozen_fp8 (resident dict) and the streaming rehydrate
    path (one tensor at a time). warn_state: shared {"warned": bool} so skip
    warnings print once per run.
    """
    warn_state = warn_state if warn_state is not None else {"warned": False}
    try:
        hf_path = map_fn(gguf_name)
    except KeyError:
        if not warn_state["warned"]:
            print(f"[act-replay] install_frozen_fp8: no HF mapping for "
                  f"{gguf_name!r} (and possibly others) — skipping. Logged once.",
                  flush=True)
            warn_state["warned"] = True
        return 0
    mod = modules.get(hf_path)
    if mod is None or not hasattr(mod, "weight"):
        if not warn_state["warned"]:
            print(f"[act-replay] install_frozen_fp8: {gguf_name!r} -> "
                  f"{hf_path!r} not a weighted module — skipping. Logged once.",
                  flush=True)
            warn_state["warned"] = True
        return 0
    if model_config is not None:
        perm = gguf_to_hf_perm(gguf_name, tuple(w.shape), model_config)
        if perm is not None:
            axis, index = perm
            w = w.index_select(axis, index.to(w.device))
    with torch.no_grad():
        mod.weight.copy_(w.to(device=device, dtype=dtype))
    return 1


# ─── checkpoint save / resume ────────────────────────────────────────────────


def save_ckpt(path, step, targets, optimizer):
    """Persist {step, cent, scl, opt}. targets: {gguf_name: AttachedTarget}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": int(step),
        "cent": {name: at.centroids.detach().cpu().clone() for name, at in targets.items()},
        "scl": {name: at.scales.detach().cpu().clone() for name, at in targets.items()},
        "opt": optimizer.state_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    return path


def load_ckpt(path, targets, optimizer):
    """Restore centroids/scales into `targets` and optimizer state. Returns step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for name, at in targets.items():
        with torch.no_grad():
            at.centroids.copy_(ckpt["cent"][name].to(at.centroids.device))
            at.scales.copy_(ckpt["scl"][name].to(at.scales.device))
    optimizer.load_state_dict(ckpt["opt"])
    return int(ckpt["step"])


# ─── training loop ───────────────────────────────────────────────────────────


def eval_kl(model, teacher, batches, hold_idx, resp_delims=None):
    """Mean holdout KL over the held-out batches (no grad).

    resp_delims: optional (start_ids, end_ids) — when given, the KL is masked to
    the assistant-response tokens of each batch (all-ones fallback for raw text).
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in hold_idx.tolist():
            ids = batches[i]
            idx, vals, tail = teacher.get(i, ids)
            logits = model(ids)
            V = logits.shape[-1]
            mask = (batch_response_mask(ids, *resp_delims).reshape(-1)
                    if resp_delims is not None else None)
            kl = kl_topk(logits.reshape(-1, V), idx, vals, tail, mask=mask)
            total += float(kl.item())
            n += 1
            # free this batch's logits + KL chain before the next iteration so the
            # eval working set stays at one batch, not the whole holdout.
            del logits, kl, idx, vals, tail, mask
    model.train()
    # eval is a natural boundary (no live graph); drain the CUDA caching allocator
    # and the glibc arenas so the next train step starts from a clean slate (NOT
    # done in the hot loop).
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    _trim_host()
    return total / max(n, 1)


def train(model, teacher, batches, train_idx, hold_idx, optimizer,
          grad_accum=8, epochs=1, steps=None, eval_interval=200,
          start_step=0, ckpt_path=None, targets=None, resp_delims=None,
          eval_batches=None, warmup_steps=0, total_steps=None,
          reassign_mode="none", reassign_interval=50, reassign_frac=0.1,
          loss_scale=1.0):
    """Run the act-replay KL training loop.

    Per train batch (in train_idx order): student logits -> kl_topk vs the
    teacher's top-K for that batch -> backward (averaged over grad_accum
    micro-steps) -> optimizer.step() every grad_accum batches. Every
    eval_interval optimizer steps, print one holdout-KL line to stdout.
    Returns the final optimizer step count.

    resp_delims: optional (start_ids, end_ids) assistant-turn delimiters. When
    given, the KL loss is masked to response tokens per batch (Design: "Loss = KL
    over response tokens"); raw-text batches with no assistant span fall back to
    an all-ones mask.

    eval_batches: optional separate batch list for the interleaved holdout eval
    (indexed by hold_idx). Defaults to `batches`. Set this when `batches` has been
    time-split for training (--train-seq-len) but the holdout must keep full-length
    sequences — train and eval then draw from different lists with their own index
    spaces.
    """
    if eval_batches is None:
        eval_batches = batches
    model.train()
    Ml8Fp8Fn.loss_scale = loss_scale
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    step = start_step
    micro = 0
    optimizer.zero_grad()
    for _epoch in range(epochs):
        for i in train_idx.tolist():
            ids = batches[i]
            idx, vals, tail = teacher.get(i, ids)
            logits = model(ids)
            V = logits.shape[-1]
            mask = (batch_response_mask(ids, *resp_delims).reshape(-1)
                    if resp_delims is not None else None)
            loss = kl_topk(logits.reshape(-1, V), idx, vals, tail, mask=mask) / grad_accum
            loss.backward()
            # The backward is done: drop every handle into this micro's autograd
            # graph (loss/logits + the teacher top-K + mask) so its ~1.2GB of
            # retained activations frees before the next micro builds its own.
            # Only floats survive across iterations (step counters); empty_cache is
            # deliberately NOT called here (hot loop) — only at eval boundaries.
            del loss, logits, idx, vals, tail, mask
            micro += 1
            if micro % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if total_steps is not None:
                    _apply_lr_schedule(optimizer, base_lrs, step, warmup_steps, total_steps)
                if reassign_mode != "none" and step % reassign_interval == 0 and targets is not None:
                    reassign_targets(list(targets.values()), reassign_mode, frac=reassign_frac)
                # Without expandable_segments (page-faults gfx1201 — see
                # alloc_conf_hint) the caching allocator ratchets reserved VRAM
                # toward the high-water mark (~33GB on the 4B, breaching the
                # 95% headroom rule). Draining every few steps costs ~10ms
                # against a multi-second optimizer step and caps the ratchet.
                if _EMPTY_CACHE_EVERY and step % _EMPTY_CACHE_EVERY == 0:
                    if torch.cuda.is_available() and torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                _memlog(f"post-train-step{step}")
                # Live-tensor census after the FIRST optimizer step: this is where
                # retained dequant STE buffers / fp32 KL chains would show up.
                if step == start_step + 1:
                    _tensor_census(f"post-train-step{step}")
                if eval_interval and step % eval_interval == 0:
                    kl = eval_kl(model, teacher, eval_batches, hold_idx, resp_delims=resp_delims)
                    print(f"[act-replay] step {step} holdout_kl {kl:.6f}", flush=True)
                    if ckpt_path is not None and targets is not None:
                        save_ckpt(ckpt_path, step, targets, optimizer)
                if steps is not None and step >= steps:
                    # flush any pending grad fraction so it isn't lost silently
                    if micro % grad_accum != 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    return step
    # flush a trailing partial accumulation window
    if micro % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    return step


# ─── blob export ─────────────────────────────────────────────────────────────


def _looks_ml8_gguf(gguf_path):
    """True if `gguf_path` looks like an already-quantized ml8 GGUF.

    Cheap precheck for the re-emit base: an ml8 GGUF carries the codebook sidecar
    tensors whose names end in ``.centroids`` (e.g. ``blk.0.attn_qkv.centroids``).
    A clean bf16/F16 base has none. We scan tensor names only (no data read).
    """
    import gguf
    reader = gguf.GGUFReader(str(gguf_path))
    return any(t.name.endswith(".centroids") for t in reader.tensors)


def _iter_untrained_ml8(gguf_path, names):
    """Yield (gguf_name, ml8 entry) for the given untrained ML8_4 names,
    re-read one at a time from the source GGUF at export time (streaming —
    nothing is held on host during training for these)."""
    want = set(names)
    if not want:
        return
    from gguf_state import open_ml8_gguf
    _, stream = open_ml8_gguf(gguf_path, frozen_mode="none")
    for kind, name, payload in stream:
        if kind == "ml8" and name in want:
            yield name, payload


def _frozen_fp8_names(gguf_path):
    """Names of the ML8_FP8 tensors in a GGUF (metadata scan only, no unpack)."""
    import gguf
    from gguf import GGMLQuantizationType

    reader = gguf.GGUFReader(str(gguf_path))
    return [t.name for t in reader.tensors
            if t.tensor_type == GGMLQuantizationType.ML8_FP8]


def _derive_tier_spec(named_tiers):
    """Reduce (tensor_name, tier|None) pairs to a 'leaf=tier,...' override spec.

    `tier` is 'ml8'/'fp8' for ML8_4/ML8_FP8 tensors, None for untiered (F32/bf16)
    tensors. The spec reproduces the source role->tier layout across the MAIN
    blk layers plus non-blk leaves (token_embd, output).

    Roles are NOT uniformly present in layer 0. A hybrid arch interleaves
    SSM/fused-qkv layers (which carry `attn_qkv`) with full-attention layers
    (which carry split `attn_q/k/v/output`), and a surgically-protected tensor
    such as `attn_v` may be fp8 in ONLY the full-attention layers — none of
    which is layer 0. Deriving from layer 0 alone DROPS those roles; the
    converter then tier-mismatches their fp8 blobs, skips them, and the
    re-emitted tensors become byte-garbage (catastrophic KL). So every main
    layer is scanned, not just blk.0.

    The trailing MTP/nextn block (identified by its nextn/eh_proj tensors)
    legitimately re-tiers shared roles and is excluded — it is not a role-table
    leaf. A role carrying two tiers across the MAIN layers raises: a
    role-uniform spec cannot express it, and failing loudly beats silently
    corrupting half the re-emit.
    """
    named = list(named_tiers)
    # MTP/nextn block layer indices: any blk.L tensor named nextn/eh_proj.
    mtp_layers = set()
    for name, _tier in named:
        parts = name.split(".")
        if parts[0] == "blk" and len(parts) >= 2 and ("nextn" in name or "eh_proj" in name):
            try:
                mtp_layers.add(int(parts[1]))
            except ValueError:
                pass
    leaf_tiers = {}  # leaf -> set of tiers seen across main layers
    for name, tier in named:
        if tier is None:
            continue
        leaf_name = name[: -len(".weight")] if name.endswith(".weight") else name
        parts = leaf_name.split(".")
        if parts[0] == "blk":
            if len(parts) != 3:
                continue
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            if layer in mtp_layers:
                continue
            leaf = parts[2]
        elif len(parts) == 1:
            leaf = parts[0]
        else:
            continue
        leaf_tiers.setdefault(leaf, set()).add(tier)
    spec = {}
    for leaf, tiers in leaf_tiers.items():
        if len(tiers) > 1:
            raise ValueError(
                f"{leaf}: mixed tier across main layers {sorted(tiers)} — a "
                "role-uniform ML8_TIER_OVERRIDE cannot reproduce this layout")
        spec[leaf] = next(iter(tiers))
    return ",".join(f"{k}={v}" for k, v in sorted(spec.items()))


def derive_tier_override(gguf_path):
    """Derive the ML8_TIER_OVERRIDE spec from a source GGUF's tensor types.

    The re-emit must reproduce the SOURCE artifact's role->tier layout, which
    may invert the default role table (the A3 bit-swap cell has ffn_up=fp8 /
    ffn_down=ml8 — without the override the converter skips every ffn_up fp8
    blob as tier-mismatched and the coverage gate refuses at 76%). The source
    GGUF already encodes the truth per tensor (ML8_4 vs ML8_FP8).

    The per-role tiers are derived across all MAIN layers (see _derive_tier_spec):
    a layer-0-only scan misses roles absent from layer 0 — e.g. the fp8-protected
    `attn_v`, which lives only in the full-attention layers of a hybrid arch and
    whose omission silently corrupts those tensors on re-emit. The trailing
    mtp/nextn block (e.g. blk.32 on the 4B) is excluded. Non-blk tensors
    (token_embd, output) map directly. Returns a 'leaf=tier,...' spec (may be empty).
    """
    import gguf
    from gguf import GGMLQuantizationType as Q

    tier_of = {Q.ML8_4: "ml8", Q.ML8_FP8: "fp8"}
    reader = gguf.GGUFReader(str(gguf_path))
    named = [(t.name, tier_of.get(t.tensor_type)) for t in reader.tensors]
    return _derive_tier_spec(named)


def _iter_frozen_fp8_raw(gguf_path):
    """Yield (gguf_name, (e4m3 fp32 [N,K], scale fp16 [N,n_b])) one at a time.

    The trainer state's `frozen` dict only keeps the *dequantized* fp8 tensors;
    to re-emit the frozen tensors as {hf_name}.fp8.pt we need the raw e4m3 lattice
    values + their per-group fp16 scales, which we recover straight from the GGUF.
    Streaming (vs materializing all ~113 tensors at once) keeps the export tail's
    host RAM bounded on the 15GB box — each tensor is unpacked, consumed by
    export_blobs, and freed before the next.
    """
    import gguf
    from gguf import GGMLQuantizationType
    from gguf_state import unpack_scaled_fp8_blocks, _logical_N_bytes, _row_major_bytes
    from ml8_to_gguf import _FP8_BLOCK_BYTES, _FP8_GROUP_SIZE

    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.ML8_FP8:
            continue
        N, nbytes = _logical_N_bytes(tensor)
        K = nbytes // _FP8_BLOCK_BYTES * _FP8_GROUP_SIZE
        packed = _row_major_bytes(tensor, N, nbytes)
        e4m3, scale = unpack_scaled_fp8_blocks(packed, N, K)
        yield tensor.name, (e4m3, scale)


def _write_ml8_blob(out_dir, hf_name, indices, centroids, scales, rotation=None):
    """Write one ml8_io-schema .pt blob (name/shape/group_size/n_centroids/
    indices int8/centroids_per_group/scale_per_group + zeroed metrics).

    indices [N,K], centroids [G,16], scales [N,G] — torch tensors on CPU.
    group_size = K // G. Centroids are snapped to the e4m3 lattice (the on-disk
    sidecar dtype) so the blob round-trips bit-exactly through ml8_to_gguf.

    rotation: optional {"h_a","a_dim","b_dim"} — the FROZEN Kronecker input
    rotation the source GGUF carried on this tensor. The trained centroids/scales
    encode the weight in the ROTATED basis (W_rot), so the re-emit MUST carry the
    rotation forward or the deployed kernel matmuls W_rot in the wrong basis
    (every ml8 GEMM wrong; ~12 KLD). ml8_to_gguf writes rotation_h_a/_meta only
    when the blob has a 'rotation' dict, so we populate it here.
    """
    indices = indices.detach().cpu().to(torch.int8)
    N, K = indices.shape
    cent = snap_to_e4m3(centroids.detach().cpu().to(torch.float32))
    G = cent.shape[0]
    scales = scales.detach().cpu().to(torch.float32)
    blob = {
        "name": hf_name,
        "shape": [int(N), int(K)],
        "group_size": int(K // G),
        "n_centroids": int(cent.shape[1]),
        "indices": indices,
        "centroids_per_group": cent,
        "scale_per_group": scales,
        "mse": 0.0,
        "w_snr_db": 0.0,
        "y_snr_db": 0.0,
        "rel_err": 0.0,
    }
    if rotation is not None:
        blob["rotation"] = {
            "kind": "kronecker_orth_sylvester",
            "a_dim": int(rotation["a_dim"]),
            "b_dim": int(rotation["b_dim"]),
            "in_features": int(K),
            "h_a": rotation["h_a"].detach().cpu().to(torch.float32),
        }
    torch.save(blob, out_dir / f"{hf_name}.pt")


def export_blobs(state, hf_names, out_dir, frozen_fp8_raw=None, untrained_ml8=None,
                 model_config=None):
    """Write each ml8 target as an ml8_io-schema blob, plus frozen fp8 tensors.

    state: {gguf_name: AttachedTarget}.
    hf_names: {gguf_name: hf_tensor_name} — the blob's `name` and filename stem.
    out_dir: destination dir (never /tmp); created if needed.
    frozen_fp8_raw: optional {gguf_name: (e4m3 fp32 [N,K], scale fp16 [N,n_b])}
        dict OR an iterable of such pairs (e.g. _iter_frozen_fp8_raw's stream) —
        written as {hf_name}.fp8.pt = canonical fp8 schema
        (name/tier/shape/group_size/e4m3/scale).
    untrained_ml8: optional {gguf_name: {"indices","scales","centroids"}} dict
        OR an iterable of such (name, entry) pairs (e.g. _iter_untrained_ml8's
        stream) — ML8_4 tensors that exist in the source GGUF but were NOT
        selected as training targets (e.g. with a narrowed --tensors-train
        glob). Without re-emitting these, a partial-training run would silently
        drop them to bf16 in the re-emitted GGUF and tank coverage.
        Re-exported verbatim from the source.

    Per ml8 target: snap_to_e4m3 the final centroids, then write the schema
    (name/shape/group_size/n_centroids/indices int8/centroids_per_group/
    scale_per_group + mse/w_snr_db/y_snr_db/rel_err = 0.0). group_size = K // G.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gguf_name, at in state.items():
        # The frozen Kronecker rotation rides on the AttachedTarget as a
        # KroneckerRotation (.h_a/.a_dim/.b_dim); normalize to the blob dict so
        # the re-emit reproduces the source's rotation_h_a/_meta sidecars.
        at_rot = getattr(at, "rotation", None)
        rot = None if at_rot is None else {
            "h_a": at_rot.h_a, "a_dim": int(at_rot.a_dim), "b_dim": int(at_rot.b_dim)}
        # attach_targets reordered GGUF->HF rows (gguf_to_hf_perm) so the student
        # trained in HF layout; INVERT that here so the re-emit writes GGUF-order
        # rows. Identity for non-linear-attn / 0.8B (num_v == num_k). Without it
        # the V-reordered ml8 tensors (attn_qkv/attn_gate) ship scrambled by
        # 90-136% — every such GEMM wrong, ~10 KLD.
        idx_out, scl_out = at.indices, at.scales
        perm = gguf_to_hf_perm(gguf_name, tuple(at.indices.shape), model_config)
        if perm is not None:
            axis, index = perm
            unp = _apply_perm_to_ml8_entry(
                {"indices": at.indices, "scales": at.scales},
                (axis, torch.argsort(index)))
            idx_out, scl_out = unp["indices"], unp["scales"]
        _write_ml8_blob(out_dir, hf_names[gguf_name],
                        idx_out, at.centroids, scl_out, rotation=rot)

    if untrained_ml8 is not None:
        pairs = (untrained_ml8.items()
                 if hasattr(untrained_ml8, "items") else untrained_ml8)
        for gguf_name, ent in pairs:
            if gguf_name in state:
                continue  # a trained target already wrote the up-to-date blob
            hf_name = hf_names.get(gguf_name)
            if hf_name is None:
                try:
                    hf_name = map_gguf_to_hf(gguf_name)
                except KeyError:
                    hf_name = gguf_name
            _write_ml8_blob(out_dir, hf_name, ent["indices"],
                            ent["centroids"], ent["scales"],
                            rotation=ent.get("rotation"))

    if frozen_fp8_raw is not None:
        from ml8_to_gguf import _FP8_GROUP_SIZE
        pairs = (frozen_fp8_raw.items()
                 if hasattr(frozen_fp8_raw, "items") else frozen_fp8_raw)
        for gguf_name, (e4m3, scale) in pairs:
            hf_name = hf_names.get(gguf_name)
            if hf_name is None:
                # frozen fp8 tensors that don't map to a known HF name keep their
                # GGUF name as the stem (sanitized for the filesystem).
                hf_name = gguf_name
            e4m3 = e4m3.detach().cpu().to(torch.float32)
            scale = scale.detach().cpu().to(torch.float16)
            N, K = int(e4m3.shape[0]), int(e4m3.shape[1])
            # Write the CANONICAL fp8 blob schema (matches calibrate_ml8_paged's
            # *.fp8.pt output): name/tier/shape/group_size + e4m3/scale. The earlier
            # export wrote only {e4m3, scale}, so ml8_to_gguf._build_fp8_blob_map —
            # which keys off blob['name'] then classify_role — skipped every fp8
            # blob with "blob has no 'name' field", collapsing re-emit coverage.
            torch.save(
                {
                    "name": hf_name,
                    "tier": "fp8",
                    "shape": [N, K],
                    "group_size": int(_FP8_GROUP_SIZE),
                    "e4m3": e4m3,
                    "scale": scale,
                },
                out_dir / f"{hf_name}.fp8.pt")
    return out_dir


# ─── CLI entry point ─────────────────────────────────────────────────────────


def alloc_conf_hint(device_str, env=None):
    """Return a launch-time WARNING string if the HIP expandable-segments
    allocator knob IS set for a CUDA/HIP run, else None.

    POLARITY FLIP (2026-06-10): expandable_segments:True page-faults the GPU on
    gfx1201 under this trainer's allocation pattern — 5/5 4B runs died with
    "Memory access fault ... Page not present" (last serialized dispatch:
    at::native::mbtopk::gatherTopK), and the identical run without the env var
    completed clean end-to-end. Fragmentation is managed instead by
    empty_cache() at eval boundaries and the pre-export GPU release.

    This module imports torch at load time, so the allocator is already
    configured by the time we run — we can only DETECT and warn, not fix.
    Returns None for CPU runs or when the var is unset/empty.
    """
    env = os.environ if env is None else env
    if not str(device_str).startswith("cuda"):
        return None
    conf = env.get("PYTORCH_HIP_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        return None
    return (
        "[act-replay] WARNING: PYTORCH_HIP_ALLOC_CONF contains "
        f"expandable_segments ({conf!r}). On gfx1201 this page-faults the GPU "
        "mid-run (mbtopk; 2026-06-10, 5/5 repro). Unset it — fragmentation is "
        "handled by eval-boundary empty_cache() + the pre-export release."
    )


# ─── env-gated layer-divergence probe ────────────────────────────────────────


def _find_decoder_layers(hf_model):
    """Locate the decoder-layer ModuleList of an HF causal-LM, agnostic to the
    wrapper depth (Qwen3.5 nests it under .model.layers or
    .model.language_model.layers depending on the config head). Returns the
    torch.nn.ModuleList of decoder blocks. Raises if not found."""
    candidates = [
        getattr(getattr(hf_model, "model", None), "layers", None),
        getattr(getattr(getattr(hf_model, "model", None), "language_model", None),
                "layers", None),
        getattr(hf_model, "layers", None),
    ]
    for c in candidates:
        if c is not None and len(c) > 0:
            return c
    # last resort: scan named_modules for the deepest ModuleList of decoder blocks
    best = None
    for name, mod in hf_model.named_modules():
        if name.endswith(".layers") and isinstance(mod, torch.nn.ModuleList):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate decoder .layers ModuleList")
    return best


def _rel_div(s, t):
    """Relative L2 divergence |s - t| / (|t| + 1e-9), fp32, scalar float.
    Tensors are flattened/cast to fp32 on whatever device they arrive."""
    s = s.detach().float()
    t = t.detach().float()
    num = (s - t).norm().item()
    den = t.norm().item() + 1e-9
    return num / den


def run_divergence_probe(student_hf, args, teacher, batches, hold_idx, device):
    """First-divergence probe (ACT_REPLAY_PROBE=1).

    Forwards ONE holdout batch through the frozen bf16 TEACHER then the attached
    ml8 STUDENT, capturing the fp32 output hidden state of every decoder layer and
    the F.linear output of each layer-0 2D-matmul submodule. Prints, per layer i,
    rel = |s_i - t_i| / (|t_i| + 1e-9); and per layer-0 module the same rel on the
    linear output. The first layer / submodule whose rel rises well above the e4m3
    quant noise floor (~1e-2..3e-2) localizes where the student leaves the teacher.

    Everything (hooks, teacher, captured activations) is freed before returning.
    This is a diagnostic: main() sys.exit()s right after.
    """
    import torch.nn.functional as F  # noqa: F401  (parity w/ student F.linear path)

    # pick one holdout batch (fall back to batch 0 if the holdout is empty)
    probe_i = int(hold_idx[0].item()) if len(hold_idx) > 0 else 0
    ids = batches[probe_i]
    print(f"[probe] using holdout batch idx={probe_i} ids.shape={tuple(ids.shape)}",
          flush=True)

    # The frozen bf16 teacher HF model. For the live teacher it is already
    # resident (teacher.model is the _LMWrap, .m is the HF model); otherwise load a
    # fresh frozen parent. NEVER the attached student (that has ml8 targets).
    teacher_hf = None
    tm = getattr(teacher, "model", None)
    if tm is not None and hasattr(tm, "m"):
        teacher_hf = tm.m
    loaded_teacher = False
    if teacher_hf is None:
        teacher_hf = load_hf_model(args.model, device, freeze=True)
        loaded_teacher = True
    teacher_hf.eval()
    student_hf.eval()

    s_layers = _find_decoder_layers(student_hf)
    t_layers = _find_decoder_layers(teacher_hf)
    n_layers = min(len(s_layers), len(t_layers))
    print(f"[probe] decoder layers: student={len(s_layers)} teacher={len(t_layers)} "
          f"probing {n_layers}", flush=True)

    # ── per-layer output-hidden-state capture ────────────────────────────────
    s_out, t_out = {}, {}

    def _layer_hook(store, i):
        def hook(_mod, _inp, out):
            # decoder layers return a tuple (hidden_state, ...) or a bare tensor
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[i] = h.detach().float().cpu()
        return hook

    handles = []
    for i in range(n_layers):
        handles.append(s_layers[i].register_forward_hook(_layer_hook(s_out, i)))
        handles.append(t_layers[i].register_forward_hook(_layer_hook(t_out, i)))

    # ── per-target layer-0 submodule F.linear-output capture ─────────────────
    # Layer 0 of qwen3.5-4B is a linear_attention block: its 2D-matmul linears are
    # linear_attn.in_proj_qkv (attn_qkv), linear_attn.in_proj_z (in_proj_z / gate),
    # linear_attn.out_proj; the MLP carries gate_proj/up_proj/down_proj. We capture
    # the Linear *output* (== F.linear(x, W)) so the diff is the per-module matmul
    # divergence the ml8 dequant-STE student introduces vs the teacher's bf16 W.
    L0_SUBMODULES = [
        ("attn_qkv", "linear_attn.in_proj_qkv"),
        ("in_proj_z", "linear_attn.in_proj_z"),
        ("out_proj", "linear_attn.out_proj"),
        ("gate", "mlp.gate_proj"),
        ("up", "mlp.up_proj"),
        ("down", "mlp.down_proj"),
        # full-attention fallbacks (in case layer 0 is ever a full_attention block)
        ("q_proj", "self_attn.q_proj"),
        ("k_proj", "self_attn.k_proj"),
        ("v_proj", "self_attn.v_proj"),
        ("o_proj", "self_attn.o_proj"),
    ]
    s_mod, t_mod = {}, {}

    def _mod_hook(store, label):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            store[label] = h.detach().float().cpu()
        return hook

    def _submods(layer0):
        d = dict(layer0.named_modules())
        return d

    s_l0 = _submods(s_layers[0])
    t_l0 = _submods(t_layers[0])
    present = []
    for label, path in L0_SUBMODULES:
        sm = s_l0.get(path)
        tm_ = t_l0.get(path)
        if sm is not None and tm_ is not None and hasattr(sm, "weight"):
            handles.append(sm.register_forward_hook(_mod_hook(s_mod, label)))
            handles.append(tm_.register_forward_hook(_mod_hook(t_mod, label)))
            present.append(label)
    print(f"[probe] layer-0 submodules hooked: {present}", flush=True)

    # ── forward both models on the same batch (no grad) ──────────────────────
    with torch.no_grad():
        ids_dev = ids.to(device)
        teacher_hf(ids_dev)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        student_hf(ids_dev)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # ── report per-layer divergence ──────────────────────────────────────────
    print("[probe] ===== per-decoder-layer output divergence "
          "rel=|s-t|/(|t|+1e-9) =====", flush=True)
    first_layer = None
    NOISE = 3e-2  # e4m3 noise-floor ceiling
    for i in range(n_layers):
        if i not in s_out or i not in t_out:
            continue
        rel = _rel_div(s_out[i], t_out[i])
        flag = "  <-- FIRST > noise" if (first_layer is None and rel > NOISE) else ""
        if first_layer is None and rel > NOISE:
            first_layer = i
        print(f"[probe] layer {i:2d}  rel={rel:.6e}{flag}", flush=True)

    # ── report layer-0 per-submodule divergence ──────────────────────────────
    print("[probe] ===== layer-0 per-submodule F.linear-output divergence =====",
          flush=True)
    for label in present:
        if label in s_mod and label in t_mod:
            rel = _rel_div(s_mod[label], t_mod[label])
            print(f"[probe] layer0.{label:10s} rel={rel:.6e}", flush=True)

    if first_layer is not None:
        print(f"[probe] VERDICT: first decoder layer with rel > {NOISE:.0e} "
              f"is layer {first_layer}", flush=True)
    else:
        print(f"[probe] VERDICT: no decoder layer exceeded the noise floor "
              f"{NOISE:.0e} (student tracks teacher within e4m3 noise)", flush=True)

    # ── free everything ──────────────────────────────────────────────────────
    for h in handles:
        h.remove()
    s_out.clear(); t_out.clear(); s_mod.clear(); t_mod.clear()
    if loaded_teacher:
        del teacher_hf
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def main(argv=None):
    """Wire the full pipeline. HF-dependent path; exercised end-to-end only with
    a real model + GPU (the unit tests import the functions above directly)."""
    args = parse_args(argv)

    # Allocator-fragmentation hint for CUDA/HIP runs (must be set at launch time —
    # torch is already imported, so we only advise, we don't try to set it here).
    _hint = alloc_conf_hint(args.device)
    if _hint:
        print(_hint, flush=True)

    out_dir = Path(args.out_dir)
    if str(out_dir).startswith("/tmp"):
        raise ValueError("refusing to write outputs under /tmp; pass a user path")

    # Re-emit base precheck (cheap, name-only): the GGUF we re-emit FROM must be a
    # clean bf16/F16 base, not an already-quantized ml8 GGUF. Refuse early — before
    # loading the model / running any training — if the base carries ml8 codebook
    # (.centroids) sidecar tensors, which would produce a corrupt double-quant.
    if _looks_ml8_gguf(args.base_gguf):
        raise ValueError(
            f"--base-gguf {args.base_gguf!r} looks like an ml8 GGUF (has "
            f".centroids tensors); pass the original bf16/F16 base GGUF instead")

    # streaming rehydrate (open_ml8_gguf) imported at the consumption site below

    # HF model + tokenizer (untested path).
    from transformers import AutoTokenizer
    from calib_corpus import collect_calibration

    device = torch.device(args.device)
    _memlog("start")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ORDER MATTERS on the 15GB host — one model resident at a time wherever
    # possible. Corpus + holdout split first (tokenizer-only work), then the
    # TEACHER (--teacher cache loads the parent once, writes its top-K shards
    # to disk and FREES it — training never holds a second model in host RAM
    # or VRAM), then the student, then the trainer-state rehydrate.
    n_samples = max(1, args.token_budget // max(args.seq_len, 1))
    batches = collect_calibration(
        tokenizer, n_samples=n_samples, seq_len=args.seq_len,
        composition=args.corpus, seed=args.seed, token_budget=args.token_budget)
    batches = [b.to(device) for b in batches]
    _memlog("post-corpus")

    train_idx, hold_idx = split_holdout(len(batches), frac=0.1, seed=args.seed)

    # --train-seq-len: time-split the TRAIN batches into shorter windows to cap
    # per-forward activation memory at 4B, while EVAL keeps full --seq-len
    # holdout sequences. train_batches/train_idx_w live in their own index
    # space; eval keeps the original (batches, hold_idx). The cached teacher is
    # content-addressed (shards keyed by ids hash), so the mixed window/full
    # sequence set is fine — no live-teacher-only restriction anymore.
    train_batches = batches
    train_idx_w = train_idx
    if args.train_seq_len is not None and args.train_seq_len < args.seq_len:
        train_batches, train_idx_w = split_batches_seq(
            batches, train_idx, args.train_seq_len)
        print(f"[act-replay] train-seq-len={args.train_seq_len}: split "
              f"{len(train_idx)} train batch(es) (seq_len={args.seq_len}) into "
              f"{len(train_idx_w)} window(s)")

    # teacher = frozen bf16 parent — a SEPARATE from_pretrained instance with NO
    # attached ml8 targets, for every strategy (live/cache/device). Distilling
    # against the trained-on student would chase a moving target; a cache built
    # from the attached model would distill KL-to-quant — the wrong target.
    # cache: built over EXACTLY the sequences training/eval will request
    # (holdout full batches + train windows); the parent is freed after the
    # build — the permanent fix for the live teacher's ~9GB host residency.
    from teacher_source import make_teacher
    cache_dir = args.teacher_cache_dir or str(out_dir / "teacher_cache")

    def teacher_loader():
        m = _LMWrap(load_hf_model(args.model, device, freeze=True), device)
        # Trim NOW, not after the multi-minute build pass: the shard-streaming
        # staging (~one safetensors shard at a time through host tensors) leaves
        # a multi-GB glibc high-water that would otherwise sit retained for the
        # whole cache build (measured 8.5GB RSS over a ~2GB working set).
        _trim_host()
        return m
    teacher_batches = ([batches[i] for i in hold_idx.tolist()]
                       + [train_batches[i] for i in train_idx_w.tolist()])
    teacher = make_teacher(
        args.teacher, model_loader=teacher_loader, K=args.topk,
        cache_dir=cache_dir, batches=teacher_batches,
        cache_key=f"{Path(args.model).name}_{args.corpus}")
    del teacher_batches
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    _trim_host()
    _memlog("post-teacher")

    # Student: a fresh bf16 model whose selected ml8 linears get monkeypatched
    # with trainable dequant-STE targets below. Loaded AFTER the teacher so the
    # cache strategy never has two models resident, but still BEFORE the
    # rehydrate (the ~8.3GB bf16 staging is transient — weights land on GPU and
    # the host copy frees; rehydrate-then-load was an 11G cgroup OOM, measured).
    # Gradient checkpointing (non-reentrant) trades recompute for activation
    # memory; --no-grad-ckpt off.
    model = load_hf_model(args.model, device, grad_ckpt=not args.no_grad_ckpt)
    wrapped = _LMWrap(model, device)
    _trim_host()
    _memlog("post-student")

    print(f"[act-replay] rehydrating trainer state from {args.gguf} (streaming)")
    # STREAMING rehydrate: one tensor at a time, GGUF -> unpack -> attach/install
    # on the GPU -> host copy freed. The old load-everything-then-attach path
    # materialized the full trainer state (~5-6GB on the 4B) on the 15GB host —
    # the worst of the phase spikes. frozen_mode="fp8": only ML8_FP8 tensors
    # stream as frozen; bf16/F32 pass-throughs stay with the HF student.
    from gguf_state import open_ml8_gguf, list_ml8_names
    modules = dict(model.named_modules())
    model_config = getattr(model, "config", None)
    selected = set(select_targets(list_ml8_names(args.gguf),
                                  train=args.tensors_train,
                                  skip=args.tensors_skip))
    no_fp8_install = bool(os.environ.get("ACT_REPLAY_NO_FP8_INSTALL"))
    targets = {}
    untrained_ml8_names = []
    n_fp8 = 0
    warn_state = {"warned": False}
    _, stream = open_ml8_gguf(args.gguf, frozen_mode="fp8")
    for kind, name, payload in stream:
        if kind == "ml8":
            if name in selected:
                targets[name] = _attach_one(modules, name, payload, model_config,
                                             fp8=args.fp8)
            else:
                # Re-read verbatim from the source GGUF at export time (names
                # only are kept) — dropping them would silently bf16 them in
                # the re-emit and tank coverage.
                untrained_ml8_names.append(name)
        elif not no_fp8_install:
            # FP8-faithful frozen weight replaces the bf16 parent weight
            # in-place (closes the faithfulness gap).
            n_fp8 += _install_one(modules, name, payload, map_gguf_to_hf,
                                  device, torch.bfloat16, model_config,
                                  warn_state)
        del payload
    del modules
    _trim_host()
    print(f"[act-replay] attached {len(targets)} ml8 targets")
    if no_fp8_install:
        print("[act-replay] fp8 install SKIPPED (ACT_REPLAY_NO_FP8_INSTALL)")
    else:
        print(f"[act-replay] installed {n_fp8} frozen fp8 tensors into the student")
    if untrained_ml8_names:
        print(f"[act-replay] {len(untrained_ml8_names)} untrained ml8 tensor(s) "
              f"will be re-emitted verbatim from the source GGUF")
    hf_names = {g: map_gguf_to_hf(g) for g in targets}
    _memlog("post-rehydrate-attach-install")

    # optimizer with separate lr groups for centroids vs scales
    cent_params = [at.centroids for at in targets.values()]
    scl_params = [at.scales for at in targets.values()]
    optimizer = torch.optim.Adam([
        {"params": cent_params, "lr": args.lr_cent},
        {"params": scl_params, "lr": args.lr_scale},
    ])

    start_step = 0
    if args.resume:
        start_step = load_ckpt(args.resume, targets, optimizer)
        print(f"[act-replay] resumed at step {start_step}")

    # assistant-response delimiters for KL masking (Design: KL over response
    # tokens). Derived from the model's own chat template, so no per-model strings.
    resp_delims = assistant_delimiters(tokenizer)
    print(f"[act-replay] response mask delimiters: start={resp_delims[0]} "
          f"end={resp_delims[1]}")

    # step-0 sanity: pre-train holdout KL (= the PTQ artifact's KL on this draw)
    kl0 = eval_kl(wrapped, teacher, batches, hold_idx, resp_delims=resp_delims)
    print(f"[act-replay] step {start_step} holdout_kl {kl0:.6f} (pre-train)", flush=True)
    _memlog("post-step0eval")

    # ── env-gated layer-divergence probe (ACT_REPLAY_PROBE=1) ────────────────
    # Goal: name the FIRST decoder layer / submodule where the attached ml8
    # student leaves the bf16 teacher. Forward ONE holdout batch through teacher
    # then student with hooks on every decoder layer capturing the fp32 output
    # hidden state, and per-target hooks on layer-0's 2D-matmul linears capturing
    # each F.linear output. Print per-layer relative L2 divergence + the layer-0
    # per-module diffs, then exit BEFORE training (this is a diagnostic, not a run).
    if os.environ.get("ACT_REPLAY_PROBE"):
        run_divergence_probe(model, args, teacher, batches, hold_idx, device)
        print("[probe] done — exiting before training (ACT_REPLAY_PROBE)", flush=True)
        sys.exit(0)

    ckpt_path = out_dir / "ckpt.pt"
    if args.steps is not None:
        _total_steps = args.steps
    else:
        _total_steps = max(1, (len(train_idx_w) * args.epochs) // args.grad_accum)
    final_step = train(
        wrapped, teacher, train_batches, train_idx_w, hold_idx, optimizer,
        grad_accum=args.grad_accum, epochs=args.epochs, steps=args.steps,
        eval_interval=args.eval_interval, start_step=start_step,
        ckpt_path=ckpt_path, targets=targets, resp_delims=resp_delims,
        eval_batches=batches,
        warmup_steps=args.lr_warmup_steps, total_steps=_total_steps,
        reassign_mode=args.reassign, reassign_interval=args.reassign_interval,
        reassign_frac=args.reassign_frac, loss_scale=args.loss_scale)
    print(f"[act-replay] training done at step {final_step}")
    save_ckpt(ckpt_path, final_step, targets, optimizer)

    # The export tail (blob writes + GGUF re-emit) is CPU-bound but previously
    # held both 8GB models near-full VRAM for its whole multi-minute duration.
    # The attached targets own their codebooks/indices outright, so the student,
    # teacher and optimizer state can go. The monkeypatched module forwards are
    # reference cycles, so gc.collect() must run before empty_cache actually
    # returns the segments.
    del wrapped, model, teacher, optimizer, batches, train_batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_host()
    _memlog("post-gpu-release")

    # export blobs (+ frozen fp8 + untrained ml8) then re-emit a GGUF
    blob_dir = out_dir / "blobs"
    # frozen fp8 + untrained-ml8 names are GGUF names; map the ones we can, leave
    # the rest as-is (the converter classifies on the blob's HF/GGUF name field).
    extra_hf = {}
    for gname in _frozen_fp8_names(args.gguf) + untrained_ml8_names:
        try:
            extra_hf[gname] = map_gguf_to_hf(gname)
        except KeyError:
            extra_hf[gname] = gname
    export_blobs(targets, {**hf_names, **extra_hf}, blob_dir,
                 frozen_fp8_raw=_iter_frozen_fp8_raw(args.gguf),
                 untrained_ml8=_iter_untrained_ml8(args.gguf, untrained_ml8_names),
                 model_config=model_config)
    print(f"[act-replay] wrote blobs to {blob_dir}")

    # Everything the re-emit needs is now ON DISK (blobs + ckpt). Drop the
    # trainer's remaining big holds (attached codebooks/indices, untrained ml8
    # entries) before the converter walks the base GGUF — run e (2026-06-10)
    # was cgroup-OOM-killed in convert_to_ml8_gguf while still carrying them.
    del targets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_host()
    _memlog("pre-convert")

    # The converter classifies blob tiers via the role table; reproduce the
    # SOURCE artifact's layout (it may invert the defaults — A3 swap config).
    # An operator-set ML8_TIER_OVERRIDE wins; otherwise derive from the GGUF.
    if not os.environ.get("ML8_TIER_OVERRIDE"):
        spec = derive_tier_override(args.gguf)
        if spec:
            os.environ["ML8_TIER_OVERRIDE"] = spec
            print(f"[act-replay] derived ML8_TIER_OVERRIDE={spec}")

    from ml8_to_gguf import convert_to_ml8_gguf
    out_gguf = out_dir / "act_replay.gguf"
    convert_to_ml8_gguf(Path(args.base_gguf), blob_dir, out_gguf)
    print(f"[act-replay] re-emitted GGUF -> {out_gguf}")


if __name__ == "__main__":
    main()
