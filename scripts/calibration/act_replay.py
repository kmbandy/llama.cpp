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
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "gguf-py"))

from act_replay_student import attach_to_linear, select_targets
from centroid_quantizer import snap_to_e4m3
from kl_loss import kl_topk


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
    p.add_argument("--lr-cent", type=float, default=1e-3, help="lr for centroid params")
    p.add_argument("--lr-scale", type=float, default=1e-4, help="lr for per-row scale params")
    p.add_argument("--grad-accum", type=int, default=8, help="grad-accumulation steps per optimizer step")
    p.add_argument("--micro-batch", type=int, default=1, help="micro-batch size (sequences per forward)")
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


# ─── attach targets to an HF model ───────────────────────────────────────────


def attach_targets(model_named_modules, state, train, skip):
    """Attach selected ml8 targets to their host linears in an HF model.

    model_named_modules: dict-like {module_path: nn.Module} (e.g. dict(model.named_modules())).
    state: an Ml8State (or anything with a `.ml8` dict of {gguf_name: target}).
    train/skip: forwarded to select_targets.

    Returns {gguf_name: AttachedTarget} for every attached target. Raises KeyError
    if a selected target's mapped HF module is not present in the model.
    """
    modules = dict(model_named_modules)
    selected = select_targets(list(state.ml8.keys()), train=train, skip=skip)
    attached = {}
    for gguf_name in selected:
        hf_path = map_gguf_to_hf(gguf_name)
        if hf_path not in modules:
            raise KeyError(
                f"target {gguf_name!r} -> {hf_path!r} not found in model modules")
        at = attach_to_linear(modules[hf_path], state.ml8[gguf_name])
        attached[gguf_name] = at
    return attached


def install_frozen_fp8(model, frozen, map_fn, device, dtype):
    """Install dequantized frozen fp8 weights INTO the student in-place.

    The HF student is loaded with its bf16 parent weights; for tensors that were
    quantized to ML8_FP8 we want the student to carry the FP8-faithful dequant,
    not the bf16 parent (closing the faithfulness gap). For each {gguf_name:
    weight} in `frozen`, map gguf_name -> HF module path via `map_fn` and copy the
    weight into that module's `.weight` in-place under no_grad. Unmapped names
    (KeyError from map_fn) are skipped with a single warning. Each frozen tensor
    is popped from `frozen` as it is consumed so its RAM is freed immediately
    (host has only 15GB). Returns the number of weights actually installed.
    """
    modules = dict(model.named_modules())
    n_installed = 0
    warned = False
    for gguf_name in list(frozen.keys()):
        w = frozen.pop(gguf_name)  # free as we go
        try:
            hf_path = map_fn(gguf_name)
        except KeyError:
            if not warned:
                print(f"[act-replay] install_frozen_fp8: no HF mapping for "
                      f"{gguf_name!r} (and possibly others) — skipping. Logged once.",
                      flush=True)
                warned = True
            continue
        mod = modules.get(hf_path)
        if mod is None or not hasattr(mod, "weight"):
            if not warned:
                print(f"[act-replay] install_frozen_fp8: {gguf_name!r} -> "
                      f"{hf_path!r} not a weighted module — skipping. Logged once.",
                      flush=True)
                warned = True
            continue
        with torch.no_grad():
            mod.weight.copy_(w.to(device=device, dtype=dtype))
        n_installed += 1
    return n_installed


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
    # so the next train step starts from a clean arena (NOT done in the hot loop).
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return total / max(n, 1)


def train(model, teacher, batches, train_idx, hold_idx, optimizer,
          grad_accum=8, epochs=1, steps=None, eval_interval=200,
          start_step=0, ckpt_path=None, targets=None, resp_delims=None,
          eval_batches=None):
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


def _read_frozen_fp8_raw(gguf_path):
    """Re-read ML8_FP8 tensors from a GGUF as (e4m3 fp32 [N,K], scale fp16 [N,n_b]).

    The trainer state's `frozen` dict only keeps the *dequantized* fp8 tensors;
    to re-emit the frozen tensors as {hf_name}.fp8.pt we need the raw e4m3 lattice
    values + their per-group fp16 scales, which we recover straight from the GGUF.
    """
    import gguf
    from gguf import GGMLQuantizationType
    from gguf_state import unpack_scaled_fp8_blocks, _logical_N_bytes, _row_major_bytes
    from ml8_to_gguf import _FP8_BLOCK_BYTES, _FP8_GROUP_SIZE

    out = {}
    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.ML8_FP8:
            continue
        N, nbytes = _logical_N_bytes(tensor)
        K = nbytes // _FP8_BLOCK_BYTES * _FP8_GROUP_SIZE
        packed = _row_major_bytes(tensor, N, nbytes)
        e4m3, scale = unpack_scaled_fp8_blocks(packed, N, K)
        out[tensor.name] = (e4m3, scale)
    return out


def _write_ml8_blob(out_dir, hf_name, indices, centroids, scales):
    """Write one ml8_io-schema .pt blob (name/shape/group_size/n_centroids/
    indices int8/centroids_per_group/scale_per_group + zeroed metrics).

    indices [N,K], centroids [G,16], scales [N,G] — torch tensors on CPU.
    group_size = K // G. Centroids are snapped to the e4m3 lattice (the on-disk
    sidecar dtype) so the blob round-trips bit-exactly through ml8_to_gguf.
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
    torch.save(blob, out_dir / f"{hf_name}.pt")


def export_blobs(state, hf_names, out_dir, frozen_fp8_raw=None, untrained_ml8=None):
    """Write each ml8 target as an ml8_io-schema blob, plus frozen fp8 tensors.

    state: {gguf_name: AttachedTarget}.
    hf_names: {gguf_name: hf_tensor_name} — the blob's `name` and filename stem.
    out_dir: destination dir (never /tmp); created if needed.
    frozen_fp8_raw: optional {gguf_name: (e4m3 fp32 [N,K], scale fp16 [N,n_b])} —
        written as {hf_name}.fp8.pt = canonical fp8 schema
        (name/tier/shape/group_size/e4m3/scale).
    untrained_ml8: optional {gguf_name: {"indices","scales","centroids"}} — ML8_4
        tensors that exist in the source GGUF but were NOT selected as training
        targets (e.g. with a narrowed --tensors-train glob). Without re-emitting
        these, a partial-training run would silently drop them to bf16 in the
        re-emitted GGUF and tank coverage. Re-exported verbatim from the source.

    Per ml8 target: snap_to_e4m3 the final centroids, then write the schema
    (name/shape/group_size/n_centroids/indices int8/centroids_per_group/
    scale_per_group + mse/w_snr_db/y_snr_db/rel_err = 0.0). group_size = K // G.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for gguf_name, at in state.items():
        _write_ml8_blob(out_dir, hf_names[gguf_name],
                        at.indices, at.centroids, at.scales)

    if untrained_ml8:
        for gguf_name, ent in untrained_ml8.items():
            if gguf_name in state:
                continue  # a trained target already wrote the up-to-date blob
            hf_name = hf_names.get(gguf_name)
            if hf_name is None:
                try:
                    hf_name = map_gguf_to_hf(gguf_name)
                except KeyError:
                    hf_name = gguf_name
            _write_ml8_blob(out_dir, hf_name, ent["indices"],
                            ent["centroids"], ent["scales"])

    if frozen_fp8_raw:
        from ml8_to_gguf import _FP8_GROUP_SIZE
        for gguf_name, (e4m3, scale) in frozen_fp8_raw.items():
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
    """Return a launch-time recommendation string if the HIP expandable-segments
    allocator knob is unset for a CUDA/HIP run, else None.

    This module imports torch at load time, so setting PYTORCH_HIP_ALLOC_CONF from
    inside the process is too late — the caching allocator is already configured.
    The only correct fix is to export it BEFORE launching python. We therefore
    just DETECT the gap and print the exact launch line; we do not pretend an
    in-process os.environ.setdefault would take effect (it wouldn't).

    Returns None for CPU runs or when the var is already set (any non-empty value).
    """
    env = os.environ if env is None else env
    if not str(device_str).startswith("cuda"):
        return None
    if env.get("PYTORCH_HIP_ALLOC_CONF"):
        return None
    return (
        "[act-replay] HINT: PYTORCH_HIP_ALLOC_CONF is unset. On the 15GB 4B host, "
        "fragmentation can OOM mid-train. torch is imported at module load, so this "
        "MUST be set BEFORE launching python — re-launch with:\n"
        "    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True python3 act_replay.py ..."
    )


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

    from gguf_state import load_ml8_gguf

    # HF model + tokenizer (untested path).
    from transformers import AutoTokenizer
    from calib_corpus import collect_calibration

    device = torch.device(args.device)
    _memlog("start")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ORDER MATTERS on the 15GB host: load the student FIRST (its ~8.3GB bf16
    # staging is transient — weights land on GPU and the host copy frees), THEN
    # rehydrate the ~4.5GB trainer state. Doing both concurrently resident was
    # an 11G cgroup OOM (measured). Teacher loads later, after gstate.ml8.clear().
    # Student: a fresh bf16 model whose selected ml8 linears get monkeypatched
    # with trainable dequant-STE targets below. Gradient checkpointing
    # (non-reentrant) trades recompute for activation memory; --no-grad-ckpt off.
    model = load_hf_model(args.model, device, grad_ckpt=not args.no_grad_ckpt)
    wrapped = _LMWrap(model, device)
    _memlog("post-student")

    print(f"[act-replay] rehydrating trainer state from {args.gguf}")
    # frozen_mode="fp8": keep ONLY ML8_FP8 tensors frozen (stored bf16), skip the
    # bf16/F32 pass-throughs — the HF student already carries those weights, so
    # materializing them here is ~10GB of pure RAM tax on a 4B model (host=15GB).
    gstate = load_ml8_gguf(args.gguf, frozen_mode="fp8")
    _memlog("post-rehydrate")

    targets = attach_targets(dict(model.named_modules()), gstate,
                             train=args.tensors_train, skip=args.tensors_skip)
    print(f"[act-replay] attached {len(targets)} ml8 targets")
    _memlog("post-attach")
    hf_names = {g: map_gguf_to_hf(g) for g in targets}

    # Install the FP8-faithful frozen weights into the student in-place, replacing
    # the bf16 parent weights of those modules (closes the faithfulness gap).
    # install_frozen_fp8 frees each frozen tensor as it goes, draining gstate.frozen.
    n_fp8 = install_frozen_fp8(
        model, gstate.frozen, map_gguf_to_hf, device=device, dtype=torch.bfloat16)
    print(f"[act-replay] installed {n_fp8} frozen fp8 tensors into the student")
    _memlog("post-install")

    # Capture the ML8_4 tensors that are NOT attached training targets (e.g. when
    # --tensors-train is a narrowed glob). These still need to be re-emitted at
    # their source codebooks, else the re-emitted GGUF silently drops them to bf16
    # and tanks coverage (the act_replay re-emit 0.3%-coverage bug class). The
    # trained targets are excluded here — their up-to-date codebooks win at export.
    untrained_ml8 = {g: ent for g, ent in gstate.ml8.items() if g not in targets}
    if untrained_ml8:
        print(f"[act-replay] {len(untrained_ml8)} untrained ml8 tensor(s) will be "
              f"re-emitted verbatim from the source GGUF")

    # The rehydrated ml8 codebooks now live in the attached targets; the remaining
    # gstate copies (now captured in untrained_ml8) are dead weight on the 15GB
    # host — free the dict's hold before the calib draw.
    gstate.ml8.clear()
    _memlog("post-clear")

    # calibration draw
    n_samples = max(1, args.token_budget // max(args.seq_len, 1))
    batches = collect_calibration(
        tokenizer, n_samples=n_samples, seq_len=args.seq_len,
        composition=args.corpus, seed=args.seed, token_budget=args.token_budget)
    batches = [b.to(device) for b in batches]
    _memlog("post-corpus")

    train_idx, hold_idx = split_holdout(len(batches), frac=0.1, seed=args.seed)

    # --train-seq-len: time-split the TRAIN batches into shorter windows to cap
    # per-forward activation memory at 4B, while EVAL keeps full --seq-len holdout
    # sequences. train_batches/train_idx_w live in their own index space; eval keeps
    # the original (batches, hold_idx).
    train_batches = batches
    train_idx_w = train_idx
    if args.train_seq_len is not None and args.train_seq_len < args.seq_len:
        if args.teacher != "live":
            print(f"[act-replay] WARNING: --train-seq-len with --teacher "
                  f"{args.teacher!r} — the cache/device teacher is keyed on the "
                  f"full-sequence batches; window-splitting is only validated with "
                  f"the live teacher. Falling back to no split.")
        else:
            train_batches, train_idx_w = split_batches_seq(
                batches, train_idx, args.train_seq_len)
            print(f"[act-replay] train-seq-len={args.train_seq_len}: split "
                  f"{len(train_idx)} train batch(es) (seq_len={args.seq_len}) into "
                  f"{len(train_idx_w)} window(s)")

    # teacher = frozen bf16 parent — a SEPARATE from_pretrained instance with NO
    # attached ml8 targets, for every strategy (live/cache/device). Distilling
    # against the trained-on student would chase a moving target; a cache built
    # from the attached model would distill KL-to-quant — the wrong target. live
    # keeps both resident (~2x model RAM); cache/device load the parent lazily.
    from teacher_source import make_teacher
    cache_dir = args.teacher_cache_dir or str(out_dir / "teacher_cache")
    teacher_loader = lambda: _LMWrap(
        load_hf_model(args.model, device, freeze=True), device)
    teacher = make_teacher(
        args.teacher, model_loader=teacher_loader, K=args.topk,
        cache_dir=cache_dir, batches=batches, cache_key=f"{args.model}_{args.corpus}")
    _memlog("post-teacher")

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

    ckpt_path = out_dir / "ckpt.pt"
    final_step = train(
        wrapped, teacher, train_batches, train_idx_w, hold_idx, optimizer,
        grad_accum=args.grad_accum, epochs=args.epochs, steps=args.steps,
        eval_interval=args.eval_interval, start_step=start_step,
        ckpt_path=ckpt_path, targets=targets, resp_delims=resp_delims,
        eval_batches=batches)
    print(f"[act-replay] training done at step {final_step}")
    save_ckpt(ckpt_path, final_step, targets, optimizer)

    # export blobs (+ frozen fp8 + untrained ml8) then re-emit a GGUF
    blob_dir = out_dir / "blobs"
    frozen_raw = _read_frozen_fp8_raw(args.gguf)
    # frozen fp8 + untrained-ml8 names are GGUF names; map the ones we can, leave
    # the rest as-is (the converter classifies on the blob's HF/GGUF name field).
    extra_hf = {}
    for gname in list(frozen_raw) + list(untrained_ml8):
        try:
            extra_hf[gname] = map_gguf_to_hf(gname)
        except KeyError:
            extra_hf[gname] = gname
    export_blobs(targets, {**hf_names, **extra_hf}, blob_dir,
                 frozen_fp8_raw=frozen_raw, untrained_ml8=untrained_ml8)
    print(f"[act-replay] wrote blobs to {blob_dir}")

    from ml8_to_gguf import convert_to_ml8_gguf
    out_gguf = out_dir / "act_replay.gguf"
    convert_to_ml8_gguf(Path(args.base_gguf), blob_dir, out_gguf)
    print(f"[act-replay] re-emitted GGUF -> {out_gguf}")


if __name__ == "__main__":
    main()
