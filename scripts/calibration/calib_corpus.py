#!/usr/bin/env python3
"""Calibration corpus compositions — content sweep for the ml8 method gauntlet.

The calibration corpus defines the Hessian H=XᵀX that BOTH the GPTQ assignment and the
heavy fine-tune descend, so its CONTENT is a first-class lever (scout A: in-domain vs
diverse mix → multi-point swings on small models; UD calibrates on code/chat/math, not
wikitext). This module loads named compositions from the mlambaformer raw corpus
(/mnt/hdd/corpus/raw, defined in mlambaformer/configs/corpus/mad160.yaml) by weight, with
memory-light random byte-offset sampling over the (multi-GB) JSONL shards.

Interface matches calibrate_ml8.collect_wikitext_calibration: returns a list of
[1, seq_len] input_id tensors. `--corpus wiki` falls back to the original wikitext-2 loader
so the content sweep's control cell is bit-identical to the size-sweep baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch

_RAW = "/mnt/hdd/corpus/raw"
_EXISTING = f"{_RAW}/existing"

# Pre-sample cache: tokenized draws are written here so the slow random-seek sampling over the
# multi-GB corpus on the SPINNING HDD (/mnt/hdd = /dev/sda1, rotational) happens ONCE. Repeat
# runs and every cell of the content sweep then read the tokens straight off NVMe (/home =
# /dev/nvme0n1). Override the location or disable (empty string) via the ML8_CALIB_CACHE env var.
_CACHE_DIR = os.environ.get("ML8_CALIB_CACHE", "/home/kmbandy/models/.calib_cache")

# Each source: (absolute jsonl path, text field). Paths mirror mad160.yaml.
_SRC = {
    # NOTE: the cleaned [HUMAN]:-format extract, NOT raw stackexchange/stackoverflow.jsonl
    # (that 24.8GB file is a raw HTML→text dump with page chrome/CSS boilerplate per record).
    "stackoverflow": (f"{_EXISTING}/stackoverflow_raw.jsonl", "text"),
    "softwareeng_se": (f"{_EXISTING}/softwareeng_se_raw.jsonl", "text"),
    "math_se":        (f"{_EXISTING}/math_se_raw.jsonl", "text"),
    "stats_se":       (f"{_EXISTING}/stats_se_raw.jsonl", "text"),
    "quant_so":       (f"{_EXISTING}/quant_so_raw.jsonl", "text"),
    "rpg_se":         (f"{_EXISTING}/rpg_se_raw.jsonl", "text"),
    "tool_calls":     (f"{_EXISTING}/tool_calls_raw.jsonl", "text"),
    "arxiv":          (f"{_RAW}/arxiv/arxiv.jsonl", "text"),
    "wikipedia":      (f"{_RAW}/wikipedia/wikipedia.jsonl", "text"),
    "fineweb":        (f"{_RAW}/fineweb/fineweb.jsonl", "text"),
}

# Named compositions → {source: relative weight}. "wiki" is special-cased to the
# original wikitext-2 loader (the control). Weights are relative; per-source sample
# counts are allocated proportionally to hit n_samples total.
COMPOSITIONS: dict[str, dict[str, float]] = {
    # the Unsloth analog — mad160 diverse blend (memory_traces dropped; needs assembly).
    "mix": {
        "wikipedia": 0.27, "fineweb": 0.03, "stackoverflow": 0.24, "softwareeng_se": 0.04,
        "arxiv": 0.18, "math_se": 0.07, "stats_se": 0.03, "quant_so": 0.02,
        "rpg_se": 0.06, "tool_calls": 0.01,
    },
    "code": {"stackoverflow": 0.7, "softwareeng_se": 0.3},
    "math": {"math_se": 0.5, "stats_se": 0.3, "quant_so": 0.2},
    "chat": {"rpg_se": 0.5, "softwareeng_se": 0.25, "tool_calls": 0.25},
}

# Held-out eval (mad160 marks it never-train) — used to detect calibration over-fit /
# eval-leakage. Flattened to a .txt for llama-perplexity by `write_holdout_eval_txt`.
_HOLDOUT = (f"{_EXISTING}/quant_so_eval_raw.jsonl", "text")


def _sample_jsonl(path, n, text_field, seq_len, tokenizer, rng, min_chars=100, token_target=None):
    """Draw tokenized [1, ≤seq_len] samples from a (possibly multi-GB) JSONL via random
    byte-offset seeking — seek to a random offset, skip the partial line, take the next full
    line. Memory-light. Stops at n samples; OR, when token_target is set, once the cumulative
    token count reaches token_target (n then only bounds the attempt budget)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibration source missing: {path}")
    size = os.path.getsize(path)
    out, tok_total, attempts = [], 0, 0
    cap = max((10000 if token_target is not None else n) * 60, 200)

    def _enough():
        return tok_total >= token_target if token_target is not None else len(out) >= n

    with open(path, "rb") as f:
        while not _enough() and attempts < cap:
            attempts += 1
            f.seek(rng.randrange(size))
            f.readline()                      # discard partial line at the offset
            line = f.readline()
            if not line:                      # landed near EOF — wrap to start
                f.seek(0); line = f.readline()
            try:
                text = json.loads(line).get(text_field) or ""
            except Exception:
                continue
            if len(text) < min_chars:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=seq_len).input_ids
            if ids.shape[1] < seq_len // 4:
                continue
            out.append(ids); tok_total += ids.shape[1]
    if not _enough():
        tgt = f"{token_target} tok" if token_target is not None else f"{n} samples"
        print(f"[calib-corpus] WARN {path}: got {len(out)} samples / {tok_total} tok "
              f"(target {tgt}) after {attempts} draws")
    return out


def collect_mixed_calibration(tokenizer, n_samples, seq_len, composition, seed=0, token_budget=None):
    """Return list of [1, ≤seq_len] input_ids drawn from a named COMPOSITION by weight,
    shuffled. Mirrors collect_wikitext_calibration's return shape. When token_budget is set,
    allocation is by TOKEN share (each source drawn to weight·budget tokens) so the blend holds
    by tokens, not sample count — the right control for a token-matched content sweep."""
    if composition not in COMPOSITIONS:
        raise ValueError(f"unknown composition {composition!r}; "
                         f"choices: {['wiki'] + list(COMPOSITIONS)}")
    spec = COMPOSITIONS[composition]
    rng = random.Random(seed)
    total_w = sum(spec.values())
    samples = []

    if token_budget is not None:
        # Per-source TOKEN target = weight share of the budget.
        for src, w in spec.items():
            tgt = token_budget * w / total_w
            path, field = _SRC[src]
            got = _sample_jsonl(path, n_samples, field, seq_len, tokenizer, rng, token_target=tgt)
            ntok = sum(t.shape[1] for t in got)
            print(f"[calib-corpus] {composition}: {src} {len(got)} samples / {ntok} tok (tgt {tgt:.0f})")
            samples.extend(got)
    else:
        # Per-source SAMPLE counts proportional to weight (largest-remainder to hit n_samples).
        raw = {s: n_samples * w / total_w for s, w in spec.items()}
        counts = {s: int(v) for s, v in raw.items()}
        rem = n_samples - sum(counts.values())
        for s in sorted(spec, key=lambda s: raw[s] - counts[s], reverse=True)[:rem]:
            counts[s] += 1
        for src, cnt in counts.items():
            if cnt <= 0:
                continue
            path, field = _SRC[src]
            got = _sample_jsonl(path, cnt, field, seq_len, tokenizer, rng)
            print(f"[calib-corpus] {composition}: {src} {len(got)}/{cnt}")
            samples.extend(got)

    rng.shuffle(samples)
    if token_budget is not None:
        # Per-source draws each overshoot their target by up to one whole document; with many
        # small-weight sources that compounds (mix overshot +37%). Trimming the SHUFFLED list to
        # the budget matches it to within one doc while preserving the blend in expectation
        # (uniform shuffle ⇒ the kept prefix is a uniform sub-sample of the composition).
        before = sum(t.shape[1] for t in samples)
        samples = _trim_to_budget(samples, token_budget)
        print(f"[calib-corpus] {composition}: trimmed {before}→"
              f"{sum(t.shape[1] for t in samples)} tok ({len(samples)} samples) to budget {token_budget}")
    return samples


def _cache_path(tokenizer, composition, n_samples, seq_len, seed, token_budget=None):
    """NVMe path for a (tokenizer, corpus, size, seq_len, seed) draw. The size key is the token
    budget when set (else the sample count), and the tokenizer identity is in the key so a
    tokenizer swap can't silently reuse stale token ids."""
    tok_id = str(getattr(tokenizer, "name_or_path", "tok"))
    size_key = f"b{token_budget}" if token_budget is not None else f"n{n_samples}"
    h = hashlib.sha1(f"{tok_id}|{composition}|{size_key}|s{seq_len}|seed{seed}".encode()).hexdigest()[:12]
    return os.path.join(_CACHE_DIR, f"{composition}_{size_key}_s{seq_len}_seed{seed}_{h}.pt")


def _trim_to_budget(samples, token_budget):
    """Keep samples (in order) until the cumulative token count reaches token_budget."""
    out, tot = [], 0
    for s in samples:
        out.append(s)
        tot += s.shape[1]
        if tot >= token_budget:
            break
    return out


def _cache_load(path):
    """Return the cached sample list, or None on miss / unreadable (caller redraws)."""
    if not os.path.exists(path):
        return None
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, list) and obj and all(torch.is_tensor(t) for t in obj):
            return obj
        print(f"[calib-corpus] cache malformed ({path}) — redrawing")
    except Exception as e:
        print(f"[calib-corpus] cache read failed ({path}): {e} — redrawing")
    return None


def _cache_save(path, samples):
    """Atomically persist the tokenized draw to NVMe (write-tmp + rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(samples, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[calib-corpus] cache write failed ({path}): {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def collect_calibration(tokenizer, n_samples, seq_len, composition="wiki", seed=0,
                        use_cache=True, token_budget=None):
    """Dispatch: 'wiki' → original wikitext-2 loader (control); else a mixed composition.

    When token_budget is set, every corpus is drawn to that many tokens (wiki trimmed, mixes
    drawn by token-share) — the token-matched control for the content sweep. Tokenized draws
    are cached to NVMe (keyed by tokenizer/corpus/budget-or-n/seq_len/seed) so the HDD-bound
    sampling runs once; set use_cache=False or ML8_CALIB_CACHE="" to bypass."""
    cache_path = (_cache_path(tokenizer, composition, n_samples, seq_len, seed, token_budget)
                  if (use_cache and _CACHE_DIR) else None)
    if cache_path is not None:
        cached = _cache_load(cache_path)
        if cached is not None:
            print(f"[calib-corpus] cache HIT {composition} "
                  f"{'budget=' + str(token_budget) if token_budget is not None else 'n=' + str(n_samples)} "
                  f"s={seq_len} seed={seed} ({len(cached)} samples, "
                  f"{sum(t.numel() for t in cached)} tok) ← {cache_path}")
            return cached

    if composition == "wiki":
        from calibrate_ml8 import collect_wikitext_calibration
        nreq = 100000 if token_budget is not None else n_samples
        samples = collect_wikitext_calibration(tokenizer, n_samples=nreq, seq_len=seq_len)
        if token_budget is not None:
            samples = _trim_to_budget(samples, token_budget)
    else:
        samples = collect_mixed_calibration(tokenizer, n_samples, seq_len, composition,
                                            seed=seed, token_budget=token_budget)

    if cache_path is not None and samples:
        _cache_save(cache_path, samples)
        print(f"[calib-corpus] cache MISS → wrote {len(samples)} samples "
              f"({sum(t.numel() for t in samples)} tok) → {cache_path}")
    return samples


def write_holdout_eval_txt(out_path, max_chars=1_500_000):
    """Flatten the never-train held-out JSONL into a plain .txt for llama-perplexity -f.
    Returns the path. Idempotent: skips if the file already exists and is non-empty."""
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    src, field = _HOLDOUT
    written = 0
    with open(src, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                text = json.loads(line).get(field) or ""
            except Exception:
                continue
            if not text:
                continue
            fout.write(text.replace("\r", "") + "\n\n")
            written += len(text)
            if written >= max_chars:
                break
    print(f"[calib-corpus] wrote held-out eval {out_path} ({written} chars)")
    return str(out_path)


if __name__ == "__main__":
    # Tiny self-test of allocation logic (no GPU / no tokenizer needed).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--composition", default="mix")
    ap.add_argument("--n-samples", type=int, default=128)
    a = ap.parse_args()
    spec = COMPOSITIONS[a.composition]; total_w = sum(spec.values())
    raw = {s: a.n_samples * w / total_w for s, w in spec.items()}
    counts = {s: int(v) for s, v in raw.items()}
    rem = a.n_samples - sum(counts.values())
    for s in sorted(spec, key=lambda s: raw[s] - counts[s], reverse=True)[:rem]:
        counts[s] += 1
    print(f"composition={a.composition} n={a.n_samples} alloc={counts} sum={sum(counts.values())}")
    for s in spec:
        p, _ = _SRC[s]
        print(f"  {s:16s} weight={spec[s]:.3f} exists={os.path.exists(p)}  {p}")
