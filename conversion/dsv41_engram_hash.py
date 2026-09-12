"""DeepSeek-V4.1 Engram n-gram hasher inputs that cannot be derived in C++.

Port of DeepSeek inference/engram.py build_compressed_token_map and
compute_hash_multipliers. The token map needs the HF `tokenizers` normalizer
chain; the multipliers come from numpy's PCG64 default_rng(10007 * layer_id).
Both are emitted into the GGUF at convert time (deepseek41.engram.token_map,
deepseek41.engram.hash_multipliers); primes and bucket offsets are derived by
the C++ loader from engram.vocab_size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def build_compressed_token_map(tokenizer_json: Path) -> tuple[list[int], int]:
    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = Tokenizer.from_file(str(tokenizer_json))
    n_vocab = backend.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = [0] * n_vocab
    for token_id in range(n_vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: list[int], max_ngram_size: int, compressed_vocab_size: int) -> np.ndarray:
    """[n_engram_layers, max_ngram_size] int64, odd, bounded so id*mult never overflows."""
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(values * 2 + 1)
    return np.stack(rows)


def engram_hash_kvs(tokenizer_json: Path, hparams: dict) -> tuple[list[int], np.ndarray]:
    """(token_map, multipliers) validated against config's compressed vocab size."""
    token_map, compressed = build_compressed_token_map(tokenizer_json)
    want = hparams.get("engram_compressed_vocab_size")
    if want is not None and int(want) != compressed:
        raise ValueError(f"compressed vocab {compressed} != config {want}; the whole table would rehash")
    mult = compute_hash_multipliers(list(hparams["engram_layer_ids"]), int(hparams["engram_max_ngram_size"]), compressed)
    return token_map, mult
