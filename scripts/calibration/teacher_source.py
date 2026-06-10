# teacher_source.py
"""TeacherSource abstraction: live / cache / device:N strategies for teacher top-K.

All strategies return identical (idx, vals, tail) for the same model+batch:
  idx  [B*T, K] long
  vals [B*T, K] fp32
  tail [B*T]    fp32

Cache shards are written under a caller-provided directory (never /tmp).
"""
import copy
import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch

from kl_loss import topk_teacher


def _ids_hash(ids):
    """Stable 16-hex-char blake2b digest of an ids tensor's content."""
    raw = ids.cpu().contiguous().numpy().tobytes()
    return hashlib.blake2b(raw).hexdigest()[:16]


def _hash_chain(ids_hashes):
    """blake2b over the concatenation of per-batch hashes -> 16-hex chain id."""
    joined = "".join(ids_hashes).encode()
    return hashlib.blake2b(joined).hexdigest()[:16]


class TeacherSource(ABC):
    """Strategy-agnostic source of teacher top-K logits for a batch."""

    @abstractmethod
    def get(self, batch_idx, ids):
        """Return (idx [B*T,K] long, vals [B*T,K] fp32, tail [B*T] fp32)."""
        raise NotImplementedError


def _compute(model, ids, K):
    """no_grad forward -> reshape [-1, V] -> topk_teacher. Model assumed eval."""
    with torch.no_grad():
        logits = model(ids)
    V = logits.shape[-1]
    return topk_teacher(logits.reshape(-1, V), K)


class LiveTeacher(TeacherSource):
    """In-process model on the same device. Model assumed eval + no_grad."""

    def __init__(self, model, K):
        self.model = model
        self.K = K

    def get(self, batch_idx, ids):
        return _compute(self.model, ids, self.K)


class DeviceTeacher(TeacherSource):
    """Model on another device; ids moved there per get, results moved back."""

    def __init__(self, model, device, K):
        self.device = torch.device(device)
        # Move a copy when crossing devices so the caller's model (which may be
        # reused by another strategy) is never mutated in place.
        params = list(model.parameters())
        on_device = bool(params) and all(p.device == self.device for p in params)
        if not on_device:
            model = copy.deepcopy(model).to(self.device)
        self.model = model
        self.K = K

    def get(self, batch_idx, ids):
        out_device = ids.device
        ids = ids.to(self.device)
        idx, vals, tail = _compute(self.model, ids, self.K)
        return (idx.to(out_device), vals.to(out_device), tail.to(out_device))


class CachedTeacher(TeacherSource):
    """Precomputed per-batch shards on disk.

    A cache hit requires every one of {key, K, n_batches, T, V, hash_chain} to
    match. The hash_chain binds the cache to the exact ids content of every
    batch, so reusing shards across a different corpus (same key/K/n_batches but
    different tokens) is detected instead of silently serving stale logits. Each
    shard additionally stores its own ids_hash; get() re-checks it on read.
    """

    def __init__(self, cache_dir, K):
        self.cache_dir = Path(cache_dir)
        self.K = K
        self._lru = OrderedDict()  # batch_idx -> (idx, vals, tail, ids_hash)

    @classmethod
    def try_open(cls, root, key, K, n_batches, T, hash_chain):
        """Return a CachedTeacher if root holds a complete matching cache, else None.

        Gate: meta {key, K, n_batches, T, hash_chain} must match AND all
        n_batches shard files must exist. V is intentionally NOT checked here so
        callers can validate a hit before a forward pass reveals V (V is still
        recorded in meta and is part of build's own gate). A None return means
        "miss": the caller should (re)build.
        """
        root = Path(root)
        meta_path = root / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if not (meta.get("key") == key
                and meta.get("K") == K
                and meta.get("n_batches") == n_batches
                and meta.get("T") == T
                and meta.get("hash_chain") == hash_chain):
            return None
        # Validate shard count: every expected shard file must be present.
        for i in range(n_batches):
            if not cls._shard_path(root, i).exists():
                return None
        return cls(root, K)

    @classmethod
    def build(cls, model, batches, cache_dir, key, K):
        """Build (or reuse) a shard cache. Only calls model.forward on a miss.

        Layout: Path(cache_dir)/f"teacher_{key}_K{K}"/
          meta.json    : {"key", "K", "n_batches", "T", "V", "hash_chain"}
          shard_{i}.pt : {"idx", "vals", "tail", "ids_hash"} per batch
        """
        root = Path(cache_dir) / f"teacher_{key}_K{K}"
        n_batches = len(batches)
        T = batches[0].shape[-1] if n_batches else 0
        ids_hashes = [_ids_hash(b) for b in batches]
        chain = _hash_chain(ids_hashes)

        # Cache hit requires the full gate to match and every shard present.
        hit = cls.try_open(root, key, K, n_batches, T, chain)
        if hit is not None:
            return hit

        # Miss: compute every shard via LiveTeacher and write atomically.
        # V (vocabulary width) comes from the first forward's logits width.
        root.mkdir(parents=True, exist_ok=True)
        live = LiveTeacher(model, K)
        V = cls._infer_vocab(model, batches)
        for i, ids in enumerate(batches):
            idx, vals, tail = live.get(i, ids)
            cls._write_shard(root, i, idx, vals, tail, ids_hashes[i])

        # meta written last, atomically, to mark the cache complete.
        meta = {"key": key, "K": K, "n_batches": n_batches,
                "T": T, "V": V, "hash_chain": chain}
        cls._atomic_write_text(root / "meta.json", json.dumps(meta))
        return cls(root, K)

    @staticmethod
    def _infer_vocab(model, batches):
        """Vocabulary width V from a forward on the first batch (no_grad)."""
        if not batches:
            return 0
        with torch.no_grad():
            logits = model(batches[0])
        return int(logits.shape[-1])

    @staticmethod
    def _shard_path(root, i):
        return Path(root) / f"shard_{i}.pt"

    @classmethod
    def _write_shard(cls, root, i, idx, vals, tail, ids_hash):
        path = cls._shard_path(root, i)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"idx": idx, "vals": vals, "tail": tail,
                    "ids_hash": ids_hash}, tmp)
        os.rename(tmp, path)

    @staticmethod
    def _atomic_write_text(path, text):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.rename(tmp, path)

    def get(self, batch_idx, ids):
        want = _ids_hash(ids)
        if batch_idx in self._lru:
            self._lru.move_to_end(batch_idx)
            idx, vals, tail, ids_hash = self._lru[batch_idx]
        else:
            shard = torch.load(self._shard_path(self.cache_dir, batch_idx))
            idx, vals, tail = shard["idx"], shard["vals"], shard["tail"]
            ids_hash = shard.get("ids_hash")
            self._lru[batch_idx] = (idx, vals, tail, ids_hash)
            self._lru.move_to_end(batch_idx)
            while len(self._lru) > 2:  # simple LRU of 2
                self._lru.popitem(last=False)
        if ids_hash != want:
            raise RuntimeError(
                f"teacher cache stale: ids hash mismatch for batch {batch_idx} "
                f"(shard {ids_hash!r} != requested {want!r})")
        return (idx, vals, tail)


def make_teacher(spec, model_loader, K, cache_dir, batches=None, cache_key=None):
    """Build a TeacherSource from a string spec.

    spec: "live" | "cache" | "device:N"
    model_loader: callable returning the model; only invoked when actually
      needed so a cache hit never constructs/loads the model.
    """
    if spec == "live":
        return LiveTeacher(model_loader(), K)

    if spec.startswith("device:"):
        n = spec.split(":", 1)[1]
        return DeviceTeacher(model_loader(), f"cuda:{n}", K)

    if spec == "cache":
        if batches is None or cache_key is None:
            raise ValueError("cache strategy requires batches and cache_key")
        # Hashing the ids is cheap CPU work; doing it BEFORE deciding lets us
        # validate the full hit gate without ever loading the model. The model
        # is loaded only on a miss. V is not part of the try_open gate (it is
        # unknown until a forward), which is fine — build re-records it.
        root = Path(cache_dir) / f"teacher_{cache_key}_K{K}"
        n_batches = len(batches)
        T = batches[0].shape[-1] if n_batches else 0
        chain = _hash_chain([_ids_hash(b) for b in batches])
        hit = CachedTeacher.try_open(root, cache_key, K, n_batches, T, chain)
        if hit is not None:
            return hit
        return CachedTeacher.build(model_loader(), batches, cache_dir,
                                   key=cache_key, K=K)

    raise ValueError(f"unknown teacher spec: {spec!r}")
