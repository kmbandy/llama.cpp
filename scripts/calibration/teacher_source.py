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
    """Precomputed shards on disk, CONTENT-ADDRESSED by ids hash.

    Shards are keyed by the blake2b hash of each sequence's token content
    (shard_{ids_hash}.pt), NOT by batch index. This makes the cache valid
    across different index spaces — the act-replay trainer requests train
    WINDOWS (--train-seq-len split) and full-length holdout batches through
    the same get() API; index-keyed shards collided between the two spaces,
    which is why cache mode used to be incompatible with the window split
    (forcing the live teacher = a second resident model = the 15GB-host RAM
    problem). Content addressing also makes rebuilds incremental: only
    sequences whose shard is missing are recomputed.

    get() verifies the shard's stored ids_hash on read as an integrity check.
    A request for a sequence that was never built raises (stale/incomplete
    cache) rather than silently recomputing.
    """

    def __init__(self, cache_dir, K):
        self.cache_dir = Path(cache_dir)
        self.K = K
        self._lru = OrderedDict()  # ids_hash -> (idx, vals, tail)

    @classmethod
    def try_open(cls, root, key, K, batches):
        """Return a CachedTeacher if root holds a shard for EVERY batch, else None.

        Gate: meta {key, K} must match and shard_{hash}.pt must exist for each
        provided sequence. A None return means "miss": the caller should
        (re)build — build is incremental, so a partial cache is not wasted.
        """
        root = Path(root)
        meta_path = root / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if not (meta.get("key") == key and meta.get("K") == K):
            return None
        for b in batches:
            if not cls._shard_path(root, _ids_hash(b)).exists():
                return None
        return cls(root, K)

    @classmethod
    def build(cls, model, batches, cache_dir, key, K):
        """Build (or extend) a content-addressed shard cache.

        Only sequences with no existing shard are forwarded through the model;
        a fully-present cache never calls model.forward. Layout:
          Path(cache_dir)/f"teacher_{key}_K{K}"/
            meta.json         : {"key", "K", "V"}
            shard_{hash}.pt   : {"idx", "vals", "tail", "ids_hash"}
        """
        root = Path(cache_dir) / f"teacher_{key}_K{K}"
        root.mkdir(parents=True, exist_ok=True)

        live = LiveTeacher(model, K)
        for ids in batches:
            h = _ids_hash(ids)
            if cls._shard_path(root, h).exists():
                continue
            idx, vals, tail = live.get(0, ids)
            cls._write_shard(root, h, idx, vals, tail)

        # meta written last, atomically, to mark the cache usable.
        cls._atomic_write_text(root / "meta.json",
                               json.dumps({"key": key, "K": K}))
        return cls(root, K)

    @staticmethod
    def _shard_path(root, ids_hash):
        return Path(root) / f"shard_{ids_hash}.pt"

    @classmethod
    def _write_shard(cls, root, ids_hash, idx, vals, tail):
        path = cls._shard_path(root, ids_hash)
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
        """batch_idx is accepted for API compatibility; lookup is by content."""
        want = _ids_hash(ids)
        if want in self._lru:
            self._lru.move_to_end(want)
            return self._lru[want]
        path = self._shard_path(self.cache_dir, want)
        if not path.exists():
            raise RuntimeError(
                f"teacher cache has no shard for ids hash {want!r} (batch "
                f"{batch_idx}) — cache was built over a different sequence set")
        shard = torch.load(path)
        if shard.get("ids_hash") != want:
            raise RuntimeError(
                f"teacher cache corrupt: shard ids hash {shard.get('ids_hash')!r} "
                f"!= filename hash {want!r}")
        out = (shard["idx"], shard["vals"], shard["tail"])
        self._lru[want] = out
        self._lru.move_to_end(want)
        while len(self._lru) > 2:  # simple LRU of 2
            self._lru.popitem(last=False)
        return out


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
        # validate the hit gate without ever loading the model. The model is
        # loaded only when at least one sequence has no shard yet (and build
        # is incremental — present shards are never recomputed).
        root = Path(cache_dir) / f"teacher_{cache_key}_K{K}"
        hit = CachedTeacher.try_open(root, cache_key, K, batches)
        if hit is not None:
            return hit
        return CachedTeacher.build(model_loader(), batches, cache_dir,
                                   key=cache_key, K=K)

    raise ValueError(f"unknown teacher spec: {spec!r}")
