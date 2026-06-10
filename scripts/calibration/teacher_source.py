# teacher_source.py
"""TeacherSource abstraction: live / cache / device:N strategies for teacher top-K.

All strategies return identical (idx, vals, tail) for the same model+batch:
  idx  [B*T, K] long
  vals [B*T, K] fp32
  tail [B*T]    fp32

Cache shards are written under a caller-provided directory (never /tmp).
"""
import copy
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch

from kl_loss import topk_teacher


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
    """Precomputed per-batch shards on disk, reused when (key, K, n_batches) match."""

    def __init__(self, cache_dir, K):
        self.cache_dir = Path(cache_dir)
        self.K = K
        self._lru = OrderedDict()  # batch_idx -> (idx, vals, tail)

    @classmethod
    def build(cls, model, batches, cache_dir, key, K):
        """Build (or reuse) a shard cache. Only calls model.forward on a miss.

        Layout: Path(cache_dir)/f"teacher_{key}_K{K}"/
          meta.json    : {"key", "K", "n_batches"}
          shard_{i}.pt : {"idx", "vals", "tail"} per batch
        """
        root = Path(cache_dir) / f"teacher_{key}_K{K}"
        meta_path = root / "meta.json"
        n_batches = len(batches)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("key") == key
                    and meta.get("K") == K
                    and meta.get("n_batches") == n_batches):
                # Cache hit: do not call the model.
                return cls(root, K)

        # Miss: compute every shard via LiveTeacher and write atomically.
        root.mkdir(parents=True, exist_ok=True)
        live = LiveTeacher(model, K)
        for i, ids in enumerate(batches):
            idx, vals, tail = live.get(i, ids)
            cls._write_shard(root, i, idx, vals, tail)

        # meta written last, atomically, to mark the cache complete.
        meta = {"key": key, "K": K, "n_batches": n_batches}
        cls._atomic_write_text(meta_path, json.dumps(meta))
        return cls(root, K)

    @staticmethod
    def _shard_path(root, i):
        return Path(root) / f"shard_{i}.pt"

    @classmethod
    def _write_shard(cls, root, i, idx, vals, tail):
        path = cls._shard_path(root, i)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"idx": idx, "vals": vals, "tail": tail}, tmp)
        os.rename(tmp, path)

    @staticmethod
    def _atomic_write_text(path, text):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.rename(tmp, path)

    def get(self, batch_idx, ids):
        if batch_idx in self._lru:
            self._lru.move_to_end(batch_idx)
            return self._lru[batch_idx]
        shard = torch.load(self._shard_path(self.cache_dir, batch_idx))
        out = (shard["idx"], shard["vals"], shard["tail"])
        self._lru[batch_idx] = out
        self._lru.move_to_end(batch_idx)
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
        # Check the cache dir/meta BEFORE calling model_loader; only load on miss.
        root = Path(cache_dir) / f"teacher_{cache_key}_K{K}"
        meta_path = root / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("key") == cache_key
                    and meta.get("K") == K
                    and meta.get("n_batches") == len(batches)):
                return CachedTeacher(root, K)
        return CachedTeacher.build(model_loader(), batches, cache_dir,
                                   key=cache_key, K=K)

    raise ValueError(f"unknown teacher spec: {spec!r}")
