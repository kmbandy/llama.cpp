"""Model sources: an HF repo streamed one shard at a time, or a local GGUF.

HFSource never holds more than the shards the caller has open: a shard is
fetched on open_shard and deleted on release_shard. `fetch` is injectable so
tests use a local directory as the hub.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

from .arch import ARCHS
from .plan import PlanError

MIN_SHARD_BYTES = 1_000_000
Fetch = Callable[[str, str, Path], Path]


class ShardReader(Protocol):
    def keys(self) -> list[str]: ...
    def get_tensor(self, name: str) -> np.ndarray: ...


class Source(Protocol):
    @property
    def is_gguf(self) -> bool: ...
    def hparams(self) -> dict: ...
    def tensor_index(self) -> dict[str, str]: ...
    def layer_shards(self, layer: int, prefix: str = "layers") -> list[str]: ...
    def open_shard(self, shard_id: str): ...
    def release_shard(self, shard_id: str) -> None: ...


def flatten_hparams(cfg: dict) -> dict:
    out = dict(cfg)
    for k, v in (cfg.get("text_config") or {}).items():
        out.setdefault(k, v)
    return out


def _hf_fetch(repo: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir),
                                token=os.environ.get("HF_TOKEN")))


class _SafetensorsReader:
    def __init__(self, f):
        self._f = f

    def keys(self) -> list[str]:
        return list(self._f.keys())

    def get_tensor(self, name: str):
        import torch
        t = self._f.get_tensor(name)
        if t.dtype == torch.bfloat16:
            return t.float().numpy()
        # MXFP4 scales are float8_e8m0, which has no numpy dtype. The lossless
        # repack views them as the raw E8M0 byte. Packed nibble weights are
        # already uint8; keep both as torch so repack_mxfp4_blocks can view them.
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None and t.dtype == e8m0:
            return t.view(torch.uint8)
        # Packed MXFP4 nibbles arrive as int8. Keep the tensor so the repack
        # can view the same bytes as uint8; numpy would be the wrong type.
        if t.dtype in (torch.uint8, torch.int8):
            return t
        return t.numpy()


class HFSource:
    is_gguf = False

    def __init__(self, repo: str, cache_dir: Path, fetch: Fetch | None = None):
        self.repo = repo
        self.cache_dir = Path(cache_dir)
        self.fetch = fetch or _hf_fetch
        self._hparams: dict | None = None
        self._index: dict[str, str] | None = None
        # Several shard files at once. One file per layer, so this is what
        # overlaps the next layers' downloads with the layer being packed.
        n = int(os.environ.get("WP_FORGE_DOWNLOADS", "8"))
        self._dl_pool = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="hf-shard")
        self._dl_lock = threading.Lock()
        self._dl_futs: dict[str, Future] = {}

    def _download(self, filename: str, min_bytes: int) -> Path:
        path = self.cache_dir / filename
        if path.is_file() and path.stat().st_size >= min_bytes:
            return path
        return self.fetch(self.repo, filename, self.cache_dir)

    def _start(self, filename: str, min_bytes: int) -> Future:
        with self._dl_lock:
            fut = self._dl_futs.get(filename)
            if fut is None:
                fut = self._dl_pool.submit(self._download, filename, min_bytes)
                self._dl_futs[filename] = fut
            return fut

    def prefetch(self, filenames: list[str]) -> None:
        """Start downloads for shards the next layers will need. No waiting."""
        for name in filenames:
            path = self.cache_dir / name
            if path.is_file() and path.stat().st_size >= MIN_SHARD_BYTES:
                continue
            self._start(name, MIN_SHARD_BYTES)

    def _ensure(self, filename: str, min_bytes: int = 1) -> Path:
        if min_bytes >= MIN_SHARD_BYTES:
            return self._start(filename, min_bytes).result()
        path = self.cache_dir / filename
        if not path.is_file() or path.stat().st_size < min_bytes:
            path = self.fetch(self.repo, filename, self.cache_dir)
        return path

    # non-weight files the stock converter needs beside the peeled shards
    AUX_FILES = (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
        "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer.model",
    )

    def aux_files(self) -> list[Path]:
        """Fetch whichever of AUX_FILES the repo has (missing ones are skipped)."""
        out: list[Path] = []
        for name in self.AUX_FILES:
            try:
                out.append(self._ensure(name))
            except Exception:  # noqa: BLE001 -- 404 on a file this repo does not ship
                continue
        return out

    def hparams(self) -> dict:
        if self._hparams is None:
            self._hparams = flatten_hparams(json.loads(self._ensure("config.json").read_text()))
        return self._hparams

    def tensor_index(self) -> dict[str, str]:
        if self._index is None:
            self._index = dict(json.loads(self._ensure("model.safetensors.index.json").read_text())["weight_map"])
        return self._index

    def layer_shards(self, layer: int, prefix: str = "layers") -> list[str]:
        # prefix is the arch's main-stack layer prefix ("layers" for the
        # DeepSeek repos, "model.language_model.layers" for Qwen3.8)
        want = f"{prefix}.{layer}."
        seen: list[str] = []
        for name, shard in self.tensor_index().items():
            if name.startswith(want) and shard not in seen:
                seen.append(shard)
        return seen

    @contextmanager
    def open_shard(self, shard_id: str) -> Iterator[ShardReader]:
        from safetensors import safe_open
        path = self._ensure(shard_id, MIN_SHARD_BYTES)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            yield _SafetensorsReader(f)

    def release_shard(self, shard_id: str) -> None:
        (self.cache_dir / shard_id).unlink(missing_ok=True)


_REMOTE = re.compile(r"^[A-Za-z0-9_.-]{2,}:")


class GGUFSource:
    is_gguf = True

    def __init__(self, spec: str):
        if _REMOTE.match(spec):
            raise PlanError("gguf: remote sources (machine:path) are not supported in v1; "
                            "run wp-forge on the machine that holds the file")
        self.path = Path(spec)
        self._hparams: dict | None = None

    # non-weight files the stock converter needs beside the peeled shards
    AUX_FILES = (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "generation_config.json", "chat_template.jinja",
        "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer.model",
    )

    def aux_files(self) -> list[Path]:
        """Fetch whichever of AUX_FILES the repo has (missing ones are skipped)."""
        out: list[Path] = []
        for name in self.AUX_FILES:
            try:
                out.append(self._ensure(name))
            except Exception:  # noqa: BLE001 -- 404 on a file this repo does not ship
                continue
        return out

    def hparams(self) -> dict:
        if self._hparams is None:
            r = gguf.GGUFReader(str(self.path))
            arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
            hp: dict = {"gguf_arch": arch}
            prefix = arch + "."
            for key, field in r.fields.items():
                if key.startswith(prefix) and len(field.types) == 1 and field.types[0] != gguf.GGUFValueType.ARRAY:
                    part = field.parts[-1]
                    hp[key[len(prefix):]] = (bytes(part).decode() if field.types[0] == gguf.GGUFValueType.STRING
                                             else part[0].item())
            spec = ARCHS.get(arch)
            if spec is None:
                raise PlanError(f"gguf: arch {arch!r} has no wp_forge arch entry (known: {sorted(ARCHS)})")
            hp["architectures"] = [spec.hf_architectures[0]]
            self._hparams = hp
        return self._hparams

    def tensor_index(self) -> dict[str, str]:
        raise NotImplementedError("GGUF sources are read by the C++ tools")

    def layer_shards(self, layer: int, prefix: str = "layers") -> list[str]:
        raise NotImplementedError("GGUF sources are read by the C++ tools")

    def open_shard(self, shard_id: str):
        raise NotImplementedError("GGUF sources are read by the C++ tools")

    def release_shard(self, shard_id: str) -> None:
        pass
