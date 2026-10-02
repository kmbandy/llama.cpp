"""wp-forge convert path: any model -> one GGUF, quantized per tensor by rules.

Sources:
  hf:<repo>     the stock converter streams every tensor straight from the Hub
                (HTTP range reads, nothing but config/tokenizer files on disk)
  dir:<path>    a local HF checkout, same converter
  gguf:<path>   an existing GGUF, re-quantized tensor by tensor

For hf:/dir: sources the stock converter (conversion/, every architecture it
knows) runs in-process with its GGUFWriter swapped for ``ForgeWriter``. The
converter adds tensors exactly as it always does (lazy, nothing read yet);
ForgeWriter only records them. After the converter has also set its KV
metadata, ``finalize`` decides each tensor's type from the plan's
``QuantRules`` (rules.py) -- with the model's KV now known, so nextn/MTP
layers and the arch are visible -- and registers the result with the real
writer as deferred tensors. Writing then materializes one source tensor at a
time (the next one is fetched in the background while the current one is
quantized), quantizes it, writes it and drops it: peak memory is about two
source tensors plus their quantized forms, whatever the model's size.

Quantizers:
  ggml types    ggml_quant.quantize (libggml-base's ggml_quantize_chunk)
  ml8_4/ml8_3   ml8.quantize_experts_ml8_4 (a dense weight is a 1-expert stack)
  ml8_fp8       ml8.quantize_experts_ml8_fp8
Rotations for ml8 follow convert_fp8_rotated.py, the runtime's contract:
dense roles are block-Hadamard (attn_output/ffn_down/ssm_out, b = local_b)
or Kronecker (attn_q/k/v/qkv/gate, ffn_gate/up, output; one rotation per
(layer, input group) so weights reading the same activation share it);
routed experts use the plan's ml8.rotation per (layer, gate_up|down) exactly
like ExpertStage. token_embd takes ml8 unrotated (the runtime's MAD-256
path). Any other tensor asked for ml8 falls back to q8_0 -- the runtime has
no rotation role for it -- and the fallback is reported.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "gguf-py"), str(_REPO), str(_REPO / "scripts" / "calibration")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gguf  # noqa: E402
from gguf.gguf_writer import TensorInfo  # noqa: E402
from gguf.lazy import LazyBase  # noqa: E402

from . import ggml_quant  # noqa: E402
from .rules import ML8_TYPES, QuantRules, RulesError, storable  # noqa: E402

GQT = gguf.GGMLQuantizationType
F32_LIKE = {GQT.F32, GQT.F64, GQT.I8, GQT.I16, GQT.I32, GQT.I64}
# architectures whose llama.cpp loader registers ml8 sidecars today
# (src/models/*.cpp calling register_ml8_weight / load_ml8_sidecars); an ml8
# GGUF of any other arch is written fine but will not load until its loader does
ML8_RUNTIME_ARCHES = {"qwen35", "qwen35moe", "deepseek41"}
_EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


class ConvertError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ml8Opts:
    rotation: str = "kronecker"   # routed experts: kronecker | block_hadamard | none
    rotation_seed: int = 0
    max_b: int = 1024
    fit_rows: int = 65536
    local_b: int = 128            # block-Hadamard width for the dense K-split roles


@dataclass
class Decision:
    name: str
    shape: tuple[int, ...]        # numpy order
    src_type: str                 # what the converter / source GGUF had
    asked: str                    # what the rules asked for
    type: str                     # what is written
    rule: str | None = None
    fallback: str | None = None   # why type != asked
    rotation: str | None = None   # ml8 only: kronecker | block_hadamard | none
    bytes: int = 0                # weight + sidecars


# --- deferred tensors ------------------------------------------------------

class _Shared:
    """One computation feeding several consecutive tensors (an ml8 weight and
    its sidecars). Computed on first use, freed after the last part is taken."""

    def __init__(self, compute: Callable[[], tuple[np.ndarray, ...]], n_parts: int):
        self._compute = compute
        self._parts: tuple[np.ndarray, ...] | None = None
        self._left = n_parts
        self._lock = threading.Lock()

    def take(self, i: int) -> np.ndarray:
        with self._lock:
            if self._parts is None:
                self._parts = self._compute()
                self._compute = None  # type: ignore[assignment]
            part = self._parts[i]
            self._left -= 1
            if self._left == 0:
                self._parts = None
            return part


class Deferred:
    """Array stand-in GGUFWriter accepts: shape/dtype/nbytes now, bytes at tofile()."""

    def __init__(self, shape: tuple[int, ...], dtype: Any, thunk: Callable[[], np.ndarray],
                 on_written: Callable[[], None] | None = None):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self._thunk = thunk
        self._on_written = on_written

    def tofile(self, fout) -> None:
        arr = np.ascontiguousarray(self._thunk())
        self._thunk = None  # type: ignore[assignment]
        if arr.nbytes != self.nbytes:
            raise ConvertError(f"deferred tensor produced {arr.nbytes} bytes, registered {self.nbytes}")
        arr.tofile(fout)
        del arr
        if self._on_written is not None:
            self._on_written()

    def byteswap(self, inplace: bool = False):  # GGUFWriter only calls this on big-endian output
        raise ConvertError("wp-forge convert writes little-endian GGUFs only")


# --- source tensors --------------------------------------------------------

@dataclass
class Pending:
    name: str
    shape: tuple[int, ...]        # logical, numpy order
    src_qtype: GQT
    load: Callable[[], np.ndarray]  # -> f32 (or the source's own bytes for "keep")
    raw: Any                        # the object to pass through unchanged on "keep"
    raw_shape: Any = None


def _logical_shape(tensor: Any, raw_dtype: GQT | None) -> tuple[int, ...]:
    shape = tuple(int(s) for s in tensor.shape)
    if raw_dtype is not None and np.dtype(tensor.dtype) == np.uint8:
        return tuple(gguf.quant_shape_from_byte_shape(shape, raw_dtype))
    return shape


def _f32_of(arr: np.ndarray, qtype: GQT | None) -> np.ndarray:
    if np.dtype(arr.dtype).kind == "f" and qtype in (None, GQT.F32, GQT.F64, GQT.F16):
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(gguf.dequantize(np.asarray(arr), qtype), dtype=np.float32)


class _Prefetch:
    """Materializes pending tensor i+1 while tensor i is being quantized."""

    def __init__(self, pend: list[Pending]):
        self._pend = pend
        self._pos = {p.name: i for i, p in enumerate(pend)}
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wp-forge-fetch")
        self._fut: dict[int, Future] = {}
        self._lock = threading.Lock()

    def _start(self, i: int) -> None:
        if 0 <= i < len(self._pend) and i not in self._fut:
            self._fut[i] = self._ex.submit(self._pend[i].load)

    def f32(self, name: str) -> np.ndarray:
        i = self._pos[name]
        with self._lock:
            self._start(i)
            fut = self._fut.pop(i)
            self._start(i + 1)
        arr = fut.result()
        p = self._pend[i]
        # drop the lazy source: after evaluation it holds its parents' eager
        # arrays, which would otherwise live until the whole model is written
        p.load = None  # type: ignore[assignment]
        p.raw = None
        return _f32_of(arr, p.src_qtype)

    def close(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)


# --- the engine ------------------------------------------------------------

class Engine:
    def __init__(self, rules: QuantRules, ml8: Ml8Opts, events: Callable[[dict], None],
                 workers: int | None = None):
        self.rules = rules
        self.ml8 = ml8
        self.events = events
        self.workers = workers
        self.decisions: list[Decision] = []
        self._imatrix: dict[str, np.ndarray] | None = None

    # imatrix: llama-imatrix's GGUF output (<name>.in_sum2 + <name>.counts)
    def imatrix_for(self, name: str) -> np.ndarray | None:
        if not self.rules.imatrix:
            return None
        if self._imatrix is None:
            self._imatrix = load_imatrix(Path(self.rules.imatrix))
        return self._imatrix.get(name)

    def decide(self, p: Pending, kv: dict[str, Any]) -> Decision:
        n_dims = len(p.shape)
        asked, rule = self.rules.decide(p.name, n_dims, p.src_qtype in F32_LIKE)
        d = Decision(p.name, p.shape, p.src_qtype.name.lower(), asked, asked, rule)
        if asked in ("keep", "drop"):
            return d
        n_per_row = p.shape[-1]
        if asked in ML8_TYPES:
            why = self._ml8_unsupported(p, kv, asked)
            if why is None and not storable(asked, n_per_row):
                why = f"row {n_per_row} not a multiple of the ml8 block"
            if why is None:
                d.rotation = self._ml8_rotation_kind(p)
                return d
            return self._fallback(d, n_per_row, why)
        if not storable(asked, n_per_row):
            return self._fallback(d, n_per_row, f"row {n_per_row} not a multiple of {asked}'s block")
        if ggml_quant.requires_imatrix(asked) and self.imatrix_for(p.name) is None:
            raise RulesError(f"{p.name}: {asked} needs an importance matrix entry (quant.imatrix)")
        return d

    def _fallback(self, d: Decision, n_per_row: int, why: str) -> Decision:
        d.type = "q8_0" if storable("q8_0", n_per_row) and d.asked != "q8_0" else "keep"
        d.fallback = why
        return d

    def _ml8_unsupported(self, p: Pending, kv: dict[str, Any], asked: str) -> str | None:
        if _EXPS_RE.match(p.name):
            return None if len(p.shape) == 3 else f"expert tensor of {len(p.shape)} dims"
        if len(p.shape) != 2:
            return f"{len(p.shape)}-dim tensor (ml8 takes 2-D weights and 3-D routed experts)"
        if p.name == "token_embd.weight":
            return "token_embd has no ml8_fp8 path" if asked == "ml8_fp8" else None
        from convert_fp8_rotated import classify_tensor
        action, role = classify_tensor(p.name, (p.shape[1], p.shape[0]), _first_nextn(kv),
                                       "ml8_4", _kv(kv, "general.architecture"))
        if action in ("rotate_kronecker", "rotate_hadamard"):
            return None
        if action == "q8_0" and role is not None and _is_nextn(p.name, kv):
            return "nextn/MTP layer (the runtime loads it without ml8 sidecars)"
        return f"role {role!r} has no ml8 rotation in the runtime"

    def _ml8_rotation_kind(self, p: Pending) -> str:
        if self.ml8.rotation == "none" or p.name == "token_embd.weight":
            return "none"
        m = _EXPS_RE.match(p.name)
        if m:
            return self.ml8.rotation
        from convert_fp8_rotated import classify_tensor
        action, _ = classify_tensor(p.name, (p.shape[1], p.shape[0]), None, "ml8_4", None)
        if action == "rotate_hadamard" and p.shape[1] % self.ml8.local_b == 0:
            return "block_hadamard"
        return "kronecker"

    # -- emission: (name, tensor, raw_shape, raw_dtype) for GGUFWriter.add_tensor
    def emit(self, d: Decision, p: Pending, fetch: _Prefetch, done: Callable[[Decision], None]):
        if d.type == "drop":
            done(d)
            return []
        if d.type == "keep":
            d.bytes = int(p.raw.nbytes)
            return [(p.name, _Passthrough(p, fetch, lambda: done(d)), p.raw_shape, p.src_qtype)]
        if d.type in ML8_TYPES:
            return self._emit_ml8(d, p, fetch, done)
        qt = GQT[d.type.upper()]
        rb = ggml_quant.row_bytes(d.type, p.shape[-1])
        byte_shape = (*p.shape[:-1], rb)
        im = self.imatrix_for(p.name)

        def thunk() -> np.ndarray:
            f32 = fetch.f32(p.name)
            if im is not None and im.ndim == 2 and f32.ndim == 3:  # per-expert imatrix
                return np.stack([ggml_quant.quantize(f32[e], d.type, im[e], self.workers) for e in range(f32.shape[0])])
            return ggml_quant.quantize(f32, d.type, im if im is None or im.ndim == 1 else im[0], self.workers)

        d.bytes = int(np.prod(byte_shape))
        return [(p.name, Deferred(byte_shape, np.uint8, thunk, lambda: done(d)), None, qt)]

    def _emit_ml8(self, d: Decision, p: Pending, fetch: _Prefetch, done: Callable[[Decision], None]):
        from . import ml8 as ml8mod
        from convert_fp8_rotated import _group_seed, _parse_layer, role_group_key, tensor_role
        from kronecker_rotation import (BLOCK_HADAMARD_KIND_ID, KRONECKER_ORTH_SYLVESTER_KIND_ID,
                                        BlockHadamardRotation, KroneckerRotation, factor_for_dim,
                                        random_orthogonal)

        experts = len(p.shape) == 3
        k = p.shape[-1]
        kind = d.rotation or "none"
        o = self.ml8
        if kind == "kronecker":
            a, b = factor_for_dim(k, max_b=o.max_b)
            kind_id = KRONECKER_ORTH_SYLVESTER_KIND_ID
        elif kind == "block_hadamard":
            if experts:
                a, b = factor_for_dim(k, max_b=o.max_b)
            else:
                a, b = k // o.local_b, o.local_b
            kind_id = BLOCK_HADAMARD_KIND_ID
        else:
            a = b = kind_id = 0

        def rotation():
            if kind == "none":
                return None
            if experts:
                m = _EXPS_RE.match(p.name)
                group = "down" if m.group(2) == "down" else "gate_up"
                return ml8mod.build_rotation(k, kind, o.rotation_seed, o.max_b, int(m.group(1)), group)[0]
            if kind == "block_hadamard":
                return BlockHadamardRotation(in_features=k, b_dim=b)
            role, _ = tensor_role(p.name)
            seed = _group_seed(o.rotation_seed, _parse_layer(p.name), role_group_key(role))
            return KroneckerRotation(h_a=random_orthogonal(a, seed=seed), b_dim=b)

        n_c = ML8_TYPES[d.type]
        n_e = p.shape[0] if experts else 1
        base = p.name[:-len(".weight")] if p.name.endswith(".weight") else p.name

        def compute():
            f32 = fetch.f32(p.name)
            stack = f32 if experts else f32[None]
            rot = rotation()
            if d.type == "ml8_fp8":
                packed = ml8mod.quantize_experts_ml8_fp8(stack, rot)
                cents = None
            else:
                packed, cents = ml8mod.quantize_experts_ml8_4(stack, rot, o.fit_rows, n_centroids=n_c)
            del stack, f32
            if not experts:
                packed = packed[0]
                cents = cents[0] if cents is not None else None
            out = [packed]
            if cents is not None:
                out.append(cents)
            if kind == "kronecker":
                out.append(ml8mod.rotation_h_a_array(rot))
            if kind != "none":
                out.append(ml8mod.rotation_meta_bytes(a, b, k, kind_id))
            return tuple(out)

        if d.type == "ml8_fp8":
            qt, row_b = GQT.ML8_FP8, k // 32 * 34
        else:
            qt, row_b = GQT.ML8_4, k // 64 * 36
        specs: list[tuple[str, tuple[int, ...], Any, GQT | None]] = [
            (p.name, (*p.shape[:-1], row_b), np.uint8, qt)]
        if d.type != "ml8_fp8":
            cshape = (n_e, k // 64, 16) if experts else (k // 64, 16)
            specs.append((f"{base}.centroids", cshape, np.uint8, GQT.F8_E4M3))
        if kind == "kronecker":
            specs.append((f"{base}.rotation_h_a", (a, a), np.float32, None))
        if kind != "none":
            specs.append((f"{base}.rotation_meta", (4,), np.int32, None))
        shared = _Shared(compute, len(specs))
        d.bytes = sum(int(np.prod(s)) * np.dtype(dt).itemsize for _, s, dt, _ in specs)
        out = []
        for i, (n, s, dt, q) in enumerate(specs):
            last = i == len(specs) - 1
            out.append((n, Deferred(s, dt, (lambda i=i: shared.take(i)),
                                    (lambda: done(d)) if last else None), None, q))
        return out


class _Passthrough(Deferred):
    """'keep': the source's own bytes, unchanged (the converter's lazy array,
    or a GGUF reader's mmap slice)."""

    def __init__(self, p: Pending, fetch: _Prefetch, on_written: Callable[[], None]):
        self.shape = tuple(int(s) for s in p.raw.shape)
        self.dtype = np.dtype(p.raw.dtype)
        self.nbytes = int(p.raw.nbytes)
        self._p = p
        self._on_written = on_written

    def tofile(self, fout) -> None:
        raw = self._p.raw
        arr = LazyBase.to_eager(raw) if isinstance(raw, LazyBase) else raw
        np.ascontiguousarray(arr).tofile(fout)
        self._p.raw = None
        self._on_written()


# --- helpers ---------------------------------------------------------------

def _kv(kv: dict[str, Any], key: str) -> Any:
    v = kv.get(key)
    return getattr(v, "value", v)


def _first_nextn(kv: dict[str, Any]) -> int | None:
    arch = _kv(kv, "general.architecture")
    bc = _kv(kv, f"{arch}.block_count")
    nn = _kv(kv, f"{arch}.nextn_predict_layers") or 0
    return int(bc) - int(nn) if bc is not None and nn else None


def _is_nextn(name: str, kv: dict[str, Any]) -> bool:
    fn = _first_nextn(kv)
    m = re.match(r"^blk\.(\d+)\.", name)
    return fn is not None and m is not None and int(m.group(1)) >= fn


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    """llama-imatrix GGUF: per weight `<name>.in_sum2` [n_mat, n_per_row] and
    `<name>.counts` [n_mat]; the importance is in_sum2 / counts per row. A 2-D
    result row per expert for MoE weights, else a 1-D vector."""
    r = gguf.GGUFReader(str(path))
    t = {x.name: x for x in r.tensors}
    out: dict[str, np.ndarray] = {}
    for n, x in t.items():
        if not n.endswith(".in_sum2"):
            continue
        w = n[:-len(".in_sum2")]
        c = t.get(w + ".counts")
        s2 = np.array(x.data, dtype=np.float32).reshape(-1, int(x.shape[0]))
        cnt = np.array(c.data, dtype=np.float32).reshape(-1, 1) if c is not None else np.ones((s2.shape[0], 1), np.float32)
        val = s2 / np.maximum(cnt, 1.0)
        out[w] = val[0] if val.shape[0] == 1 else val
    return out


# --- the converter's writer ------------------------------------------------

class ForgeWriter(gguf.GGUFWriter):
    """GGUFWriter that holds the converter's tensors until finalize()."""

    _forge_pending: list[Pending]
    _forge_names: set[str]

    @classmethod
    def adopt(cls, w: gguf.GGUFWriter) -> "ForgeWriter":
        w.__class__ = cls
        w._forge_pending = []  # type: ignore[attr-defined]
        w._forge_names = set()  # type: ignore[attr-defined]
        return w  # type: ignore[return-value]

    def add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None, tensor_endianess=None):
        if name in self._forge_names:
            raise ConvertError(f"duplicated tensor {name!r}")
        self._forge_names.add(name)
        qt = raw_dtype if raw_dtype is not None else _np_qtype(tensor.dtype)
        shape = _logical_shape(tensor, raw_dtype)

        def load(t=tensor) -> np.ndarray:
            return LazyBase.to_eager(t) if isinstance(t, LazyBase) else np.asarray(t)

        self._forge_pending.append(Pending(name, shape, qt, load, tensor, raw_shape))

    def get_total_parameter_count(self):
        saved = self.tensors
        self.tensors = [{p.name: TensorInfo(shape=p.shape, dtype=p.src_qtype, nbytes=0)
                         for p in self._forge_pending}]
        try:
            return super().get_total_parameter_count()
        finally:
            self.tensors = saved

    def pending(self) -> list[Pending]:
        return self._forge_pending


def _np_qtype(dt) -> GQT:
    return {np.dtype(np.float32): GQT.F32, np.dtype(np.float16): GQT.F16,
            np.dtype(np.float64): GQT.F64, np.dtype(np.int8): GQT.I8,
            np.dtype(np.int16): GQT.I16, np.dtype(np.int32): GQT.I32,
            np.dtype(np.int64): GQT.I64}.get(np.dtype(dt), GQT.F32)


# --- the job ---------------------------------------------------------------

@dataclass
class ConvertResult:
    out_path: str
    bytes: int
    secs: float
    decisions: list[Decision] = field(default_factory=list)

    def summary(self) -> dict:
        by: dict[str, dict] = {}
        for d in self.decisions:
            s = by.setdefault(d.type, {"tensors": 0, "bytes": 0})
            s["tensors"] += 1
            s["bytes"] += d.bytes
        return {"out": self.out_path, "bytes": self.bytes, "secs": round(self.secs, 1),
                "by_type": by, "fallbacks": sum(1 for d in self.decisions if d.fallback)}


CONVERTER_OPTS = ("mtp", "fuse_qkv", "fuse_gate_up_exps", "fp8_as_q8")


def _hf_model(source: str, cache_dir: Path | None, opts: dict[str, Any] | None = None):
    """(model_instance, writer) for an hf:/dir: source, converter not yet run.
    opts: the plan's converter: block. mtp: auto (default) exports MTP/nextn
    layers unless the config says there are none; true / false force it."""
    opts = dict(opts or {})
    import torch  # noqa: F401
    from conversion import ModelBase, ModelType, get_model_architecture, get_model_class

    if source.startswith("hf:"):
        from huggingface_hub import snapshot_download
        repo = source[3:]
        local = snapshot_download(repo_id=repo, cache_dir=str(cache_dir) if cache_dir else None,
                                  allow_patterns=["LICENSE", "*.json", "*.md", "*.txt", "*.jinja",
                                                  "tokenizer.model"])
        dir_model, remote = Path(local), repo
    else:
        dir_model, remote = Path(source[4:]), None
    hparams = ModelBase.load_hparams(dir_model, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    cls = get_model_class(arch, mmproj=False)
    mtp = str(opts.pop("mtp", "auto")).lower()
    if getattr(cls, "supports_mtp_export", False):
        flat = {**hparams, **(hparams.get("text_config") or {})}
        cls.no_mtp = mtp == "false" or (mtp == "auto" and flat.get("mtp_num_hidden_layers") == 0)
        cls.mtp_only = False
    elif mtp == "true":
        raise ConvertError(f"converter.mtp: {arch} has no MTP export")
    model = cls(dir_model, gguf.LlamaFileType.GUESSED, Path("unused.gguf"),
                remote_hf_model_id=remote, hparams=hparams,
                **{k: bool(v) for k, v in opts.items()})
    return model, ForgeWriter.adopt(model.gguf_writer)


def _gguf_pending(path: Path, w: gguf.GGUFWriter) -> list[Pending]:
    from convert_fp8_rotated import _SKIP_FIELDS, _copy_field
    r = gguf.GGUFReader(str(path))
    for name, f in r.fields.items():
        if name not in _SKIP_FIELDS:
            _copy_field(w, name, f)
    out = []
    for t in r.tensors:
        shape = tuple(int(s) for s in reversed(list(t.shape)))  # ne -> numpy order
        out.append(Pending(t.name, shape, t.tensor_type, (lambda t=t: np.asarray(t.data)), t.data, None))
    return out


def convert(source: str, out_path: Path, rules: QuantRules, ml8: Ml8Opts = Ml8Opts(), **kw) -> ConvertResult:
    """See the module docstring. kw: events, dry_run, cache_dir, workers, stamp, converter."""
    import torch
    with torch.inference_mode():
        return _convert(source, out_path, rules, ml8, **kw)


def _convert(source: str, out_path: Path, rules: QuantRules, ml8: Ml8Opts, *,
             events: Callable[[dict], None] = lambda e: None, dry_run: bool = False,
             cache_dir: Path | None = None, workers: int | None = None,
             stamp: dict[str, Any] | None = None, converter: dict[str, Any] | None = None) -> ConvertResult:
    t0 = time.time()
    if source.startswith(("hf:", "dir:")):
        model, writer = _hf_model(source, cache_dir, converter)
        model.prepare_tensors()
        model.prepare_metadata(vocab_only=False)
        pend = writer.pending()
        writer.tensors = [{}]
    elif source.startswith("gguf:"):
        r_arch = gguf.GGUFReader(source[5:]).fields["general.architecture"].contents()
        writer = gguf.GGUFWriter(path=None, arch=r_arch)
        pend = _gguf_pending(Path(source[5:]), writer)
    else:
        raise ConvertError(f"source {source!r}: want hf:<repo>, dir:<path> or gguf:<path>")

    kv = {k: v for d in writer.kv_data for k, v in d.items()}
    arch = _kv(kv, "general.architecture")
    eng = Engine(rules, ml8, events, workers)
    decisions = [eng.decide(p, kv) for p in pend]
    for d in decisions:
        if d.fallback:
            events({"kind": "fallback", "tensor": d.name, "asked": d.asked, "type": d.type, "why": d.fallback})
    if any(d.type in ML8_TYPES for d in decisions) and arch not in ML8_RUNTIME_ARCHES:
        events({"kind": "warning", "msg": f"arch {arch!r}: llama.cpp does not load ml8 sidecars for this "
                                         f"arch yet (supported: {sorted(ML8_RUNTIME_ARCHES)})"})

    fetch = _Prefetch([p for p, d in zip(pend, decisions) if d.type not in ("keep", "drop")])
    n_done = [0]
    n_total = sum(1 for d in decisions if d.type != "drop")

    def done(d: Decision) -> None:
        n_done[0] += 1
        events({"kind": "tensor_done", "tensor": d.name, "type": d.type, "bytes": d.bytes,
                "n": n_done[0], "of": n_total, "secs": round(time.time() - t0, 1)})

    for p, d in zip(pend, decisions):
        for name, tensor, raw_shape, raw_dtype in eng.emit(d, p, fetch, done):
            gguf.GGUFWriter.add_tensor(writer, name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)
    writer.add_string("wp_forge.recipe", json.dumps({
        "source": source, "default": rules.default,
        "rules": [{"match": r.match, "type": r.type} for r in rules.rules],
        "ml8": asdict(ml8), "imatrix": rules.imatrix}))
    for k, v in (stamp or {}).items():
        writer.add_string(f"wp_forge.{k}", str(v))

    total = sum(d.bytes for d in decisions)
    if dry_run:
        fetch.close()
        return ConvertResult(str(out_path), total, time.time() - t0, decisions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    try:
        writer.write_header_to_file(path=tmp)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
    finally:
        fetch.close()
    tmp.replace(out_path)
    return ConvertResult(str(out_path), out_path.stat().st_size, time.time() - t0, decisions)
