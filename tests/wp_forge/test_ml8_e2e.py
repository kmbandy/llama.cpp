"""ml8 (rotated ML8_FP8 / ML8_4) acceptance test: the REAL pipeline.

wp_forge builds a tiny one-layer GGUF with ml8 rotation (kronecker and
block_hadamard) -> llama-wp-repack -> llama-wp-expert-descriptor ->
llama-wp-expert-worker (CPU, the real binary, as a subprocess) serves a real
compute request over the wire protocol (via the llama-wp-worker-verify C++
client, which reuses pipe-protocol.h so this file does not reimplement the
wire format) -> the worker's output is compared to a numpy reference built
from the ORIGINAL unrotated f32 expert weights and unrotated input:

    y = sum_e route_weight_e * down_e(silu(gate_e @ x) * (up_e @ x))

This is the acceptance test for the default per-expert compute path wired in
tools/wp-expert-worker/wp-expert-worker.cpp (compute_batch's per-expert
loop): rotate gate/up's shared input once, rotate down's independent
rotation (if any) after swiglu, route ML8_4 roles through
ggml_ml8_mul_mat(weight, centroids, x) and everything else through plain
ggml_mul_mat.

Skipped unless the CPU tools are built (Tools.discover(), plus
llama-wp-expert-worker / llama-wp-worker-verify under build-cpu/bin).
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
from safetensors import safe_open

from conversion.wp_forge.arch import ARCHS
from conversion.wp_forge.plan import LayerRange, Ml8Settings, SetSpec
from conversion.wp_forge.sink import LocalSink
from conversion.wp_forge.source import HFSource
from conversion.wp_forge.stages import ExpertStage
from conversion.wp_forge.tools import REPO_ROOT, Tools
from synth import make_synthetic_hf_repo

# ml8_4's QK_ML8=64 needs n_embd (gate/up K) and n_ff (down K) both %64==0.
# max_b=64 -> factor_for_dim gives a=4 (gate/up, K=256) and a=2 (down,
# K=128) -- both >1, so the rotation is non-trivial (not a K=1,b=K no-op).
N_LAYER, NEXTN, N_EXPERT, N_FF, N_EMBD = 1, 1, 4, 128, 256
N_EXPERT_USED = N_EXPERT
MAX_B = 64
N_TOKENS = 3


def _worker_bin() -> Path:
    return REPO_ROOT / "build-cpu" / "bin" / "llama-wp-expert-worker"


def _verify_bin() -> Path:
    return REPO_ROOT / "build-cpu" / "bin" / "llama-wp-worker-verify"


@pytest.fixture
def env(tmp_path: Path):
    try:
        tools = Tools.discover()
    except FileNotFoundError:
        pytest.skip("wp-forge C++ tools not built (Tools.discover raised)")
    if not _worker_bin().exists() or not _verify_bin().exists():
        pytest.skip("llama-wp-expert-worker / llama-wp-worker-verify not built "
                     "(cmake --build build-cpu --target llama-wp-expert-worker llama-wp-worker-verify)")

    hub = make_synthetic_hf_repo(
        tmp_path / "hub", n_layer=N_LAYER, n_expert=N_EXPERT,
        n_ff=N_FF, n_embd=N_EMBD, nextn=NEXTN, engram=False,
    )

    def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    source = HFSource("fake/repo", tmp_path / "cache", fetch=fetch)

    spine = tmp_path / "spine.gguf"
    w = gguf.GGUFWriter(str(spine), arch="deepseek41")
    w.add_block_count(N_LAYER + NEXTN)
    w.add_embedding_length(N_EMBD)
    w.add_expert_count(N_EXPERT)
    w.add_expert_feed_forward_length(N_FF)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_name("wp-forge-ml8-e2e-spine")
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    rplan = SimpleNamespace(
        arch=ARCHS["deepseek41"],
        hparams=source.hparams(),
        spine_path=str(spine),
        sidecar_paths={},
        plan=SimpleNamespace(source="fake/repo", name="synthetic", quant={}),
    )
    return SimpleNamespace(source=source, rplan=rplan, tools=tools, tmp_path=tmp_path, hub=hub)


# --------------------------------------------------------------------- build

def _build_layer(env, quant: str, rotation: str, tag: str) -> tuple[Path, Path]:
    """Runs the real ExpertStage pipeline (quant -> 1-layer GGUF ->
    llama-wp-repack -> stitch -> llama-wp-expert-descriptor) and returns
    (manifest_path, descriptor_path)."""
    set_id = f"L0-{tag}"
    set_spec = SetSpec(
        id=set_id, role="experts", layers=LayerRange(0, N_LAYER - 1),
        slice_index=None, widths=None, machine="box-a", dir=f"/models/{set_id}",
        output_base=set_id, quant=quant, est_bytes=0,
        ml8=Ml8Settings(rotation=rotation, rotation_seed=7, max_b=MAX_B, fit_rows=N_EXPERT * 8),
    )
    sink = LocalSink(str(env.tmp_path / set_id))
    sink.mkdir()
    events: list[dict] = []
    stage = ExpertStage(
        [set_spec], env.rplan, env.source, env.tools, {set_id: sink},
        events.append, env.tmp_path / f"work-{tag}",
    )
    res = stage.run()
    r = res[set_id]
    assert r.descriptor_rel is not None, f"descriptor tool rejected the manifest: {events}"
    manifest = env.tmp_path / set_id / r.manifest_rel
    descriptor = env.tmp_path / set_id / r.descriptor_rel
    assert manifest.is_file() and descriptor.is_file()
    return manifest, descriptor


def _strip_rotation(descriptor_path: Path, out_path: Path) -> None:
    """Copies a descriptor with every role's "rotation" object removed, but
    the weight bytes (rotation-baked at quantize time) left untouched -- the
    "what if the runtime rotation had never cancelled" control case."""
    desc = json.loads(descriptor_path.read_text())
    for layer in desc["layers"]:
        for role in layer["roles"].values():
            role.pop("rotation", None)
    out_path.write_text(json.dumps(desc))


# ------------------------------------------------------------------- worker

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Worker:
    def __init__(self, manifest: Path, descriptor: Path, slots: int = 8):
        self.port = _free_port()
        self.proc = subprocess.Popen(
            [str(_worker_bin()),
             "--shard-manifest", str(manifest),
             "--descriptor", str(descriptor),
             "--device", "CPU",
             "--listen", f"127.0.0.1:{self.port}",
             "--slots", str(slots)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # Fail fast (rather than waiting out llama-wp-worker-verify's own
        # ~10s connect retry) if the worker exits immediately, e.g. bad JSON.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out, err = self.proc.communicate()
                raise RuntimeError(
                    f"worker exited early rc={self.proc.returncode}\nstdout={out}\nstderr={err}")
            time.sleep(0.05)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def _write_request(
        path: Path, layer: int, n_tokens: int, n_embd: int, swiglu_clamp: float,
        assignments: list[tuple[int, np.ndarray]], activations: np.ndarray) -> None:
    lines = [f"{layer} {n_tokens} {n_embd} {swiglu_clamp!r}", str(len(assignments))]
    for eid, weights in assignments:
        assert len(weights) == n_tokens
        lines.append(f"{eid} " + " ".join(f"{float(w)!r}" for w in weights))
    lines.append(" ".join(f"{float(v)!r}" for v in activations.reshape(-1)))
    path.write_text("\n".join(lines) + "\n")


def _dispatch(worker: _Worker, tmp_path: Path, tag: str, layer: int, n_tokens: int,
              n_embd: int, assignments: list[tuple[int, np.ndarray]],
              activations: np.ndarray) -> np.ndarray:
    req_path = tmp_path / f"req-{tag}.txt"
    out_path = tmp_path / f"out-{tag}.f32"
    _write_request(req_path, layer, n_tokens, n_embd, 0.0, assignments, activations)
    proc = subprocess.run(
        [str(_verify_bin()), "127.0.0.1", str(worker.port), str(req_path), str(out_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"wp-worker-verify failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    data = np.fromfile(out_path, dtype="<f4")
    assert data.size == n_tokens * n_embd
    return data.reshape(n_tokens, n_embd)


# ---------------------------------------------------------------- reference

def _load_expert_weights(hub: Path) -> list[dict[str, np.ndarray]]:
    # write() in synth.py numbers shards 1-based in draw order: layer 0's
    # shard is always the first one written (shard_name(0)).
    n_shards = N_LAYER + NEXTN + 1
    shard = hub / f"model-{1:05d}-of-{n_shards:05d}.safetensors"
    weights = []
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for e in range(N_EXPERT):
            weights.append({
                "gate": f.get_tensor(f"layers.0.ffn.experts.{e}.w1.weight").float().numpy(),
                "up":   f.get_tensor(f"layers.0.ffn.experts.{e}.w3.weight").float().numpy(),
                "down": f.get_tensor(f"layers.0.ffn.experts.{e}.w2.weight").float().numpy(),
            })
    return weights


def _reference(weights: list[dict[str, np.ndarray]], x: np.ndarray,
                assignments: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """y = sum_e route_weight_e(t) * down_e(silu(gate_e @ x) * (up_e @ x)),
    computed in f64 from the ORIGINAL (unrotated, unquantized) weights and
    the ORIGINAL (unrotated) input -- no rotation, no quantization."""
    n_tokens, _ = x.shape
    y = np.zeros((n_tokens, weights[0]["down"].shape[0]), dtype=np.float64)
    xf = x.astype(np.float64)
    for eid, route_w in assignments:
        w = weights[eid]
        gate = xf @ w["gate"].astype(np.float64).T   # [n_tokens, n_ff]
        up   = xf @ w["up"].astype(np.float64).T     # [n_tokens, n_ff]
        silu = gate / (1.0 + np.exp(-gate))
        hidden = silu * up
        out = hidden @ w["down"].astype(np.float64).T   # [n_tokens, n_embd]
        y += out * np.asarray(route_w, dtype=np.float64)[:, None]
    return y


def _rel_l2(actual: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(actual.astype(np.float64) - ref) / np.linalg.norm(ref))


# ------------------------------------------------------------------- cases

# (quant type, rotation, asserted max relative L2 error). Bounds are
# measured-with-margin: report the actual numbers on failure via the assert
# message rather than encoding tight thresholds that would be brittle to
# unrelated calibration changes.
CASES = [
    # Measured 2026-09-27 at this test's tiny geometry (n_embd=256, n_ff=128,
    # 4 experts): ML8_FP8 ~0.0475/0.0474 (kronecker/block_hadamard), ML8_4
    # ~0.171/0.173. Bounds below keep comfortable margin above those.
    ("ml8_fp8", "kronecker", 0.08),
    ("ml8_fp8", "block_hadamard", 0.08),
    ("ml8_4", "kronecker", 0.30),
    ("ml8_4", "block_hadamard", 0.30),
]


@pytest.mark.parametrize("quant,rotation,max_rel_err", CASES)
def test_ml8_default_path_matches_float_reference(env, quant, rotation, max_rel_err):
    manifest, descriptor = _build_layer(env, quant, rotation, f"{quant}-{rotation}")
    weights = _load_expert_weights(env.hub)

    rng = np.random.default_rng(1234)
    x = rng.standard_normal((N_TOKENS, N_EMBD)).astype(np.float32)
    assignments = [
        (e, np.full(N_TOKENS, 0.15 + 0.05 * e, dtype=np.float32))
        for e in range(N_EXPERT)
    ]
    ref = _reference(weights, x, assignments)

    worker = _Worker(manifest, descriptor)
    try:
        actual = _dispatch(worker, env.tmp_path, "correct", 0, N_TOKENS, N_EMBD, assignments, x)
    finally:
        worker.close()

    rel_err = _rel_l2(actual, ref)
    assert rel_err < max_rel_err, (
        f"{quant}/{rotation}: relative L2 error {rel_err:.4g} exceeds bound {max_rel_err} "
        f"(actual[0,:4]={actual[0, :4]}, ref[0,:4]={ref[0, :4]})"
    )

    # --- prove the rotation actually matters: same manifest (same,
    # rotation-baked weight bytes), same request, but a descriptor with the
    # rotation stripped -- the worker then feeds an UNROTATED activation
    # into weights quantized/calibrated in the ROTATED basis. If the
    # default path's rotation wiring were a no-op (or wrong), this second
    # run would score about as well as the first; it should instead be
    # dramatically worse.
    stripped = env.tmp_path / f"descriptor-norot-{quant}-{rotation}.json"
    _strip_rotation(descriptor, stripped)
    worker_norot = _Worker(manifest, stripped)
    try:
        actual_norot = _dispatch(
            worker_norot, env.tmp_path, "norot", 0, N_TOKENS, N_EMBD, assignments, x)
    finally:
        worker_norot.close()
    rel_err_norot = _rel_l2(actual_norot, ref)
    assert rel_err_norot > 5.0 * max(rel_err, 1e-6) and rel_err_norot > 0.3, (
        f"{quant}/{rotation}: stripping the rotation from the descriptor did not make the "
        f"output much worse (with-rotation rel_err={rel_err:.4g}, without={rel_err_norot:.4g}) "
        f"-- the runtime rotation wiring may not be doing anything"
    )
