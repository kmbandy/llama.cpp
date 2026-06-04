import subprocess, sys, hashlib
from pathlib import Path
import pytest

CALIB = Path(__file__).parent / "calibrate_ml8_paged.py"
MODEL = Path("/home/kmbandy/models/Qwen3.5-0.8B-hf")
GGUF  = Path("/home/kmbandy/models/Qwen3.5-0.8B-bf16.gguf")

def _run(out_dir):
    env_tier = "token_embd=ml8,ssm_out=fp8,ffn_down=fp8,attn_v=fp8"
    cmd = [sys.executable, str(CALIB),
           "--model", str(MODEL), "--gguf", str(GGUF), "--arch", "qwen35",
           "--device", "cpu", "--strategy", "dense", "--output-dir", str(out_dir),
           "--rotation", "kronecker", "--group-size", "64", "--n-centroids", "16",
           "--percdamp", "0.05", "--fit-loss", "mse", "--dense-coverage", "full",
           "--faithful-acts", "--faithful-weights", "--awq", "none",
           "--corpus", "wiki", "--seq-len", "512", "--token-budget", "2048",
           "--no-resume", "--hessian-mode", "per-target", "--max-layers", "1",
           "--resident"]
    import os
    env = {**os.environ, "ML8_DETERMINISTIC": "1", "ML8_TIER_OVERRIDE": env_tier,
           "PYTHONPATH": str(Path(__file__).parents[2] / "gguf-py")}
    subprocess.run(cmd, check=True, env=env, cwd=str(Path(__file__).parents[2]))

def _blob_hashes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).glob("*.pt"))}

@pytest.mark.slow
def test_per_target_deterministic_and_refactor_safe(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _run(a); _run(b)
    ha, hb = _blob_hashes(a), _blob_hashes(b)
    assert ha and ha == hb, f"per-target blobs not bit-identical across runs: {ha} vs {hb}"
