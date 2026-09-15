from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import gguf
from conversion.wp_forge.plan import PlanError
from conversion.wp_forge.source import GGUFSource, HFSource
from synth import make_synthetic_hf_repo


@pytest.fixture
def hf(tmp_path: Path):
    hub = make_synthetic_hf_repo(tmp_path / "hub")
    calls: list[str] = []

    def fetch(repo: str, filename: str, dest_dir: Path) -> Path:
        calls.append(filename)
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    return HFSource("fake/repo", tmp_path / "cache", fetch=fetch), calls


def test_hparams_index_and_layer_shards(hf) -> None:
    src, _ = hf
    hp = src.hparams()
    assert hp["num_hidden_layers"] == 2 and hp["architectures"] == ["DeepseekV41ForCausalLM"]
    idx = src.tensor_index()
    assert "layers.0.ffn.experts.0.w1.weight" in idx and "mtp.0.attn.wq.weight" in idx
    assert src.layer_shards(1) == ["model-00002-of-00004.safetensors"]
    assert not src.is_gguf


def test_open_shard_reads_bf16_as_f32(hf) -> None:
    src, _ = hf
    with src.open_shard(src.layer_shards(0)[0]) as r:
        a = r.get_tensor("layers.0.ffn.experts.0.w1.weight")
        assert a.shape == (64, 32) and a.dtype == np.float32
        assert r.keys()


def test_release_shard_refetches(hf) -> None:
    src, calls = hf
    shard = src.layer_shards(0)[0]
    with src.open_shard(shard):
        pass
    assert (src.cache_dir / shard).is_file()
    src.release_shard(shard)
    assert not (src.cache_dir / shard).exists()
    with src.open_shard(shard):
        pass
    assert calls.count(shard) == 2


def test_gguf_source(tmp_path: Path) -> None:
    p = tmp_path / "spine.gguf"
    w = gguf.GGUFWriter(str(p), arch="deepseek41")
    w.add_block_count(2)
    w.add_embedding_length(32)
    w.add_expert_count(4)
    w.add_expert_feed_forward_length(64)
    w.add_expert_used_count(2)
    w.add_tensor("output_norm.weight", np.ones(32, dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    src = GGUFSource(str(p))
    hp = src.hparams()
    assert src.is_gguf and hp["architectures"] == ["DeepseekV41ForCausalLM"]
    assert hp["block_count"] == 2 and hp["expert_feed_forward_length"] == 64
    with pytest.raises(NotImplementedError):
        src.tensor_index()
    with pytest.raises(PlanError, match="remote"):
        GGUFSource("mad-lab-2026:/x.gguf")
