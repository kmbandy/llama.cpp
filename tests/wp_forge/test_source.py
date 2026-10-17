from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import gguf
from conversion.wp_forge import source as source_mod
from conversion.wp_forge.cli import make_source
from conversion.wp_forge.convert import ConvertError, _hf_model
from conversion.wp_forge.plan import PlanError
from conversion.wp_forge.source import GGUFSource, HFSource, parse_hf_spec
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
    # 5 shards: 2 layers + 1 MTP + head, then the engram shard. layer_shards() matches the
    # layer's engram tensors too, so layer 1 also lists the engram shard.
    assert src.layer_shards(1) == ["model-00002-of-00005.safetensors", "model-00005-of-00005.safetensors"]
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


def test_parse_hf_spec() -> None:
    assert parse_hf_spec("hf:deepseek-ai/DeepSeek-V4.1-Flash") == ("deepseek-ai/DeepSeek-V4.1-Flash", None)
    assert parse_hf_spec("hf:org/name@main") == ("org/name", "main")
    assert parse_hf_spec("hf:org/name@" + "a" * 40) == ("org/name", "a" * 40)
    # splits on the LAST '@'
    assert parse_hf_spec("hf:x/y@v1@abc") == ("x/y@v1", "abc")
    for bad in ("hf:org/name@", "hf:@main", "hf:@"):
        with pytest.raises(ValueError):
            parse_hf_spec(bad)
    with pytest.raises(PlanError):
        parse_hf_spec("gguf:/x.gguf")


def test_hf_fetch_passes_revision(tmp_path: Path, monkeypatch) -> None:
    seen: list[dict] = []

    def fake_download(**kw):
        seen.append(kw)
        p = Path(kw["local_dir"]) / kw["filename"]
        p.write_text("{}")
        return str(p)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    source_mod._hf_fetch("org/name", "config.json", tmp_path / "d", revision="v2")
    source_mod._hf_fetch("org/name", "config.json", tmp_path / "d")
    assert [k["revision"] for k in seen] == ["v2", None]


def test_pinned_revision_reaches_every_fetch(tmp_path: Path, monkeypatch) -> None:
    hub = make_synthetic_hf_repo(tmp_path / "hub")
    calls: list[tuple[str, str, str | None]] = []

    def fake_fetch(repo, filename, dest_dir, revision=None):
        calls.append((repo, filename, revision))
        dest_dir.mkdir(parents=True, exist_ok=True)
        return Path(shutil.copy(hub / filename, dest_dir / filename))

    monkeypatch.setattr(source_mod, "_hf_fetch", fake_fetch)
    monkeypatch.delenv("WP_FORGE_HF_LOCAL_HUB", raising=False)
    src = make_source("hf:fake/repo@abc123", tmp_path / "cache")
    assert isinstance(src, HFSource) and src.repo == "fake/repo" and src.revision == "abc123"
    src.hparams()
    src.tensor_index()
    with src.open_shard(src.layer_shards(0)[0]):
        pass
    assert {f for _, f, _ in calls} >= {"config.json", "model.safetensors.index.json"}
    assert all(c[0] == "fake/repo" and c[2] == "abc123" for c in calls)


def test_unpinned_spec_fetches_default_branch(tmp_path: Path, monkeypatch) -> None:
    seen: list[str | None] = []
    monkeypatch.setattr(source_mod, "_hf_fetch",
                        lambda repo, filename, dest_dir, revision=None: seen.append(revision) or Path(dest_dir) / filename)
    monkeypatch.delenv("WP_FORGE_HF_LOCAL_HUB", raising=False)
    src = make_source("hf:fake/repo", tmp_path / "cache")
    assert src.revision is None
    src.fetch("fake/repo", "config.json", tmp_path / "cache")
    assert seen == [None]


def test_revision_refused_with_local_hub(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WP_FORGE_HF_LOCAL_HUB", str(tmp_path))
    with pytest.raises(PlanError, match="revision"):
        make_source("hf:fake/repo@abc", tmp_path / "cache")


def test_revision_refused_on_convert_path(tmp_path: Path) -> None:
    with pytest.raises(ConvertError, match="revision"):
        _hf_model("hf:fake/repo@abc", tmp_path)
