from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from conversion.wp_forge.tools import (
    REPO_ROOT,
    STDERR_TAIL_CHARS,
    BINARIES,
    ToolError,
    Tools,
)


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    d = tmp_path / "bin"
    d.mkdir()
    for name in BINARIES:
        (d / name).write_text("#!/bin/sh\n")
        (d / name).chmod(0o755)
    return d


@pytest.fixture
def tools(build_dir: Path) -> Tools:
    return Tools.discover(build_dir)


def _patch(monkeypatch, state: dict | None = None) -> list[list[str]]:
    """Monkeypatch subprocess.run to record argv; rc/stderr come from state."""
    state = state if state is not None else {"rc": 0, "stderr": ""}
    calls: list[list[str]] = []

    def run(argv, *a, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, state["rc"], stdout="", stderr=state["stderr"])

    monkeypatch.setattr(subprocess, "run", run)
    return calls


# ---------------------------------------------------------------- discovery


def test_discover_from_explicit_build_dir(build_dir: Path):
    t = Tools.discover(build_dir)
    for name, field in [
        ("llama-wp-repack", "repack_bin"),
        ("llama-wp-expert-descriptor", "descriptor_bin"),
        ("llama-quantize", "quantize_bin"),
        ("llama-wp-dense-extract", "dense_extract_bin"),
    ]:
        assert (build_dir / name).is_file() and getattr(t, field).is_file(), name
    assert t.convert_hf_bin == REPO_ROOT / "convert_hf_to_gguf.py"
    assert t.convert_hf_bin.is_file()


def test_discover_env_var(monkeypatch, tmp_path: Path):
    d = tmp_path / "envbin"
    d.mkdir()
    for name in BINARIES:
        (d / name).write_text("x")
    monkeypatch.setenv("WP_FORGE_BUILD_DIR", str(d))
    t = Tools.discover()  # no explicit arg -> env wins
    assert t.repack_bin == d / "llama-wp-repack"
    assert t.dense_extract_bin == d / "llama-wp-dense-extract"


def test_discover_explicit_dir_beats_env(monkeypatch, tmp_path: Path, build_dir: Path):
    monkeypatch.setenv("WP_FORGE_BUILD_DIR", str(tmp_path / "elsewhere"))
    t = Tools.discover(build_dir)
    assert t.repack_bin == build_dir / "llama-wp-repack"


def test_discover_missing_error_names_targets(tmp_path: Path):
    d = tmp_path / "partial"
    d.mkdir()
    (d / "llama-quantize").write_text("x")  # only one of the four binaries
    with pytest.raises(FileNotFoundError) as e:
        Tools.discover(d)
    msg = str(e.value)
    for missing in ("llama-wp-repack", "llama-wp-expert-descriptor", "llama-wp-dense-extract"):
        assert f"cmake target {missing}" in msg
    assert "llama-quantize" not in msg  # the one that exists
    assert str(d) in msg


def test_discover_repo_default_requires_build_cpu_bin(monkeypatch):
    """No env var: discovery falls back to <repo>/build-cpu/bin. This worktree
    has none by design, so it must error naming every target. (If a build
    ever exists there, the fallback is what we want -- skip instead of
    asserting against a built tree.)"""
    monkeypatch.delenv("WP_FORGE_BUILD_DIR", raising=False)
    default = REPO_ROOT / "build-cpu" / "bin"
    if default.exists():
        pytest.skip("build-cpu/bin exists in this worktree; fallback path is live")
    with pytest.raises(FileNotFoundError) as e:
        Tools.discover()
    for name in BINARIES:
        assert f"cmake target {name}" in str(e.value)
    assert str(default) in str(e.value)


# ------------------------------------------------------------------ repack


def test_repack_plain_argv(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    model = tmp_path / "model.gguf"
    base = tmp_path / "out"
    got = tools.repack(model, base)
    assert got == Path(str(base) + "-experts-manifest.json")
    assert calls == [[str(tools.repack_bin), str(model), str(base)]]


def test_repack_full_argv_and_slice_manifests(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    model, base = tmp_path / "m.gguf", tmp_path / "b"
    got = tools.repack(
        model,
        base,
        layer_ranges="0-32,33-46",
        allow_partial=True,
        expert_slices=[1024, 512, 256, 256],
        slice_output_split=True,
    )
    assert calls == [
        [
            str(tools.repack_bin),
            str(model),
            str(base),
            "--layer-ranges", "0-32,33-46",
            "--allow-partial",
            "--expert-slices", "1024,512,256,256",
            "--slice-output-split",
        ]
    ]
    # one manifest per slice, named as wp-repack writes them, index 5-wide zero-padded
    assert got == [
        Path(f"{base}-eslice-slice-0000{i}-experts-manifest.json") for i in range(4)
    ]


def test_repack_slice_split_without_slices_raises(tools: Tools, monkeypatch, tmp_path: Path):
    _patch(monkeypatch)
    with pytest.raises(ValueError):
        tools.repack(tmp_path / "m.gguf", tmp_path / "b", slice_output_split=True)


def test_repack_slice_manifests_override(build_dir: Path, monkeypatch, tmp_path: Path):
    t = Tools(
        repack_bin=build_dir / "llama-wp-repack",
        descriptor_bin=build_dir / "llama-wp-expert-descriptor",
        quantize_bin=build_dir / "llama-quantize",
        dense_extract_bin=build_dir / "llama-wp-dense-extract",
        convert_hf_bin=REPO_ROOT / "convert_hf_to_gguf.py",
        slice_manifests=(1, 3),
    )
    calls = _patch(monkeypatch)
    base = tmp_path / "b"
    got = t.repack(tmp_path / "m.gguf", base, expert_slices=[4, 4], slice_output_split=True)
    assert got == [
        Path(f"{base}-eslice-slice-00001-experts-manifest.json"),
        Path(f"{base}-eslice-slice-00003-experts-manifest.json"),
    ]
    assert calls[0][-1] == "--slice-output-split"


# -------------------------------------------------------------- descriptor


def test_descriptor_argv_and_default_output(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    spine = tmp_path / "spine-00001-of-00007.gguf"
    manifest = tmp_path / "shards" / "b-experts-manifest.json"
    got = tools.descriptor(spine, manifest)
    assert calls == [
        [
            str(tools.descriptor_bin),
            "--model", str(spine),
            "--shard-manifest", str(manifest),
        ]
    ]
    # tool default: manifest stem + ".expert-descriptor.json" in the manifest dir
    assert got == manifest.parent / "b-experts-manifest.expert-descriptor.json"


# ---------------------------------------------------------------- quantize


def test_quantize_argv(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    src, dst = tmp_path / "s.gguf", tmp_path / "d.gguf"
    got = tools.quantize(src, dst, "q8_0", {"ffn_gate": "mxfp4", "ffn_down": "mxfp4"})
    assert calls == [
        [
            str(tools.quantize_bin), str(src), str(dst), "q8_0",
            "--tensor-type", "ffn_gate=mxfp4",
            "--tensor-type", "ffn_down=mxfp4",
        ]
    ]
    assert got == dst


def test_quantize_no_tensor_types(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    src, dst = tmp_path / "s.gguf", tmp_path / "d.gguf"
    tools.quantize(src, dst, "f16", {})
    assert calls == [[str(tools.quantize_bin), str(src), str(dst), "f16"]]


# ----------------------------------------------------------- dense_extract


def test_dense_extract_argv(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    first, out = tmp_path / "model-00001-of-00002.gguf", tmp_path / "dense.gguf"
    got = tools.dense_extract(first, out)
    assert calls == [
        [
            str(tools.dense_extract_bin),
            "--model", str(first),
            "--output", str(out),
        ]
    ]
    assert got == out


# -------------------------------------------------------------- convert_hf


def test_convert_hf_argv(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    mdir, out = tmp_path / "hf", tmp_path / "out.gguf"
    got = tools.convert_hf(mdir, out)
    assert calls == [
        [
            "python3", str(tools.convert_hf_bin),
            str(mdir),
            "--outfile", str(out),
            "--outtype", "bf16",
            "--use-temp-file",
        ]
    ]
    assert got == out


def test_convert_hf_outtype(tools: Tools, monkeypatch, tmp_path: Path):
    calls = _patch(monkeypatch)
    tools.convert_hf(tmp_path / "hf", tmp_path / "o.gguf", outtype="q8_0")
    assert calls[0][calls[0].index("--outtype") + 1] == "q8_0"


# ---------------------------------------------------------------- ToolError


def test_tool_error_carries_argv_rc_stderr_tail(tools: Tools, monkeypatch, tmp_path: Path):
    state = {"rc": 3, "stderr": "first line\n" + "x" * 10 + "\nLAST LINE"}
    calls = _patch(monkeypatch, state)
    with pytest.raises(ToolError) as e:
        tools.dense_extract(tmp_path / "m.gguf", tmp_path / "d.gguf")
    err = e.value
    assert err.tool == "llama-wp-dense-extract"
    assert err.rc == 3
    assert err.stderr_tail == "first line\n" + "x" * 10 + "\nLAST LINE"
    assert err.argv == calls[0]
    assert "LAST LINE" in str(err)


def test_tool_error_tails_long_stderr(tools: Tools, monkeypatch, tmp_path: Path):
    state = {"rc": 2, "stderr": ("y" * (STDERR_TAIL_CHARS * 3)) + "TAIL"}
    _patch(monkeypatch, state)
    with pytest.raises(ToolError) as e:
        tools.quantize(tmp_path / "s.gguf", tmp_path / "d.gguf", "q8_0", {})
    assert len(e.value.stderr_tail) == STDERR_TAIL_CHARS
    assert e.value.stderr_tail.endswith("TAIL")


def test_no_real_subprocess_calls_at_all(tools: Tools, monkeypatch, tmp_path: Path):
    """Defense in depth: nothing in this module may reach the real
    subprocess.run -- a stray call (not just an un-patched one) fails the run."""

    def deny(argv, *a, **kw):
        raise AssertionError(f"real subprocess.run called: {argv}")

    monkeypatch.setattr(subprocess, "run", deny)
    with pytest.raises(AssertionError):
        tools.repack(tmp_path / "m.gguf", tmp_path / "b")
