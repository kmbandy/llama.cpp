"""Thin wrappers around the external llama.cpp CPU tools.

`Tools` resolves binary paths once per run (WP_FORGE_BUILD_DIR env, then the
repo's build-cpu/bin). Each wrapper builds one exact argv, runs it, and
returns the path(s) the tool produced, raising `ToolError` on non-zero exit.
No policy lives here: what to repack, which quant types to pick, and what to
do with the outputs are all the caller's calls.

`repack` with `slice_output_split=True` returns the per-slice manifest list.
The tool writes those manifests as
`<base>-eslice-slice-NNNNN-experts-manifest.json` (NNNNN = zero-padded slice
index, 5 wide, see slice_output_base in tools/wp-repack/wp-repack.cpp), so
the list is derived from `len(expert_slices)`; if a prior run produced the
set, override `slice_manifests` with the list read from the real manifest
before calling.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STDERR_TAIL_CHARS = 4000

# binary -> the cmake target (in build-cpu) that produces it
BINARIES: dict[str, str] = {
    "llama-wp-repack": "llama-wp-repack",
    "llama-wp-expert-descriptor": "llama-wp-expert-descriptor",
    "llama-quantize": "llama-quantize",
    "llama-wp-dense-extract": "llama-wp-dense-extract",
}


class ToolError(RuntimeError):
    def __init__(self, tool: str, argv: list[str], rc: int, stderr_tail: str):
        self.tool = tool
        self.argv = argv
        self.rc = rc
        self.stderr_tail = stderr_tail
        super().__init__(f"{tool} failed rc={rc}: {' '.join(argv)}\n{stderr_tail}")


@dataclass(frozen=True)
class Tools:
    repack_bin: Path
    descriptor_bin: Path
    quantize_bin: Path
    dense_extract_bin: Path
    convert_hf_bin: Path

    # Slice indices as written into the manifest names; 0-based by default.
    slice_manifests: tuple[int, ...] | None = None

    @classmethod
    def discover(
        cls,
        build_dir: Path | str | None = None,
        *,
        slice_manifests: tuple[int, ...] | None = None,
    ) -> "Tools":
        if build_dir is None:
            env = os.environ.get("WP_FORGE_BUILD_DIR")
            build_dir = Path(env) if env else REPO_ROOT / "build-cpu" / "bin"
        else:
            build_dir = Path(build_dir)
        missing = [
            f"{name} (cmake target {target})"
            for name, target in BINARIES.items()
            if not (build_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"wp-forge: missing tool(s) in {build_dir}: {', '.join(missing)}; "
                f"build them first (see plan Global Constraints)"
            )
        convert_hf = REPO_ROOT / "convert_hf_to_gguf.py"
        if not convert_hf.is_file():
            raise FileNotFoundError(
                f"wp-forge: missing {convert_hf} (stock converter at repo root)"
            )
        return cls(
            repack_bin=build_dir / "llama-wp-repack",
            descriptor_bin=build_dir / "llama-wp-expert-descriptor",
            quantize_bin=build_dir / "llama-quantize",
            dense_extract_bin=build_dir / "llama-wp-dense-extract",
            convert_hf_bin=convert_hf,
            slice_manifests=slice_manifests,
        )

    # ------------------------------------------------------------------ run

    def _run(self, tool: str, argv: list[str]) -> None:
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ToolError(
                tool, argv, proc.returncode, proc.stderr[-STDERR_TAIL_CHARS:]
            )

    # -------------------------------------------------------------- wrappers

    def repack(
        self,
        model_gguf: Path,
        output_base: Path,
        *,
        layer_ranges: str | None = None,
        allow_partial: bool = False,
        expert_slices: list[int] | None = None,
        slice_output_split: bool = False,
    ) -> Path | list[Path]:
        """Run llama-wp-repack. Returns the manifest path, or the per-slice
        manifest list when slice_output_split is set."""
        argv = [str(self.repack_bin), str(model_gguf), str(output_base)]
        if layer_ranges is not None:
            argv += ["--layer-ranges", layer_ranges]
        if allow_partial:
            argv.append("--allow-partial")
        if expert_slices is not None:
            argv += ["--expert-slices", ",".join(str(n) for n in expert_slices)]
        if slice_output_split:
            argv.append("--slice-output-split")
        self._run("llama-wp-repack", argv)
        if not slice_output_split:
            return Path(str(output_base) + "-experts-manifest.json")
        if expert_slices is None:
            raise ValueError("slice_output_split requires expert_slices")
        indices = (
            self.slice_manifests
            if self.slice_manifests is not None
            else tuple(range(len(expert_slices)))
        )
        return [
            Path(f"{output_base}-eslice-slice-{i:05d}-experts-manifest.json")
            for i in indices
        ]

    def expert_shard(self, src_manifest: Path, out_base: Path, first: int, last: int, layer: int) -> tuple[Path, Path]:
        """Keep whole experts first..last of one layer. Returns (index, blob)."""
        binary = self.repack_bin.parent / "llama-wp-expert-shard"
        argv = [
            str(binary),
            "--src-manifest", str(src_manifest),
            "--out-base", str(out_base),
            "--experts", f"{first}-{last}",
            "--layers", f"{layer}-{layer}",
        ]
        self._run("llama-wp-expert-shard", argv)
        blob = Path(f"{out_base}-experts-00001-of-00001.wpb")
        index = Path(f"{out_base}-experts-00001-of-00001.wpi.json")
        if not blob.is_file() or not index.is_file():
            raise ToolError("llama-wp-expert-shard", argv, 0, f"missing {blob} or {index}")
        return index, blob

    def descriptor(self, spine_gguf: Path, manifest: Path) -> Path:
        """Run llama-wp-expert-descriptor. Returns the descriptor path
        (default: <manifest stem>.expert-descriptor.json in the manifest's dir)."""
        argv = [
            str(self.descriptor_bin),
            "--model", str(spine_gguf),
            "--shard-manifest", str(manifest),
        ]
        self._run("llama-wp-expert-descriptor", argv)
        return manifest.parent / (manifest.stem + ".expert-descriptor.json")

    def quantize(
        self, src_gguf: Path, dst_gguf: Path, default_type: str, tensor_types: dict[str, str]
    ) -> Path:
        """Run llama-quantize: `src dst default_type [--tensor-type n=t ...]`.
        Returns dst_gguf."""
        argv = [str(self.quantize_bin), str(src_gguf), str(dst_gguf), default_type]
        for name, qtype in tensor_types.items():
            argv += ["--tensor-type", f"{name}={qtype}"]
        self._run("llama-quantize", argv)
        return dst_gguf

    def dense_extract(self, first_gguf: Path, out: Path) -> Path:
        """Run llama-wp-dense-extract on the first shard of a multi-file model.
        Returns out."""
        argv = [
            str(self.dense_extract_bin),
            "--model", str(first_gguf),
            "--output", str(out),
        ]
        self._run("llama-wp-dense-extract", argv)
        return out

    def convert_hf(self, model_dir: Path, outfile: Path, outtype: str = "bf16") -> Path:
        """Run the stock convert_hf_to_gguf.py with --use-temp-file. Passes
        nothing that alters the converter's tensor mapping. Returns outfile."""
        argv = [
            "python3", str(self.convert_hf_bin),
            str(model_dir),
            "--outfile", str(outfile),
            "--outtype", outtype,
            "--use-temp-file",
        ]
        self._run("convert_hf_to_gguf.py", argv)
        return outfile
