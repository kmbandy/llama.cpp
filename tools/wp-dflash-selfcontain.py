#!/usr/bin/env python3
"""
wp-dflash-selfcontain.py -- make a DFlash/DFlash2 speculative-draft GGUF self-contained
by copying its borrowed token-embedding and output-projection tensors in from the target
model, so it no longer needs `ctx_other` at load time.

Background
----------
A DFlash/DFlash2 draft GGUF (LLM_ARCH_DFLASH) ships no `token_embd.weight` / `output.weight`
of its own -- both are created TENSOR_NOT_REQUIRED in
src/models/dflash.cpp:load_arch_tensors() (see lines ~238-239, plus the pre-dsv4 duplicate at
~176-177) and, when absent, the draft borrows the target's copies at graph-build time via
`cparams.ctx_other` (src/models/dflash.cpp:1113-1115 for tok_embd, :1301-1302/:1427-1431/
:1543-1548 for the various output-projection call sites). Whether `ctx_other` is armed at all
is decided purely by tensor presence -- no KV metadata flag is involved:

    src/llama-context.cpp:652-659:
        if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
            if (model.tok_embd == nullptr || model.output == nullptr) {
                if (params.ctx_other == nullptr) {
                    throw std::runtime_error(...);
                }
                cparams.ctx_other = params.ctx_other;
            }
        }

Under `-sm tensor` (tensor-parallel) the target's `output.weight` lives in a Meta-split
buffer that a draft scheduler cannot co-schedule, so the ctx_other borrow path aborts
(ggml/src/ggml-backend.cpp:1243, "pre-allocated tensor ... in Meta buffer"). If the draft
GGUF instead carries its OWN `token_embd.weight` and `output.weight`, `model.tok_embd` and
`model.output` are both non-null after loading, `ctx_other` is never required and never
armed, and the draft runs the graph entirely out of its own (non-Meta) tensors -- see the
`model.tok_embd != nullptr` / `output == nullptr` checks that gate the borrow in
src/models/dflash.cpp:1109-1116 and :1297-1305.

This tool produces that self-contained draft GGUF by copying `token_embd.weight` and
`output.weight` (raw quantized bytes, no dequant) from the target GGUF into a copy of the
draft GGUF, leaving every other draft tensor and all draft KV metadata untouched.

RAM safety
----------
Both input files are opened with gguf.GGUFReader, which memory-maps them (np.memmap) --
tensor.data is a view over the mapped pages, never a full-file copy. Tensor bytes are
streamed straight to the output file with GGUFWriter.write_tensor_data() as soon as each
tensor's info is written, so at most one tensor's worth of *page-cache* data is resident at
a time; nothing approaching the target's 29 GB is ever materialized in the process's own
heap. Peak additional heap usage is metadata-sized (KB), not tensor-sized.

Usage
-----
    PYTHONPATH=gguf-py python3 tools/wp-dflash-selfcontain.py \\
        --draft  ~/models/Qwen3.8-27B-DFlash2-Q8_0.gguf \\
        --target ~/models/Qwen3.8-27B-Q8_0.gguf \\
        --out    ~/models/Qwen3.8-27B-DFlash2-Q8_0.selfcontained.gguf

    # dry run (metadata only, no output file written):
    PYTHONPATH=gguf-py python3 tools/wp-dflash-selfcontain.py \\
        --draft ~/models/Qwen3.8-27B-DFlash2-Q8_0.gguf \\
        --target ~/models/Qwen3.8-27B-Q8_0.gguf --list
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load the in-tree gguf-py package the same way gguf-py's own scripts do.
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

import gguf  # noqa: E402

logger = logging.getLogger("wp-dflash-selfcontain")

# The two tensors src/models/dflash.cpp borrows from the target via ctx_other when the
# draft does not carry them itself. output_norm.weight is NOT in this list: DFlash always
# creates its own output_norm (required, not TENSOR_NOT_REQUIRED -- dflash.cpp's
# "output_norm = create_tensor(..., 0)" a few lines above the tok_embd/output pair), so it
# is never borrowed and every DFlash draft already ships one.
BORROWED_TENSORS = ("token_embd.weight", "output.weight")

DFLASH_ARCH_NAMES = ("dflash",)


def human_bytes(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0:
            return f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{v:.2f} PiB"


def get_arch(reader: gguf.GGUFReader) -> str:
    field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if field is None:
        raise ValueError("input GGUF has no general.architecture key")
    return str(field.contents())


def get_kv_scalar(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field is not None else None


def tensors_by_name(reader: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    return {t.name: t for t in reader.tensors}


def validate(reader_draft: gguf.GGUFReader, reader_target: gguf.GGUFReader) -> dict[str, gguf.ReaderTensor]:
    """Return {tensor_name: target ReaderTensor} for the tensors that need copying, after
    sanity-checking that the draft and target agree on the shapes those tensors must have."""

    draft_arch = get_arch(reader_draft)
    if draft_arch not in DFLASH_ARCH_NAMES:
        logger.warning(
            "draft general.architecture = %r, expected one of %r -- "
            "this tool was written against LLM_ARCH_DFLASH's loader; proceeding anyway",
            draft_arch, DFLASH_ARCH_NAMES,
        )

    draft_tensors = tensors_by_name(reader_draft)
    target_tensors = tensors_by_name(reader_target)

    to_copy: dict[str, gguf.ReaderTensor] = {}
    for name in BORROWED_TENSORS:
        if name in draft_tensors:
            logger.info("draft already carries %r (%s %s) -- leaving it as-is, nothing to copy",
                        name, draft_tensors[name].tensor_type.name, tuple(int(x) for x in draft_tensors[name].shape))
            continue
        if name not in target_tensors:
            raise ValueError(f"target GGUF has no tensor named {name!r} -- cannot make draft self-contained")
        to_copy[name] = target_tensors[name]

    if not to_copy:
        logger.info("draft is already fully self-contained (both %s present) -- nothing to do", BORROWED_TENSORS)
        return to_copy

    # n_embd sanity check: dflash.<n>.embedding_length (whichever <n> block-name the arch
    # uses -- read generically via the "embedding_length" suffix) must equal the target's
    # token_embd.weight row count (dim 0, in elements not bytes).
    n_embd_draft = get_kv_scalar(reader_draft, f"{get_arch(reader_draft)}.embedding_length")
    n_embd_target = get_kv_scalar(reader_target, f"{get_arch(reader_target)}.embedding_length")

    for name, rt in to_copy.items():
        # ReaderTensor.shape is ne[] in ggml order (fastest-varying first); for a 2D weight
        # matrix stored as {n_embd, n_vocab} that's shape = (n_embd, n_vocab).
        n_embd_tensor = int(rt.shape[0])
        if n_embd_draft is not None and int(n_embd_draft) != n_embd_tensor:
            raise ValueError(
                f"target tensor {name!r} has n_embd={n_embd_tensor} but draft "
                f"{get_arch(reader_draft)}.embedding_length={n_embd_draft} -- refusing to splice mismatched tensors")
        if n_embd_target is not None and int(n_embd_target) != n_embd_tensor:
            raise ValueError(
                f"target tensor {name!r} has n_embd={n_embd_tensor} but target's own "
                f"{get_arch(reader_target)}.embedding_length={n_embd_target} -- target GGUF looks inconsistent")

    # vocab sanity check: token_embd.weight and output.weight must agree on n_vocab, and
    # (when the draft has its own tokenizer.ggml.tokens list, i.e. it isn't relying on a
    # d2t reduced-vocab mapping) that list's length should match too.
    vocab_sizes = {name: int(rt.shape[1]) for name, rt in to_copy.items()}
    if len(set(vocab_sizes.values())) > 1:
        raise ValueError(f"target's borrowed tensors disagree on n_vocab: {vocab_sizes}")

    if "d2t" not in [t.name for t in reader_draft.tensors]:
        tok_field = reader_draft.get_field(gguf.Keys.Tokenizer.LIST)
        if tok_field is not None:
            n_tok = len(tok_field.data)
            for name, n_vocab in vocab_sizes.items():
                if n_tok != n_vocab:
                    logger.warning(
                        "draft tokenizer.ggml.tokens has %d entries but target %r has n_vocab=%d "
                        "(no d2t mapping present in draft -- expected these to match)",
                        n_tok, name, n_vocab)

    return to_copy


def do_list(reader_draft: gguf.GGUFReader, reader_target: gguf.GGUFReader, to_copy: dict[str, gguf.ReaderTensor]) -> None:
    print(f"draft arch          : {get_arch(reader_draft)}")
    print(f"draft tensor count  : {len(reader_draft.tensors)}")
    print(f"target arch         : {get_arch(reader_target)}")
    print(f"target tensor count : {len(reader_target.tensors)}")
    print()
    if not to_copy:
        print("Nothing to copy -- draft is already self-contained.")
        return
    print("Tensors that would be copied from target -> draft-out:")
    total = 0
    for name, rt in to_copy.items():
        total += rt.n_bytes
        print(f"  {name:24s} type={rt.tensor_type.name:8s} shape={tuple(int(x) for x in rt.shape)!s:20s} "
              f"nbytes={human_bytes(rt.n_bytes)}")
    draft_bytes = sum(t.n_bytes for t in reader_draft.tensors)
    print()
    print(f"draft tensor bytes (unchanged) : {human_bytes(draft_bytes)}")
    print(f"copied-in tensor bytes         : {human_bytes(total)}")
    print(f"expected output size (approx)  : {human_bytes(draft_bytes + total)}  (plus KV/header overhead)")


def copy_kv(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    """Copy every KV pair from the draft into the writer, unchanged. general.architecture is
    skipped because GGUFWriter.add_architecture() (called from its constructor) already
    wrote it from the `arch` argument, and GGUF.* are writer-internal virtual fields."""
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def do_write(reader_draft: gguf.GGUFReader, reader_target: gguf.GGUFReader,
             to_copy: dict[str, gguf.ReaderTensor], out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists (pass --force to overwrite)")

    arch = get_arch(reader_draft)
    writer = gguf.GGUFWriter(out_path, arch, use_temp_file=False, endianess=reader_draft.endianess)

    copy_kv(reader_draft, writer)

    # tensor infos: all original draft tensors, in their original order, plus the borrowed
    # ones appended at the end.
    for t in reader_draft.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, rt in to_copy.items():
        writer.add_tensor_info(name, rt.data.shape, rt.data.dtype, rt.data.nbytes, rt.tensor_type)

    total_bytes = sum(t.n_bytes for t in reader_draft.tensors) + sum(rt.n_bytes for rt in to_copy.values())
    logger.info("writing %s (%s across %d tensors)", out_path, human_bytes(total_bytes),
                len(reader_draft.tensors) + len(to_copy))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    written = 0
    for t in reader_draft.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader_draft.endianess)
        written += t.n_bytes
    for name, rt in to_copy.items():
        writer.write_tensor_data(rt.data, tensor_endianess=reader_target.endianess)
        written += rt.n_bytes

    writer.close()
    logger.info("done: wrote %s (%d bytes)", out_path, written)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", required=True, type=Path, help="DFlash/DFlash2 draft GGUF (input)")
    ap.add_argument("--target", required=True, type=Path, help="target model GGUF to copy tok_embd/output from (input, mmap only)")
    ap.add_argument("--out", type=Path, help="self-contained draft GGUF (output). Required unless --list.")
    ap.add_argument("--list", action="store_true", help="print what would be copied and exit; reads headers only, writes nothing")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    if not args.list and args.out is None:
        ap.error("--out is required unless --list is given")

    reader_draft = gguf.GGUFReader(args.draft, mode="r")
    reader_target = gguf.GGUFReader(args.target, mode="r")

    to_copy = validate(reader_draft, reader_target)

    if args.list:
        do_list(reader_draft, reader_target, to_copy)
        return

    if not to_copy:
        logger.info("nothing to do; not writing %s", args.out)
        return

    do_write(reader_draft, reader_target, to_copy, args.out, args.force)


if __name__ == "__main__":
    main()
