#!/usr/bin/env python3
"""Build the static DS4 n-gram prefetch table from falsifier_dataset.npz."""

import argparse
import os
import struct

import numpy as np


MAGIC = b"WPNGRAM\0"
VERSION = 1
DEFAULT_LAYERS = 43
DEFAULT_EXPERTS = 256
DEFAULT_TOP_K = 16


def top_entries(counts, top_k):
    nonzero = np.flatnonzero(counts)
    ranked = sorted(nonzero.tolist(), key=lambda expert: (-int(counts[expert]), expert))[:top_k]
    return [(expert, int(counts[expert])) for expert in ranked]


def write_row(output, counts, top_k):
    total = int(counts.sum())
    entries = top_entries(counts, top_k)
    if total <= 0 or not entries or total > 0xFFFFFFFF:
        raise ValueError("table row has no counts or exceeds the u32 total")
    output.write(struct.pack("<IHH", total, len(entries), 0))
    for expert, count in entries:
        if count > 0xFFFFFFFF:
            raise ValueError("table entry exceeds the u32 count")
        output.write(struct.pack("<HI", expert, count))


def build_counts(dataset, n_layers, n_experts):
    r_layer = dataset["r_layer"]
    r_ntok = dataset["r_ntok"]
    r_step = dataset["r_step"]
    sel_flat = dataset["sel_flat"]
    sel_off = dataset["sel_off"]
    s_task = dataset["s_task"]
    s_tokens_flat = dataset["s_tokens_flat"]
    s_tokens_off = dataset["s_tokens_off"]
    n_tasks = int(dataset["n_tasks"][0])

    if len(r_layer) != len(r_ntok) or len(r_layer) != len(r_step) or len(sel_off) != len(r_layer) + 1:
        raise ValueError("inconsistent per-record arrays")
    if len(s_tokens_off) != len(s_task) + 1:
        raise ValueError("inconsistent per-step token arrays")

    step_first_rec = np.full(len(s_task), -1, np.int64)
    for record, layer in enumerate(r_layer):
        if int(layer) == 0:
            step = int(r_step[record])
            if step < 0 or step >= len(step_first_rec):
                raise ValueError("layer-0 record has an invalid step id")
            step_first_rec[step] = record
    if np.any(step_first_rec < 0):
        raise ValueError("a step has no layer-0 record")

    train_task = np.array([(task % 5) not in (3, 4) for task in range(n_tasks)], dtype=bool)
    popularity = np.zeros((n_layers, n_experts), dtype=np.uint64)
    rows = {}

    for step in range(len(s_task)):
        task = int(s_task[step])
        if task < 0 or task >= n_tasks or not train_task[task]:
            continue
        tokens = s_tokens_flat[int(s_tokens_off[step]):int(s_tokens_off[step + 1])]
        base = int(step_first_rec[step])
        for layer in range(n_layers):
            record = base + layer
            if record >= len(r_layer) or int(r_layer[record]) != layer or int(r_step[record]) != step:
                raise ValueError(f"step {step} is missing layer {layer}")
            n_tokens = int(r_ntok[record])
            selected_count = int(sel_off[record + 1] - sel_off[record])
            if n_tokens <= 0 or selected_count % n_tokens != 0:
                raise ValueError(f"record {record} has an invalid selection shape")
            n_expert_used = selected_count // n_tokens
            selected = sel_flat[int(sel_off[record]):int(sel_off[record + 1])].reshape(n_tokens, n_expert_used)
            for position in range(n_tokens):
                experts = selected[position].astype(np.int64)
                if np.any(experts < 0) or np.any(experts >= n_experts):
                    raise ValueError(f"record {record} has an out-of-range expert")
                np.add.at(popularity[layer], experts, 1)
                if position >= len(tokens):
                    continue
                token = int(tokens[position])
                if token < 0 or token > 0x7FFFFFFF:
                    raise ValueError(f"step {step} has an out-of-range token")
                key = (token, layer)
                counts = rows.get(key)
                if counts is None:
                    counts = np.zeros(n_experts, dtype=np.uint32)
                    rows[key] = counts
                np.add.at(counts, experts, 1)

    if np.any(popularity.sum(axis=1) == 0):
        raise ValueError("a popularity layer has no training observations")
    return popularity, rows


def write_table(path, popularity, rows, top_k):
    n_layers, n_experts = popularity.shape
    with open(path, "wb") as output:
        output.write(MAGIC)
        output.write(struct.pack("<IIIIQ", VERSION, n_layers, n_experts, top_k, len(rows)))
        for layer in range(n_layers):
            write_row(output, popularity[layer], top_k)
        for token, layer in sorted(rows):
            output.write(struct.pack("<IHH", token, layer, 0))
            write_row(output, rows[(token, layer)], top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="path to falsifier_dataset.npz")
    parser.add_argument("--output", required=True, help="output WPNGRAM binary table")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--experts", type=int, default=DEFAULT_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    if args.layers <= 0 or args.layers > 0xFFFF:
        parser.error("--layers must be in 1..65535")
    if args.experts <= 0 or args.experts > 0xFFFF:
        parser.error("--experts must be in 1..65535")
    if args.top_k <= 0 or args.top_k > DEFAULT_TOP_K or args.top_k > args.experts:
        parser.error("--top-k must be in 1..16 and no larger than --experts")

    dataset = np.load(args.dataset, allow_pickle=False)
    popularity, rows = build_counts(dataset, args.layers, args.experts)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    write_table(args.output, popularity, rows, args.top_k)
    size = os.path.getsize(args.output)
    print(f"wrote {args.output}: {len(rows)} token-layer rows, {size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
