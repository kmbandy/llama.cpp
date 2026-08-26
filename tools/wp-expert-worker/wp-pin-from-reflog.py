#!/usr/bin/env python3
"""Generate static expert pins and compare offline replacement policies."""

import argparse
import collections
import math
from pathlib import Path


def read_requests(path):
    requests = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] != "R" or len(fields) < 3:
                continue
            try:
                layer = int(fields[1])
                experts = []
                for field in fields[2:]:
                    if field.startswith("nt="):
                        break
                    experts.append(int(field))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: bad R line") from error
            if experts:
                requests.append([(layer, expert) for expert in experts])
    return requests


def miss_rate(misses, references):
    return 100.0 * misses / references if references else 0.0


def simulate_lfu_aging(requests, slots, history=False, halflife=4096):
    resident = {}
    slot_pages = [None] * slots
    uses = [0] * slots
    ticks = [0] * slots
    history_uses = collections.defaultdict(int)
    history_references = 0
    tick = 0
    evict_age = 0
    misses = 0
    references = 0

    for request in requests:
        for page in request:
            references += 1
            if history:
                history_uses[page] += 1
                history_references += 1
                if history_references == halflife:
                    for key in list(history_uses):
                        history_uses[key] //= 2
                    history_references = 0
            slot_index = resident.get(page)
            if slot_index is not None:
                tick += 1
                ticks[slot_index] = tick
                uses[slot_index] += 1
                continue

            misses += 1
            empty = next((i for i, value in enumerate(slot_pages) if value is None), None)
            if empty is not None:
                slot_index = empty
            else:
                if history:
                    slot_index = min(
                        range(slots),
                        key=lambda i: (history_uses[slot_pages[i]], ticks[i]),
                    )
                else:
                    slot_index = min(range(slots), key=lambda i: (uses[i], ticks[i]))
                old_page = slot_pages[slot_index]
                evict_age = uses[slot_index]
                del resident[old_page]

            slot_pages[slot_index] = page
            resident[page] = slot_index
            tick += 1
            ticks[slot_index] = tick
            uses[slot_index] = history_uses[page] if history else evict_age + 1

    return misses, references


def simulate_lru(requests, slots):
    resident = {}
    slot_pages = [None] * slots
    ticks = [0] * slots
    tick = 0
    misses = 0
    references = 0
    for request in requests:
        for page in request:
            references += 1
            slot_index = resident.get(page)
            if slot_index is None:
                misses += 1
                slot_index = next((i for i, value in enumerate(slot_pages) if value is None), None)
                if slot_index is None:
                    slot_index = min(range(slots), key=lambda i: ticks[i])
                    del resident[slot_pages[slot_index]]
                slot_pages[slot_index] = page
                resident[page] = slot_index
            tick += 1
            ticks[slot_index] = tick
    return misses, references


def simulate_belady(requests, slots):
    references = [page for request in requests for page in request]
    positions = collections.defaultdict(collections.deque)
    for index, page in enumerate(references):
        positions[page].append(index)
    resident = set()
    misses = 0
    for index, page in enumerate(references):
        positions[page].popleft()
        if page in resident:
            continue
        misses += 1
        if len(resident) == slots:
            victim = max(
                resident,
                key=lambda candidate: positions[candidate][0]
                if positions[candidate]
                else math.inf,
            )
            resident.remove(victim)
        resident.add(page)
    return misses, len(references)


def simulate_static_pin(requests, slots, pinned):
    pinned = set(pinned[:slots])
    dynamic_slots = slots - len(pinned)
    resident = {}
    slot_pages = [None] * dynamic_slots
    ticks = [0] * dynamic_slots
    tick = 0
    misses = 0
    references = 0
    for request in requests:
        for page in request:
            references += 1
            if page in pinned:
                continue
            slot_index = resident.get(page)
            if slot_index is None:
                misses += 1
                if dynamic_slots == 0:
                    continue
                slot_index = next((i for i, value in enumerate(slot_pages) if value is None), None)
                if slot_index is None:
                    slot_index = min(range(dynamic_slots), key=lambda i: ticks[i])
                    del resident[slot_pages[slot_index]]
                slot_pages[slot_index] = page
                resident[page] = slot_index
            tick += 1
            ticks[slot_index] = tick
    return misses, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--total-pages", type=int)
    parser.add_argument("--pin-slots", type=int)
    parser.add_argument("--pin-file", type=Path, required=True)
    parser.add_argument("--halflife", type=int, default=4096)
    args = parser.parse_args()
    if args.slots <= 0 or args.halflife <= 0:
        parser.error("--slots and --halflife must be positive")
    pin_slots = args.slots if args.pin_slots is None else args.pin_slots
    if pin_slots < 0:
        parser.error("--pin-slots must not be negative")

    requests = read_requests(args.input)
    references = [page for request in requests for page in request]
    counts = collections.Counter(references)
    first_seen = {}
    for index, page in enumerate(references):
        first_seen.setdefault(page, index)
    ranked = sorted(counts, key=lambda page: (-counts[page], first_seen[page]))
    pinned = ranked[:pin_slots]
    with args.pin_file.open("w") as stream:
        for layer, expert in pinned:
            stream.write(f"{layer} {expert}\n")

    cycles = sum(
        1 for previous, current in zip(requests, requests[1:])
        if current[0][0] <= previous[0][0]
    ) + (1 if requests else 0)
    print(f"input={args.input} R_lines={len(requests)} references={len(references)} "
          f"unique_pages={len(counts)} cycles={cycles}")
    total_pages = len(counts) if args.total_pages is None else args.total_pages
    if total_pages <= 0:
        parser.error("--total-pages must be positive")
    print(f"slots={args.slots} total_pages={total_pages} "
          f"residency={args.slots / total_pages * 100.0:.3f}% "
          f"pin_slots={len(pinned)} halflife={args.halflife} pin_file={args.pin_file}")

    policies = [
        ("current-lfu-aging", simulate_lfu_aging(requests, args.slots)),
        ("lru", simulate_lru(requests, args.slots)),
        ("lfu-history", simulate_lfu_aging(
            requests, args.slots, history=True, halflife=args.halflife)),
        ("belady", simulate_belady(requests, args.slots)),
        ("static-pin+lru", simulate_static_pin(requests, args.slots, pinned)),
    ]
    for name, (misses, total) in policies:
        print(f"{name}: misses={misses} references={total} "
              f"miss_rate={miss_rate(misses, total):.6f}%")


if __name__ == "__main__":
    main()
