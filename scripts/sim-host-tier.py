"""Offline cache-replacement simulation for the wp-expert-worker HOST RAM
tier (wp::HostArena, src/weight-pager/wp-host-arena.*) -- NOT the VRAM slot
pool sim-evict.py already covers. Same problem shape (cyclic prefill sweeps
far bigger than the cache, decode's skewed low-reuse traffic) but a
different cache: the host tier is a read-through NVMe page cache, sized in
tens of GB, one page = one MXFP4 expert (~18.8 MB).

Two things are simulated:
  1. Synthetic traces shaped like the real worker (prefill sweeps, decode
     Zipf routing with per-prompt drift, and a mixed trace), built from the
     CLI args below.
  2. freq_admit(), a page-for-page transliteration of
     HostArena::admit_landed_locked_ / borrow() as implemented in
     src/weight-pager/wp-host-arena.cpp -- same tie-favors-incumbent rule,
     same "landing does not self-record" rule -- so this simulator's numbers
     are the ones that policy actually produces, not an idealized version of
     it. lru() is the pre-existing (default, WP_HOST_TIER_POLICY unset)
     behaviour: HostArena's retention-cap trim is plain LRU with no
     frequency gate.

Can also replay a real captured trace via WP_PAGEIN_LOG-format lines
("<layer> <expert> ... <timestamp-ish ignored>") -- see load_pagein_log().
That format only has per-line "which pages this ubatch/token touched", no
n_tokens marker (unlike sim-evict.py's WP_REF_LOG format), so phase can only
be inferred from --pagein-log-prefill-ubatches if the caller knows how many
leading lines were prefill.
"""
import argparse
import random
import sys
from collections import OrderedDict


# --- policies ---------------------------------------------------------------
# Every policy takes (refs, cap) where refs is a flat list of page keys (one
# entry per page TOUCHED, i.e. already expanded -- a request that touches 6
# experts contributes 6 entries) and cap is the tier size in PAGES. Returns
# the hit count (not miss count, unlike sim-evict.py) because it reads more
# directly against the worker's own ram_hit_rate metric.

def lru(refs, cap):
    """HostArena's actual default: plain LRU retention-cap trim. Byte for
    byte what WP_HOST_TIER_POLICY unset does."""
    c = OrderedDict()
    hits = 0
    for p in refs:
        if p in c:
            c.move_to_end(p)
            hits += 1
        else:
            if len(c) >= cap:
                c.popitem(last=False)
            c[p] = True
    return hits


def freq_admit(refs, cap):
    """WP_HOST_TIER_POLICY=freq_admit. Mirrors HostArena exactly:
      - cache: OrderedDict, front = LRU (next victim), back = MRU.
      - freq: persists across eviction (unlike the cache dict), incremented
        ONLY on a HIT (a page that is still resident being re-referenced) --
        never on landing a fresh miss. This is the load-bearing asymmetry:
        recording on landing would make every page in a uniform cyclic sweep
        tie at "seen once per pass" forever, degenerating to plain LRU.
      - on a miss with the cache already at cap: compare the newcomer's
        freq to the LRU victim's. A tie (both 0, the common case on a cold
        or purely-uniform sweep) goes to the INCUMBENT -- the newcomer is
        not cached at all this reference. Only a STRICTLY higher freq
        newcomer evicts the incumbent.
    No sketch here (an unbounded dict is fine for a finite simulated trace
    and matches the C++ CM-sketch's behavior exactly whenever there are no
    hash collisions, which is the CM-sketch's whole design point); the
    aging/halving HostArena's sketch does is likewise omitted since none of
    these traces run long enough for it to matter (worker default: halve
    every 8x the entry count, i.e. thousands of ubatches).
    """
    c = OrderedDict()
    freq = {}
    hits = 0
    for p in refs:
        if p in c:
            c.move_to_end(p)
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue
        if len(c) < cap:
            c[p] = True
            continue
        victim = next(iter(c))
        if freq.get(p, 0) <= freq.get(victim, 0):
            continue   # loses the tie/comparison: not cached, incumbent kept
        del c[victim]
        c[p] = True
    return hits


def freq_admit_windowed(refs, cap, window_frac=0.1):
    """freq_admit's decode regression (see below) fixed with a small
    always-admit LRU WINDOW in front of the frequency-gated MAIN segment --
    the same W in W-TinyLFU. window_cap = window_frac of cap (min 1); main
    cap = the rest.

    freq_admit's flaw: a page can only accumulate freq while resident, but a
    losing page is evicted (or, in the C++, self-evicted) on its FIRST
    admission attempt -- it never gets a chance to earn the hit that would
    let it win. On a purely uniform cyclic sweep that is fine (nothing SHOULD
    win, ties are the correct outcome). Under decode's Zipf traffic it is not
    fine: whichever ~cap pages happened to be resident when main first
    filled are permanently favored by the incumbent tie-break, even after
    the true popularity ranking has moved on, because a genuinely hotter
    challenger can't out-earn an incumbent it never gets to run.

    The fix: every miss lands in the window first (unconditionally -- never
    gated), where it gets real hits like any LRU page. Only when the window
    itself needs to evict (window full, plain LRU order) does the evicted
    window page get a shot at MAIN: promoted for free if main isn't at cap
    yet, otherwise compared against main's LRU victim by freq (same
    tie-favors-incumbent rule as before). A prefill sweep page still loses
    almost every time (it is touched once, ages out of the small window
    before earning a second hit, same as before) so prefill's scan
    resistance is preserved; a decode page gets window_cap references'
    worth of real opportunity to prove itself before facing the gate.
    """
    window_cap = max(1, int(cap * window_frac))
    main_cap = max(1, cap - window_cap)
    window = OrderedDict()
    main = OrderedDict()
    freq = {}
    hits = 0
    for p in refs:
        if p in window:
            window.move_to_end(p)
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue
        if p in main:
            main.move_to_end(p)
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue
        # miss: always lands in the window first.
        if len(window) >= window_cap:
            demoted = next(iter(window))
            del window[demoted]
            if len(main) < main_cap:
                main[demoted] = True
            else:
                victim = next(iter(main))
                if freq.get(demoted, 0) > freq.get(victim, 0):
                    del main[victim]
                    main[demoted] = True
                # else: demoted loses the gate, evicted for good.
        window[p] = True
    return hits


def freq_admit_phase(refs_nt, cap):
    """WP_HOST_TIER_POLICY=freq_admit, PHASE-AWARE. Takes (page, n_tokens)
    pairs (n_tokens > 1 => prefill, == 1 => decode) instead of a flat page
    list -- the worker has this for free, it is the request's own
    ubatch/token count.

    freq_admit (unqualified, above) and freq_admit_windowed both hurt DECODE
    relative to plain LRU (measured: ~40-55% relative hit-rate loss at 8-16
    GB tiers) even though decode was never the problem -- gating ANY demand
    admission, even a legitimate short-term-reuse decode page, slows how
    fast the cache can track decode's own natural (and already LRU-friendly)
    locality. The prefill sweep is the only phase that actually needs
    gating: it is a one-shot scan by construction (every page touched
    exactly once per pass), so nothing is lost by refusing it easy wins.

    So: gate ONLY prefill (n_tokens > 1) landings with the same tie-favors-
    incumbent frequency comparison as freq_admit(). Decode (n_tokens == 1)
    landings are UNGATED -- always MRU, identical to plain LRU -- so decode
    gets zero regression by construction, not just empirically. This also
    gives prefill sweeps the SAME protection against evicting decode's own
    resident set: a decode page a session keeps re-hitting has real
    (nonzero) freq, which beats any prefill candidate's freq (always 0,
    since a fresh sweep page has never been a hit) under the tie-break, so a
    resident decode favorite survives a prefill sweep that runs on top of
    it.
    """
    c = OrderedDict()
    freq = {}
    hits = 0
    for p, nt in refs_nt:
        if p in c:
            c.move_to_end(p)
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue
        if len(c) < cap:
            c[p] = True
            continue
        if nt <= 1:
            # decode: ungated, always admits hot -- byte-for-byte plain LRU.
            victim = next(iter(c))
            del c[victim]
            c[p] = True
            continue
        victim = next(iter(c))
        if freq.get(p, 0) <= freq.get(victim, 0):
            continue
        del c[victim]
        c[p] = True
    return hits


POLICIES = {"LRU": lru, "FREQ_ADMIT": freq_admit, "FREQ_ADMIT_W": freq_admit_windowed}
PHASE_POLICIES = {"FREQ_ADMIT_PHASE": freq_admit_phase}


# --- synthetic traces --------------------------------------------------------

def gen_prefill(n_ubatches, n_layers, experts_per_layer, rng):
    """K ubatches, each sweeping layers 0..L-1 in order, touching every
    expert of every layer (a full prefill ubatch routes to ~all experts --
    see the problem statement). experts_per_layer is THIS worker's share
    (already partitioned across workers upstream)."""
    refs = []
    for _ in range(n_ubatches):
        for layer in range(n_layers):
            for expert in range(experts_per_layer):
                refs.append((layer, expert))
    return refs


def gen_decode(n_tokens, n_layers, experts_per_layer, k, rng, zipf_s=1.1,
               drift_every=0, drift_frac=0.1):
    """Per token, per layer: k of experts_per_layer drawn from a Zipf-like
    skew (rank^-zipf_s, normalized), independently per layer. drift_every>0
    permutes drift_frac of each layer's rank->expert mapping every
    drift_every tokens, modeling the "some per-prompt drift" the popularity
    distribution has across prompts within a session."""
    weights = [1.0 / ((r + 1) ** zipf_s) for r in range(experts_per_layer)]
    total = sum(weights)
    weights = [w / total for w in weights]
    rank_to_expert = [list(range(experts_per_layer)) for _ in range(n_layers)]

    refs = []
    for t in range(n_tokens):
        if drift_every and t > 0 and t % drift_every == 0:
            for layer in range(n_layers):
                n_swap = max(1, int(experts_per_layer * drift_frac))
                idx = rng.sample(range(experts_per_layer), n_swap)
                vals = [rank_to_expert[layer][i] for i in idx]
                rng.shuffle(vals)
                for i, v in zip(idx, vals):
                    rank_to_expert[layer][i] = v
        for layer in range(n_layers):
            ranks = rng.choices(range(experts_per_layer), weights=weights, k=k * 3)
            chosen = []
            seen = set()
            for r in ranks:
                e = rank_to_expert[layer][r]
                if e not in seen:
                    seen.add(e)
                    chosen.append(e)
                if len(chosen) == k:
                    break
            while len(chosen) < k:
                e = rng.randrange(experts_per_layer)
                if e not in seen:
                    seen.add(e)
                    chosen.append(e)
            for e in chosen:
                refs.append((layer, e))
    return refs


def gen_mixed(n_prefills, prefill_ubatches, decode_tokens, n_layers,
              experts_per_layer, k, rng):
    refs = []
    for _ in range(n_prefills):
        refs += gen_prefill(prefill_ubatches, n_layers, experts_per_layer, rng)
        refs += gen_decode(decode_tokens // max(n_prefills, 1), n_layers,
                            experts_per_layer, k, rng, drift_every=64)
    return refs


def load_pagein_log(path):
    """WP_PAGEIN_LOG format: '<layer> <expert>' per page-in, one per line
    (see the worker's own comment at the WP_PAGEIN_LOG call site). Flattens
    straight to the same (layer, expert) key freq_admit()/lru() expect."""
    refs = []
    for line in open(path):
        f = line.split()
        if len(f) < 2:
            continue
        refs.append((f[0], f[1]))
    return refs


PAGE_MB = 18.8


def caps_for(tier_gb_list):
    return [(gb, max(1, int(gb * 1024 / PAGE_MB))) for gb in tier_gb_list]


def run(name, refs, tier_gb_list, refs_nt=None):
    """refs_nt, if given, is the same trace as [(page, n_tokens), ...] and
    additionally runs the phase-aware policy (gated only on n_tokens > 1
    landings -- see freq_admit_phase's docstring for why decode must never
    be gated)."""
    n = len(refs)
    distinct = len(set(refs))
    print("\n===== %s" % name)
    print("  %d references, %d distinct pages (%.1f GB distinct footprint)" %
          (n, distinct, distinct * PAGE_MB / 1024))
    for gb, cap in caps_for(tier_gb_list):
        print("  tier %2d GB (%5d pages):" % (gb, cap))
        for pname, fn in POLICIES.items():
            hits = fn(refs, cap)
            print("     %-16s hit_rate=%.4f (%d/%d)" % (pname, hits / n if n else 0.0, hits, n))
        if refs_nt is not None:
            for pname, fn in PHASE_POLICIES.items():
                hits = fn(refs_nt, cap)
                print("     %-16s hit_rate=%.4f (%d/%d)" % (pname, hits / n if n else 0.0, hits, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts-per-layer", type=int, default=48,
                     help="this worker's share of the 384 routed experts/layer")
    ap.add_argument("--k", type=int, default=6, help="experts/token/layer selected by the router")
    ap.add_argument("--prefill-ubatches", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=20000)
    ap.add_argument("--tier-gb", type=int, nargs="+", default=[8, 16, 32, 40])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pagein-log", type=str, default=None,
                     help="replay a captured WP_PAGEIN_LOG file instead of synthetic traces")
    args = ap.parse_args()

    if args.pagein_log:
        run("captured: " + args.pagein_log, load_pagein_log(args.pagein_log), args.tier_gb)
        return

    rng = random.Random(args.seed)
    prefill_refs = gen_prefill(args.prefill_ubatches, args.layers, args.experts_per_layer, rng)
    run("prefill (%d ubatches x %d layers x %d experts, cyclic sweep)" %
        (args.prefill_ubatches, args.layers, args.experts_per_layer),
        prefill_refs, args.tier_gb, refs_nt=[(p, 8) for p in prefill_refs])

    rng = random.Random(args.seed)
    decode_refs = gen_decode(args.decode_tokens, args.layers, args.experts_per_layer, args.k, rng,
                              drift_every=64)
    run("decode (%d tokens, k=%d, zipf, drift every 64 tokens)" % (args.decode_tokens, args.k),
        decode_refs, args.tier_gb, refs_nt=[(p, 1) for p in decode_refs])

    rng = random.Random(args.seed)
    mixed_nt = []
    for _ in range(2):
        mixed_nt += [(p, 8) for p in gen_prefill(args.prefill_ubatches, args.layers,
                                                 args.experts_per_layer, rng)]
        mixed_nt += [(p, 1) for p in gen_decode(args.decode_tokens // 2, args.layers,
                                                args.experts_per_layer, args.k, rng,
                                                drift_every=64)]
    run("mixed (2 prefills then long decode)",
        [p for p, _ in mixed_nt], args.tier_gb, refs_nt=mixed_nt)


if __name__ == "__main__":
    sys.exit(main())
