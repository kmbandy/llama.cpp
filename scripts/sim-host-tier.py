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


def freq_admit_phase_reject(refs_nt, cap):
    """WP_HOST_TIER_POLICY=freq_admit as of the 2026-09-25 reject_lru_
    revision (HostArena::admit_landed_locked_ / begin_read_locked_ /
    evict_one_locked_ in src/weight-pager/wp-host-arena.cpp). Supersedes
    freq_admit_phase() above, which modeled the ORIGINAL (commit 2ccf93034)
    version of the policy -- kept for comparison, not because it is still
    what the code does.

    Two things changed, both found by tracing what a live run actually did
    (not by assumption -- see the design note this function's diff summary
    cites):

    1. THE RESERVATION-TIME EVICTION BUG. admit_landed_locked_ only ever
       decided WHERE a page that had ALREADY been read through NVMe landed
       (finish_read time). It said nothing about begin_read_locked_'s OWN
       eviction, which runs BEFORE that decision, to physically free a slot
       for the read to land in. Under sustained load the tier sits at its
       byte cap essentially always, so begin_read_locked_'s eviction fires on
       almost every miss regardless of what admission will later decide --
       i.e. every "rejected" candidate still evicted a REAL resident to get
       read at all, which is exactly what a live run (2026-09-25) measured:
       ram_evictions ~= ram_lookups, near-0 hit rate even with the tier sized
       to hold >10% of the working set. This function models that: a miss
       ALWAYS evicts something (from `reject`, the disposable list, if it
       has anything, else from `main`, the real resident set) the moment it
       needs a slot, not only when admission accepts it.
    2. reject_lru_ (this function's `reject` dict): a losing prefill landing
       goes into ITS OWN small disposable resident list instead of the front
       of `main`. Eviction always drains `reject` before ever touching
       `main`. Combined with (1), this is what actually fixes the bug: once
       ANY page has been rejected, every subsequent miss's unavoidable
       eviction takes its victim from `reject` (cheap, by construction) and
       `main`'s real residents stop being touched at all -- the
       evictions-that-hit-`main` count (the thing to watch live as
       ram_evictions_lru) should fall to ~0 once the first rejection has
       landed, while total evictions (ram_evictions) stays high because
       `reject` keeps legitimately churning to serve every requested-but-
       not-worth-caching page.
    3. RECORD ON EVERY ACCESS, HIT OR MISS (both prefill and decode): the
       original policy recorded frequency only on a hit, reasoning that a
       fresh miss is as uninformative as the victim it displaces. With
       reservation eviction no longer clobbering `main` on every miss, a
       losing prefill candidate still needs SOME way to grow its own
       estimate (once per sweep pass, from its own misses) so it can be told
       apart from a real incumbent -- otherwise every comparison after the
       first pass would need the OLD 0-vs-0 tie-break to be doing literally
       all the work, which is fragile. Recording on every access is also
       what lets a decode page's real repeat-borrow frequency keep beating a
       same-once-per-pass prefill sweep page after eviction and re-read.

    No physically separate "staging" scratch region is modeled (or built in
    the C++): reject_lru_ produces the same practical effect for the metric
    that matters (`main`/lru_ evictions -> 0) without a second bounded pool,
    duplicate-read races for concurrently-requested pages, or touching every
    call site that finalizes a page-in's arena hold in wp-expert-worker.cpp
    (see the diff summary for the full reservation-vs-staging tradeoff
    analysis) -- the existing budget_bytes = tier_bytes + margin headroom
    (read_inflight_max entries) already absorbs concurrent in-flight reads
    exactly as it did before this change.
    """
    main = OrderedDict()
    reject = OrderedDict()
    freq = {}
    hits = 0
    evictions_main = 0
    evictions_reject = 0
    for p, nt in refs_nt:
        if p in main:
            main.move_to_end(p)
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue
        if p in reject:
            # A demand hit on a previously-rejected page promotes it, same
            # shape as HostArena::borrow()'s new reject_lru_ promotion path.
            del reject[p]
            main[p] = True
            freq[p] = freq.get(p, 0) + 1
            hits += 1
            continue

        # Miss: record on EVERY access, admitted or not (point 3 above).
        freq[p] = freq.get(p, 0) + 1

        total = len(main) + len(reject)
        if total < cap:
            main[p] = True
            continue

        # Need to evict to land: reject first (cheap, disposable), main only
        # if reject has nothing to give up (point 1/2 above).
        if nt <= 1:
            # Decode: never gated -- plain LRU straight into main.
            if reject:
                victim = next(iter(reject)); del reject[victim]
                evictions_reject += 1
            else:
                victim = next(iter(main)); del main[victim]
                evictions_main += 1
            main[p] = True
            continue

        # Prefill: always compare against the WEAKEST currently resident
        # page -- reject's own front if it has anything, else main's front.
        # BUG FOUND VIA THIS SIMULATOR (2026-09-25): an earlier version of
        # this function (and, it turned out, the C++ it mirrors) skipped the
        # comparison entirely whenever `reject` was non-empty, reasoning
        # that disposable capacity already existing meant nothing needed
        # gating. That is wrong: one-in-one-out churn keeps `reject`
        # non-empty almost permanently once the first rejection has
        # happened, so the gate stopped firing after landing #1 and every
        # later candidate was admitted unconditionally -- measured as EXACT
        # 0% hit rate on the pure prefill trace below (a plain sliding
        # window, no incumbent protection at all), not the
        # tier_size/sweep_size convergence the design intends. Comparing
        # against reject's own front (not skipping the comparison) is what
        # fixes it: reject entries are no stronger than any main entry, so
        # testing a candidate against one is still a real admission test.
        if reject:
            victim = next(iter(reject))
        else:
            victim = next(iter(main))
        if freq.get(p, 0) <= freq.get(victim, 0):
            # Candidate loses/ties: it lands in reject, and the unavoidable
            # eviction (to physically free a slot) takes exactly the page it
            # just lost to.
            if victim in reject:
                del reject[victim]; evictions_reject += 1
            else:
                del main[victim]; evictions_main += 1
            reject[p] = True
        else:
            # Candidate wins: it lands hot in main, displacing the page it
            # just beat (from reject if that's where the victim was, else
            # from main itself).
            if victim in reject:
                del reject[victim]; evictions_reject += 1
            else:
                del main[victim]; evictions_main += 1
            main[p] = True
    return hits, evictions_main, evictions_reject


POLICIES = {"LRU": lru, "FREQ_ADMIT": freq_admit, "FREQ_ADMIT_W": freq_admit_windowed}
PHASE_POLICIES = {
    "FREQ_ADMIT_PHASE (old, 2ccf93034)": freq_admit_phase,
    "FREQ_ADMIT_PHASE_REJECT (new, reject_lru_)": freq_admit_phase_reject,
}


# --- spec-land-then-promote (the 2026-09-25 live-traffic finding) ----------
#
# The traces and policies above all model page-ins as a single event: a
# demand read either hits or misses. Live traffic under
# WP_PREFILL_LAYER_AHEAD is NOT shaped like that: a coordinator-run live test
# (three back-to-back 24.5k-token prefills, freq_admit engaged) measured
# n_pagein(demand)=2462 against ~60k layer-ahead SPECULATIVE page-ins on the
# same worker -- ~96% of traffic. Every one of those speculative pages lands
# Resident-but-speculative through begin_read/finish_read (which never gates
# a speculative landing -- admit_landed_locked_'s condition is
# `!e.speculative`), and only becomes a genuine demand (admitted) resident
# through HostArena::borrow()'s spec->demand promotion on its first real
# (prefill) reference. The FREQ_ADMIT_PHASE_REJECT policy above only ever
# models a FRESH DEMAND MISS being gated -- it has nothing to say about a
# page that was already resident (as a speculative landing) before its first
# demand reference, which is what the live traffic actually mostly is. That
# gap is exactly why the live counters showed the gate provably working on
# the minority path it covered (ram_admission_cold_landed=754, nonzero) while
# hit rate stayed ~0 (ram_lookup_hits=34 of 61641) and ram_evictions_lru
# tracked ram_lookups almost 1:1: the promotion path -- where nearly all
# traffic actually entered demand standing -- was completely unguarded.
def spec_promote_old(refs, cap):
    """Before the 2026-09-25 promotion-gate fix: borrow()'s spec->demand
    promotion NEVER consulted admission at all -- every promoted page landed
    hot in lru_ unconditionally, regardless of frequency. For a trace where
    every page lands speculatively and is promoted on its very next
    reference (one land-then-promote event per page per pass, exactly what a
    single-pass-per-prompt prefill layer-ahead sweep looks like), an
    unconditional-always-hot promotion is BYTE FOR BYTE plain LRU -- there is
    no frequency-gated code left in the loop at all. This is the function
    that reproduces the live ~0% hit rate."""
    return lru(refs, cap)


def spec_promote_new(refs, cap):
    """After the fix: the promotion is gated with the SAME comparison a
    fresh demand landing uses (HostArena::loses_admission_locked_, shared by
    admit_landed_locked_ and borrow()'s promotion path). For this trace
    shape (land speculatively, promote on the very next -- i.e. essentially
    simultaneous -- reference), landing-then-immediate-promotion collapses
    to exactly one gated admission decision per page reference, so this
    reuses freq_admit_phase_reject()'s mechanics directly rather than
    duplicating them: the RESULT is what matters here, not a byte-identical
    reproduction of the two separate HostArena calls (begin_read/finish_read
    for the speculative landing, then borrow() for the promotion) --
    reject_lru_'s eviction preference (spec_lru_ -> reject_lru_ -> lru_,
    unconditional on the source of the landing) already means the
    speculative landing itself never has to touch lru_ either, so the two
    HostArena calls and this one combined step reach the same steady state.
    """
    hits, _, _ = freq_admit_phase_reject([(p, 8) for p in refs], cap)
    return hits


SPEC_PROMOTE_POLICIES = {
    "SPEC_PROMOTE (old, promotion ungated)": spec_promote_old,
    "SPEC_PROMOTE (new, promotion gated)":   spec_promote_new,
}


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
                result = fn(refs_nt, cap)
                if isinstance(result, tuple):
                    hits, ev_main, ev_reject = result
                    print("     %-42s hit_rate=%.4f (%d/%d)  evictions_main=%d evictions_reject=%d"
                          % (pname, hits / n if n else 0.0, hits, n, ev_main, ev_reject))
                else:
                    hits = result
                    print("     %-42s hit_rate=%.4f (%d/%d)"
                          % (pname, hits / n if n else 0.0, hits, n))


def run_spec_promote(name, refs, tier_gb_list):
    """Report for SPEC_PROMOTE_POLICIES (spec-land-then-promote traffic --
    see the block comment above spec_promote_old/spec_promote_new). Separate
    from run() because this shape doesn't need the LRU/FREQ_ADMIT/etc.
    comparison policies at all -- the point here is specifically old vs new
    promotion-gate behaviour on traffic shaped like the live finding."""
    n = len(refs)
    distinct = len(set(refs))
    print("\n===== %s" % name)
    print("  %d references, %d distinct pages (%.1f GB distinct footprint)" %
          (n, distinct, distinct * PAGE_MB / 1024))
    for gb, cap in caps_for(tier_gb_list):
        print("  tier %2d GB (%5d pages, tier/sweep=%.4f):" % (gb, cap, cap / distinct if distinct else 0.0))
        for pname, fn in SPEC_PROMOTE_POLICIES.items():
            hits = fn(refs, cap)
            print("     %-38s hit_rate=%.4f (%d/%d)" % (pname, hits / n if n else 0.0, hits, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts-per-layer", type=int, default=48,
                     help="this worker's share of the 384 routed experts/layer")
    ap.add_argument("--k", type=int, default=6, help="experts/token/layer selected by the router")
    ap.add_argument("--prefill-ubatches", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=20000)
    ap.add_argument("--tier-gb", type=int, nargs="+", default=[8, 16, 24, 32])
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

    # Reuses prefill_refs (already the same cyclic-sweep shape
    # WP_PREFILL_LAYER_AHEAD produces): the live finding this section
    # reproduces is specifically about layer-ahead SPECULATIVE prefill
    # traffic, and old/new only differ on whether the spec->demand promotion
    # is gated -- see the block comment above spec_promote_old/_new.
    run_spec_promote(
        "prefill, spec-land-then-promote (%d ubatches x %d layers x %d experts -- "
        "matches the live finding: WP_PREFILL_LAYER_AHEAD makes ~96%% of page-ins "
        "speculative landings promoted via borrow(), not fresh demand misses)" %
        (args.prefill_ubatches, args.layers, args.experts_per_layer),
        prefill_refs, args.tier_gb)


if __name__ == "__main__":
    sys.exit(main())
