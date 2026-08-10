"""Offline cache-replacement simulation on the captured reference stream.

The reference stream is policy-independent, so every policy below sees exactly
the same sequence of page requests. Belady/OPT gives the true ceiling: no online
policy can beat it, so it bounds what any replacement work could ever be worth.

Format (2026-08-08, D5): each line is
    <layer> <expert> <expert> ... nt=<n_tokens>
where the trailing `nt=` field is the request's n_tokens (1 = decode, >1 =
prefill), added so the prefill-aware / sweep-boundary policies can tell the two
phases apart. The sentinel makes the column self-describing: a legacy capture
(all-integer lines) is still parsed -- experts = f[1:], n_tokens assumed 1 --
with a loud stderr warning, because a bare trailing integer is indistinguishable
from an expert id and silently misparsing old captures is worse than assuming
decode.
"""
import sys
import zlib
from collections import OrderedDict, Counter, deque


def load(path):
    """-> flat list of (page_key, n_tokens) in request order."""
    refs = []
    legacy = 0
    for line in open(path):
        f = line.split()
        if len(f) < 2:
            continue
        layer = f[0]
        if f[-1].startswith("nt="):
            n_tokens = int(f[-1][3:])
            experts = f[1:-1]
        else:
            legacy += 1
            n_tokens = 1
            experts = f[1:]
        for e in experts:
            refs.append(((layer, e), n_tokens))
    if legacy:
        print("WARNING: %s: %d legacy lines without the nt= column -- treated as "
              "decode (n_tokens=1); the PREFILL policy is meaningless on this "
              "capture, regenerate with the current worker" % (path, legacy),
              file=sys.stderr)
    return refs


def lru(refs, cap):
    c = OrderedDict(); miss = 0
    for p, _ in refs:
        if p in c: c.move_to_end(p)
        else:
            miss += 1
            if len(c) >= cap: c.popitem(last=False)
            c[p] = 1
    return miss


def fifo(refs, cap):
    c = set(); q = deque(); miss = 0
    for p, _ in refs:
        if p not in c:
            miss += 1
            if len(c) >= cap: c.discard(q.popleft())
            c.add(p); q.append(p)
    return miss


def lfu(refs, cap):
    """LFU with LRU tie-break; frequency persists after eviction (that is the
    point -- it is what lets a hot page come back quickly)."""
    c = {}; freq = Counter(); miss = 0; clock = 0
    for p, _ in refs:
        clock += 1; freq[p] += 1
        if p in c: c[p] = clock
        else:
            miss += 1
            if len(c) >= cap:
                victim = min(c, key=lambda k: (freq[k], c[k]))
                del c[victim]
            c[p] = clock
    return miss


def two_q(refs, cap):
    """Simplified 2Q: new pages enter a small FIFO probation; only on a SECOND
    reference do they earn a slot in the main LRU. Scan-resistant -- a page
    touched once cannot evict a hot page."""
    kin = max(1, cap // 4)
    a1 = OrderedDict(); am = OrderedDict(); miss = 0
    for p, _ in refs:
        if p in am: am.move_to_end(p); continue
        if p in a1:
            del a1[p]
            if len(am) >= cap - kin: am.popitem(last=False)
            am[p] = 1
            continue
        miss += 1
        if len(a1) >= kin: a1.popitem(last=False)
        a1[p] = 1
    return miss


def s3_fifo(refs, cap):
    """S3-FIFO (2026-08-08, D5; rewritten 08-08 morning -- the first version
    crashed on main-queue eviction and fed the ghost from the wrong queue).
    Scan-resistant: new pages enter a small FIFO (~10% of cap); a page evicted
    from small WITHOUT a re-reference goes to the ghost (an admission-history
    FIFO of keys only, no data); one that WAS re-referenced promotes to main.
    A miss whose key is in the ghost admits straight to main. Main keeps a
    one-bit re-reference flag: a re-referenced page gets a second chance
    instead of eviction on its first pass. 2Q's fragile queue sizing is
    replaced by the small-probation FIFO + the ghost."""
    smax = max(1, cap // 10)
    S = deque(); M = deque()
    s_bit = {}; m_bit = {}          # p -> re-referenced since insertion
    G = OrderedDict()               # ghost: keys only, FIFO, bounded to cap
    miss = 0

    def evict_main():
        while M:
            ev = M.popleft()
            if m_bit.get(ev):
                m_bit[ev] = False
                M.append(ev)        # second chance
            else:
                del m_bit[ev]
                return

    def insert_main(p):
        m_bit[p] = False; M.append(p)
        while len(s_bit) + len(m_bit) > cap:
            evict_main()

    def evict_small():
        ev = S.popleft()
        if s_bit.pop(ev):
            insert_main(ev)         # re-referenced in probation -> main
        else:
            G[ev] = 1               # one-hit wonder -> ghost (scan resistance)
            while len(G) > cap:
                G.popitem(last=False)

    for p, _ in refs:
        if p in m_bit:
            m_bit[p] = True
            continue
        if p in s_bit:
            s_bit[p] = True
            continue
        miss += 1
        if p in G:
            del G[p]
            insert_main(p)          # ghost hit: it came back -- it is not a scan
        else:
            s_bit[p] = False; S.append(p)
            while len(S) > smax or len(s_bit) + len(m_bit) > cap:
                evict_small()
    return miss


def doorkeeper_lru(refs, cap):
    """WTinyLFU-style admission gate in front of LRU. A count-min sketch tracks
    coarse access frequency; when the cache is full a page is admitted only once
    its sketch count clears a small threshold. Scan-resistant: a never-reused
    page cannot displace a hot one, because it is rejected at admission.
    Deterministic (crc32-seeded lanes -- builtin hash() is randomized per
    process) and aged (counters halve every 8*cap accesses, the TinyLFU reset,
    so ancient scans cannot buy admission forever)."""
    W = 65536; D = 4
    sketch = [0] * (D * W)

    def lanes(p):
        x0 = zlib.crc32(repr(p).encode())
        out = []
        for i in range(D):
            x = (x0 + i * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
            x ^= x >> 31
            out.append(i * W + x % W)
        return out

    c = OrderedDict(); miss = 0
    threshold = 2
    sample = max(8 * cap, 1024); ops = 0
    for p, _ in refs:
        ops += 1
        if ops % sample == 0:
            sketch = [v >> 1 for v in sketch]
        for h in lanes(p):
            sketch[h] += 1
        if p in c:
            c.move_to_end(p); continue
        miss += 1
        if len(c) >= cap:
            # full: admit only if the newcomer looks hot enough to matter
            if min(sketch[h] for h in lanes(p)) < threshold:
                continue
            c.popitem(last=False)
        c[p] = 1
    return miss


def prefill_band(refs, cap):
    """PREFILL-BAND ADMISSION (D5 / P5). The prefill sweep is a scan -- every page
    read once, never re-referenced -- running through the SAME pool decode lives
    in. It is also a *sweep*: pages read late in a sweep are the ones decode is
    LEAST likely to need first. Admit prefill page-ins (n_tokens > 1) at the
    coldest LRU rank (the eviction end) so a tail sweep cannot flush the pages
    decode is about to need. Decode references (n_tokens == 1) are admitted hot
    as usual. (08-08 morning fix: the first version inserted both phases at the
    hot end -- OrderedDict insertion appends -- making it bit-for-bit LRU.)"""
    c = OrderedDict(); miss = 0
    for p, ntok in refs:
        if p in c:
            c.move_to_end(p); continue
        miss += 1
        if len(c) >= cap:
            c.popitem(last=False)
        c[p] = 1
        if ntok > 1:
            c.move_to_end(p, last=False)   # prefill: coldest rank, evicted first
    return miss


def opt(refs, cap):
    """Belady. Evict whichever resident page is used furthest in the future.
    (08-08 morning fix: next-use must key on the PAGE, not the (page, n_tokens)
    tuple -- the tuple key made a page whose next reference is in the other
    phase look never-referenced-again, deflating the ceiling.)"""
    nxt = [0] * len(refs)
    last = {}
    for i in range(len(refs) - 1, -1, -1):
        p = refs[i][0]
        nxt[i] = last.get(p, float('inf'))
        last[p] = i
    c = {}; miss = 0
    for i, (p, _) in enumerate(refs):
        if p in c: c[p] = nxt[i]; continue
        miss += 1
        if len(c) >= cap:
            victim = max(c, key=lambda k: c[k])
            if c[victim] < nxt[i]:
                continue  # every resident page is needed sooner: skip caching
            del c[victim]
        c[p] = nxt[i]
    return miss


POLICIES = {
    "LRU": lru, "FIFO": fifo, "LFU": lfu, "2Q": two_q,
    "S3FIFO": s3_fifo, "DKLRU": doorkeeper_lru, "PREFILL": prefill_band,
    "OPT": opt,
}

for path in sys.argv[1:]:
    refs = load(path)
    n = len(refs)
    distinct = len(set(p for p, _ in refs))
    n_prefill = sum(1 for _, nt in refs if nt > 1)
    print("\n===== %s" % path)
    print("  %d references, %d distinct pages (%d prefill refs)" % (n, distinct, n_prefill))
    for cap in (500, 1000, 2000, 2200):
        base = lru(refs, cap)
        row = [(name, fn(refs, cap)) for name, fn in POLICIES.items()]
        print("  cap %4d slots:" % cap)
        for name, m in row:
            d = 100.0 * (base - m) / base if base else 0.0
            print("     %-7s misses %6d  (%.1f%% miss rate)  %+6.1f%% vs LRU" %
                  (name, m, 100.0 * m / n, d))
