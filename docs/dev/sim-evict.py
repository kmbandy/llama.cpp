"""Offline cache-replacement simulation on the captured reference stream.

The reference stream is policy-independent, so every policy below sees exactly
the same sequence of page requests. Belady/OPT gives the true ceiling: no online
policy can beat it, so it bounds what any replacement work could ever be worth.
"""
import sys
from collections import OrderedDict, Counter, deque


def load(path):
    """-> flat list of page keys, in request order."""
    refs = []
    for line in open(path):
        f = line.split()
        if len(f) < 2:
            continue
        layer = f[0]
        for e in f[1:]:
            refs.append((layer, e))
    return refs


def lru(refs, cap):
    c = OrderedDict(); miss = 0
    for p in refs:
        if p in c: c.move_to_end(p)
        else:
            miss += 1
            if len(c) >= cap: c.popitem(last=False)
            c[p] = 1
    return miss


def fifo(refs, cap):
    c = set(); q = deque(); miss = 0
    for p in refs:
        if p not in c:
            miss += 1
            if len(c) >= cap: c.discard(q.popleft())
            c.add(p); q.append(p)
    return miss


def lfu(refs, cap):
    """LFU with LRU tie-break; frequency persists after eviction (that is the
    point -- it is what lets a hot page come back quickly)."""
    c = {}; freq = Counter(); miss = 0; clock = 0
    for p in refs:
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
    for p in refs:
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


def opt(refs, cap):
    """Belady. Evict whichever resident page is used furthest in the future."""
    nxt = [0] * len(refs)
    last = {}
    for i in range(len(refs) - 1, -1, -1):
        nxt[i] = last.get(refs[i], float('inf'))
        last[refs[i]] = i
    c = {}; miss = 0
    for i, p in enumerate(refs):
        if p in c: c[p] = nxt[i]; continue
        miss += 1
        if len(c) >= cap:
            victim = max(c, key=lambda k: c[k])
            if c[victim] < nxt[i]:
                continue  # every resident page is needed sooner: skip caching
            del c[victim]
        c[p] = nxt[i]
    return miss


for path in sys.argv[1:]:
    refs = load(path)
    n = len(refs)
    distinct = len(set(refs))
    print("\n===== %s" % path)
    print("  %d references, %d distinct pages" % (n, distinct))
    for cap in (500, 1000, 2000, 2200):
        base = lru(refs, cap)
        row = [("LRU", base), ("FIFO", fifo(refs, cap)), ("LFU", lfu(refs, cap)),
               ("2Q", two_q(refs, cap)), ("OPT", opt(refs, cap))]
        print("  cap %4d slots:" % cap)
        for name, m in row:
            d = 100.0 * (base - m) / base if base else 0.0
            print("     %-5s misses %6d  (%.1f%% miss rate)  %+6.1f%% vs LRU" %
                  (name, m, 100.0 * m / n, d))
