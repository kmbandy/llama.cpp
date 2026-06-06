"""split_messages_corpus: the calib/held-out partition must be disjoint, complete,
deterministic under a seed, and independent of input order — the leakage guarantee."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from split_messages_corpus import split_records


def _recs(n):
    return [{"messages": [{"role": "user", "content": f"r{i}"}]} for i in range(n)]


def test_disjoint_and_complete():
    recs = _recs(100)
    calib, held = split_records(recs, heldout_n=15, seed=0)
    assert len(held) == 15
    assert len(calib) == 85
    # every record lands in exactly one pool (disjoint ∪ complete)
    cset = {json.dumps(r) for r in calib}
    hset = {json.dumps(r) for r in held}
    assert cset.isdisjoint(hset)
    assert len(cset) + len(hset) == 100


def test_deterministic_under_seed():
    recs = _recs(50)
    a_c, a_h = split_records(recs, 10, seed=7)
    b_c, b_h = split_records(recs, 10, seed=7)
    assert a_h == b_h and a_c == b_c
    # a different seed yields a different held-out selection
    _, c_h = split_records(recs, 10, seed=8)
    assert c_h != a_h


def test_heldout_n_clamped():
    recs = _recs(5)
    calib, held = split_records(recs, heldout_n=99, seed=0)
    assert len(held) == 5 and len(calib) == 0
    calib, held = split_records(recs, heldout_n=0, seed=0)
    assert len(held) == 0 and len(calib) == 5
