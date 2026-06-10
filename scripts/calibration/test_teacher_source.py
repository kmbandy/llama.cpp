import pytest
import torch, torch.nn as nn
from teacher_source import LiveTeacher, CachedTeacher, DeviceTeacher, make_teacher

class StubLM(nn.Module):
    """Maps ids [B,T] -> logits [B,T,V] deterministically; counts forward calls."""
    def __init__(self, V=32):
        super().__init__()
        self.V = V; self.calls = 0
        self.emb = nn.Embedding(V, V)
        torch.manual_seed(0); nn.init.normal_(self.emb.weight)
    def forward(self, ids):
        self.calls += 1
        return self.emb(ids) * 3.0

def _batches(n=3, B=2, T=5, V=32):
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, V, (B, T), generator=g) for _ in range(n)]

def test_live_matches_topk():
    from kl_loss import topk_teacher
    m, ids = StubLM(), _batches(1)[0]
    lt = LiveTeacher(m, K=8)
    idx, vals, tail = lt.get(0, ids)
    ridx, rvals, rtail = topk_teacher(m(ids).reshape(-1, m.V), 8)
    assert torch.equal(idx, ridx) and torch.equal(vals, rvals) and torch.equal(tail, rtail)

def test_cache_equals_live_and_hits(tmp_path):
    m = StubLM(); batches = _batches()
    live = LiveTeacher(m, K=8)
    ref = [live.get(i, b) for i, b in enumerate(batches)]
    ct = CachedTeacher.build(StubLM(), batches, tmp_path, key="k1", K=8)
    for i, b in enumerate(batches):
        for a, r in zip(ct.get(i, b), ref[i]):
            assert torch.equal(a, r)
    m2 = StubLM()
    ct2 = CachedTeacher.build(m2, batches, tmp_path, key="k1", K=8)
    assert m2.calls == 0  # cache hit: model never called
    for i, b in enumerate(batches):
        for a, r in zip(ct2.get(i, b), ref[i]):
            assert torch.equal(a, r)

def test_device_teacher_equals_live():
    m, ids = StubLM(), _batches(1)[0]
    ref = LiveTeacher(m, K=8).get(0, ids)
    dt = DeviceTeacher(m, "cpu", K=8)
    for a, r in zip(dt.get(0, ids), ref):
        assert torch.equal(a, r)

def test_make_teacher_factory(tmp_path):
    m = StubLM()
    assert isinstance(make_teacher("live", lambda: m, K=8, cache_dir=tmp_path, batches=None), LiveTeacher)
    assert isinstance(make_teacher("device:0", lambda: m, K=8, cache_dir=tmp_path, batches=None), DeviceTeacher)
    ct = make_teacher("cache", lambda: m, K=8, cache_dir=tmp_path, batches=_batches(), cache_key="kx")
    assert isinstance(ct, CachedTeacher)

def test_changed_batch_rebuilds(tmp_path):
    # Same key/K/n_batches, but one token flipped in one batch. The ids-content
    # hash_chain must differ from the cached one, forcing a rebuild (model called)
    # rather than silently serving stale shards from the prior corpus.
    batches = _batches()
    CachedTeacher.build(StubLM(), batches, tmp_path, key="kc", K=8)

    flipped = [b.clone() for b in batches]
    flipped[0][0, 0] ^= 1  # flip one token -> different ids content
    m = StubLM()
    CachedTeacher.build(m, flipped, tmp_path, key="kc", K=8)
    assert m.calls > 0  # cache must NOT hit: a rebuild forward happened

def test_get_wrong_ids_raises(tmp_path):
    batches = _batches()
    ct = CachedTeacher.build(StubLM(), batches, tmp_path, key="kw", K=8)
    # Correct ids: fine.
    ct.get(0, batches[0])
    # Wrong ids for that batch index: per-batch hash mismatch must raise.
    wrong = batches[0].clone()
    wrong[0, 0] ^= 1
    with pytest.raises(RuntimeError, match="ids hash mismatch"):
        ct.get(0, wrong)
