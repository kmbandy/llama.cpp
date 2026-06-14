import torch
import ml8_runtime

def _unpack(indices_packed, N, K, lo_first=True):
    # inverse of the lo-first nibble packing: [K//2, N] uint8 -> [N, K] uint8
    p = indices_packed.t().cpu().numpy()              # [N, K//2]
    out = torch.zeros(N, K, dtype=torch.uint8)
    out[:, 0::2] = torch.from_numpy((p & 0x0F).copy())
    out[:, 1::2] = torch.from_numpy(((p >> 4) & 0x0F).copy())
    return out

def test_layer_from_components_roundtrips():
    N, K, G, NC = 4, 8, 2, 16
    gsize = K // G
    centroids = torch.randn(G, NC).to(torch.float8_e4m3fn).float()   # e4m3 lattice vals
    scales = torch.rand(N, G) + 0.5
    indices = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    gidx = torch.arange(K) // gsize                                  # uniform grouping
    layer = ml8_runtime.layer_from_components(
        centroids=centroids, scales=scales, indices=indices, gidx=gidx)
    assert layer.n_rows == N and layer.n_cols == K
    assert layer.group_size == gsize and layer.n_centroids == NC
    assert layer.indices_packed.shape == (K // 2, N)
    assert layer.centroids_fp8.shape == (G, NC) and layer.centroids_fp8.dtype == torch.float8_e4m3fn
    assert layer.scales_fp32.shape == (G, N)
    # packed indices unpack back to originals
    assert torch.equal(_unpack(layer.indices_packed, N, K), indices)
    # scales transposed correctly
    assert torch.allclose(layer.scales_fp32, scales.t())


def _numpy_pack_reference(indices):
    """The deployed kernel's lo-first nibble packing, via numpy (the spec).

    [N, K] uint8 (vals 0..15) -> [K//2, N] uint8.
    """
    idx_np = indices.to(torch.uint8).cpu().contiguous().numpy()   # [N, K]
    lo = idx_np[:, 0::2]
    hi = idx_np[:, 1::2]
    packed_n_kp = (lo & 0x0F) | ((hi & 0x0F) << 4)                # [N, K//2]
    return torch.from_numpy(packed_n_kp.T.copy()).contiguous()    # [K//2, N]


def test_pack_indices_on_device_matches_numpy_reference():
    # On-device torch bit-op packing must be byte-identical to the numpy spec,
    # across shapes (odd N, K a multiple of 2) and the full 0..15 value range.
    torch.manual_seed(0)
    for N, K in [(4, 8), (16, 64), (33, 128), (1, 2)]:
        indices = torch.randint(0, 16, (N, K), dtype=torch.uint8)
        ref = _numpy_pack_reference(indices)
        out = ml8_runtime._pack_indices_lo_first(indices, device="cpu")
        assert out.shape == (K // 2, N)
        assert out.dtype == torch.uint8
        assert torch.equal(out, ref), f"mismatch at N={N}, K={K}"


def test_pack_indices_uses_no_numpy_host_roundtrip(monkeypatch):
    # The on-device packer must not fall back to numpy / .cpu() host packing.
    # Guard against regression: numpy() on the indices tensor would defeat the
    # whole point (#223 — the .cpu().numpy() repack was the host bottleneck).
    indices = torch.randint(0, 16, (8, 16), dtype=torch.uint8)
    called = {"numpy": False}
    real_numpy = torch.Tensor.numpy

    def spy_numpy(self, *a, **k):
        called["numpy"] = True
        return real_numpy(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "numpy", spy_numpy)
    ml8_runtime._pack_indices_lo_first(indices, device="cpu")
    assert not called["numpy"], "_pack_indices_lo_first must pack with torch bit-ops, not numpy"


def test_layer_caches_packed_indices_until_mutated():
    # layer_from_components must reuse the packed-indices buffer across calls when
    # indices is unchanged (the frozen-during-Axis-A case), and re-pack only when
    # indices is mutated in place (reassign step). Invalidation keys on the tensor's
    # version counter, not id() alone (id is stable across in-place mutation).
    N, K, G, NC = 4, 8, 2, 16
    gsize = K // G
    centroids = torch.randn(G, NC).to(torch.float8_e4m3fn).float()
    scales = torch.rand(N, G) + 0.5
    indices = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    gidx = torch.arange(K) // gsize

    def build():
        return ml8_runtime.layer_from_components(
            centroids=centroids, scales=scales, indices=indices, gidx=gidx)

    l1 = build()
    l2 = build()
    # Same underlying packed buffer reused (cache hit) — not just equal, identical.
    assert l1.indices_packed.data_ptr() == l2.indices_packed.data_ptr()

    # In-place reassign-style mutation bumps indices._version → cache invalidates.
    with torch.no_grad():
        indices[0, 0] = (int(indices[0, 0].item()) + 1) % NC
    l3 = build()
    assert l3.indices_packed.data_ptr() != l1.indices_packed.data_ptr()
    # And the re-packed layout is still correct for the mutated indices.
    assert torch.equal(_unpack(l3.indices_packed, N, K), indices)


def test_gidx_validation_still_rejects_nonuniform():
    # The uniform-contiguous-grouping invariant must still be enforced — the ml8
    # kernel only supports it. A scrambled gidx raises ValueError.
    N, K, G, NC = 4, 8, 2, 16
    gsize = K // G
    centroids = torch.randn(G, NC).to(torch.float8_e4m3fn).float()
    scales = torch.rand(N, G) + 0.5
    indices = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    bad_gidx = (torch.arange(K) // gsize).flip(0)                    # non-contiguous
    try:
        ml8_runtime.layer_from_components(
            centroids=centroids, scales=scales, indices=indices, gidx=bad_gidx)
        raise AssertionError("expected ValueError for non-uniform gidx")
    except ValueError:
        pass


def test_gidx_validated_once_across_repeated_calls(monkeypatch):
    # gidx is invariant during training; validating it on every forward forced a
    # DtoH sync (~200/micro). It must be validated once per buffer, then skipped.
    N, K, G, NC = 4, 8, 2, 16
    gsize = K // G
    centroids = torch.randn(G, NC).to(torch.float8_e4m3fn).float()
    scales = torch.rand(N, G) + 0.5
    indices = torch.randint(0, NC, (N, K), dtype=torch.uint8)
    gidx = torch.arange(K) // gsize

    calls = {"equal": 0}
    real_equal = torch.equal

    def spy_equal(a, b):
        calls["equal"] += 1
        return real_equal(a, b)

    monkeypatch.setattr(torch, "equal", spy_equal)
    for _ in range(3):
        ml8_runtime.layer_from_components(
            centroids=centroids, scales=scales, indices=indices, gidx=gidx)
    assert calls["equal"] == 1, f"gidx validated {calls['equal']}x, expected once"
