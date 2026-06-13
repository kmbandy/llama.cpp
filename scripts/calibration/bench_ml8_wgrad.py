"""Re-time the wgrad path: fused kernel vs the old index_put_/index_add_ scatter,
across representative 4B ml8 shapes. Prints per-shape ms and the speedup; exits
nonzero if the kernel is not faster than the old scatter on any shape."""
import sys, time
import torch
from ml8_backward_kernels import ml8_wgrad_triton, ml8_wgrad_torch
from test_ml8_backward_kernels import _reference_grads, _mk_case

SHAPES = [("attn", 1024, 2560, 20), ("mlp_up", 9728, 2560, 20),
          ("mlp_down", 2560, 9728, 76)]  # (name, N, K, G) — N=out, K=in for wgrad


def t(fn, n=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3


def main():
    assert torch.cuda.is_available()
    ok = True
    for name, N, K, G in SHAPES:
        dW_raw, indices, gidx, cent, scales, gsz = _mk_case(N, K, G, "cuda", seed=N)
        old = t(lambda: _reference_grads(dW_raw, indices, gidx, cent, scales))
        new = t(lambda: ml8_wgrad_triton(dW_raw, indices, cent, scales, gsz))
        tor = t(lambda: ml8_wgrad_torch(dW_raw, indices, cent, scales, gsz))
        print(f"[{name:9s} N={N} K={K} G={G}] old_scatter={old:7.3f}ms "
              f"torch_fallback={tor:7.3f}ms triton={new:7.3f}ms  "
              f"speedup_vs_old={old/new:5.2f}x")
        ok = ok and (new < old)
    print("PASS" if ok else "FAIL: kernel not faster than old scatter on some shape")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
