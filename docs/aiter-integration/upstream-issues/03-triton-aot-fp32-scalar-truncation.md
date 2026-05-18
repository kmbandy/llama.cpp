# DRAFT: triton-lang/triton — AOT launcher truncates fp32 scalar args (silently zeros them)

**Status:** Draft. Ready to file at https://github.com/triton-lang/triton/issues — needs an account with submit access to that repo.

**Suggested title:** `[AOT] Generated launcher declares fp32 scalar args as 'double' — kernel reads lower 4 bytes, value silently corrupted`

**Severity:** HIGH — produces silently wrong results for any kernel that takes a non-zero fp32 scalar argument.

---

## Summary

`python -m triton.tools.compile` emits a C launcher that declares `fp32` kernel-signature scalars as **`double`** in the C function prototype, then passes `&scalar` (an 8-byte `double*`) to `hipModuleLaunchKernel`. The HIP/CUDA launcher copies `sizeof(fp32)` = 4 bytes from each arg pointer, so the kernel actually receives **the low 4 bytes of the double's IEEE 754 representation** — which for typical scalar values is a denormal or near-zero float.

The result is that the kernel sees the wrong value for the scalar (typically ≈0), but everything compiles cleanly and the kernel runs without crashing. The bug is silent: it only manifests as numerically wrong output.

## Reproduction

AITER's `unified_attention` 3D kernel takes `scale: float32` (used as `qk_scale = scale * RCP_LN2`, then `S = qk_scale * tl.dot(Q, K)`):

```python
# unified_attention.py:387+
@triton.jit
def kernel_unified_attention_3d(
    # ... pointer args ...
    scale,  # float32
    # ...
):
    qk_scale = scale * RCP_LN2
    # ...
    S = qk_scale * tl.dot(Q, K)
```

AOT-compile with `*fp32` for the scale slot in `--signature`:

```bash
python -m triton.tools.compile unified_attention.py \
    --kernel-name kernel_unified_attention_3d \
    --target hip:gfx1201:32 \
    --signature "*fp32:16, *fp32:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, *i32:16, *i32:16, *fp16:16, *fp16:16, fp32, *fp32:16, *fp32:16, *fp32:16, fp32, ..." \
    --grid "num_seqs/2 + num_seqs, 2, 32" \
    ...
```

The generated `.c` declares the signature as:

```c
hipError_t uattn_3d_..._0d3d4d5d(
    hipStream_t stream, ...,
    double scale,        // ← fp32 in Triton signature, but C type is double
    ...,
    double softcap,      // ← same issue
    ...);
```

and packs args as:

```c
void *args[29] = { ..., &scale, ..., &softcap, ... };
hipModuleLaunchKernel(func, ..., args, NULL);
```

When the caller passes `scale = 0.0883883` (a typical attention softmax scale = `1/sqrt(128)`), the kernel reads the lower 4 bytes of the IEEE 754 `double` representation of `0.0883883`:

- `0.0883883` as `double` = `0x3FB6A09D9F4CD96A`
- Lower 4 bytes = `0x9F4CD96A`
- Reinterpreted as `float` = `-4.34e-20` (denormal, effectively 0)

So inside the kernel:

```
qk_scale = (~0) * RCP_LN2 ≈ 0
S        = (~0) * tl.dot(Q, K) ≈ 0
softmax(0, 0, 0, ...) = uniform 1/N
```

The kernel produces a uniform-attention output regardless of K, instead of the intended scaled softmax.

## Why this is silent

The bug only produces wrong values when softmax with non-zero scale gives non-uniform weights — i.e., when QK scores differ across tokens. Three plausible test inputs that **hide the bug:**

- **K uniform across tokens** → QK uniform → softmax uniform regardless of scale → output correct
- **Q = 0** → QK = 0 → softmax uniform → output correct
- **scale = 0 by design (e.g., debug)** → output uniformly weighted V, which matches the broken case exactly

I encountered this while building AITER attention into a custom inference engine. Three correctness tests with uniform K passed bit-exact (max_err < 1e-3). The first test with non-uniform K showed `actual = (2/3) * expected`, where `2/3` is the ratio of uniform-softmax-over-all-tokens to softmax-restricted-to-active-tokens-only.

## Reproducer

A minimal stand-alone reproducer is at https://github.com/<repo>/.../aiter-integration/wrappers/POC-test_uattn_3d_longctx.cpp — uses `ctx=1024` with K split into two classes (half zero, half one) so softmax should heavily favor one half.

## Suggested fix

Match the C launcher's declared type to the kernel signature:

```c
// for fp32:
float scale,
float softcap,

// for fp64:
double scale,
double softcap,
```

The fix is in `python/triton/tools/compile.py` where the C signature is emitted. Locate the type-mapping that's emitting `double` for `fp32` and produce `float` instead.

The args-pointer construction (`&scale`) is already correct once the local variable has the right type; the only change needed is the declared parameter type in the function signature and the local-variable declaration.

If backward-compat with existing callers that pass `double` is a concern, an alternative inside the launcher is to convert at the boundary:

```c
hipError_t uattn_3d_...(hipStream_t stream, ..., double scale_in, ..., double softcap_in, ...) {
    float scale = (float)scale_in;
    float softcap = (float)softcap_in;
    void *args[N] = { ..., &scale, ..., &softcap, ... };
    return hipModuleLaunchKernel(...);
}
```

This second form is what I patched in my vendored copy to validate the fix, but the cleaner long-term fix is to emit `float` in the signature directly.

## Same bug applies to all fp32 scalars

This is not specific to `scale` — any kernel argument declared as `fp32` (not pointer) in the Triton signature will hit this. Other affected scalars in `unified_attention` include `softcap` and `softmax_scale` variants. The bug is invisible whenever the scalar happens to be 0 (since the low 4 bytes of `0.0` as double are also 0).

Same likely applies to `i16`, `i8`, `u16`, `u8`, `f16`, `bf16` non-pointer args — Triton signature types narrower than 4 bytes promoted to `int` or `double` in C ABI would also misread.

## Environment

- Triton 3.7.0+git4768da5e (mainline, built from source 2026-05-17)
- Python 3.14.4
- ROCm 7.2.3 / HIP 7.2.53211
- AMD Radeon AI PRO R9700 (gfx1201)

## Note

This is the third Triton AOT issue encountered while building AITER into llama.cpp's HIP backend. The other two are filed separately:
- #1: `--grid` expression with Python `//` operator emits broken C
- #2: Generated `.h` missing `extern "C"` guards
