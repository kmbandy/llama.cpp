# DRAFT: triton-lang/triton — `--grid` expression with Python `//` operator emits broken C

**Status:** Draft. Ready to file at https://github.com/triton-lang/triton/issues — needs an account with submit access to that repo.

**Suggested title:** `[AOT] --grid expression with Python // operator emits broken C (becomes line comment)`

---

## Summary

`python -m triton.tools.compile` inlines the `--grid` expression string verbatim into the generated C launcher (`compile.py:191-193`), with no Python→C operator translation. Python's integer-division operator `//` becomes a C single-line-comment, silently corrupting the grid computation.

## Reproduction

A minimal kernel:

```python
# vector_add.py
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Compile with a natural Python-style grid:

```bash
python -m triton.tools.compile vector_add.py \
    --kernel-name add_kernel \
    --target hip:gfx1201:32 \
    --signature "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
    --grid "n_elements // 1024, 1, 1" \
    --num-warps 4 --num-stages 1 \
    --out-name vector_add --out-path /tmp/out/vector_add
```

The generated `.c` contains (line ~56):

```c
unsigned int gX = n_elements // 1024;
unsigned int gY =  1;
unsigned int gZ =  1;
```

In C99+ that parses as `gX = n_elements;` followed by `// 1024;` (line comment), so `gX` ends up as the raw token count rather than `ceil(n_elements / 1024)`. The kernel then launches `n_elements` blocks instead of `n_elements / BLOCK_SIZE`, producing wildly wrong results — or, more commonly, exceeding the GPU's grid limits and silently failing.

The workaround — use C-syntax `/` for int division — is fine once you know about it, but the failure mode is silent and only manifests at runtime (or with `hipcc`'s warning-as-error mode at compile).

## Suggested fix

Two reasonable options:

1. **Auto-translate `//` to `/` in the grid expression** when emitting C. Targets the specific incident; small change to the template emitter in `python/triton/tools/compile.py` around `gridX/Y/Z`.
2. **Parse the grid expression with a small AST evaluator** before emitting, validating it's expressible in C and producing canonical C output. More work but catches other Python-isms (e.g. `**`, named args, etc.).

Either way, **document the C-syntax requirement** in `python/triton/tools/compile.py`'s `desc` docstring near the `--grid` description. The current docstring example (`'compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py'`) doesn't include a `--grid` example, so users naturally try Python-style expressions.

## Environment

- Triton 3.7.0+git4768da5e (mainline, built from source 2026-05-17)
- Python 3.14.4
- ROCm 7.2.3 / HIP 7.2.53211
- AMD Radeon AI PRO R9700 (gfx1201)
