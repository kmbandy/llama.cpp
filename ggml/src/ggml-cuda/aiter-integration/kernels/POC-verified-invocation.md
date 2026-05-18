# Verified Triton AOT invocation — POC reference

End-to-end verified 2026-05-17 against:
- Triton **3.7.0+git4768da5e** (mainline post-3.7.0 release, built from source)
- ROCm **7.2.3** / HIP **7.2.53211**
- AMD clang **22.0.0git** (`/opt/rocm/lib/llvm/bin/clang++`)
- AMD Radeon **R9700 (gfx1201)** + Radeon **6900 XT (gfx1030)** AOT artifacts produced

## Exact AOT compile invocation

```bash
source ~/venvs/pytorch/bin/activate   # has triton mainline editable install

python -m triton.tools.compile \
    /path/to/kernels/vector_add.py \
    --kernel-name add_kernel \
    --target hip:gfx1201:32 \                    # or hip:gfx1030:32
    --signature "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
    --grid "n_elements / 1024, 1, 1" \           # C-syntax — NOT Python //
    --num-warps 4 \
    --num-stages 1 \
    --out-name vector_add \
    --out-path /tmp/out/vector_add
```

Produces:
- `vector_add.b87d3d74_0d1d2d.c` (~42 KB on gfx1201, ~41 KB on gfx1030)
- `vector_add.b87d3d74_0d1d2d.h` (~650 B, declares the launcher)

`HSACO_NAME[6696]` on gfx1201 (different blob size on gfx1030); same C
symbol names across arches — link separately per arch (see ARCHITECTURE.md
§10.2).

## Exact hipcc invocation to compile + link

```bash
hipcc test_vector_add.cpp vector_add.b87d3d74_0d1d2d.c -o test_vector_add \
    -I/opt/rocm/include -I. \
    --offload-arch=gfx1201    # required on LINK too, not just AOT
```

The C++ test must wrap the generated `.h` include in `extern "C"` since
Triton's generated header lacks `#ifdef __cplusplus` guards.

## Test result (2026-05-17, R9700)

```
✓ vector_add AOT test PASSED — N=4096, all elements = 3.0
```

See `wrappers/POC-test_vector_add.cpp` for the exact test source.

## Gotchas to remember when writing similar invocations

1. **`--grid` is inlined verbatim into the C launcher.** Use `/` not `//`
   for int division. Use `,` to separate the 3 grid dims. Reference
   kernel parameters by name (`n_elements`).
2. **`--target` syntax is `<backend>:<arch>:<warp_size>`.** For HIP:
   - RDNA (gfx10/11/12): warp_size=32
   - CDNA (gfx90a/942/950): warp_size=64
3. **`--signature` accepts alignment hints** via `:16` suffix on pointer
   types. The `1024` at the end of our signature is `BLOCK_SIZE` as a
   compile-time constant — folded into the kernel, NOT in the launcher arg list.
4. **The generated launcher takes `hipDeviceptr_t` for pointers**, not
   typed pointers like `float*`. The C++ caller must `(hipDeviceptr_t)` cast.
5. **Lazy module load.** First call invokes `hipModuleLoadData` internally;
   subsequent calls just dispatch. One `hipModule_t` per launcher symbol
   per process.
