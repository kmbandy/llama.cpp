# DRAFT: triton-lang/triton — AOT-generated `.h` missing `extern "C"` guards

**Status:** Draft. Ready to file at https://github.com/triton-lang/triton/issues — needs an account with submit access to that repo.

**Suggested title:** `[AOT] Generated header missing extern "C" guards — breaks C++ consumers`

---

## Summary

The C header emitted by `python -m triton.tools.compile` declares its launcher and load/unload functions with plain C linkage but **lacks `#ifdef __cplusplus` / `extern "C"` guards**. C++ consumers that naively `#include` the header hit linker errors because the launcher symbol in the generated `.c` (compiled as C) has unmangled linkage, but the C++ consumer sees a C++-mangled declaration.

## Reproduction

After AOT-compiling any kernel, the generated `<name>.<spec-hash>.h` looks like:

```c
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc.

#pragma once

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

void unload_vector_add_b87d3d74_0d1d2d(void);
void load_vector_add_b87d3d74_0d1d2d(void);
hipError_t vector_add_b87d3d74_0d1d2d(hipStream_t stream, ...);
```

Including this from a `.cpp`:

```cpp
#include "vector_add.b87d3d74_0d1d2d.h"

int main() {
    vector_add_b87d3d74_0d1d2d(stream, ...);  // unresolved at link
}
```

Compile/link fails:

```
ld.lld: error: undefined symbol: vector_add_b87d3d74_0d1d2d(ihipStream_t*, void*, void*, void*, int)
>>> referenced by main.cpp.o:(main)
>>> did you mean: extern "C" unload_vector_add_b87d3d74_0d1d2d
>>> defined in: vector_add-...o
```

Workaround on the consumer side:

```cpp
extern "C" {
#include "vector_add.b87d3d74_0d1d2d.h"
}
```

This works but is friction the header could remove for the consumer.

## Suggested fix

Wrap the declarations in a standard extern-C block. Template change to `python/triton/tools/compile.py` (the part that emits the header):

```c
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc.

#pragma once

#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void unload_vector_add_b87d3d74_0d1d2d(void);
void load_vector_add_b87d3d74_0d1d2d(void);
hipError_t vector_add_b87d3d74_0d1d2d(hipStream_t stream, ...);

#ifdef __cplusplus
}
#endif
```

This is the standard pattern. Zero behavior change for C consumers; fixes C++ consumers transparently. Tiny PR.

## Environment

- Triton 3.7.0+git4768da5e (mainline, built from source 2026-05-17)
- Python 3.14.4
- ROCm 7.2.3 / HIP 7.2.53211
- AMD Radeon AI PRO R9700 (gfx1201)

## Note

This came up while building Triton AOT into a C++ inference engine (llama.cpp's HIP backend). Encountered alongside a related grid-expression bug (separate issue).
