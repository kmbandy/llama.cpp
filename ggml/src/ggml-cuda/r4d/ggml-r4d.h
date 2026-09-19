// ggml-side trampoline for the vendored R4D kernel library (see README-ORIGIN.txt in this
// directory for provenance). Not part of upstream R4D -- this file is ggml-specific glue.
//
// R4D's attention entry points are compiled for exactly one geometry (head_dim 256, gqa 6, paged
// block size 16) and exactly one architecture (gfx1201); everything else the kernels reject at the
// call site (see r4d.h). This header is always safe to include: when GGML_HIP_R4D is not defined
// (the kernels were not compiled into this build, e.g. non-gfx1201-only AMDGPU_TARGETS), the R4D
// entry points themselves are not declared and ggml_cuda_r4d_available() unconditionally returns
// false, so callers can gate on it without an #ifdef of their own.
#pragma once

#include <hip/hip_runtime.h>
#include <cstring>

#ifdef GGML_HIP_R4D
#include "r4d.h"   // r4d/r4d.h -- already wraps its declarations in extern "C"
#endif

// True iff the R4D kernels were compiled into this build (GGML_HIP_R4D) AND the current HIP
// device is gfx1201 -- the only architecture the vendored kernels are compiled for. Checked via
// hipGetDevice + hipGetDeviceProperties().gcnArchName rather than a compile-time macro because the
// device actually selected at runtime is what matters (a multi-GPU / heterogeneous host could have
// a non-gfx1201 device current). Cheap enough to call per-dispatch: hipGetDeviceProperties is a
// driver query, not a kernel launch, but callers on a hot path should still cache the result rather
// than call this every time.
static inline bool ggml_cuda_r4d_available() {
#ifdef GGML_HIP_R4D
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) {
        return false;
    }
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
        return false;
    }
    // gcnArchName looks like "gfx1201" or "gfx1201:sramecc+:xnack-"; match the prefix.
    return strncmp(prop.gcnArchName, "gfx1201", 7) == 0;
#else
    return false;
#endif
}
