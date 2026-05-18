# vector_add.py
#
# Smallest possible Triton @jit kernel — used as the proof-of-concept that
# validates the end-to-end AOT pipeline (Triton → HSACO → hipModuleLoad →
# launch). Once this works on gfx1201, we have the toolchain for
# unified_attention and beyond.
#
# Status: DRAFT, target of MAD-187 milestone 1.

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,            # pointer to first input vector
    y_ptr,            # pointer to second input vector
    out_ptr,          # pointer to output vector
    n_elements,       # number of elements in each vector
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)
