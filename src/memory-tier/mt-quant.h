#pragma once

// INT4 / INT8 quantization helpers for cold-tier KV serialization.
//
// Stateless functions. Each {quantize, dequantize} pair is a round-trip
// for inputs already in roughly [-1, +1]. Quality is "fine for cold
// tier" — you would not use these for active inference.
//
// IMPORTANT (history): an earlier int4 implementation had a sign-bias
// bug — storing -8 round-tripped to 0 because the encoder packed
// `(v & 0x0F)` instead of `(v + 8) & 0x0F`. These helpers use the
// correct +8 bias so the full [-8, +7] range round-trips faithfully.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mt {

// Encode `n` floats from `src` (range ~[-1, +1]) into 4-bit packed
// signed integers. Output size = ceil(n / 2). The returned buffer
// owns its bytes.
std::vector<uint8_t> quantize_int4(const float * src, size_t n);

// Decode `n` 4-bit signed values from `src` back to floats in [-1, +1].
// `dst` must have space for at least `n` floats. Returns false if any
// argument is invalid (null or n == 0 with non-empty buffer).
bool dequantize_int4(const uint8_t * src, float * dst, size_t n);

// Encode `n` floats from `src` (range ~[-1, +1]) into 8-bit signed
// integers. Output size = n.
std::vector<uint8_t> quantize_int8(const float * src, size_t n);

// Decode `n` 8-bit signed values from `src` back to floats.
bool dequantize_int8(const uint8_t * src, float * dst, size_t n);

// MAD-135: per-block int4 with explicit scale. Reads `n` floats from
// `src`, computes scale = max(abs(x[i])), normalizes each into
// [-1, +1], int4-quantizes. Output: `*scale_out` is the per-block
// scale; `dst` (size at least ceil(n/2)) holds packed int4 nibbles.
// scale==0 means the entire block is zero — quant bytes are still
// written (all zero).
//
// Cold-tier use case: bound the dynamic range without an external
// scale assumption. F16 KV values in attention can have magnitudes
// well outside [-1, +1]; per-block scale recovers a usable range.
bool quantize_block_int4_with_scale(const float * src, size_t n,
                                     float * scale_out, uint8_t * dst);

// Inverse: reads `*scale_in` + ceil(n/2) packed int4 bytes,
// dequantizes into `dst` (size at least n).
bool dequantize_block_int4_with_scale(const uint8_t * src, float scale_in,
                                       float * dst, size_t n);

}  // namespace mt
