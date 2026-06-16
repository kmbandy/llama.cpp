// fp8_oracle.h
#pragma once
#include <cstdint>
// Decode one OCP e4m3 byte (1 sign, 4 exp bias-7, 3 mantissa; no inf; 0xFF/0x7F = NaN).
float fp8_e4m3_to_float(uint8_t b);
// Reference D = A*B + C. A,B are 16x16 row-major e4m3 bytes; C,D are 16x16 row-major f32.
void wmma_ref_16x16x16(const uint8_t* A, const uint8_t* B, const float* C, float* D);
