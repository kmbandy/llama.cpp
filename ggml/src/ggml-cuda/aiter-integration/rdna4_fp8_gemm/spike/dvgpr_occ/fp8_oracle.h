// fp8_oracle.h
#pragma once
#include <cstdint>
// Decode one OCP e4m3 byte (1 sign, 4 exp bias-7, 3 mantissa; no inf; 0xFF/0x7F = NaN).
float fp8_e4m3_to_float(uint8_t b);
// Reference D = A*B + C. A,B are 16x16 row-major e4m3 bytes; C,D are 16x16 row-major f32.
void wmma_ref_16x16x16(const uint8_t* A, const uint8_t* B, const float* C, float* D);

// Tiered oracle comparison (DSWS v2). Generalizes the inline gate fabs(got-ref) > rel*fabs(ref)+abs_.
//   Tier 1 (n_kseg==1): TIGHT = {rel 5e-3, abs 1e-2}  (the proven gate).
//   Tier 2 (n_kseg>1):  LOOSE = {rel 3e-2, abs 2e-2}  (absorbs split-K reassociation).
struct OracleCmp { bool ok; long bad; double max_rel; };
OracleCmp oracle_compare(const float* got, const float* ref, long n, float rel, float abs_);
