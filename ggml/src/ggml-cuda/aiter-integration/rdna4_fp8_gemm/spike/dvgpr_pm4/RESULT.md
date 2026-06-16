# RESULT: dynamic VGPR ARMED on gfx1201 (RDNA4) COMPUTE via raw PM4 — world-first

**Date:** 2026-06-15. **Hardware:** AMD R9700 / gfx1201 (RDNA4), KFD node 1
(GFXIP 12.0.1, 128 FCompute SIMDs, headless dGPU). **ROCm:** 7.2.3.

## Claim

`COMPUTE_PGM_RSRC2` bit 6 (== `DYNAMIC_VGPR` on GFX12) written into a raw PM4
indirect buffer on a KFD **compute** queue causes a gfx1201 wave to launch in
dynamic-VGPR mode — wave `STATUS[30]` (`DYN_VGPR_EN`) reads **1**. This bypasses
the MES firmware, which drops the bit on the HIP/AQL path. Previously concluded
unreachable / AMD-side (see `../dvgpr_probe/RESULT.md`); that verdict is **overturned**.

## Method

- `probe.s` → `probe.bin`: 32-byte hand-written gfx1201 raw ISA. Reads its own
  `STATUS[30]` via `s_getreg_b32 hwreg(HW_REG_WAVE_STATUS, 30, 1)` and stores it
  to `*s[0:1]` (output ptr preloaded from `COMPUTE_USER_DATA_0/1`). NOT a .hsaco —
  raw ISA loaded at a GPU VA and dispatched by raw PM4 (no kernel descriptor, no MES).
- `pm4_dispatch.cpp`: opens KFD, finds the gfx12 node, allocs ISA/output/fence/ring
  (host-visible GTT), creates an `HSA_QUEUE_COMPUTE` queue, and places a raw PM4 IB
  **directly in the ring** mirroring kfdtest `Dispatch::BuildIb`
  (ACQUIRE_MEM → 7×SET_SH_REG → DISPATCH_DIRECT → RELEASE_MEM fence), rings the
  64-bit doorbell, CPU-polls the EOP fence, reads `DYN_VGPR_EN`.
- The lift: `COMPUTE_PGM_RSRC2` is built by us in userspace; `--dynvgpr` ORs in
  `1<<6`. Everything else is byte-identical between baseline and lift.
- Packet encoders are vendored verbatim from kfdtest `PM4Packet.cpp`
  (ROCm/ROCR-Runtime @ **ba56a24c**); only the gtest scaffolding is shimmed away.
  Every register offset / bit shift transcribed from pinned upstream (no guesses);
  the four assembled dwords were cross-checked against an independent recompute.

## Result — deterministic A/B

| dispatch            | PGM_RSRC1   | PGM_RSRC2   | bit6 | `DYN_VGPR_EN` | fence |
|---------------------|-------------|-------------|------|---------------|-------|
| baseline (×3)       | 0x000c0004  | 0x00002888  | 0    | **0**         | clean |
| **lift** (×3)       | 0x000c0004  | 0x000028c8  | 1    | **1**         | clean |
| lift `--priv`       | 0x001c0004  | 0x000028c8  | 1    | **1**         | clean |

`0,1,0,1` across repeated runs. No hang, no GPU reset, GPU healthy throughout.
The baseline `0` reproduces the old AQL result (static VGPR) and proves both the
PM4 vehicle and the probe are correct; the lift `1` is the unlock.

## Why it works (the layer map, now complete)

| layer | status |
|---|---|
| silicon `SQ_DYN_VGPR` | ✅ enabled (`0xff`, WAVE_LIMIT=15) — read via umr |
| `COMPUTE_PGM_RSRC2.DYNAMIC_VGPR` (bit 6) | ✅ documented GFX120 enable, **now proven to arm** |
| LLVM codegen (amdgpu_kernel) | ❌ won't emit the bit (cs_chain-only) |
| ROCr | ❌ no concept, passes KD through |
| MES (AQL/HIP) | ❌ reprograms dispatch, drops bit 6 |
| **raw PM4 on KFD compute queue (this spike)** | ✅ **CP consumes RSRC2 verbatim — MES bypassed** |

## What this does and does NOT prove

- **Proven:** the per-dispatch dynamic-VGPR *arming* gate — the prerequisite the
  whole occupancy-unlock campaign hung on — is reachable from userspace on RDNA4
  compute. The wave runs in dyn-VGPR mode (`DYN_VGPR_EN=1`).
- **Not yet proven (next phase):** that a real fp8 GEMM emitting `s_alloc_vgpr`
  /`s_dealloc_vgpr` actually grows/shrinks its VGPR block at runtime, that
  occupancy rises, and that TFLOPS climbs toward the 307 TF WMMA ceiling. That is
  now an *engineering* path (kernel + dispatch via this proven PM4 vehicle), not an
  AMD-dependency wall.

## Reproduce

```
./build.sh                 # assembles probe.bin + builds pm4_dispatch (RAM-capped)
./pm4_dispatch             # BASELINE -> DYN_VGPR_EN must be 0
./pm4_dispatch --dynvgpr   # LIFT     -> DYN_VGPR_EN == 1  (dynamic VGPR armed)
```

Sources committed; `probe.bin` + `pm4_dispatch` are reproducible build artifacts
(gitignored). Vendored kfdtest encoder under `vendor/` (pinned ba56a24c, MIT — the
minimal subset that compiles). The full reference originals (`Dispatch.cpp`, the
queue classes, `gfx_7_2_sh_mask.h`) live under `ref/` locally and are gitignored;
they are re-fetchable verbatim from the same pinned commit.
