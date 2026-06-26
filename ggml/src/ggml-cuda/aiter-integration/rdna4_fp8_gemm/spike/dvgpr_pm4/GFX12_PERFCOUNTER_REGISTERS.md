# gfx12 / gfx1201 (RDNA4, R9700) perfcounter register reference

Verified register offsets, field bit-layouts, event IDs, and the PM4 programming
sequence for reading GL2C (RDNA L2) hardware perfcounters from **inside our own
libhsakmt KFD compute ring** — the only way to instrument our raw-PM4 kernels,
which every ROCr-based profiler (rocprofv3 / RGP / rocprofiler-sdk device
counting) is blind to.

> **Why this doc exists:** these values were non-trivial to pin down and a wrong
> offset can hang the compute queue (which on this box also drives the displays).
> Everything here was reproduced **by hand from primary sources**, with
> independent sources agreeing exactly. Don't re-derive — look here first.

## Provenance

| What | Primary source | Local cached copy |
|---|---|---|
| Register absolute offsets | Mesa `src/amd/registers/gfx12.json` | `ref_gfx12/mesa_gfx12.json` |
| (cross-check) offsets + BASE_IDX | kernel `drivers/gpu/drm/amd/include/asic_reg/gc/gc_12_0_0_offset.h` | `ref_gfx12/gc_12_0_0_offset.h` |
| Field bit-masks | kernel `gc_12_0_0_sh_mask.h` | `ref_gfx12/gc_12_0_0_sh_mask.h` |
| GL2C event PERF_SEL ids | ROCm rocprofiler-sdk `counter_defs.yaml` (`architectures: gfx1201, block: GL2C`) | `/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml` |

Mesa JSON gives a flat **byte** offset (`"map":{"at":N}`); **dword offset = N/4**.
Kernel header gives `reg<NAME>` + `<NAME>_BASE_IDX` where absolute dword =
(GC `BASE_IDX 1` segment base **0xA000**) + the per-register offset. The two
reconcile exactly (e.g. GRBM_GFX_INDEX: kernel `0x2200`+0xA000 = `0xC200` =
Mesa `198656/4`).

## Recipe — pull MORE registers later

```bash
REF=ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_pm4/ref_gfx12
# 1. absolute dword offset of any register NAME:
python3 - "$REF/mesa_gfx12.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for m in d["register_mappings"]:
    if m.get("name")=="GRBM_GFX_INDEX":          # <-- change name
        at=m["map"]["at"]; print(hex(at//4), "uconfig:", hex(at//4-0xC000))
PY
# 2. field bit-masks of that register:
grep -E '<NAME>__.*_MASK' "$REF/gc_12_0_0_sh_mask.h"
# 3. a block's event PERF_SEL ids (by name + arch, never raw number):
#    grep the rocprofiler counter_defs.yaml for the metric, take the entry whose
#    architectures: list contains gfx1201.
```

All perfcounter registers below are **uconfig space** (dword ≥ 0xC000). A
`WRITE_DATA → MEM_MAPPED_REGISTER` packet takes the **absolute** dword offset
(the values below, directly). A `SET_UCONFIG_REG` packet takes **offset − 0xC000**.

---

## Register offsets (absolute dword)

| Register | dword | uconfig (−0xC000) | notes |
|---|---|---|---|
| `GRBM_GFX_INDEX` | `0xC200` | `0x0200` | instance/SE bank-select |
| `CP_PERFMON_CNTL` (`=CP_PERFMON_CNTL_1`) | `0xD808` | `0x1808` | global reset/start/stop |
| `GL2C_PERFCOUNTER0_SELECT` | `0xDB80` | `0x1B80` | SELECT0..3 = `0xDB80,82,84,86` |
| `GL2C_PERFCOUNTER0_SELECT1` | `0xDB81` | `0x1B81` | packs 2 more events/counter |
| `GL2C_PERFCOUNTER1_SELECT` | `0xDB82` | `0x1B82` | |
| `GL2C_PERFCOUNTER2_SELECT` | `0xDB84` | `0x1B84` | |
| `GL2C_PERFCOUNTER3_SELECT` | `0xDB86` | `0x1B86` | |
| `GL2C_PERFCOUNTER0_LO` / `_HI` | `0xD380` / `0xD381` | `0x1380/81` | counter i LO/HI = `0xD380 + 2*i` (+1=HI) |
| `GL2C_PERFCOUNTER1_LO` / `_HI` | `0xD382` / `0xD383` | | |
| `GL2C_PERFCOUNTER2_LO` / `_HI` | `0xD384` / `0xD385` | | |
| `GL2C_PERFCOUNTER3_LO` / `_HI` | `0xD386` / `0xD387` | | |

---

## Field bit-layouts (from `gc_12_0_0_sh_mask.h`)

### `GRBM_GFX_INDEX` (0xC200)
| field | bits | mask | note |
|---|---|---|---|
| `INSTANCE_INDEX` | [6:0] | `0x0000007F` | **gfx12 = 7 bits** (gfx10/11 were 8). |
| `SA_INDEX` | [9:8] | `0x00000300` | |
| `SE_INDEX` | [19:16] | `0x000F0000` | |
| `SA_BROADCAST_WRITES` | 29 | `0x20000000` | |
| `INSTANCE_BROADCAST_WRITES` | 30 | `0x40000000` | |
| `SE_BROADCAST_WRITES` | 31 | `0x80000000` | |

GL2C is a **GLOBAL block** → addressed flat by `INSTANCE_INDEX` (NOT per-SE).
- Select instance `i`: `i | SE_BROADCAST | SA_BROADCAST` (INSTANCE_BROADCAST=0).
- Full broadcast: `SE_BROADCAST | SA_BROADCAST | INSTANCE_BROADCAST`.

### `CP_PERFMON_CNTL` (0xD808)
| field | bits | mask |
|---|---|---|
| `PERFMON_STATE` | [3:0] | `0x0000000F` |
| `SPM_PERFMON_STATE` | [7:4] | `0x000000F0` |
| `PERFMON_ENABLE_MODE` | [9:8] | `0x00000300` |
| `PERFMON_SAMPLE_ENABLE` | 10 | `0x00000400` |

`PERFMON_STATE` enum: **0 = DISABLE_AND_RESET · 1 = START_COUNTING · 2 = STOP_COUNTING.**
`PERFMON_ENABLE_MODE`: 0 = ALWAYS_COUNT · 2 = COUNT_CONTEXT_TRUE · 3 = COUNT_CONTEXT_FALSE.

### `GL2C_PERFCOUNTERx_SELECT` (0xDB80 + 2x)
| field | bits | mask |
|---|---|---|
| `PERF_SEL` | [9:0] | `0x000003FF` |
| `PERF_SEL1` | [19:10] | `0x000FFC00` |
| `CNTR_MODE` | [23:20] | `0x00F00000` |
| `PERF_MODE1` | [27:24] | `0x0F000000` |
| `PERF_MODE` | [31:28] | `0xF0000000` |

### `GL2C_PERFCOUNTERx_SELECT1` (0xDB81)
| field | bits | mask |
|---|---|---|
| `PERF_SEL2` | [9:0] | `0x000003FF` |
| `PERF_SEL3` | [19:10] | `0x000FFC00` |
| `PERF_MODE2` | [27:24] | `0x0F000000` |
| `PERF_MODE3` | [31:28] | `0xF0000000` |

For a single accumulating event per counter: write `SELECT.PERF_SEL = <event>`,
all MODE/CNTR_MODE = 0. To pack 4 events into one counter pair use PERF_SEL/1 in
SELECT and PERF_SEL2/3 in SELECT1.

---

## GL2C event PERF_SEL ids — **gfx1201 specific**

> **gfx12 renumbers events vs gfx11.** Never reuse gfx11 numbers and never index
> by raw number across arches: gfx11 `RDREQ_32B`=99 but gfx12=146; gfx12 `MISS`=42
> collides with gfx11 `HIT`=42. Always look up by **name + architecture**.

| event | PERF_SEL (dec/hex) | use |
|---|---|---|
| `GL2C_EA_RDREQ_32B` | 146 / 0x92 | FETCH bytes |
| `GL2C_EA_RDREQ_64B` | 147 / 0x93 | FETCH bytes |
| `GL2C_EA_RDREQ_128B` | 148 / 0x94 | FETCH bytes |
| `GL2C_EA_WRREQ` | 108 / 0x6C | WRITE bytes (umbrella; 64B-only is `GL2C_EA_WRREQ_64B`=114) |
| `GL2C_HIT` | 41 / 0x29 | L2 hit rate |
| `GL2C_MISS` | 42 / 0x2A | L2 hit rate |

**gfx12 FETCH bytes** (no 96B bucket on gfx12):
`FETCH = RDREQ_32B·32 + RDREQ_64B·64 + RDREQ_128B·128`, summed over all instances.

## Instances

gfx1201 (Navi48) = **32 GL2C instances** (PAL `gfx12Device numGl2c=32`; aqlprofile
`gfx1201::Gl2cCounterBlockNumInstances=32` — overrides the gfx1200 generic 16).
GL2C is GLOBAL → loop `INSTANCE_INDEX = 0..31`, read each, **sum** for total
L2↔memory traffic. To confirm on a specific board: `AMD_DEBUG=info` →
`memory_channels = N (TCC blocks)`.

---

## PM4 packet encodings (compute ring)

Implemented in `pm4_perf.h`. dst_sel/src_sel values are gfx9..gfx12 stable.

- **Write a register** — `WRITE_DATA` (IT=0x37): `dst_sel=MEM_MAPPED_REGISTER(0)`,
  `engine_sel=ME(0)`, address field = **absolute reg dword offset**, then data.
- **Read a counter** — `COPY_DATA` (IT=0x40): `src_sel=PERFCOUNTERS(4)`,
  `dst_sel=MEMORY(1)`, src field = **reg dword offset directly** (NOT byte ×4),
  `count_sel=64BIT` to grab LO+HI together, dst = result-BO GPU VA.
- **Write a register via the perf window (fallback)** — `COPY_DATA`:
  `src_sel=IMMEDIATE(5)`, `dst_sel=PERFCOUNTERS(4)`, src = value, dst = reg offset.

## Programming sequence (bracket a dispatch)

```
ACQUIRE_MEM
WRITE_REG CP_PERFMON_CNTL = DISABLE_AND_RESET(0)        # reset
# program SELECTs (broadcast-program once is valid for a GLOBAL block):
WRITE_REG GRBM_GFX_INDEX = broadcast-all
WRITE_REG GL2C_PERFCOUNTER0_SELECT = PERF_SEL 146  (RDREQ_32B)
WRITE_REG GL2C_PERFCOUNTER1_SELECT = PERF_SEL 147  (RDREQ_64B)
WRITE_REG GL2C_PERFCOUNTER2_SELECT = PERF_SEL 148  (RDREQ_128B)
WRITE_REG GL2C_PERFCOUNTER3_SELECT = PERF_SEL 108  (WRREQ)
WRITE_REG CP_PERFMON_CNTL = START_COUNTING(1)          # start
  ... SET_SH_REG x N ; DISPATCH_DIRECT ...             # the workload
CS_PARTIAL_FLUSH                                        # wait for waves to retire
WRITE_REG CP_PERFMON_CNTL = STOP_COUNTING(2) | SAMPLE_ENABLE  # stop + latch
for i in 0..31:                                         # per-instance read
    WRITE_REG GRBM_GFX_INDEX = select-instance(i)
    COPY_DATA GL2C_PERFCOUNTER0_LO (0xD380) -> resultBO[i].rd32   (64-bit)
    COPY_DATA GL2C_PERFCOUNTER1_LO (0xD382) -> resultBO[i].rd64
    COPY_DATA GL2C_PERFCOUNTER2_LO (0xD384) -> resultBO[i].rd128
    COPY_DATA GL2C_PERFCOUNTER3_LO (0xD386) -> resultBO[i].wr
WRITE_REG GRBM_GFX_INDEX = broadcast-all
RELEASE_MEM (fence)                                     # CPU polls, then reduces
```

## Known risks / gotchas

- **Write-method (load-bearing):** Mesa writes SELECT/CNTL via `SET_UCONFIG_REG`;
  AMD **PAL writes them via `COPY_DATA → perfcounters`** because gfx12 brackets
  perfcounter regs in a `PERF_COUNTER_WINDOW`. If the plain register writes read
  back as **zero** on the KFD compute queue, switch the SELECT/CNTL writes to the
  COPY_DATA-immediate→perfcounters fallback. The **read** path is identical either
  way. *Validate against the known `--bw` probe (~45 GB/s) before trusting the GEMM.*
- **GC base 0xA000** is inferred from source agreement, not a kernel constant
  (the kernel fills it from on-GPU IP discovery at boot). The absolute offsets are
  directly confirmed by Mesa regardless.
- **`GL2C_EA_WRREQ`=108 and the RDREQ totals are single-sourced** (PAL); the four
  programmed (146/147/148/108) + HIT/MISS (41/42) are double-sourced.
```
