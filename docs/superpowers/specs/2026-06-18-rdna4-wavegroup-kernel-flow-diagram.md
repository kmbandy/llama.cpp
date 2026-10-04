# RDNA4 Wave-Group fp8 GEMM Kernel Flow Diagram

**Date:** 2026-06-18  
**Target:** AMD Radeon AI PRO R9700 / gfx1201, RDNA4, wave32  
**Kernel under investigation:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wggemm2.s`

This diagram shows the current PM4 wave-group GEMM path from host request to completed output, plus the
measured probes that have ruled stages in or out as the dominant slowdown.

## End-to-End Flow

```mermaid
flowchart TD
    A["llama.cpp / harness request\nGEMM shape: M x N x K\nTarget probe: e.g. 4096^2 x K16384"] --> B["Host setup\nocc_dispatch.cpp\n- allocate A/Bshuf/C/occ buffers\n- build PM4 dispatch packet\n- set COMPUTE_PGM_RSRC1/2\n- set LDS_SIZE, VGPR field, USER_SGPRs"]

    B --> C["Raw PM4 dispatch through KFD/libhsakmt\nNo HSA/AQL kernel launch\nTGID_X delivery unavailable on this path"]

    C --> D["Shader entry\n4-wave workgroup, 128 threads\nwave32, TWM=2, TWN=2\nPer-wave tile: FM x FN, usually 4x4"]

    D --> E["Tile assignment\nleader lane atomic_add(counter, BAND)\nLDS broadcast base_ti via ds_store_b32\ns_barrier_signal/wait\nall 4 waves derive same output tile"]

    E --> F["Per-output-tile setup\nDecode tile row/col\nCompute A row base and Bshuf base\nInitialize 16 f32 accumulator chains per wave"]

    F --> G["K loop over TBK=32 slices\n512 loop iterations for K=16384"]

    G --> H["A feed path\nGlobal A load: global_load_b128\nWait: s_wait_loadcnt\nPublish: ds_store_b128 to LDS\nOrder: s_wait_dscnt + s_barrier"]

    H --> I["A fragment path\nRead shared A from LDS\nInstruction: ds_load_b64\nWait: s_wait_dscnt\nFragments feed v_wmma"]

    G --> J["B feed path\nPreshuffled B tile-major buffer\nInstruction: global_load_tr_b64\nAddress: Bshuf + tile_col*2048 + wave_n*1024 + kk*NT*256 + ni*256\nWait: coarse or fine s_wait_loadcnt ladder"]

    I --> K["Compute path\nv_wmma_f32_16x16x16_fp8_fp8\n4x4 per wave = 16 accumulators\n2 kk groups => 32 WMMAs per K-loop iteration"]
    J --> K

    K --> L["Tail ordering\nAdvance B pointer by 2*NT*256\nBarrier before LDS A buffer reuse\nLoop until K complete"]
    L --> G

    K --> M["Output store\nDiagnostic/probe path may STORE=0\nCorrectness path stores C fragments\nHost oracle checks acc/frags"]

    M --> N["Dispatch completion\nPM4 EOP/fence observed\nHarness reports TF, WMMA/cyc, maxlive, claims, acc00/oracle"]
```

## Current Real-Kernel Hot Loop Shape

The slow fed path is the real `DBUF==1` A-ping-pong loop unless a probe selects another path.

```mermaid
flowchart LR
    A0["A[t] already in LDS buffer cur"] --> A1["ds_load A[t] frags"]
    A1 --> B0["global_load_tr_b64 B[t] frags"]
    B0 --> W0["wait B / wait A frags"]
    W0 --> P0["prefetch A[t+1]\nglobal_load_b128"]
    P0 --> C0["32x v_wmma or FEEDONLY no-op"]
    C0 --> W1["wait A[t+1]\ns_wait_loadcnt"]
    W1 --> S0["ds_store A[t+1]\ninto LDS buffer next"]
    S0 --> BAR["s_barrier_signal/wait\npublish next LDS buffer"]
    BAR --> NEXT["t++\nswap buffers"]
    NEXT --> A1
```

Important: the real path remains slow even when the WMMA body is removed (`FEEDONLY`), so the current
wall is in loop sequencing or a hidden side effect, not in matrix arithmetic.

## Measurements Attached To Each Stage

| Stage / probe | What it tested | Result | Current reading |
|---|---|---:|---|
| HIP winner, fed | Reference 4-wave HIP kernel, same shape family | **161.1 TF** @ `4096^2 x K16384` | G2 parity bar |
| HIP winner, NOFEED | Same HIP lever kernel with feed removed | **272.0 TF** | Matrix stream can be dense when feed instructions are removed |
| PM4 real fed `DBUF==1` | Current 4-wave PM4 GEMM path | **~1.4 TF** | Broken for performance |
| PM4 real `FEEDONLY` | Same feed loop, WMMA removed | **~1.4 TF-equivalent** | WMMA is not on the critical path |
| PM4 real NOFEED | Load once, reuse operands, no per-K feed | **~104-107 TF** | Compute path is viable but below HIP NOFEED |
| 2x2 NOFEED | Higher occupancy, smaller tile | **32.8 TF** | Shrinking tile destroys issue density |
| KUNROLL NOFEED U=1..8 | Longer back-to-back WMMA run | **~100-105 TF flat** | Not backedge/run-length bound |
| Band claim sweep | Fewer atomics in real GEMM | NOFEED **107 -> 101** for band 1->4 | Atomic claim is secondary in real path |
| Feed-only depth-P probe | 1-wave, no LDS, no barrier, normal loads | P=1 **123 GB/s**, P=16 **48 GB/s** | Raw load stream is not the 2.7 GB/s wall |
| Coupling ladder r1-r5 | Add 4-wave WG, barriers, LDS A round-trip | **1073-2382 GB/s** synthetic | 4-wave shape, barrier cadence, LDS publication are individually fast |
| B transpose rung 6 | `global_load_tr_b64`, real Bshuf addressing, real residency | **137-274 GB/s** | B transpose load is not pathological |
| PROFILE rung 7 | Add sampled realtime timers to real loop | **~96 TF** anomaly | Profiler side effect makes slow real loop fast |
| STAGGER rung 8 | Inert per-WG delay before K loop | **~1.4 TF flat** | One-time per-tile stagger does not reproduce PROFILE speed-up |

## Current Investigation State

```mermaid
flowchart TD
    S["Observed wall\nReal FED == FEEDONLY == ~1.4 TF"] --> A["Not WMMA compute\nRemoving WMMA changes nothing"]
    A --> B["Not raw global load BW\nfeedpipe/rung ladders are 50x-800x faster"]
    B --> C["Not LDS/barrier as isolated ops\nr3-r5 remain fast"]
    C --> D["Not B global_load_tr_b64\nrung 6 remains 50x+ faster than real feed"]
    D --> E["Not simple per-tile desync\nrung 8 stagger flat"]
    E --> F["Open lead\nPROFILE path changes something and jumps to ~96 TF"]

    F --> G["Next bisection target\nIdentify which PROFILE ingredient caused the jump"]
    G --> G1["s_sendmsg_rtn_b64 + s_wait_kmcnt per K-tile"]
    G --> G2["profiler token atomic only"]
    G --> G3["branch/control-flow skeleton only"]
    G --> G4["per-K inert delay control"]
```

## Stage Glossary

| Term | Meaning in this kernel |
|---|---|
| Raw PM4 | Direct command submission through KFD/libhsakmt, bypassing normal HSA/AQL launch behavior |
| TGID_X | Hardware workgroup id. It is not delivered to SGPRs in this raw-PM4 path, so tile assignment uses an atomic queue |
| 4-wave workgroup | 128-thread workgroup, four wave32 waves cooperate on one logical 128x128 output tile |
| Atomic claim | Leader lane claims tile index `ti` with `global_atomic_add` |
| LDS broadcast | Leader writes `ti` or A tile data to LDS; other waves read after `s_barrier_signal/wait` |
| A feed | Global A rows loaded by `global_load_b128`, staged into LDS by `ds_store_b128`, then consumed by `ds_load_b64` fragments |
| B feed | Preshuffled B read directly with `global_load_tr_b64`; no fp8 `global_load_tr_b128` path exists |
| `s_wait_loadcnt` | Waits for outstanding global/vector memory loads tracked by loadcnt |
| `s_wait_dscnt` | Waits for LDS/DS operations tracked by dscnt |
| `v_wmma_f32_16x16x16_fp8_fp8` | RDNA4 fp8 WMMA instruction producing f32 accumulators |
| `DBUF==1` | Current default A ping-pong LDS path; this is the canonical slow fed path under investigation |
| `FEEDONLY` | Probe variant that preserves feed/waits/barriers but removes WMMA, proving compute is not on the critical path |
| `PROFILE` | Probe variant with sampled realtime timers; unexpectedly speeds the real loop to ~96 TF |

## Simplified Mental Model

The current evidence says individual hardware operations are fast when isolated. The slow behavior only
appears when they are composed in the real `DBUF==1` loop. The actionable anomaly is the `PROFILE`
variant: a small code-path perturbation changes the same real loop from ~1.4 TF to ~96 TF. The next
diagram update should replace the open `PROFILE` box with the specific side effect once bisection finds it.
