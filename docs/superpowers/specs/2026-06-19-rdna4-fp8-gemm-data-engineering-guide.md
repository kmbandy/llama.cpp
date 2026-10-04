# RDNA4 fp8 GEMM Kernel Guide for a Data Engineer

**Date:** 2026-06-19  
**Hardware:** AMD Radeon AI PRO R9700 / gfx1201, RDNA4, wave32  
**Primary kernel:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/occ_kernel_wggemm2.s`  
**Companion diagram:** `docs/superpowers/specs/2026-06-18-rdna4-wavegroup-kernel-flow-diagram.md`

This is a step-by-step explanation of the current PM4 fp8 GEMM path, what each stage does, what knobs we can
change, and how to think about each stage using AWS/data-engineering analogies.

The useful mental model:

```text
GPU GEMM kernel = a distributed streaming job

Host/harness      = Airflow/Step Functions launching a job
PM4 dispatch      = low-level job submission API
Workgroups        = ECS tasks / Spark executors
Waves             = workers inside each executor
LDS               = executor-local scratch/cache
VGPRs             = per-worker in-memory state/register file
Global memory     = S3/data lake, except much faster when actually in VRAM
WMMA              = specialized vectorized transform operator
Barriers          = stage synchronization points
waitcnt           = waiting for async IO futures to complete
```

## Current Ground Truth

The campaign had one major measurement reset:

- Early bad fed numbers were contaminated because operands were allocated in system RAM and streamed over PCIe.
- The harness now needs to assert device-local VRAM for perf buffers before any performance result is trusted.
- Correctness-only work before that remains useful; bandwidth/feed conclusions before the VRAM fix are suspect.

Current clean picture:

| Item | Current meaning |
|---|---|
| `NOFEED ~297 TF` | The WMMA/matrix path can run near silicon peak when operands are already resident. |
| `FED ~145-150 TF` | Current best fed path is roughly half peak, near hipBLASLt/HIP-class behavior. |
| `FED == FEEDONLY` | Removing WMMA does not change time much; the current wall is operand feed, not matrix compute. |
| Barriers | Removing both barriers in the current feed-only path only bought ~1-2%; barriers are not the main wall now. |
| Main live lever | Reduce feed-load operations per output, mostly by increasing reuse or changing A/B feed layout. |
| `global_load_tr_b128` | Not a fp8 drop-in. ISA says `tr_b128` is for 16-bit matrix transpose; fp8 uses `tr_b64`. |
| Static VGPR cap | ISA says normal shaders may have up to 256 VGPRs per wave. Dynamic VGPR does not make 512 logical VGPRs addressable. |

## Step 0: Host Request and Shape Selection

### What Happens

The harness receives a GEMM shape:

```text
C[M, N] = A[M, K] x B[K, N]
```

Typical target:

```text
M=N=32768 or 65536
K=16384
fp8 inputs, f32 accumulation
```

### Decision: Target Prefill and Training Shapes

This campaign is targeting **prefill** and **training / weight-gradient** workloads, where the goal is to make
the fp8 GEMM compute-bound and push toward the matrix-core ceiling. We are **not** optimizing this kernel
for decode first; decode is usually a skinnier, bandwidth-dominated path and should be treated as a separate
bandwidth/kernel problem.

Use shape selection to answer two different questions:

1. **Peak/saturation:** can the kernel approach the silicon ceiling when the GPU is fully fed with work?
2. **Real workload relevance:** does the same optimization help the prefill/training shapes llama.cpp actually cares about?

Target shape families:

| Family | Shape examples | Why it matters |
|---|---|---|
| Peak stress / saturation | `32768 x 32768 x K16384`, `65536 x 65536 x K16384` | Keeps the GPU fully occupied and exposes peak behavior. |
| Training / weight-gradient | `4096 x 4096 x K16384`, `4096 x 14336 x K16384`, `14336 x 4096 x K16384`, `14336 x 14336 x K16384` | Matches large-K gradient/update style GEMMs. |
| Prefill projection | `4096 x 4096 x K4096`, `4096 x 14336 x K4096`, `8192 x 4096 x K4096`, `8192 x 14336 x K4096` | Matches token-batch by model-weight projection GEMMs. |
| Long-context / large-batch prefill | `16384 x 4096 x K4096`, `16384 x 14336 x K4096` | Tests more saturated prefill behavior without using artificial square-only stress. |

Standard evaluation ladder for a new optimization:

```text
1. Small oracle shape
   e.g. 512^2 or 1024^2 with manageable K

2. Peak stress shape
   32768^2 x K16384 or 65536^2 x K16384

3. One training shape
   4096^2 x K16384 or 4096 x 14336 x K16384

4. One prefill shape
   4096 x 14336 x K4096 or 8192 x 14336 x K4096
```

Only broaden the sweep after the optimization wins on the main target class. Avoid comparing two kernel
variants across different shape families unless the table explicitly reports `M`, `N`, `K`, tile size, resident
waves, saturation, and clock.

The host prepares:

- `A`
- `Bshuf` or other preshuffled B layout
- `C`
- scratch/result buffers such as `occ`
- PM4 dispatch packets

### Data Engineering Analogy

This is like defining a Spark/Glue job:

```text
source tables: A, B
output table: C
partitioning: tile grid
job parameters: executor count, memory, shuffle layout
```

### Important Variables

| Variable / config | Meaning | Data-engineering analogy | What changing it does |
|---|---|---|---|
| `M`, `N`, `K` | Matrix dimensions | Input table sizes and join key depth | Affects saturation, reuse, and total work. |
| `TBK` / K slice | K chunk per inner loop, usually 32 fp8 elements | Micro-batch size along join key | Larger would reduce loop count, but fp8 WMMA is fixed at `16x16x16`; no larger fp8 K WMMA exists. |
| Saturation size | Enough tiles to keep all workers busy | Enough partitions to fill an EMR cluster | Underfilled runs lie; always print total work and claims/pool. |
| Clock pin | Fixed GPU clock, e.g. 2350 MHz | Pinning instance performance / disabling autoscaling noise | Required for comparing small deltas. |

### Alterable

- Shape size for saturation.
- K dimension.
- Whether B is preshuffled and how.
- Whether A also gets an `A-shuf` layout.
- Perf mode versus oracle mode.

## Step 1: Memory Allocation

### What Happens

The host allocates buffers for A/B/C and metadata. For performance, operand buffers must be device-local VRAM.

### Decision: Hot Buffers Must Be Device-Local VRAM

For the prefill/training GEMM target, all hot operands, outputs, and hot-loop workspaces must live in
device-local VRAM:

```text
A
Bshuf
future Ashuf, if used
C
hot-loop workspace / counters / scratch buffers when they affect timing
```

Host-visible or system RAM is allowed only for:

```text
initial staging
CPU oracle/reference data
result readback
small control buffers that are not read in the hot loop
```

No performance result is valid unless the harness proves physical placement before dispatch. Required guard:

```text
assert_device_local(A)
assert_device_local(Bshuf)
assert_device_local(C)
assert_device_local(Ashuf)       # if present
assert_device_local(workspace)   # if used by the timed kernel
```

Every perf table should print a placement row, for example:

```text
placement: A=VRAM Bshuf=VRAM C=VRAM workspace=VRAM
```

This is not optional. The earlier `~1.4 TF` "wall" was caused by A/B being allocated in system RAM and
fed over PCIe. Correctness still passed, but the performance result was invalid.

### Data Engineering Analogy

This is the difference between:

```text
reading data from local NVMe on the worker
vs
reading every split across the public internet from a remote S3 region
```

The code was accidentally doing the equivalent of streaming from the wrong storage tier.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `deviceLocal=true` / `NonPaged=1` | Allocate in GPU VRAM | Put data on local worker storage | Mandatory for perf. Missing this caused fake PCIe bottlenecks. |
| Host-visible staging | CPU-accessible memory | S3/landing-zone staging area | Good for setup/copy, bad for hot kernel operands. |
| VRAM guard | Abort if perf buffers are not local | Data quality check on input location | Prevents silent invalid benchmarks. |

### What We Learned

If this stage is wrong, every downstream conclusion is wrong. The kernel can be correct but performance data is meaningless.

## Step 2: B Preshuffle / Layout Preparation

### What Happens

B is not fed to WMMA in ordinary row-major order. It is stored in a layout compatible with:

```asm
global_load_tr_b64
```

This instruction loads a 16x16 matrix of 8-bit data and transposes it into the VGPR fragment layout WMMA expects.

### Decision: Use `Bshuf + global_load_tr_b64` and Cache the Packing Plan

The current B path is:

```text
logical B[K, N]
  -> Bshuf packed layout in VRAM
  -> global_load_tr_b64
  -> WMMA-ready B fragment
```

Use `global_load_tr_b64` as the production fp8 B feed path. Do **not** treat
`global_load_tr_b128` as a drop-in replacement: the RDNA4 ISA defines `tr_b128` as a 16-bit matrix
transpose, while fp8 uses the 8-bit `tr_b64` path. Any `tr_b128` B path would require a separate custom
packing format and a fragment-forensics oracle.

For training, cache the **Bshuf packing plan** by shape/config. The plan is reusable; the packed bytes are
only reusable while the underlying B values are unchanged.

```text
Reusable:
  logical -> physical offset mapping
  specialized pack kernel
  Bshuf strides/constants
  required workspace size

Not reusable when values change:
  the actual Bshuf byte contents
```

Suggested cache key:

```text
K
N
dtype
tile_M / tile_N
FM / FN
TWM / TWN
Bshuf format version
global_load_tr mode
```

Data-engineering analogy:

```text
packing plan  = cached query plan / table layout spec
Bshuf bytes   = materialized view contents
```

If source B changes, the materialized view must be refreshed, but the plan does not need to be rediscovered.
For dynamic training operands, prefer eventually fusing the producer into the Bshuf layout, e.g. fp8
cast/quantization writes directly to Bshuf rather than writing row-major B and repacking it.

### Data Engineering Analogy

This is like pre-bucketing and sorting a fact table before a join:

```text
Without preshuffle: each worker performs expensive random reshaping at runtime.
With preshuffle: the runtime scan emits records directly in the join/operator format.
```

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `Bshuf` | Preshuffled B buffer | Pre-sorted/bucketed table | Required for correct/fast `global_load_tr_b64`. |
| `global_load_tr_b64` | 8-bit transpose load | Scan operator that also pivots data | Correct fp8 B feed path. |
| `global_load_tr_b128` | 16-bit transpose load | Similar-looking operator for a different schema | Not a fp8 drop-in; micro-oracle proved mismatch. |
| `tile_col`, `wave_n`, `ni` | B tile offsets | Partition id and column shard | Bugs here produce wrong quadrants/fragments. |

### Alterable

- Bshuf layout.
- Whether B goes through global transpose load or LDS.
- B tile geometry and N-wave count.

### Current Verdict

B via `Bshuf + global_load_tr_b64` is the known-good path. B-in-LDS was correct but did not beat the current baseline because LDS/barrier cost outweighed B dedup.

## Step 3: PM4 Dispatch

### What Happens

The kernel is not launched through normal HIP/HSA. It is launched through raw PM4 packets via KFD/libhsakmt.

PM4 sets hardware state:

- shader code address
- user SGPRs
- workgroup dimensions
- VGPR/LDS resource fields
- dispatch size
- fences/events

### Data Engineering Analogy

HIP is like using managed Glue/EMR. PM4 is like directly calling low-level ECS/Fargate/EC2 APIs and wiring the job runtime yourself.

You get more control, but you lose managed defaults.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `COMPUTE_PGM_RSRC1` | VGPR/SGPR/wave resource config | Executor memory/CPU config | Affects admission/occupancy. |
| `COMPUTE_PGM_RSRC2` | LDS/workgroup/dyn flags | Runtime feature flags | Wrong bits can break TGID/LDS/dyn behavior. |
| `LDS_SIZE` | Declared LDS allocation | Executor-local scratch disk/RAM | Too high reduces occupancy; too low corrupts. |
| `USER_SGPR` count | Scalar args passed to shader | Environment variables / task args | Wrong indices produce silent bad state. |
| `TGID_X` | Hardware workgroup ID | Partition id assigned by scheduler | Not delivered in this raw PM4 path; use atomic queue instead. |

### Current Verdict

Raw PM4 is necessary for some low-level experiments, but it is easy to misconfigure. TGID is unavailable here, so tile assignment uses an atomic queue.

## Step 4: Workgroup and Wave Geometry

### What Happens

A workgroup contains multiple wave32 waves. The classic current tile:

```text
TWM=2, TWN=2
4 waves per workgroup
logical tile: 128x128
per-wave accumulator tile: 4x4 WMMA fragments
```

Larger N geometry:

```text
TWM=2, TWN=4
8 waves per workgroup
logical tile: 128x256
```

### Data Engineering Analogy

Workgroup = Spark executor.  
Wave = worker thread inside the executor.  
Tile = partition of the output table.

`TWM/TWN` decide how many workers collaborate on one output partition.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `TWM` | Number of wave tiles in M dimension | Workers assigned to row shards | More M waves can share B, changes A/B reuse. |
| `TWN` | Number of wave tiles in N dimension | Workers assigned to column shards | More N waves share A, larger output tile. |
| `FM`, `FN` | Per-wave fragment tile | Per-worker local batch size | Bigger means more reuse but more VGPRs. |
| `waves_per_wg` | `TWM*TWN` | Threads per executor | Must compare resident waves, not just resident workgroups. |
| `maxliveWG` | Resident workgroups | Active executors | Misleading across geometries unless converted to resident waves. |
| `residentWaves` | `maxliveWG * waves_per_wg` | Total active workers | Correct occupancy comparison unit. |

### Current Verdict

The 128x256 tile gave only a modest saturated win once compared by resident waves. It is not the main path to 250-300 by itself.

## Step 5: Tile Assignment

### What Happens

Because `TGID_X` is unavailable, workgroups claim output tiles from a global counter:

```asm
global_atomic_add(counter, BAND)
```

The claimed tile index `ti` is broadcast to all waves in the workgroup through LDS.

### Data Engineering Analogy

This is a work queue:

```text
SQS/Kinesis shard/atomic counter = global tile queue
Worker grabs next task
Processes one or more output partitions
Repeats
```

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `BAND` / `CLAIMCHUNK` | Number of tiles claimed per atomic | Batch size per SQS receive | Reduces queue overhead; too large can imbalance work. |
| `claims` | Number of claims performed | Queue receives | Helps detect under-saturation or too much scheduler overhead. |
| `TOTAL` | Total tiles | Number of partitions | Must greatly exceed resident workers for saturation. |
| `grabs/pool` | Work per resident worker | Partitions per executor | Low values under-saturate and lie. |

### Current Verdict

Atomics can matter, but after saturation and VRAM fixes they are not the dominant wall in the current best fed path.

## Step 6: A Feed Path

### Current LDS-A Path

A currently goes through LDS:

```text
global_load_b128 A rows
s_wait_loadcnt
ds_store_b128 into LDS
s_wait_dscnt
s_barrier
ds_load_b64 A fragments
s_wait_dscnt
WMMA consumes A fragments
```

### Data Engineering Analogy

This is like repartitioning data into executor-local cache before a join:

```text
Read from S3/Parquet
write into executor-local shuffle/cache
sync all local workers
read from local cache in join-ready layout
```

The cache saves duplicate reads and fixes layout, but it costs local IO and synchronization.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `KWIN` | Number of K slices staged per LDS window | Micro-batch/window size | Amortizes barriers and publish cost. Best seen around `KWIN=4`. |
| Publish width `pw1/pw2/pw4` | Number of A loads in flight per publish wait | Concurrent S3 reads per task | Wider publish helped once clock/saturation were correct. |
| `LDS` bytes | A ring size | Local cache size | More LDS can reduce occupancy. |
| `ds_store_b128` | Write A to LDS | Write to local shuffle/cache | Part of A publish overhead. |
| `ds_load_b64` | Read A fragments | Read from local cache | Feed operation per WMMA fragment. |
| `ds_load_2addr_b64` | Potential two-frag LDS read | Vectorized local-cache read | ISA-supported way to reduce A LDS read instruction count without contiguous b128 layout. |
| LDS-free A | Direct global A fragments | Skip local cache/repartition | Only works if correct fragment layout is preserved. |
| `A-shuf` | Proposed preshuffled A layout | Pre-bucketed/pre-pivoted A table | Could allow direct `global_load_tr_b64` A, but must be micro-oracle verified. |

### Current Verdict

A-in-LDS is not just bandwidth optimization; it is also a layout/coalescing mechanism. A plain LDS-free load can be fast but wrong, or correct but uncoalesced. The active structural question is whether an `A-shuf + global_load_tr_b64` path can be both correct and faster.

## Step 7: B Feed Path

### What Happens

B is already fed directly:

```text
Bshuf global memory
global_load_tr_b64
s_wait_loadcnt ladder or wait
WMMA consumes B fragments
```

### Data Engineering Analogy

B is the table that has already been pre-sorted and stored in the exact format the compute operator wants, so each worker scans it directly.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `global_load_tr_b64` | fp8 transpose load | Scan + pivot operator | Correct B feed. |
| B preshuffle | Global layout | Precomputed materialized view | Required for correctness/perf. |
| B prefetch | Load next B slice early | Async prefetch | Hurt in current kernel; occupancy already hides latency. |
| B-in-LDS | Stage B for sharing | Local cache shared by workers | Correct but net loss due to LDS/barrier cost. |
| `global_load_tr_b128` | 16-bit transpose load | Operator for different schema | Not valid as fp8 drop-in. |

### Current Verdict

Bshuf + `global_load_tr_b64` is the known-good B path. The big remaining B-side win would need a new layout/instruction trick, not a simple b128 swap.

## Step 8: Waits and Barriers

### What Happens

GPU memory instructions are asynchronous. The shader must wait before using results:

```asm
s_wait_loadcnt N
s_wait_dscnt N
s_barrier_signal -1
s_barrier_wait -1
```

### Data Engineering Analogy

`waitcnt` is like waiting on futures/promises:

```text
Wait until async S3 read futures are complete before using data.
```

Barrier is like a Spark stage boundary:

```text
All workers in the executor must finish stage A before any proceed to stage B.
```

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `s_wait_loadcnt` | Wait for global loads | Await remote read futures | Too early serializes; too late corrupts. |
| `s_wait_dscnt` | Wait for LDS ops | Await local-cache writes/reads | Required for LDS correctness. |
| Barrier count | Workgroup synchronization | Stage boundaries | Removing unsafe barriers caused real LDS races. |
| `KWIN` | Barrier amortization window | Process multiple micro-batches per stage | Big win up to LDS/occupancy limits. |

### Current Verdict

Barriers are necessary for LDS correctness, but in the current best feed-only test removing both only bought ~1-2%. The wall is now feed operations per output, not barrier latency by itself.

## Step 9: WMMA Compute

### What Happens

The core matrix instruction:

```asm
v_wmma_f32_16x16x16_fp8_fp8
```

Each WMMA consumes A/B fragments and accumulates into f32 registers.

### Data Engineering Analogy

This is the specialized vectorized transform operator, like a highly optimized native Spark SQL expression or Redshift vectorized join kernel. Once inputs are in the right format, compute is extremely fast.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `FM`, `FN` | Number of A/B fragments per wave tile | Local batch dimensions per worker | Bigger increases reuse and accumulators. |
| Accumulators | f32 output fragments | Worker-local aggregate state | Consume most VGPRs. |
| `NOFEED` | Reuse operands already loaded | Compute-only benchmark | Proves matrix ceiling. Current ~297 TF. |
| `FEEDONLY` | Remove WMMA, keep feed | IO-only benchmark | If equal to FED, compute is free/hidden. |
| VGPR count | Register footprint | Worker memory per task | Higher can reduce occupancy; cap is 256 logical VGPRs per wave. |

### Current Verdict

WMMA is not the current fed wall. `FED == FEEDONLY` means the matrix work is hidden behind feed. This makes larger reuse tiles attractive, because extra WMMA may be nearly free while feed per output falls.

## Step 10: Output Store

### What Happens

The kernel stores f32 accumulator fragments to C. Many perf probes use minimal or diagnostic stores so output traffic does not hide the inner-loop signal.

### Data Engineering Analogy

This is writing the transformed table back to S3/Redshift. For debugging, we sometimes write a small audit sample instead of the full table.

### Important Variables

| Variable / config | Meaning | Analogy | What changing it does |
|---|---|---|---|
| `STORE=0/1` | Disable or enable full stores | Write audit sample vs full table | Isolates compute/feed from output bandwidth. |
| `acc00` | Lightweight correctness sample | Row-count/sample checksum | Fast but can miss races. |
| Full oracle | Full fragment correctness | Full data quality validation | Required for race-prone/layout changes. |

### Current Verdict

For performance exploration, stores are often reduced. Any claimed speedup must still have correctness or traffic proof appropriate to the probe.

## Step 11: Measurement and Guardrails

### Required Columns

Every meaningful run should report:

```text
TF
GB/s or feed-equivalent when relevant
correctness/proof
maxliveWG
waves_per_wg
residentWaves
claims / saturation
clock
VGPR field
LDS bytes
```

### Data Engineering Analogy

This is your observability layer:

```text
CloudWatch metrics
data quality checks
partition counts
cluster utilization
job duration
input location validation
```

Without this, a run can look fast because it skipped work, read from the wrong storage tier, or underfilled the cluster.

### Known Measurement Traps

| Trap | What happened | Guardrail |
|---|---|---|
| System RAM operands | PCIe path made fake feed wall | Assert device-local VRAM. |
| Under-saturation | Too few tiles made pool look slow | Print grabs/pool and resident waves. |
| Clock throttling | Unpinned clock invalidated deltas | Pin and print clock. |
| `acc00` too weak | LDS race could pass sample | Use full oracle for race/layout changes. |
| Broken fast kernel | PROFILE looked fast but wrong | No speedup trusted without correctness/traffic proof. |
| Workgroups vs waves | Geometry comparison was wrong | Always print resident waves. |

## Knob Summary

| Knob | Layer | Expected effect | Risk |
|---|---|---|---|
| `deviceLocal` allocation | Host memory | Mandatory perf correctness | Silent if not guarded. |
| Shape / saturation | Work distribution | Keeps GPU full | Underfilled runs lie. |
| `BAND` / chunking | Scheduler | Reduces atomic queue overhead | Load imbalance if too large. |
| `TWM/TWN` | Workgroup geometry | Changes tile size/reuse | Resident waves change; compare carefully. |
| `FM/FN` | Per-wave reuse | Reduces feed per output | VGPR cap/occupancy pressure. |
| `KWIN` | A LDS window | Amortizes barriers/publish | Larger LDS can cut occupancy. |
| Publish width | A feed | More A loads in flight per wait | VGPR/register pressure. |
| `ds_load_2addr_b64` | A LDS consume | Fewer LDS read instructions | Must match fragment layout. |
| A-shuf + `tr_b64` | A feed layout | Potentially removes LDS path | Requires new preshuffle and oracle. |
| B-in-LDS | B sharing | Deduplicates B feed | LDS/barrier cost outweighed gain so far. |
| `global_load_tr_b128` | B feed | Tempting wider load | Wrong data type for fp8 transpose. |
| Dynamic VGPR | Allocation | Elastic allocation within 256 cap | Volatile/unsafe on current cap-flip path. |

## Current Strategic Interpretation

The kernel is no longer mysterious:

```text
NOFEED near 300 TF proves matrix throughput exists.
FED ~= FEEDONLY proves the current wall is feed.
No-barrier barely helps, so synchronization is not the main current cost.
The live path is reducing feed operations per useful output.
```

The most important active ideas are:

1. Larger per-wave reuse tiles that still fit under the 256 logical VGPR cap.
2. A correct coalesced LDS-free A path via `A-shuf + global_load_tr_b64`.
3. ISA-supported A LDS read reduction via `ds_load_2addr_b64`.
4. Continued strict measurement discipline: VRAM, clock, saturation, correctness.

## Practical AWS Analogy: The Whole Kernel as a Join Pipeline

Think of this GEMM as a massive join/aggregation:

```text
A rows join B columns on K
Each output tile is a partition of the final aggregate table C
Each wave owns a local aggregation state in VGPRs
WMMA is the vectorized aggregation operator
```

The current bottleneck is not the aggregation operator. It is preparing records into the exact schema/layout that the operator needs.

In data-engineering terms:

```text
The compute UDF is basically free.
The expensive part is shuffle, repartition, local cache writes, and scan layout.
```

So the winning strategy is not "make the UDF faster." It is:

```text
read each input record fewer times
pre-layout data so workers don't reshape it at runtime
increase local batch/reuse size
avoid unnecessary stage boundaries
prove every optimization with data quality checks
```

That maps directly to the GPU levers:

```text
read each fragment fewer times      -> bigger FM/FN reuse tile
pre-layout data                     -> Bshuf, possible A-shuf
increase local batch size           -> KWIN, larger tiles
avoid stage boundaries              -> fewer safe barriers / no LDS path if correct
data quality checks                 -> oracle, checksum, acc00 only for smoke
```
