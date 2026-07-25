> # ⚠️ PRIORITIES SUPERSEDED — 2026-07-13 (same day, later)
> **The vendor MEASUREMENTS in this doc are good and still stand** (§9 ml8/mlambaformer baselines, the
> `GSU1` finding, the Tensile teardown). **Its PRIORITIES are WRONG.**
>
> This doc argues hard that the split-K flush (4.0 LDS ops/WMMA), vector width (4.1 vs 16 B/insn) and
> tile size are the levers. The **phase profile then measured `FLUSH` at 6.4%** while **`FOLLOW_WAIT`
> was 81%** — the compute waves were idle, waiting on a pipeline that had been serialized by a
> hardcoded constant.
>
> **DO NOT touch SEGK / flush cost / vector width / tile size until the flow economy runs.**
> They are all downstream of a machine that was not running.
>
> **→ Read `DSWS_POOL_UNLOCK_2026-07-13.md` first.**

# hipBLASLt teardown — what the 230 TF actually is, and what it means for DSWS
**2026-07-13.** Trigger: kmbandy benchmarked stock torch 2.13 / ROCm 7.2.3 (hipBLASLt) fp8 on the R9700
and it beat our best kernel. This doc records what we found when we took it apart. Read this before
resuming any DSWS perf work.

---

## 1. The numbers that started it

Stock torch (hipBLASLt), gfx1201 / R9700, fp8 vs bf16:

| shape | bf16 | **fp8** | speedup |
|---|---|---|---|
| FFN up `[4096,768]@[768,2048]` | 81.2 | **141.8** | 1.75x |
| FFN down | 61.4 | **151.4** | 2.46x |
| attn proj | 54.1 | **115.6** | 2.13x |
| lm_head | 90.4 | **176.4** | 1.95x |
| 4096^3 square | 122.4 | **230.3** | 1.88x |
| 8192^3 square | 124.7 | **219.0** | 1.76x |

Our numbers for comparison (fp8 ceiling on this card = **307 TF**):
- **static 8x2 KWINBPF kernel: 165.7 TF** on *square* shapes = 54% of peak.
- our own HIP 4x4-dbuf reference: 161 TF.
- **DSWS (adaptive): has never posted a number.** 165.7 is NOT DSWS.

### 1a. The process failure (say it out loud)
We benchmarked our hand-written PM4 kernel against **our own HIP reference** (161 TF) and were pleased to
beat it by 3%. We never once ran it against the vendor library that was installed on the same box. hipBLASLt
has shipped **162 tuned gfx1201 fp8 kernel libraries** in `/opt/rocm/lib/hipblaslt/library/` since the
**2026-05-11** ROCm 7.2.3 upgrade. It was there the whole time.
**RULE: the baseline is the best thing that already exists on the machine, not the thing we wrote.**

---

## 2. Roofline: none of this is bandwidth. It's all configuration.

Machine balance = 307 TF / 644.6 GB/s = **476 FLOP/byte**. Every shape above runs at AI 599..4096 =>
**all six are COMPUTE-bound.** Not one is near the memory wall.

| shape | arith. intensity | roofline | hipBLASLt | % of roofline |
|---|---:|---:|---:|---:|
| FFN up | 599 | 307 TF | 141.8 | 46.2% |
| FFN down | 793 | 307 TF | 151.4 | 49.3% |
| **attn proj** | **1170** | 307 TF | **115.6** | **37.7%** |
| lm_head | 1489 | 307 TF | 176.4 | 57.5% |
| 4096^3 | 2048 | 307 TF | 230.3 | 75.0% |
| 8192^3 | 4096 | 307 TF | 219.0 | 71.3% |

**The tell:** performance does NOT track arithmetic intensity. attn proj has 2x the AI of FFN-up and runs
SLOWER (115.6 vs 141.8). A memory effect would be monotonic. This is **tile quantization** — Tensile picking
a pre-tuned config that fits the squares and misfits everything else.

### => THE THESIS (kmbandy). Flatness beats peak.
hipBLASLt's envelope is **115.6 -> 230.3 TF, a 2.0x swing driven entirely by shape-fit.** DSWS wins if its
FLOOR is high, not if its peak is. **A kernel that merely holds ~165 flat already beats hipBLASLt on attn
proj (115.6) and FFN-down (151.4).** And the 230 on squares is the *ceiling proof*: 75% of peak is reachable
on this silicon, so our 54% is not a hardware limit either.

Falsification condition: if DSWS's own shape-variance is just as wide, or it's flat-but-slow, the thesis dies.
**We have never measured DSWS's variance curve. That is the load-bearing untested assumption.**

---

## 3. Teardown: WHY hipBLASLt gets 230 (this is the formula we were handed)

Read straight out of the Tensile solution library
(`TensileLibrary_B8B8_..._gfx1201.dat`, 3116 kernels; dominant config = **436x `MT128x128x64`**):

```
Cijk_Alik_Bljk_B8B8S_..._MT128x128x64_MI16x16x1_..._MIWT4_4_..._WG32_4_1
                         ^^^^^^^^^^^^  ^^^^^^^^^      ^^^^^^^      ^^^^^^^^
                         macro tile    WMMA 16x16     wave tile    4 waves/WG
        ..._GRVWA8_GRVWB8_..._LRVW16_..._PLR1_..._LBSPPA256_LBSPPB128_..._WGM8_...
             ^^^^^^^^^^^^^      ^^^^^^      ^^^^      ^^^^^^^^^^^^^^^^^      ^^^^
             global read VW 8   LDS read VW 16  prefetch-LR   LDS padding    L2 swizzle
```

### 3a. IT IS NOT THE TILE. (My first hypothesis; REFUTED — do not re-chase.)
`MIWT4_4` = each wave computes a **4x4 frag tile** = 16 frags = 128 accumulator VGPRs. That is the **SAME
per-wave geometry as our own HIP 4x4-dbuf reference, which scores 161 TF.** They get 230 from an identical
accumulator tile. Tile size is not the lever. (I extrapolated an 8x8 tile predicting 232 TF and it matched
their 230 almost exactly — a coincidence from a one-point calibration. It was wrong. Model fit != mechanism.)

### 3b. IT IS INSTRUCTION EFFICIENCY — vector width on data movement.
Issue-port accounting (our own model, calibrated on the 165.7 measurement: ~31 feed instrs per 32 WMMAs):

| kernel | TF | WMMA/cyc (of 15.9) | **feed instrs / WMMA** |
|---|---:|---:|---:|
| static 8x2 (KWINBPF) | 165.7 | 8.58 | **0.85–0.97** |
| HIP 4x4-dbuf ref | 161.0 | 8.34 | 0.91 |
| **hipBLASLt squares** | **230.3** | **11.93** | **0.33** |
| fp8 ceiling | 307.0 | 15.90 | 0.00 |

They issue **~1/3 the feed traffic per unit of math.** Where does it come from? Bytes moved per
data-movement instruction:

| kernel | B / data-movement insn |
|---|---:|
| **hipBLASLt** (`LRVW16`, `GRVWA8/B8`) | **16.0** |
| our static coop kernel (already uses `global_store_b128`) | 9.9 |
| **our DSWS flow kernel** | **4.1** |

**The flow kernel — the adaptive one we are actively building — moves 4 bytes per instruction. It is 2.4x
worse than our OWN static kernel.** That is a self-inflicted regression, not a hardware limit.

### 3c. The single worst offender: the flow kernel's C-store loop
`occ_kernel_dsws_flow.s` (~:1841-1849), executed **384 times per tile**:
```asm
ds_load_b32      v13, v12 offset:(frag*1024 + e*4)
s_wait_dscnt 0x0                                     // <-- a FULL LDS DRAIN, PER DWORD
global_store_b32 v10, v13, s[28:29] offset:...
```
4-byte load + full wait + 4-byte store, per element. The coop kernel already emits `global_store_b128` for
the same job. Fix = load 4 dwords, ONE wait, ONE b128 store.

**HONEST CAVEAT (do not over-promise):** widening the flow kernel closes the gap **to our own static kernel**,
not automatically to 230. The static kernel is at 9.9 B/insn and still scores only 165.7 — so vector width
alone does NOT explain the full 165->230. The remainder is in the other Tensile levers (3d).

### 3d. The rest of the checklist (unexplored levers, straight from their config)
- `GRVWA8` / `GRVWB8` — global read vector width 8
- `LRVW16` — LDS read vector width 16 (`ds_read_b128`)
- `PLR1` — prefetch local read (software pipelining of LDS->VGPR)
- `WGM8` — workgroup swizzle for L2 locality
- `LBSPPA256` / `LBSPPB128` — LDS padding (bank-conflict avoidance)
- `MT128x128x64` — K-depth 64 per LDS stage

**Every one of these is a knob a hand-written kernel can set — and an ADAPTIVE kernel can VARY PER SHAPE.
That is the DSWS pitch in one line: Tensile picks one config from a lookup table; DSWS picks its config at
runtime.**

---

## 4. What survives, what dies

**DIES:** "our hand-written square GEMM beats the reference." It doesn't; the reference was a strawman we
wrote. 165.7 vs 230.3 on the same shapes, same silicon.

**SURVIVES (and is sharper than before):**
- The dyn-VGPR moat is untouched — HIP still cannot express `s_alloc_vgpr`; AMD's own profiler stack is blind
  to our KFD/PM4 kernels. What died was the *demo* (beat everyone at square GEMM), not the mechanism.
- The adaptive thesis is now **quantified**: a 2.0x shape-variance envelope in the incumbent, on shapes that
  are all compute-bound, caused by static configuration. That is a real, measured hole in the market leader.
- Our own phase profile already said WMMA is only **2–4%** of compute-wave time. We are not math-bound. The
  gap is flow/feed overhead — exactly what this teardown says hipBLASLt solved with vector width + pipelining.

---

## 5. Work stack (dependency order)

1. **Confirm the instrumentation fix** (GPU, tonight). *No perf number from DSWS is trustworthy until this
   lands* — every timing we have was taken with counters now known to corrupt the data path
   (see `~/dsws_gpu_logs/2026-07-13_instrumentation_is_the_bug.md`).
2. **Widen the flow kernel's data movement.** C-store loop first (§3c) — most egregious, and the coop kernel
   is the template. Then the A/B staging loads. Orthogonal to everything else; pure win.
3. **Shape sweep, DSWS vs hipBLASLt**, same six shapes, same methodology → get OUR variance curve and overlay
   it. This is the experiment that validates or kills the thesis.
4. Then the remaining Tensile levers (§3d), made **adaptive** rather than fixed.

## 6. Standing rules learned today
- **Benchmark against the best thing on the machine, not against your own reference.**
- A model that fits the number is not the same as the mechanism (§3a — I fit 232 vs their 230 with the wrong
  cause). Read the artifact; don't extrapolate to it.
- Measuring apparatus must be provably inert on the path it measures.

---

# 7. THE BIG ONE: hipBLASLt DOES NOT SPLIT K. (found while doing the C-store fix)

`strings TensileLibrary_B8B8_*_gfx1201.dat | grep -oE '_GSU[0-9]+_' | sort | uniq -c`
```
    612 _GSU1_          <-- Global Split-U = 1.  EVERY kernel.  ZERO exceptions.
```
**All 612 of their gfx1201 fp8 kernels hold the accumulators in VGPRs across the ENTIRE K loop and store
C exactly once.** No LDS accumulator banks. No per-segment reduction. Zero flushes.

## 7a. What our split-K costs, in their currency
Per rowblk-segment (SEGK=32 => KSEG_STEPS=2, FM=2, FN=4):
- WMMAs issued        : KSEG_STEPS * FM * FN = **16**
- LDS reduce ops       : FM*FN*8 = **64** (`ds_add_f32`, one per accumulator dword)
- => **4.0 LDS ops PER WMMA, of pure flush**, before a single operand load.

hipBLASLt's **entire** feed budget is **0.33 instrs/WMMA**. We spend **12x their whole budget** just
flushing accumulators. At n_kseg=64 we flush the accumulators to LDS **64 times per tile**; they flush zero.

**This independently confirms our own phase profile, which we never connected to a cause:**
> FLUSH (split-K C-writes, 32x amp) = **26–51%** of compute-wave time — the #1 cost.

## 7b. But the cost is a DIAL WE ALREADY OWN
flush LDS-ops per WMMA = `128 / SEGK`. And **SEGK is already a `-defsym`.**

| SEGK | n_kseg (K=2048) | flush LDS/WMMA | flushes/tile |
|---:|---:|---:|---:|
| **32 (current)** | 64 | **4.00** | 64 |
| 64 | 32 | 2.00 | 32 |
| 128 | 16 | 1.00 | 16 |
| 256 | 8 | 0.50 | 8 |
| 512 | 4 | 0.25 | 4 |
| 2048 (no split) | 1 | 0.06 | 1 |

SEGK=256 is **8x cheaper**; SEGK=512 is **16x**. We have never swept this knob against the oracle.

## 7c. THE DESIGN TENSION, NOW QUANTIFIED (this is the DSWS thesis, sharpened)
Split-K in DSWS is **NOT for parallelism** — it exists to **CREATE THE DYN-VGPR HEADROOM**: short compute
bursts mean the VGPR budget is frequently freeable, so feed waves can grow into it on demand instead of
compute camping the SIMD for all of K. (KG: the v2 vision doc.)

- **Short segments** (small SEGK) = frequent budget release = maximal adaptivity = **expensive flush**.
- **Long segments**  (big SEGK)   = cheap flush = **compute camps the SIMD**, adaptivity starves.

=> **The optimal SEGK is a shape-dependent trade.** A Tensile lookup table picks GSU1 for everything and
eats the 2.0x shape-variance we measured in §2. An adaptive kernel can *pick its own point on this curve
per shape, at runtime.* **That is the DSWS pitch, and §7 is the first quantified evidence for it.**

## 7d. Immediate experiment (free — SEGK is a build flag)
Sweep SEGK = {32, 64, 128, 256, 512} against the oracle + the TF counter, at ACC_N=6 and ACC_N=3.
Costs nothing but GPU time. **We have never done this.** Expect: TF climbs steeply as the flush cost falls,
until adaptivity/occupancy starts to bind — and the knee is the DSWS operating point.

---

# 8. C-store widening — APPLIED (2026-07-13, offline, uncommitted)
`occ_kernel_dsws_flow.s` completer C-store: per-dword `ds_load_b32 + s_wait_dscnt 0x0 + global_store_b32`
(x8 elems x FM*FN frags x ACC_N banks) -> **two `ds_load_b128` + ONE wait + two `global_store_b128` per frag.**
Safe by layout: a lane's 8 f32 in a frag are contiguous 32B on BOTH sides (LDS vaddr `v10=lane*32` + bank base,
global vaddr `v10=lane*32`), and every offset is 16B-aligned (ACC_BASE/ACC_STRIDE are 1024B multiples).
`v16..v23` are free in the lean block (only the TRACE row-builder touches them; one-shot, dead at its store).

Measured (ACC_N=6, STAGINSTR=1, llvm-objdump):
| | before | after |
|---|---:|---:|
| data-movement instrs | 965 | **389** (2.48x fewer) |
| `s_wait_dscnt` (full LDS drains) | 502 | **166** (336 removed) |
| bytes / instr | 4.1 | **10.3** (coop=9.9, hipBLASLt=16.0) |
| .text | 17040B | 9936B |
Bytes moved: **3988 -> 3988** (identical; pure instruction-count win, semantics preserved).

**HONEST SIZING:** the C-store runs ONCE PER TILE, not per segment. Execution-weighted it only buys
**0.15 feed/WMMA** (0.97 -> 0.82) — worth doing, and the 336 removed serializing drains are worth more than
the issue-slot model shows, but **this is not where 165->230 lives. §7 is.**

---

# 9. hipBLASLt ON THE REAL ml8 SHAPES — MEASURED (2026-07-13). THIS IS THE BAR.
Harness + data: `~/dsws_gpu_logs/bench_hipblaslt_ml8.py`, `hipblaslt_ml8_baseline.{json,log}`
R9700 gfx1201, torch 2.13 / hip 7.13, `torch._scaled_mm` (= the hipBLASLt fp8 path).

## 9a. HEADLINE: they never exceed 190.2 TF on ANY ml8 shape.
**The 230.3 TF that started this investigation is a 4096^3 SQUARE number and is NOT AVAILABLE on the
shapes ml8 runs.** The bar is 190, not 230 — and on most shapes it is far lower.

## 9b. On shapes large enough to judge fairly (>=2 GFLOP): 9%–62% of their OWN roofline.
A **6.6x efficiency swing**, and **8 of 11 leave >50% of the machine on the floor**. Every one of these is
COMPUTE-bound, so this is Tensile lookup-table miss, not physics.

| shape | M | fp8 TF | % of roofline |
|---|---:|---:|---:|
| dense ffn_down | 2048 | **190.2** | 62% *(their best ml8 case — still under their 75% on squares)* |
| dense ffn_gate/up | 2048 | 188.9 | 62% |
| dense attn_o | 2048 | 156.4 | 51% |
| dense attn_q | 2048 | 148.2 | 48% |
| dense ffn_gate/up | 512 | 142.5 | 46% |
| dense ffn_down | 512 | 118.4 | 39% |
| dense attn_q | 512 | 93.6 | 31% |
| dense attn_o | 512 | 81.7 | 27% |
| dense attn_kv | 2048 | 79.5 | 26% |
| moe attn_q | 512 | 67.5 | 22% |
| moe attn_o | 512 | 65.5 | 21% |
| **dense attn_kv** | **512** | **28.6** | **9%** <- worst; compute-bound (AI=539) and they get 9% |

## 9c. fp8 BUYS NOTHING on 9 of 20 shapes.
All the MoE expert GEMMs, plus dense attn_kv@512. **`moe ffn_down` M=512 runs 0.69x — hipBLASLt's fp8 is
SLOWER THAN bf16.** For the 35B-A3B MoE (our headline inference model) the vendor fp8 GEMM is worthless
or actively harmful.

## 9d. CAVEATS — do not overclaim
1. **The M=64 / sub-2-GFLOP shapes are NOT a fair library test.** 0.1-GFLOP problems, launch-overhead
   dominated; excluded from 9b. Real MoE uses a **grouped** GEMM (all experts in one launch), so
   per-expert dispatch is unrealistic. **A grouped-GEMM measurement is required** before claiming
   anything about the MoE floor.
2. **We still have NO DSWS number at these shapes.** We have measured the incumbent precisely and
   ourselves not at all.
3. Even their best ml8 case (62%) is under their 75% on squares — ml8 geometry is hard for everyone.

## 9e. WHAT THIS MEANS
First hard evidence, **on our own shapes**, for the flatness thesis (§2): a **6.6x spread on compute-bound
work**. **A kernel merely holding ~150 TF FLAT beats hipBLASLt on 8 of the 11 measurable ml8 shapes.**
**DSWS's job is to BEAT THE FLOOR, NOT THE CEILING.**

## 9f. THE CHEAPEST SHOT WE HAVE NOT TAKEN
Our **static 8x2 KWINBPF kernel already does 165.7 TF on squares.** If it holds anywhere near that on the
ml8 shapes, **it may ALREADY beat the vendor library across most of the ml8 workload** — and we have been
sitting on that result for months without knowing. **That is ONE RUN.** Do it before any further DSWS
optimization work.

## 9g. Revised work stack
1. Confirm the instrumentation fix (correctness). *No DSWS perf number is trustworthy until this lands.*
2. **Static kernel on the ml8 shapes** — the cheap shot in 9f. Might already be the win.
3. Grouped-GEMM hipBLASLt baseline for the MoE experts (fair-test the floor).
4. DSWS flow kernel on the ml8 shapes: first honest TF + coast-frac + **grow-fail** profile.
   *If grow-fail is still 0 at a real shape, the adaptive economy is not engaging — that is the
   foundational bug, bigger than any tuning.*
5. Per-shape SEGK ladders (§7b + the legal-SEGK table: n_kseg must be a power of 2, SEGK a multiple of 16;
   K=2560 cannot do n_kseg=64 at all). Find the knee = largest SEGK where grow-fail still binds.

---

# 10. TIME-WEIGHTED TARGET — mlambaformer + ml8 shapes (measured 2026-07-13)
Harness: `~/dsws_gpu_logs/bench_hipblaslt_ml8.py` (33 shapes). Log: `hipblaslt_all_shapes.log`.
**HOST numbers (TheRock ROCm 7.13). The mlambaformer doc's numbers are CONTAINER (mainline 7.2). Close but
NOT identical (in_proj bf16 72.1 here vs 81.2 there). Do not mix the two sets.**

## 10a. TFLOPS is the wrong lens. Rank by ms/step.
mlambaformer forward, best-of(bf16,fp8) = what you would actually ship:

| shape | GF/step | best TF | dtype | ms/step | **% of GEMM time** |
|---|---:|---:|---|---:|---:|
| **MoE expert fc2** `[512,1536]@[1536,768]` | 231.9 | 19.9 | bf16 | 11.67 | **28.1%** |
| **MoE expert fc1** `[512,768]@[768,1536]` | 231.9 | 20.1 | bf16 | 11.53 | **27.7%** |
| **mamba in_proj** `[4096,768]@[768,4200]` | 475.6 | 72.1 | bf16 | 6.60 | **15.9%** |
| router MLP | 25.8 | 9.4 | bf16 | 2.74 | 6.6% |
| mamba out_proj | 173.9 | 66.6 | fp8 | 2.61 | 6.3% |
| router down_proj | 38.7 | 22.3 | fp8 | 1.74 | 4.2% |
| lm_head | 201.3 | 169.3 | fp8 | 1.19 | 2.9% |
| router out | 0.4 | 0.3 | bf16 | 1.15 | 2.8% |
| attn o_proj | 58.0 | 50.9 | fp8 | 1.14 | 2.7% |
| **TOTAL** | **1451.9** | | | **41.57 ms** | |

**TOP 3 = 72% of all GEMM time. THE TWO MoE EXPERTS ALONE = 55.8%.**

## 10b. THE MoE EXPERTS ARE THE PRIZE — and they are a DSWS-shaped problem
They run at **~20 TF against a ~250-300 TF roofline = 5-8% of what the shape allows.** fp8 makes them
WORSE (0.67x / 0.70x), so they ship bf16.
**Why:** M=512 (top-1 of 8 -> each expert sees only its ROUTED tokens), invoked **192x per step**
(8 experts x 24 layers) at ~1.2 GFLOP a call. A library launching 192 separate small GEMMs cannot beat
~5% of roofline — that is launch overhead + tile quantization, not physics. **Persistent kernel,
small-M, many-call, adaptive: that is exactly the DSWS shape.**

## 10c. ml8 is STRUCTURALLY the right medicine for exactly these shapes
Under fp8 (8-bit weights) the MoE experts are **MEMORY-bound** — M=512 does not amortize the weight matrix.
Under **ml8's 4.25 bpv** weights their roofline rises to **~300 TF and they become COMPUTE-bound**
(fc1 300.3T, fc2 307T). **This is the only place ml8's compression materially helps:** at M=4096 the weights
are already amortized over 4096 tokens and ml8 raises AI only ~1.02x.
**RULE: ml8's compression pays when M is SMALL (MoE experts, decode), not at large M.**

## 10d. FREE WIN — ship today, no kernel work
`mamba in_proj` is **N=4200, not divisible by 16**, so `torch._scaled_mm` **REJECTS it** — hipBLASLt has no
fp8 path and it falls back to bf16 (72.1 TF). **Pad N 4200 -> 4208** (next multiple of 16; WMMA needs 16 —
ml8's group_size=64 constrains K, not N) and fp8 runs at **107.3 TF = 1.52x**.
=> **5.2% of TOTAL GEMM time, for a one-line weight-tensor shape change.**

## 10e. Envelopes (>=2 GFLOP only)
| | floor | ceiling | swing |
|---|---|---|---|
| hipBLASLt on **ml8** shapes | dense attn_kv M=512 @ 29.1 TF = **9.5%** of roofline | dense ffn_down M=2048 @ 190.9 TF = 62.2% | **6.6x** |
| hipBLASLt on **mlambaformer** shapes | attn o_proj M=4096 @ 50.9 TF = **16.6%** | lm_head M=4096 @ 169.3 TF = 55.1% | **3.3x** |

**fp8 BUYS NOTHING OR LOSES on 16 of 33 shapes** — including EVERY MoE expert GEMM in BOTH models.
