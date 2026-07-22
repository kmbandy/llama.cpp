# ⛔ RETRACTED AND CORRECTED — DSWS vs hipBLASLt on the REAL ml8 / mlambaformer shapes
## 2026-07-21 · the first version of this file was WRONG. Read §0 before citing anything here.

---

# §0 — RETRACTION. THE ORIGINAL DSWS COLUMN WAS NOT THROUGHPUT.

**Every DSWS TF number published in the first version of this file was manufactured by a parser bug.**
Not noisy, not optimistic — a different quantity entirely.

`sweep_dsws_realshapes.sh` matched `'<number> TF'`. The kernel prints `TF=<number>`. That pattern
**never matched on any shape**, so every row silently fell through to a fallback that took the
**LAST decimal on the throughput line**. That is:

- the **`spread N%`** field of the `SUSTAINED` line (shapes where reps ran), or
- the **`N% of 307 TF fp8 peak`** field of the `THROUGHPUT` line (shapes where they didn't).

| shape | published | log actually said | what was really captured |
|---|---:|---:|---|
| `ml8 moe attn_kv M=64` | 10.87 (**"6.39x WIN"**) | `TF=0.0` | `spread 16.3%` |
| `ml8 dense attn_o M=512` | 9.07 | `TF=1.4` | `spread 10.2%` |
| `mlmf lm_head` | 0.20 | `TF=0.6` | `0.2% of peak` |
| `ml8 dense ffn_gate/up M=512` | 1.51 | `TF=1.5` | `spread 1.7%` — coincidentally close |

**WHAT IS RETRACTED:**
1. **All four claimed wins over hipBLASLt.** There is **no** MoE-decode corner where DSWS beats the
   vendor. The three `M=64` MoE shapes read **`TF=0.0`** — 137 chunks for a 0.1 GFLOP problem, so
   dispatch/chunk overhead is the entire measurement.
2. **The flatness thesis.** "DSWS CV 0.700 vs hipBLASLt 0.922, we are flatter" was computed over the
   spread-percentage column. Corrected, **DSWS CV = 1.128 vs hipBLASLt 0.905 — we are LESS flat than
   the vendor.** The data now *contradicts* the thesis it was cited as proving.
3. **"Our mean is 11.5x lower."** True figure is **~80x** (0.87 vs 69.18).

**WHAT SURVIVES:** the hipBLASLt column (separate harness, `bench_hipblaslt_ml8.py`, untouched by this
bug); WORK-EXACT and oracle-CLEAN on all 26 shapes (correctness was never in question — only speed);
and the non-pow2 n_kseg fix, without which half these shapes would not run at all.

**FIXED:** `sweep_dsws_realshapes.sh` now anchors on `TF=` and takes the FIRST match on the line,
preferring `SUSTAINED` over `THROUGHPUT`. Verified against the archived logs to reproduce the true
values. The corrected table below was rebuilt **entirely from those archived per-shape logs** — no GPU
time was needed, because `~/dsws_gpu_logs/rs_*.log` survived.

**THE LESSON, GENERALIZED:** a harness that extracts a number by positional regex will happily return a
*different, plausible number* forever. Both defects found today (this, and an unverifiable "bin sha"
propagated into three documents) are the same failure: **a value was published without ever being
checked against its source.** Anchor extraction on a unique key, and spot-check the harness against a
raw log before trusting a single row.

---

# §1 — THE CORRECTED TABLE (config of record, WAVES=30)

Raw: `~/dsws_gpu_logs/rs_*.log` · rebuilt: `~/dsws_gpu_logs/dsws_vs_hipblaslt_CORRECTED.json`
fp8 roofline on R9700 gfx1201 = 307 TF. DSWS TF corrected to **real FLOP** (M padded up to the 96-row
super-tile, then scaled back by `M/Mpad`) — padding counts AGAINST us, never for us.

| shape | M | N | K | n_kseg | **DSWS TF (true)** | ~~retracted~~ | hipBLASLt TF | DSWS/hipBLASLt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ml8 dense ffn_down | 2048 | 2560 | 9216 | 36 | **4.36** | ~~1.16~~ | 189.3 | 0.023 |
| ml8 dense ffn_down | 512 | 2560 | 9216 | 36 | **2.40** | ~~4.71~~ | 123.3 | 0.020 |
| ml8 dense attn_o | 2048 | 2560 | 4096 | 16 | **2.33** | ~~3.30~~ | 159.2 | 0.015 |
| ml8 dense ffn_gate/up | 2048 | 9216 | 2560 | 10 | **1.55** | ~~0.48~~ | 186.7 | 0.008 |
| ml8 dense attn_q | 2048 | 4096 | 2560 | 10 | **1.55** | ~~4.07~~ | 159.6 | 0.010 |
| ml8 dense ffn_gate/up | 512 | 9216 | 2560 | 10 | **1.33** | ~~1.51~~ | 135.1 | 0.010 |
| ml8 dense attn_q | 512 | 4096 | 2560 | 10 | **1.24** | ~~3.56~~ | 84.8 | 0.015 |
| ml8 dense attn_o | 512 | 2560 | 4096 | 16 | **1.24** | ~~9.07~~ | 79.4 | 0.016 |
| ml8 dense attn_kv | 2048 | 1024 | 2560 | 10 | **1.16** | ~~5.43~~ | 97.3 | 0.012 |
| ml8 moe attn_o | 512 | 2048 | 4096 | 16 | **1.07** | ~~8.36~~ | 72.2 | 0.015 |
| ml8 moe attn_q | 512 | 4096 | 2048 | 8 | **0.98** | ~~5.07~~ | 70.0 | 0.014 |
| mlmf mamba out_proj | 4096 | 768 | 1536 | 6 | **0.69** | ~~2.18~~ | 68.9 | 0.010 |
| mlmf lm_head | 4096 | 32000 | 768 | 3 | **0.60** | ~~0.20~~ | 167.9 | 0.004 |
| ml8 dense attn_kv | 512 | 1024 | 2560 | 10 | **0.36** | ~~7.56~~ | 37.5 | 0.010 |
| mlmf attn o_proj | 4096 | 768 | 768 | 3 | **0.30** | ~~3.77~~ | 45.7 | 0.007 |
| ml8 moe attn_o | 64 | 2048 | 4096 | 16 | **0.20** | ~~4.80~~ | 9.6 | 0.021 |
| mlmf router down_proj | 4096 | 256 | 768 | 3 | **0.20** | ~~10.32~~ | 15.7 | 0.013 |
| ml8 moe ffn_gate/up | 512 | 512 | 2048 | 8 | **0.18** | ~~20.36~~ | 15.4 | 0.012 |
| ml8 moe ffn_down | 512 | 2048 | 512 | 2 | **0.18** | ~~4.36~~ | 12.4 | 0.015 |
| ml8 moe attn_kv | 512 | 512 | 2048 | 8 | **0.18** | ~~8.36~~ | 15.2 | 0.012 |
| mlmf MoE expert fc1 | 512 | 1536 | 768 | 3 | **0.18** | ~~6.22~~ | 14.2 | 0.013 |
| mlmf MoE expert fc2 | 512 | 768 | 1536 | 6 | **0.18** | ~~5.87~~ | 17.3 | 0.010 |
| ml8 moe attn_q | 64 | 4096 | 2048 | 8 | **0.13** | ~~9.80~~ | 16.9 | 0.008 |
| ml8 moe ffn_gate/up | 64 | 512 | 2048 | 8 | **0.00** | ~~6.60~~ | 1.7 | 0.000 |
| ml8 moe ffn_down | 64 | 2048 | 512 | 2 | **0.00** | ~~8.00~~ | 1.6 | 0.000 |
| ml8 moe attn_kv | 64 | 512 | 2048 | 8 | **0.00** | ~~10.87~~ | 1.7 | 0.000 |

|  | **DSWS (true)** | ~~retracted~~ | hipBLASLt |
|---|---:|---:|---:|
| mean | **0.87** | ~~6.00~~ | 69.18 |
| median | **0.48** | ~~5.25~~ | 57.30 |
| stdev | **0.98** | ~~4.12~~ | 62.57 |
| **CV** | **1.128** | ~~0.687~~ | **0.905** |
| min / max | 0.00 / 4.36 | ~~0.20 / 20.36~~ | 1.60 / 189.30 |
| **wins (ratio > 1)** | **0 / 26** | ~~4 / 26~~ | — |

**Not measured:** `mlmf mamba in_proj` N=4200 / `in_proj_ML8PAD` N=4208 — UNSUPPORTED (`N % 64 != 0`).
`attn_linear_k` / `val_proj1` / `router_out` not reached (sweep halted at `router_MLP`, n_kseg=1).

---

# §2 — ⭐ WHAT THE CORRUPT COLUMN WAS HIDING: THROUGHPUT TRACKS `n_kseg` ⭐

Sorted by the true numbers, the table orders itself almost perfectly by `n_kseg = K / SEGK`:

| n_kseg | 36 | 16 | 10 | 8 | 6 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| true TF | 4.36, 2.40 | 2.33, 1.24, 1.07, 0.20 | 1.55, 1.55, 1.33, 1.24, 1.16, 0.36 | 0.98, 0.18, 0.18, 0.13, 0.00 | 0.69, 0.18 | 0.60, 0.30, 0.20, 0.18 | 0.18, 0.00 |

**The published column showed no such structure — the bug was masking the single most important
signal in the dataset.**

**MECHANISM (derived from source, `occ_kernel_dsws_flow.s`, before these runs):**
1. A reservation is legal only while `r < DA_ZDONE` (:3983).
2. `DA_ZDONE` advances by **one field width** (`2^shift`) per group boundary (:4151).
3. That boundary requires `DRAIN >= ASSIGN` (:4086) **and** `GSTORED >= z>>shift` (:4093) — the
   previous group's C-store must have drained. Banks are reused (`zero_banks`, :4144), which is *why*
   groups serialize.
4. One reservation = one `ksi`, carried by ONE wave across all `ACC_N=3` rowblks (:4358, :4487).

=> **Instantaneous compute parallelism per workgroup = `min(WAVES, n_kseg)`.**

At `WAVES=30`, on the real shapes: `K=768 -> n_kseg=3 -> 3 of 30 waves can hold work (90% of the
workgroup idle BY CONSTRUCTION)`. `WAVES=30` was tuned on the **deep-K synthetic (K=524288 ->
n_kseg=2048)**, where units always outnumbered waves and this ceiling could not appear. It is the same
root cause as the FLUSH artifact (33.7% synthetic vs 1.4% real): **the synthetic hid a per-group supply
problem that only shows up when `n_kseg` falls into single digits.**

---

# §3 — MEASURED: REMOVING WAVES IS WORTH ~4.3x

`FLOW_WAVES` selects the binary by name (`occ_dsws2_w<N>_flow_gd.bin`), so each point is its own build.
All runs WORK-EXACT + oracle CLEAN. TF below is **as printed by the kernel** (padded M), read directly
off the `TF=` field, NOT through the sweep script.

| shape | n_kseg | W=30 | W=10 | W=5 | gain |
|---|---:|---:|---:|---:|---:|
| `ml8_dense_ffn_gate_up M=512 N=9216 K=2560` | 10 | 1.5 | 4.1 | **6.5** | **4.3x** |
| `mlmf_lm_head M=4096 N=32000 K=768` | 3 | 0.6 | — | **2.6** | **4.3x** |

Identical 4.3x at `n_kseg=3` and `n_kseg=10` => **not shape-specific.**

Counters move the right way (`ffn_gate_up`, W=10 -> W=5): coast **93.5% -> 64.0%**; boundary bails
`occ[97]` **754,475 -> 205,288**; starvation iters `occ[86]` **5.86M -> 1.21M**; feed-stages **0 ->
1,568**. `door1 NOTHING-STAGED` remains 100% of coast at every wave count — the supply of units is the
wall, exactly as §2 predicts. `door3 FAT-PEAK-FULL` and `door4 GROW-FAIL` are **0** throughout: the
dyn-VGPR moat never engages, so it is not paying for itself here.

**PREDICTION MADE AND FALSIFIED (recorded because it was registered in advance):** I predicted the
optimum would sit AT `n_kseg` and that going below it would lose throughput to lost parallelism.
`W=5 > W=10` on a `n_kseg=10` shape falsifies that — contention among starved waves outweighs the
parallelism they provide, so the curve had not yet turned over at 5. **`WAVES=4` is unbuildable**:
`NCOMPUTE = WAVES-3 = 1` gives `BATON_MAGIC = 2^32/1`, not representable in 32 bits. The
`.if NCOMPUTE < 1` guard (:780) catches 0 but not 1. It fails loud at assembly time — a gap, not a
hazard.

**COUNTER-FREE ASSIGN REMAINS THE PLANNED WORK.** (I briefly wrote here that this result "retired" it.
That was wrong on the merits AND wrong to decide unilaterally — cancelling planned architecture is
kmbandy's call, not a conclusion to draw from one measurement. Retracted; see §3a.)

## §3a — TWO FOLLOW-UPS THAT INVERT THE DIAGNOSIS: THE SHARED CURSOR IS THE BOTTLENECK

| change | result |
|---|---|
| `SEGK=256 -> 64` (4x MORE units: n_kseg 10 -> 40, all 30 waves feedable) | **1.5 -> 1.2 TF, WORSE**; coast ROSE 93.5% -> 97.5% |
| `BATCH=2` at `WAVES=5` (more work per CAS) | **ABORTED** — chunk 0.81s vs ~0.08s at BATCH=1, >=10x slower |

**Units are NOT the wall — `min(WAVES, n_kseg)` is dead as a throughput explanation.** More units means
each reservation carries LESS work (1/4 at SEGK=64), so CAS traffic per unit of output goes UP.
`door1 NOTHING-STAGED = 100%` was never evidence about supply: under SELFSERVE that is the **vestigial
ring** door, which reads 100% regardless.

`BATCH=2` was catastrophic at `WAVES=30` AND at `WAVES=5`, so a wave holding the shared `SSWIN` window
while it drains serially is **intrinsic to the shared cursor**, not a wave-count artifact.

**All three results fit one explanation: the single shared `ASSIGN` cursor CAS is the bottleneck.**
Adding units adds CAS traffic; batching deadlocks on the shared window; the only thing that helps is
fewer waves fighting over it. **=> counter-free assign — which removes the shared thing rather than
working around it — is the indicated fix, now with `WAVES=5` as a far cleaner starting point.**

---

# §4 — NEXT

1. **Re-sweep at the better wave count with the FIXED extractor** to get a true baseline table.
   Projected from the 4.3x: ~4 TF typical, ~19 TF best. Real, still far from the vendor, honest.
2. **Wave count wants to be SHAPE-AWARE** (`~min(WAVES, n_kseg)`-driven). The per-wave-count binary
   naming already supports it — build a small set of `w*` binaries, let the dispatcher pick by n_kseg.
3. **BUILD COUNTER-FREE ASSIGN** (design: brief §6 + KG `efa5d89f`). §3a shows the shared `ASSIGN`
   cursor is the bottleneck: more units makes it worse, batching on it aborts, and only removing waves
   helps. This is the one lever that removes the shared thing. **"More units per group" is REFUTED** —
   `SEGK=64` gave 4x the units and was *slower*.
4. Open, smaller: `n_kseg==1` (K=256) fail-safe; `N%64` shapes (mamba in_proj N=4200); the `occ[20]`
   over-claim (benign, unexplained).
