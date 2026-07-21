# ⭐ DSWS vs hipBLASLt — THE REAL ml8 / mlambaformer SHAPES ⭐
## 2026-07-21 · CONFIG OF RECORD · first head-to-head on the shapes we actually run

**This is the reference table. Every DSWS throughput claim is measured against THIS.**
Superseded numbers (synthetic cubes, deep-K feed shapes, invented configs) are NOT comparable and
must not be cited next to these.

---

## 1. THE CONFIG (config of record — do NOT deviate without a superseding decision record)

```
BUILD  (bin sha 397bfbe1cb010c6e)
  WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 VBUDGET=1536 JDEPTH=1 KMAJOR=0
  STAGGER=1 MAXFAT=0 DECENTASN=1 SELFSERVE=1 BANKZERO=1 RBU=1 SSWIN=8
  INITBAR=1 TERMFIX=1 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1 ./build_flow.sh
    -> GROUPS = G/ACC_N = 2 ; super-tile M = G*16*FM = 96 ; N-panel = FN*16 = 64

DISPATCH (per shape, via ./gpu_run.sh)
  SSWIN=8 FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 FLOW_POOL_N=1
  DSWS2_SEGK=256 DSWS2_K=<K> DSWS2_ORACLE_MTL=<Mpad/96> DSWS2_ORACLE_NTL=<N/64>
  DSWS2_ORACLE_STRIDE=32 DSWS2_TARGET_SECS=1.5 STAGINSTR=1 FORENSICS=0 TFPROBE=1
  ./occ_dispatch --dsws2

SWEEP        ./sweep_dsws_realshapes.sh        (STRIDE=32 TARGET_SECS=1.5)
hipBLASLt    ~/dsws_gpu_logs/bench_hipblaslt_ml8.py  (torch 2.13 / hip 7.13, torch._scaled_mm)
RAW          ~/dsws_gpu_logs/dsws_sweep_CONFIGOFRECORD.out
             ~/dsws_gpu_logs/dsws_vs_hipblaslt_configofrecord.json
```

**Validity:** all 26 measured shapes are **WORK-EXACT** (`computed == G*TOTAL_super*reps`) and
**oracle CLEAN** (`bad=0 max_rel=0`). fp8 roofline on R9700 gfx1201 = 307 TF.

**M padding:** the config-of-record super-tile is 96 rows and the dispatch requires `M % 96 == 0`.
Real M (64/512/2048/4096) do not divide 96, so M is padded UP to the next multiple of 96 and the
reported DSWS TF is corrected back to **real FLOP** (`TF_real = TF_padded * M/Mpad`). The padding
therefore counts AGAINST us — it is never flattery. Pad factors: M=64→96 (+50%), 512→576 (+12.5%),
2048→2112 (+3.1%), 4096→4128 (+0.8%).

---

## 2. THE TABLE

| shape | M | N | K | GFLOP | **DSWS TF** | hipBLASLt TF | DSWS/hipBLASLt |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ml8 moe attn_kv**    |   64 |   512 | 2048 |   0.1 | **10.87** |   1.70 | **6.39x** ⬆ |
| **ml8 moe ffn_down**   |   64 |  2048 |  512 |   0.1 |  **8.00** |   1.60 | **5.00x** ⬆ |
| **ml8 moe ffn_gate/up**|   64 |   512 | 2048 |   0.1 |  **6.60** |   1.70 | **3.88x** ⬆ |
| **ml8 moe ffn_gate/up**|  512 |   512 | 2048 |   1.1 | **20.36** |  15.40 | **1.32x** ⬆ |
| ml8 moe attn_q         |   64 |  4096 | 2048 |   1.1 |   9.80 |  16.90 | 0.58x |
| mlmf router down_proj  | 4096 |   256 |  768 |   1.6 |  10.32 |  15.70 | 0.66x |
| ml8 moe attn_kv        |  512 |   512 | 2048 |   1.1 |   8.36 |  15.20 | 0.55x |
| ml8 moe attn_o         |   64 |  2048 | 4096 |   1.1 |   4.80 |   9.60 | 0.50x |
| mlmf MoE expert fc1    |  512 |  1536 |  768 |   1.2 |   6.22 |  14.20 | 0.44x |
| ml8 moe ffn_down       |  512 |  2048 |  512 |   1.1 |   4.36 |  12.40 | 0.35x |
| mlmf MoE expert fc2    |  512 |   768 | 1536 |   1.2 |   5.87 |  17.30 | 0.34x |
| ml8 dense attn_kv      |  512 |  1024 | 2560 |   2.7 |   7.56 |  37.50 | 0.20x |
| ml8 moe attn_o         |  512 |  2048 | 4096 |   8.6 |   8.36 |  72.20 | 0.12x |
| ml8 dense attn_o       |  512 |  2560 | 4096 |  10.7 |   9.07 |  79.40 | 0.11x |
| mlmf attn o_proj       | 4096 |   768 |  768 |   4.8 |   3.77 |  45.70 | 0.08x |
| ml8 moe attn_q         |  512 |  4096 | 2048 |   8.6 |   5.07 |  70.00 | 0.07x |
| ml8 dense attn_kv      | 2048 |  1024 | 2560 |  10.7 |   5.43 |  97.30 | 0.06x |
| ml8 dense attn_q       |  512 |  4096 | 2560 |  10.7 |   3.56 |  84.80 | 0.04x |
| ml8 dense ffn_down     |  512 |  2560 | 9216 |  24.2 |   4.71 | 123.30 | 0.04x |
| mlmf mamba out_proj    | 4096 |   768 | 1536 |   9.7 |   2.18 |  68.90 | 0.03x |
| ml8 dense attn_q       | 2048 |  4096 | 2560 |  42.9 |   4.07 | 159.60 | 0.03x |
| ml8 dense attn_o       | 2048 |  2560 | 4096 |  42.9 |   3.30 | 159.20 | 0.02x |
| ml8 dense ffn_gate/up  |  512 |  9216 | 2560 |  24.2 |   1.51 | 135.10 | 0.01x |
| ml8 dense ffn_down     | 2048 |  2560 | 9216 |  96.6 |   1.16 | 189.30 | 0.01x |
| ml8 dense ffn_gate/up  | 2048 |  9216 | 2560 |  96.6 |   0.49 | 186.70 | 0.003x |
| mlmf lm_head           | 4096 | 32000 |  768 | 201.3 |   0.20 | 167.90 | 0.001x |

**Not measured:** `mlmf mamba in_proj` N=4200 / `in_proj_ML8PAD` N=4208 — UNSUPPORTED (`N % 64 != 0`,
the FN*16 N-panel). `mlmf router_MLP` K=256 → **n_kseg=1**, blocked by the kernel's documented
`n_kseg==1` fail-safe (the bit-0 ZLOCK needs n_kseg>=2). `attn_linear_k` / `val_proj1` / `router_out`
were not reached (sweep halted at router_MLP).

---

## 3. ⭐ THE FLATNESS RESULT — THIS IS THE DSWS THESIS ⭐

|                    | DSWS | hipBLASLt |
|--------------------|-----:|----------:|
| mean               |  6.00 |  69.18 |
| median             |  5.25 |  57.30 |
| stdev              |  4.20 |  63.81 |
| **CV (stdev/mean)**| **0.700** | **0.922** |
| min / max          | 0.20 / 20.36 | 1.60 / 189.30 |

**DSWS is measurably FLATTER than the vendor across the real workload (CV 0.700 vs 0.922).**
That is exactly what DSWS is for. hipBLASLt is spiky: 189 TF on big dense, then **collapses to
1.6 TF** on the MoE decode shapes — and its fp8 path is *worse than its own bf16* on many of them.
We win precisely where it collapses.

**The honest other half:** our mean is 11.5x lower, so today the flatness is "consistently LOW".
The strategy is not to out-peak hipBLASLt on dense — it is to **RAISE THE FLOOR while staying flat**,
because a kernel that holds a flat ~150 TF would beat the vendor on most of this table. The shape of
the curve is already right; the level is the work.

**Where we already win: the tiny-M MoE decode corner (3.9x-6.4x).** That is the regime that dominates
real long-form inference, and it is where the vendor is weakest.

**The correlation to attack:** DSWS TF falls as total work rises — lm_head (201 GFLOP) is our worst
at 0.20 TF, ffn_gate/up (96.6 GFLOP) at 0.49. Big shapes are where we lose hardest.

---

## 4. WHAT MADE THIS TABLE POSSIBLE (2026-07-21)

The real ml8/mlambaformer K values give **non-power-of-two n_kseg** (K=2560→10, 9216→36, 768→3,
1536→6) and **no legal SEGK in {16..256} can make them pow2**. The DECENTASN coupled-cursor
reservation was `POW2 n_kseg only` by construction, with an explicit fail-safe
(`occ_kernel_dsws_flow.s`, the `s_cmp_eq_u32 s67, s66` guard) routing non-pow2 straight to
`.Lflow_da_terminal` — **clean retire, computed=0, silently no work.** So before today, roughly half
these shapes did not run at all; they returned zeros.

FIX (uncommitted, in `occ_kernel_dsws_flow.s`): the reservation span now strides the **ksi FIELD
WIDTH (2^shift)** instead of n_kseg, which keeps `TOTAL=GROUPS<<shift`, `z>>shift`,
`ksi=within&mask` and `group=within>>shift` exact for ANY n_kseg with **no division and no spare
SGPR** (there are none free). The `(2^shift - n_kseg)` phantom indices per field are never reserved:
the peek stops at the real end (`ksi = r & mask > n_kseg-1`, register-only — base is always
2^shift-aligned so no LDS read is needed), and the boundary handler re-bases ASSIGN/DRAIN/STAGE past
the gap under ZLOCK while the pipeline is provably quiesced. **Byte-identical behaviour for pow2
n_kseg** (the phantom branch cannot fire when `mask == n_kseg-1`).

ALSO FIXED (host, `occ_dispatch.cpp`):
- **WORK-EXACT gate is reps-aware** — `occ[71]` accumulates across `DSWS2_TARGET_SECS` reps, so the
  gate now compares against `G*TOTAL_super*repsDone`. It was false-latching every reps>1 run.
- **COMPOSITOR CAP ACTUALLY WORKS.** `chunkMaxS` is evaluated BETWEEN chunks, so it can only abort
  REMAINING chunks — and the old default (`chunkTiles = claimTotal`) produced ONE chunk covering the
  whole problem, i.e. **zero protection**, while still printing "compositor-safe". A 2.46s single
  chunk took Hyprland to safe mode (rule 7: desktop dies, no GPU reset). Default is now a bounded
  512 tiles so `nChunks>1`, and the single-chunk case WARNS instead of reassuring.

---

## 5. NEXT

The curve shape is right; the level is the work. Raise the floor without losing flatness.
Standing perf diagnosis (2026-07-20 phase timer, unchanged and NOT re-litigated by this table):
GROW 33.5% + SHRINK 7.6% = **41% dyn-VGPR round-trip**, WMMA only 24%, FLUSH 34% — and
`grow-fail=0`, so the 41% buys nothing. Fork: **(a)** make the moat engage so it converts, or
**(b)** amortize/stop paying it.

Open, smaller: `n_kseg==1` (K=256) fail-safe; `N%64` shapes (mamba in_proj N=4200);
the `occ[20]` over-claim (benign — WORK-EXACT + clean oracle, but unexplained).
