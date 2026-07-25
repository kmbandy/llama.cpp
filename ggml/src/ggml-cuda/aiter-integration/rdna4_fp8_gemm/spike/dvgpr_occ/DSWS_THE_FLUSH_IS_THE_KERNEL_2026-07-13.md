# DSWS — THE FLUSH IS THE KERNEL (2026-07-13, evening + night)

**START AT `DSWS_MORNING_2026-07-14.md`** — it has every lever, its valid range, and the current
value. This doc is the *story* (and all the retractions). `DSWS_POOL_UNLOCK_2026-07-13.md` is
**superseded** and banner-stamped.

**FINAL: 0.4 -> 36.9 TF, bit-exact, in one day (92x).** The TL;DR table below stops at 32.0
(K-depth J). The last 32.0 -> 36.9 came from **unlocking `G`** — see the addendum at the end.

---

## TL;DR  —  0.4 -> 32.0 TF, BIT-EXACT, IN ONE DAY (80x)

**The C flush WAS 97.4% of the runtime. It is now ZERO. `K-DEPTH J` killed it outright.**

| | TF | note |
|---|---|---|
| this morning (WOFLUSH atomics) | **0.4** | 1.21 BILLION `global_atomic_add_f32` |
| banked LDS reduce | 2.5 | 6.7x — the reduce I had deleted |
| + tile-for-K-depth trade (G=4 SEGK=128) | 6.3 | |
| + clock-committed measurement | 8.8 | (5.3 vs 8.8 was DVFS, not architecture — see §0.1) |
| **+ K-DEPTH J (J=512, W=24)** | **32.0** | **flush ablation now moves TF by 0.0** |

**Every row bit-exact (`bad=0 max_rel=0`).**

### The config (32.0 TF, bit-exact)

```bash
WAVES=24 G=4 SEGK=64 ACC_N=4 POOL_N=2 JDEPTH=512 WOFLUSH=0 BANKZERO=1 \
  STAGINSTR=1 PHASEPROBE=0 DEADMAN=1 TFPROBE=1 DIAG=0 ./build_flow.sh

ML8_YIELD_MS=5 ML8_YIELD_EVERY_MS=50 DSWS2_FLOW=1 DSWS2_SEGK=64 DSWS2_ACC_N=4 DSWS2_G=4 \
FLOW_POOL_N=2 FLOW_WAVES=24 DSWS2_ORACLE_MTL=256 DSWS2_ORACLE_NTL=512 DSWS2_NKSEG=512 \
DSWS2_ORACLE_STRIDE=65536 ML8_POOL=64 ML8_COOP_CHUNK=8192 ML8_COOP_STREAM=1 \
ML8_COOP_CHUNK_MAXS=3.0 timeout 900 ./occ_dispatch --dsws2
```

---

## THE NEW WALL: THE TILE IS TOO SMALL (memory-bound on the operand feed)

The flush is gone; ablating it now changes TF by **nothing** (32.1 with, 32.1 without).
The roofline arithmetic says exactly what is left:

- Super-tile is **128 x 64**. `AI = 2*TM*TN/(TM+TN) = 85 FLOP/byte`.
- **Roofline at that tile = 85 * 644.6 GB/s = 54.8 TF. We are at 32.0 = 58% of it.**
- Measured: 67.1M super-tiles x 12,288 B = **824 GB of operands in 2.2 s = 375 GB/s**,
  i.e. **58% of the card's 644.6 GB/s.** We are genuinely DRAM-bound now.
- The 93.6% coast-frac is waves waiting on **memory**, not waves being idle.

### ...and J UNLOCKED THE FIX FOR FREE

**At `J = n_kseg`, SPLIT-K IS GONE**: one wave computes ALL of K for its rowblk, so each
accumulator bank receives **exactly one contribution**. **The LDS banks are dead weight.**
They are also the ONLY reason a bigger tile does not fit (`banks = ACC_N * FM*FN*1024`
explodes past 64KB the moment FN grows).

| tile | AI | roofline |
|---|---|---|
| 128 x 64 (today) | 85 | **54.8 TF** |
| 128 x 128 | 128 | 82.5 TF |
| 256 x 128 | 170 | **110 TF** |
| 256 x 256 | 256 | **165 TF** |

**THE CHAIN: J -> no split-K -> no banks -> big tile -> 110-165 TF roofline.**
That is kmbandy's original "pivot to tile size" plan, and J is what unblocked it.

---

## K-DEPTH J — WHAT IT IS AND THE BUG THAT ALMOST HID IT

A wave holds ACC **in registers across J consecutive ksi of the SAME rowblk** and flushes
**ONCE**. J-fold fewer `ds_add_f32`, **ZERO extra LDS** -- it walks the same POOL_N slot
buffers sequentially as they re-stage, instead of demanding J resident.

> **J is strictly better than SEGK.** Both raise K-per-flush, but `OPSTRIDE = SEGK*16*(FN+G*FM)`
> makes SEGK cost LDS *linearly* (which is why SEGK=256 is unreachable). **J costs nothing.**

### Measured (G=4 SEGK=64 ACC_N=4 POOL_N=2, K=32768, n_kseg=512) — ALL BIT-EXACT

| J | W=8 | W=16 | W=24 |
|---|---|---|---|
| 1 | 8.8 | | |
| 8 | 15.6 | 20.1 | 15.3 |
| 16 | 18.2 | 24.7 | 24.8 |
| 32 | 19.8 | 27.9 | 28.1 |
| 64 | 20.6 | 29.8 | 30.1 |
| 128 | | 30.9 | 31.1 |
| 256 | | 31.4 | 31.8 |
| **512** (= n_kseg) | | 31.8 | **32.0** |

**W BECAME A LEVER *BECAUSE* OF J** (W=8 -> W=24 is +45% at J=64). At high J the carriers sit
**FAT, waiting** in `.Lflow_jwait` for their next segment; more lean waves stage it faster.
This coupling did not exist before J.

### *** THE INVARIANT: DRAIN MUST NEVER PASS AN UNFLUSHED SEGMENT ***

First J build measured **`bad=64` — every fragment wrong, at J=2/4/8.** Cause: I retired each
slot (RBDONE++ -> DRAIN++) right after its WMMA, **while the sum was still in registers.** At a
tile's end that fires `DRAIN == ASSIGN` with sums unflushed -> the coordinator calls
`zero_banks` and opens the NEXT tile -> the carriers then flush into the **freshly-zeroed banks
of the wrong tile.** Every tile lost its last group AND poisoned its successor.

**FIX:** mid-group slots retire early (this is what keeps J's LDS cost at zero); the group's
**LAST** slot is settled by the shared **post-flush** path, exactly as J=1 always did.

### Ownership model (the other subtle part)

The carrier holds rowblk `r` across slots **without re-claiming it**, so a *fresh* wave must not
claim `r` again at a mid-group slot. The coordinator **POISONS `SL_RBNEXT = ACC_N` on every
non-lead slot** (`ksi % J != 0`) and the existing `r >= ACC_N -> tryadv` check turns fresh waves
away for free. A lead-gate before the grow stops them burning a grow/shrink pair on the
`(J-1)/J` of slots they cannot claim.

**Guard:** `WAVES >= 2*ACC_N` (`.error` otherwise) — ACC_N carriers sit FAT waiting, so at least
as many waves must stay LEAN to stage for them, or the carriers deadlock the feeder.

---

## 0. *** THE MEASUREMENT RULE — READ THIS FIRST OR YOU WILL DRAW FALSE CONCLUSIONS ***

### 0.1 NEVER trust a TF number from a run shorter than ~1 SECOND. The clock is not committed.

The card is `perf_level=manual`: it idles at ~1147 MHz and only boosts to ~2350 MHz under
sustained load. **The boost threshold is right around 0.5 s.** Measured, holding the kernel
and shape geometry fixed and varying ONLY the duration:

| duration | TF |
|---|---|
| 0.03 - 0.41 s | **5.2 - 5.3** (every run, rock stable) |
| 0.50 s - 8.0 s | **8.8** (every run, rock stable) |

It is **bimodal**, and it is the *clock*, not the kernel. Proven by crossing the factors:

- LONG run at SMALL N (2.0 s, N=4096) -> **8.8**
- SHORT run at BIG N (0.41 s, N=32768) -> **5.3**

So N, M, tiles/WG and total work are all IRRELEVANT to this jump. Only duration matters.

> **This artifact produced a FALSE "the kernel grows with shape size" conclusion tonight,
> which I nearly shipped.** It also silently suppressed every real-ml8-shape number by ~1.65x.

### 0.2 NEVER trust an ARCHITECTURE conclusion from an under-fed run.

Every measurement before ~19:00 ran in **33 ms**: 3 chunks x 64 tiles over 64 workgroups =
**one tile per workgroup.** The economy never reached steady state. I was profiling spin-up
and drain, and drawing architectural conclusions from it. Feeding it inverted the diagnosis:

| run length | "assign-starved" | verdict |
|---|---|---|
| 0.03 s | 76.1% | ASSIGN-BOUND |
| 6.64 s | **1.8%** | STAGE-BOUND |

**Rule: feed it to >=1 s of steady state BEFORE forming any verdict. If a test finishes
instantly, that is a bug in the test, not a result.** (kmbandy has now been right about this
three separate times — see the KG "feed it" feedback entry.)

---

## 1. WHAT IS TRUE (measured, bit-exact unless noted)

### 1.1 The flush dominates, and it is the COUNT, not contention

- `NOCFLUSH=1`: **664.4M -> 17.0M ticks (39x).** Flush = 97.4% of runtime under WOFLUSH.
- `KMAJOR=1` (spreads `ksi` across C cells to cut atomic contention): **665.1M ticks — ZERO
  change.** Not contention. The raw op count.
- Atomics = `computed x 64` = `(tiles x n_kseg x ACC_N) x 64`. At n_kseg=1024 that is
  **1.21 BILLION `global_atomic_add_f32`** — 64 atomics per 16 WMMAs.

### 1.2 flush/WMMA = 128 / SEGK  <- the master equation

```
flush/WMMA = 8/KSEG_STEPS = 8/(SEGK/16) = 128/SEGK
```

**`FM*FN` CANCELS.** Growing the tile does NOT improve the ratio. **`SEGK` (K-depth per
super-tile) is the ONLY knob on flush:compute**, because `n_kseg = K/SEGK`.

### 1.3 ...and TILE SIZE is what BUYS SEGK (kmbandy called this)

```
OPSTRIDE = SEGK * 16 * (FN + G*FM)              cap 32768B (group-segment .error)
LDS      = OP_BASE + POOL_N*OPSTRIDE + ACC_N*8192   cap 65536B
```
`SEGK=128` at G=6 needs 32,768B and **overruns the 32KB cap by 512 bytes.** Drop G 6->4 and
it fits. **Tile size buys K-depth; K-depth is the only currency that pays down the flush.**

### 1.4 Geometry sweep (banked, POOL_N=1, GROUPS=1, K=32768, all bit-exact)

| G | SEGK | n_kseg | span | TF |
|---|---|---|---|---|
| 6 | 32 | 1024 | 98.8M | 2.5 |
| 4 | 64 | 512 | 63.7M | 3.9 |
| 4 | 128 | 256 | 39.2M | 6.3 |
| 3 | 128 | 256 | 45.7M | 5.4 |

**G=3 REGRESSES** — a genuine optimum, not "smaller is better." (All of these are *short*
runs, hence clock-suppressed; the RANKING holds, the absolute values are ~0.6x true.)
`SEGK=256` is **unreachable** (needs ~49KB operand pool; cannot coexist with the banks).

### 1.5 THE POOL WORKS. (RETRACTION — see 3.1)

Banked, all bit-exact, W=16:

| SEGK | POOL_N | coast-frac | TF |
|---|---|---|---|
| 128 | 1 | 96.9% | 6.2 |
| 64 | 2 | 92.3% | 4.2 |
| 32 | 4 | **87.3%** | 2.5 |

Deeper pool -> **less coasting**, monotonically. The waves DO get fed. TF falls only because
pool depth is paid for with K-depth out of the same LDS budget. **Pipelining works; it is a
BUDGET problem, not a design failure.**

### 1.6 WAVE COUNT IS IRRELEVANT once fed. Hardware cap = 32.

At the 32768^3 cube (70 TFLOP, 8 s):

| waves | 8 | 16 | 24 | 30 |
|---|---|---|---|---|
| TF | 8.8 | 8.8 | 8.7 | 8.7 |

**Dead flat.** Every earlier "more waves hurts" result (4.8 -> 3.5 at 1x work) was a
**starvation artifact**, and so was every wave-count "optimum" derived from it.

**A workgroup maxes at 1024 work-items = 32 waves at wave32. There is no going higher.**
We run 30 because the kernel squats `ROLE[30]/[31]` for coordinator state (`.error` at >30).
`grow-fail` first appears at W=30 (~1.5M) and does **not** help TF — VGPR pressure is not a
useful self-balancer.

### 1.7 LDS caps us at ONE workgroup per CU

At 57,856B / WG against 64KB of LDS per CU, **no second workgroup can co-reside.** A stalled
WG has nothing to switch to. Getting under **32,768B** would allow 2 WGs/CU. The accumulator
banks (`ACC_N*8192` = 32KB at G=4) are what blocks it. **UNTESTED — this is the co-residency
lever, and it is the most plausible reason coast-frac never drops below ~71%.**

---

## 2. WHERE WE ACTUALLY STAND vs hipBLASLt (both measured tonight, same card)

| ml8 MoE shape (M=512) | **DSWS** | hipBLASLt fp8 | hipBLASLt % of ITS OWN roofline |
|---|---|---|---|
| ffn_gate/up (K2048 N512) | 0.6 | **14.8** | 5.6% |
| ffn_down (K512 N2048) | 0.7 | **12.6** | 6.2% |
| attn_q (K2048 N4096) | 1.8 | **70.6** | 23.0% |
| attn_kv (K2048 N512) | 0.5 | **15.2** | 5.7% |
| attn_o (K4096 N2048) | 1.9 | **60.7** | 19.8% |

All DSWS runs bit-exact. DSWS runs are 1.6-14 ms => **clock-suppressed** (~1.65x low), which
does not change the verdict.

**Stated plainly: our clock-committed PEAK is 8.8 TF. hipBLASLt's WORST fp8 number on any
real ml8 shape is 12.6. We are below their floor.** We are 8x-39x behind.

**But the bar is soft, and that is the opportunity:** hipBLASLt hits only **5.6%-23% of its
own roofline** on these shapes, and its fp8 path **LOSES to bf16** on ffn_gate/up and
ffn_down (0.93x). The 230 TF figure is a 4096^3 square nobody runs.

### 2.1 SHAPES THE KERNEL STILL CANNOT RUN AT ALL

- **K must be a power-of-2 multiple of SEGK** -> excludes ml8 DENSE (K=2560, K=9216) and
  BOTH mlambaformer MoE experts (K=768/1536 — 56% of that model's GEMM time).
- **M must be a multiple of G*FM*16** (=128 at G=4) -> excludes the M=64 decode regime.
- Fix: magic-div in `DECODE_STI` (2 SALU -> 3 SALU = **+0.06 instr/WMMA**, essentially free;
  the machinery already exists in the `KMAJOR` path).

---

## 3. RETRACTIONS — things I asserted today that MEASUREMENT later killed

### 3.1 "POOL_N is not a throughput lever" — WRONG, RETRACTED

I measured the POOL_N sweep **under WOFLUSH, where the flush was 97% of the clock.** A knob
that controls pipelining CANNOT be detected when 97% of time is spent elsewhere. That sweep
could not have found a pool effect if it were enormous. I then carried "POOL_N does nothing"
forward as settled fact and ran POOL_N=1 for hours on the strength of it.
**See 1.5: the pool works.**

### 3.2 "acc_base_of is missing a slot term — that's the bug" — WRONG, RETRACTED

`acc_base_of` has **no slot term ON PURPOSE.** A bank must accumulate **all n_kseg segments
of a rowblk**, and the coordinator emits a tile's segments consecutively — so several
in-flight slots holding *different ksi of the same tile* all `ds_add_f32` into the same bank.
**That IS the split-K sum. Correct by construction.** The old `bad=96/116` at POOL_N=2/3 was
the *completer race*, already fixed by the tile-scoped `TILEDONE` completer (`BANKZERO=1`).
Nobody had retested it. It works.

### 3.3 "The kernel grows with shape size" — WRONG, RETRACTED. It was the CLOCK. (See §0.1)

### 3.4 "grow-fail is a self-stagger that recovers TF at W=30" — WRONG, killed by repetition

A single run showed a W=24 dip / W=30 recovery. **3 reps each: clean monotonic decline, no
recovery.** The dip was a glitch (an identical outlier appeared at W=28 rep 1).
**Repeat before believing a non-monotonic result.**

---

## 4. DEAD — DO NOT REBUILD (each killed by measurement, not argument)

| idea | why it's dead |
|---|---|
| **THE STAGGER** | `grow-fail = 0` on a clean build. No VGPR contention exists to phase-offset. The `1588` that motivated it was a `PHASEPROBE=1` artifact (that probe slows the machine **~44x**). And it exists to make MORE waves resident — wave count is **irrelevant** (1.6). R0 fails. |
| **WOFLUSH=1** | **My mistake.** Deletes the LDS reduction for 1.21B global atomics. **6.7x SLOWER.** |
| **KMAJOR** | zero change. It's the count, not the placement. |
| **SLEEPN back-off** | 2 -> 128, zero change. |
| **Coordinator / ASSIGN starvation** | cold-start artifact (76% -> 1.8% once fed). |
| **SEGK=256** | does not fit LDS alongside the banks at any useful G. |

**`FLUSH = 0.2%` from the phase profiler was the most confidently wrong number I produced
today** — measured on the 33 ms toy. `PHASEPROBE` percentages from short runs are worthless.

---

## 5. NEXT (in order)

### 5.1 TRUE K-DEPTH `J` — the whole ballgame

A wave holds ACC **in registers across `J` consecutive `ksi` of the SAME rowblk**, flushes
**once**. Cuts `ds_add_f32` by exactly `J×` **with no extra LDS**. The pool supplies the `J`
staged super-tiles. The flush is still **~57% of the clock** at the best config, so this is
the only thing standing between us and hipBLASLt's MoE floor.

Kernel line 128 already names it (*"K-depth J + KMAJOR"*). Never built. Gate behind a
`JDEPTH` defsym so `J=1` stays byte-identical to the known-good 8.8 TF config.

**NOTE: until J lands, the adaptive economy is UNTESTED, not validated.** It is 2.6% of the
clock — we literally cannot see whether it is good or bad behind the flush.

### 5.2 LDS < 32KB -> 2 workgroups per CU (§1.7). Untested co-residency lever.

### 5.3 Magic-div K (§2.1) — unblocks ml8 dense + both mlambaformer MoE experts.

### 5.4 Free win, unclaimed: mlambaformer `mamba in_proj` N=4200 -> pad to 4208 -> fp8 runs
at 107.3 TF (1.52x) for a one-line change = 5.2% of that model's total GEMM time.

---

## 6. INSTRUMENTATION THAT LIES

**"Zeros that were never measurements" bit me FOUR times today.** Before trusting a counter,
`grep` for its **CALL SITE**, not its definition:

| knob | trap |
|---|---|
| `STINSTR_FEED` | zero call sites. `feed-stages=0` was never a measurement. |
| `NOCFLUSH` | **defined, never referenced.** **NOW WIRED** — produced the 39x ablation. |
| `CSTORE` | **still dead.** Definition only. |
| `DIAG` | `DIAG=1` and `DIAG=0` produce **BYTE-IDENTICAL BINS.** The `FRONTIER ASSIGN/STAGE/DRAIN` counters are gated on **`FORENSICS`** (which triggers the RDNA4 register-file corruption), not `DIAG`. They have always been dark. |

- **`PHASEPROBE=1` slows the kernel ~44x** (151.5M vs 3.4M ticks) and starves it. Its banner
  claims "zero memory perturbation." That is FALSE.
- **The `[dsws2 STARVATION]` percent I added has a bad denominator** (`feedMT` counts
  feed-path entries from any wave; `coast` counts only compute-wave coasts) — it printed
  **196.2%**. Direction is fine; the number is not. **FIX THE DENOMINATOR.**

---

## 7. OPEN BUGS

- **`GROUPS > 1` IS NUMERICALLY BROKEN.** `ok=24 bad=24 max_rel=1` at G=6/SEGK=64/ACC_N=3.
  Everything shipped today is `GROUPS=1` (`ACC_N == G`). Unfixed.
- The `STARVATION` denominator (above).

---

## 8. TREE STATE (all uncommitted)

| file | change |
|---|---|
| `occ_kernel_dsws_flow.s` | `PH_WMMA`/`PH_FLUSH` stamps hoisted so they exist under WOFLUSH; `PH_SHRINK` on the tryadv path; **`OP_BASE` 256->512** + unconditional `.error` guard; **`NOCFLUSH` wired**. |
| `occ_dispatch.cpp` | **guard-page fix (`IsaMapBytes`) — commit-worthy alone, zero bricks in ~150 dispatches**; `kOpBase=512` + `static_assert`; fatal LDS>64KB check (replaces the old *silent never-launch*); geometry allow-list widened to `G in [2,6] x SEGK in {32,64,128,256}` on the flow path; `[dsws2 STARVATION]` print. |
| `build_flow.sh` | `KMAJOR` defsym; `rm -f`s its bin on build failure. |

**`OP_BASE` IS A HOST/KERNEL CO-CHANGE.** `OP_BASE` (kernel) must equal `kOpBase` (host) or
the host under-allocates LDS and **the workgroup SILENTLY NEVER LAUNCHES** (all counters 0 —
looks like a hang, is a dispatch that could not fit). Both sides now guard.

---

## 9. HARDWARE FACTS

- **RDNA4 dyn-VGPR hazard (undocumented):** a VALU VGPR write adjacent to `s_alloc_vgpr`
  corrupts the register file. Not in the ISA wait-state tables. Workaround: `FORENSICS=0`.
  Any gauge near a resize must be **pure SALU**.
- **SQC instruction-prefetch guard page:** exactly-page-rounded ISA mapping -> fault ->
  MODE1 reset -> brick. Fixed by `IsaMapBytes()`.
- **Clock:** `perf_level=manual`, idles ~1147 MHz, boosts ~2350 MHz, threshold ~0.5 s (§0.1).
- **Workgroup cap: 1024 work-items = 32 waves @ wave32.**
- Machine balance: 307 TF / 644.6 GB/s = **476 FLOP/byte**.


---

# ADDENDUM (late night): `G` WAS THE LAST LEVER, AND THE STAGGER WAS NOT

## 32.0 -> 36.9 TF: unlock `G`

**`G` (== `ACC_N`) is the number of waves that can EVER compute concurrently** — `SL_RBNEXT`
hands out rowblks `0..ACC_N-1`, so at `G=4` **only 4 of 24 waves per WG could ever run a WMMA.
83% of the fleet was GEOMETRICALLY FORBIDDEN from computing.** THAT is what the 93.6% coast-frac
actually was. Not memory. Not laziness. Geometry.

**`G` was capped at 4 by the LDS accumulator banks** (`ACC_N*8192` = 64KB at G=8). **K-depth J
made the banks unnecessary** (WOFLUSH becomes cheap once J cuts the atomic count J-fold), which
freed `G`:

| G | W | LDS | TF | grow-fail |
|---|---|---|---|---|
| 4 | 16 | 12,800 | 16.3 | 10 |
| 8 | 24 | 20,992 | 20.3 | 0 |
| 12 | 24 | 29,184 | 25.8 | 0 |
| **15** | **30** | 35,328 | **36.9** | **21,906** |

**G is NOT exhausted.** G=15 is just where `WAVES >= 2*ACC_N` ran out of waves at the 30-wave cap.

## The stagger: built, correct, worth ~0.3%

`STAGGER=1 MAXFAT=15` took **`grow-fail` 162,599 -> 66**, bit-exact. **TF 31.6 -> 31.7.**
The refused `s_alloc_vgpr`s cost essentially nothing — the waves eating them were idle anyway.

**Why the traveling-peak premise does not apply:** waves **launch LEAN (32 VGPRs)**, so we are
already at **~94% of wave-slot occupancy** before any staggering. Phase-offsetting peaks cannot
admit waves that are **already resident**. Staggering does not create registers — and grow/shrink
already time-multiplexes them.

### THE RULE (proven): `MAXFAT < ACC_N` REQUIRES `JDEPTH <= POOL_N`

J=1 and J=2 are CLEAN at MAXFAT=8; **J=64 is BAD.** A carrier walking J segments needs slots
`lead..lead+J-1` staged; `ASSIGN <= DRAIN + POOL_N`; and **DRAIN cannot pass `lead` until
`RBDONE == ACC_N`** — which needs all ACC_N rowblks claimed, which needs ACC_N fat tokens. With
`MAXFAT < ACC_N` and `J > POOL_N` the carriers stall **holding their tokens** -> deadlock ->
deadman -> unflushed ACC -> garbage.

**=> THE TRAVELING PEAK AND FREE K-DEPTH ARE MUTUALLY EXCLUSIVE.** `J <= POOL_N` means the pool
must hold J slots resident — exactly the LDS cost that made J free. So in this architecture
**concurrent-fat == ACC_N == G. The stagger knob and the G knob ARE THE SAME KNOB.**

## Two bugs found tonight

- **`FATTOK` shipped UNINITIALISED.** Signature: TF pinned to the *same* value (27.9) at
  MAXFAT=4/6/8/10/12 and a BAD oracle. **An uninitialised counter does not look like a broken
  cap; it looks like a cap that ISN'T THERE.** (kmbandy caught it: "capped at 27.9... seems like
  something's holding everything back.")
- **`CNT_FATFULL` (occ[87]) WRAPS** — read 3.1B / 436M / 3.6B non-monotonically. u32, billions of
  spin iterations. **Do not read it.**
