# DSWS — MORNING PICKUP, 2026-07-14

> **⚠️ FRAMING CORRECTION (2026-07-15): 36.9 TF is NOT a "good"/"best" result and this config is NOT a target.**
> It was measured on a synthetic ~square shape with `G=15 ACC_N=15 JDEPTH=1024` — a config that DEFEATS the
> DSWS adaptive purpose (G==ACC_N==max concurrent compute waves, so this is a static-parallelism setting; big
> JDEPTH hides the flush on a shape big enough to hide everything). **The target is to BEAT THE FLOOR on the
> real ml8/mlambaformer shapes** (hipBLASLt there is 12.6–70.6 TF, many <30; a kernel merely holding ~150 TF
> FLAT wins on 8/11 measurable ml8 shapes). Read "CURRENT BEST" below as "highest synthetic-shape TF so far",
> not as a goal. See KG b700609a for the authoritative framing.

**Where we ended:** `0.4 -> 36.9 TF on a SYNTHETIC shape, bit-exact` (correctness milestone, NOT a perf target — see banner).
**Read with:** `DSWS_THE_FLUSH_IS_THE_KERNEL_2026-07-13.md` (the full story + all retractions).
**Ignore:** `DSWS_POOL_UNLOCK_2026-07-13.md` (superseded, banner-stamped).

---

## 0. THE TWO MEASUREMENT RULES — VIOLATE THESE AND EVERY NUMBER IS A LIE

1. **NO TF NUMBER FROM A RUN SHORTER THAN ~1 SECOND.** The card is `perf_level=manual`:
   idles ~1147 MHz, boosts ~2350 MHz, **threshold ~0.5 s**. Measured bimodal: every run
   <0.41 s reads 5.3; every run >=0.50 s reads 8.8. **Same kernel, same shape.** This
   produced a false "the kernel grows with shape size" conclusion I nearly shipped.
2. **NO ARCHITECTURE VERDICT FROM AN UNDER-FED RUN.** At 33 ms (1 tile/WG) the economy never
   spins up; you profile launch+drain. "ASSIGN-starved 76%" -> 1.8% purely by feeding it.

**If a test finishes instantly, that is a bug in the test, not a result.**

---

## 1. THE CURRENT BEST (bit-exact, 36.9 TF, clock-committed)

```bash
WAVES=30 G=15 SEGK=32 ACC_N=15 POOL_N=2 JDEPTH=1024 WOFLUSH=1 BANKZERO=0 \
  STAGGER=0 STAGINSTR=1 PHASEPROBE=0 DEADMAN=1 TFPROBE=1 DIAG=0 ./build_flow.sh

ML8_YIELD_MS=5 ML8_YIELD_EVERY_MS=50 DSWS2_FLOW=1 DSWS2_SEGK=32 DSWS2_ACC_N=0 DSWS2_G=15 \
FLOW_POOL_N=2 FLOW_WAVES=30 DSWS2_ORACLE_MTL=68 DSWS2_ORACLE_NTL=512 DSWS2_NKSEG=1024 \
DSWS2_ORACLE_STRIDE=65536 ML8_POOL=64 ML8_COOP_CHUNK=8192 ML8_COOP_STREAM=1 \
ML8_COOP_CHUNK_MAXS=3.0 timeout 900 ./occ_dispatch --dsws2
```

**`DSWS2_ACC_N=0` under `WOFLUSH=1`** — the host must NOT reserve bank bytes. Get this wrong
and the WG **silently never launches** (all counters 0; looks like a hang, is a dispatch that
could not fit).

---

## 2. *** EVERY LEVER: RANGE, CONSTRAINT, CURRENT VALUE ***

### 2.1 KERNEL defsyms (`./build_flow.sh`, all `-defsym`)

| lever | valid range | NOW | what it does / the constraint that binds it |
|---|---|---|---|
| **`G`** | **2..16** (host allow-list) | **15** | Rowblks per super-tile. **== the number of waves that can EVER compute concurrently.** THE BIG LEVER — 16.3 TF @ G=4 -> 36.9 @ G=15. Capped by LDS + `WAVES >= 2*ACC_N`. |
| **`ACC_N`** | **must == G** | **15** | Rowblk banks. `GROUPS = G/ACC_N`, and **`GROUPS>1` IS NUMERICALLY BROKEN** (open bug). Always set `ACC_N = G`. |
| **`SEGK`** | 32 / 64 / 128 / 256 | **32** | K per super-tile. `n_kseg = K/SEGK`. **Costs LDS linearly** (`OPSTRIDE = SEGK*16*(FN+G*FM)`, 32KB cap). **SEGK=256 is unreachable.** Since J took over the flush, **SEGK should be SMALL** — spend the LDS on G. |
| **`JDEPTH`** | **power of 2, divides n_kseg** | **1024** | K-DEPTH J. Wave holds ACC in REGISTERS across J ksi, flushes once. **ZERO LDS cost.** THE flush killer. `J = n_kseg` = whole K in registers = split-K gone. |
| **`POOL_N`** | **{1,2,3,4}** (`slot_of` .errors else) | **2** | Pipeline depth. Costs `POOL_N*OPSTRIDE` LDS. Deeper = less coasting. |
| **`WOFLUSH`** | 0 / 1 | **1** | 1 = atomic straight to C, **NO LDS banks**. 0 = banked LDS reduce. **WOFLUSH=1 only makes sense WITH a big J** (at J=1 it is 1.21B atomics = 0.4 TF). Banks are what capped G at 4. |
| **`BANKZERO`** | 0 / 1 | **0** | Pre-zero the banks (tile-scoped completer). Only meaningful when `WOFLUSH=0`. |
| **`WAVES`** | **8..30** | **30** | Waves/WG. **HARD CAP 32** (1024 work-items @ wave32); we stop at 30 (kernel squats `ROLE[30]/[31]`). **`WAVES >= 2*ACC_N` is a hard `.error` when JDEPTH>1.** |
| **`STAGGER`** | 0 / 1 | **0** | Fat-wave admission control. **MEASURED WORTH ~0.3%** — see §4. Leave OFF. |
| **`MAXFAT`** | 0 (=ACC_N) .. ACC_N | 0 | Concurrent-fat cap. **`MAXFAT < ACC_N` REQUIRES `JDEPTH <= POOL_N`** or it DEADLOCKS (§4). |
| `SLEEPN` | any | 2 | Feed-empty backoff. **Swept 2..128: ZERO effect.** Don't bother. |
| `COORD_PERIOD` | any | 64 | Coordinator cadence. Untested as a lever. |
| `NOCFLUSH` | 0 / 1 | **0** | **PERF PROBE ONLY.** 1 = delete the C flush -> oracle MUST fail. **This is the ablation that cracked the whole day.** (I had to WIRE it — it was a dead knob.) |
| `KMAJOR` | 0 / 1 | 0 | Spread ksi across C cells. **ZERO effect** — the flush was COUNT, not contention. |
| `PHASEPROBE` | 0 / 1 | **0** | **SLOWS THE KERNEL ~44x AND STARVES IT.** Its "zero perturbation" banner is FALSE. Never form a verdict from probe-on numbers. |
| `FORENSICS` | 0 / 1 | **0** | **LEAVE AT 0.** Triggers the RDNA4 dyn-VGPR register-file corruption (VALU VGPR write adjacent to `s_alloc_vgpr`). |
| `STAGINSTR` | 0 / 1 | 1 | SALU counters (coast/computed/feed/grow-fail). Pure `s_add_u32`, ~free. Keep ON. |
| `DIAG` | 0 / 1 | 0 | **DEAD KNOB — `DIAG=1` and `DIAG=0` produce BYTE-IDENTICAL BINS.** The frontier counters are gated on `FORENSICS`, not `DIAG`. |
| `CSTORE` | — | — | **DEAD KNOB.** Defined, no call site. |
| `DEADMAN` | 0 / 1 | 1 | Watchdog -> clean retire instead of a queue wedge (= anti-brick). Keep ON. |
| `TFPROBE` | 0 / 1 | 1 | In-kernel span timer. Keep ON. |
| `FM` / `FN` | hardcoded 2 / 4 | 2 / 4 | Per-wave frag tile. **ACC VGPRs = FM*FN*8 = 64; NFV = 112.** Raising FN is THE untried tile lever (§5). |

### 2.2 HOST env vars (`./occ_dispatch --dsws2`)

| var | NOW | note |
|---|---|---|
| `DSWS2_G` / `DSWS2_SEGK` | 15 / 32 | **MUST MATCH the built bin** or the host REFUSES (a geometry/bin mismatch would silently mis-address). If a run prints nothing, look for `*** REFUSE`. |
| **`DSWS2_ACC_N`** | **0** | **0 under WOFLUSH=1.** Else = ACC_N. Wrong value -> silent no-launch. |
| `FLOW_POOL_N` / `FLOW_WAVES` | 2 / 30 | must match POOL_N / WAVES. |
| `DSWS2_ORACLE_MTL` | 68 | **M = MTL * G*FM*16.** M must be a multiple of `G*FM*16` (=480 at G=15). |
| `DSWS2_ORACLE_NTL` | 512 | **N = NTL * FN*16** (=64). N must be a multiple of 64. |
| `DSWS2_NKSEG` | 1024 | **n_kseg = K/SEGK. MUST BE A POWER OF 2.** |
| `DSWS2_ORACLE_STRIDE` | 65536 | Oracle sampling. Raise it on big shapes or the CPU reference becomes the bottleneck. |
| `ML8_COOP_CHUNK` | 8192 | Tiles per dispatch. **Raise it to keep each run >= 1 s (see §0).** |
| `ML8_COOP_CHUNK_MAXS` | 3.0 | Abort a chunk over this many seconds (display-safety). |
| `ML8_POOL` | 64 | Workgroups. |
| `ML8_YIELD_MS` / `_EVERY_MS` | 5 / 50 | Compositor yield. Leave alone.

### 2.3 THE BINDING EQUATIONS — memorize these

```
OPSTRIDE  = SEGK * 16 * (FN + G*FM)                    <= 32768   (operand-pool .error)
LDS_TOTAL = OP_BASE(512) + POOL_N*OPSTRIDE
                         + (WOFLUSH ? 0 : ACC_N*FM*FN*1024)   <= 65536
LDS <= 32768  =>  2 workgroups per CU  (UNTESTED co-residency lever)

n_kseg  = K / SEGK           (power of 2)
flush   ∝ n_kseg / JDEPTH    <- J and SEGK both pay this; ONLY SEGK costs LDS
VGPR    : lean 32, fat NFV=112 (= FM*FN*8 + frags, rounded to 16)
          per-SIMD budget ~1536; waves spread over 4 SIMDs; 16 wave-slots/SIMD

M % (G*FM*16) == 0        N % (FN*16) == 0        K % (SEGK * 2^n) == 0
WAVES >= 2*ACC_N          (hard .error when JDEPTH > 1)
MAXFAT < ACC_N            requires JDEPTH <= POOL_N   (else DEADLOCK -- see §4)
```

---

## 3. WHAT IS DEAD — DO NOT REBUILD (each killed by a measurement)

| idea | verdict |
|---|---|
| **WOFLUSH=1 at J=1** | 1.21 BILLION atomics = **97.4% of runtime**. Only viable WITH big J. |
| **KMAJOR** | zero change. The flush was COUNT, not contention. |
| **SLEEPN backoff** | 2 -> 128, zero change. |
| **Coordinator / ASSIGN starvation** | cold-start artifact (76% -> 1.8% once fed). |
| **SEGK=256** | does not fit LDS at any useful G. |
| **"POOL_N is not a lever"** | **RETRACTED** — I measured it under a 97% flush mask. The pool works (coast 96.9% -> 71.3%). |
| **"acc_base_of is missing a slot term"** | **RETRACTED** — it has no slot term ON PURPOSE (the bank accumulates ALL ksi of a tile; that IS the split-K sum). |
| **"the kernel grows with shape size"** | **RETRACTED** — it was the clock (§0). |
| **THE STAGGER** | mechanism works, **worth ~0.3%.** See §4. |

---

## 4. THE STAGGER — BUILT, CORRECT, AND WORTH ~0.3%

**It works.** `STAGGER=1 MAXFAT=15` (== ACC_N) took `grow-fail` from **162,599 -> 66**, bit-exact.
**TF went 31.6 -> 31.7.** The 162k refused `s_alloc_vgpr`s were *not* costing us anything — the
waves eating them were idle regardless.

**Why the traveling-peak premise does not apply here:** waves **launch LEAN (32 VGPRs)**, so we
are already resident at **~15 waves/SIMD of a 16-slot max (~94% occupancy)** *before* any
staggering. Phase-offsetting peaks cannot admit waves that are **already there**. Staggering
does not create registers; it time-multiplexes them — and grow/shrink already does that.

### *** THE RULE: `MAXFAT < ACC_N` REQUIRES `JDEPTH <= POOL_N` ***

Proven by measurement (J=1 and J=2 clean at MAXFAT=8; **J=64 BAD**):
a carrier walking J segments needs slots `lead..lead+J-1` staged; `ASSIGN <= DRAIN + POOL_N`;
and **DRAIN cannot pass `lead` until `RBDONE == ACC_N`**, which needs all ACC_N rowblks claimed,
which needs ACC_N fat tokens. With `MAXFAT < ACC_N` and `J > POOL_N` the carriers stall holding
their tokens -> **deadlock -> deadman -> unflushed ACC -> garbage.**

**CONSEQUENCE: the traveling peak and free K-depth are MUTUALLY EXCLUSIVE.** `J <= POOL_N` means
the pool must hold J slots resident — **exactly the LDS cost that made J free.** So in this
architecture **concurrent-fat == ACC_N == G: the stagger knob and the G knob ARE THE SAME KNOB.**

> The waves were never held back by *when* they were fat. They were held back by **not being
> allowed to compute at all** (`ACC_N = G = 4` -> only 4 of 24 waves could ever run a WMMA).
> Unlocking G was the win. kmbandy's read of the coast number was right; the fix was G, not phase.

**NOTE:** `CNT_FATFULL` (occ[87]) **WRAPS** — it read 3.1B / 436M / 3.6B non-monotonically. It is
a u32 accumulating billions of spin iterations. **Do not read it.** Fix or drop it.

---

## 5. THE OPEN QUESTION FOR THE MORNING: WHAT HOLDS 36.9 BACK?

**We do not know.** Say that out loud before theorising. What we DO know:

- **The flush is gone.** `NOCFLUSH=1` at J=512 changes TF by **0.0** (32.1 -> 32.1).
- **Wave count is saturated** (~94% of wave slots).
- **grow-fail is not the cost** (eliminating it entirely = +0.3%).
- **`G` is still climbing** — 16.3 (G=4) -> 20.3 (G=8) -> 25.8 (G=12) -> 36.9 (G=15). **We never
  found the top; G=15 is just where `WAVES >= 2*ACC_N` ran out of waves at the 30-wave cap.**

### Ranked candidates

1. **`G` IS NOT EXHAUSTED.** The `WAVES >= 2*ACC_N` guard exists because carriers sit FAT waiting
   in `.Lflow_jwait` while lean waves stage for them. **Is 2x really needed?** If `WAVES >= ACC_N + 4`
   suffices, G could go to ~26 at W=30. **Cheapest possible experiment: relax the guard, sweep.**
2. **THE TILE (`FN`).** `FN=4` -> super-tile N is only 64. `AI = 2*TM*TN/(TM+TN)`.
   `FN=8` doubles N -> AI 85 -> 128. Costs VGPRs (`ACC = FM*FN*8` = 128, NFV ~176) and LDS.
   **NOTE:** my "we are DRAM-bound at 375 GB/s" claim is **UNVERIFIED** — it assumes ZERO cache
   reuse. hipBLASLt does 230 TF on 4096^3, which *exceeds* my own roofline model, so the model is
   wrong. **MEASURE before acting on it (NOFEED ablation).**
3. **NOFEED ablation.** Same trick as NOCFLUSH: skip the operand loads, compute garbage. If NOFEED
   ~= 36.9 we are compute/issue-bound and the tile will not save us. If NOFEED is 150+, we are
   feed-bound and the tile is exactly right. **This is the one measurement that settles #2.**
4. **2 WGs/CU co-residency.** At `LDS <= 32768` two workgroups fit per CU. G=15/SEGK=32/POOL_N=2
   is **35,328 B — 2.5 KB over.** Drop POOL_N to 1 (17,920 B) and it fits. **Never tested.**

---

## 6. STILL BROKEN / STILL OWED

- **`GROUPS > 1` IS NUMERICALLY BROKEN** (`ok=24 bad=24 max_rel=1`). Everything shipped is
  `GROUPS=1` (`ACC_N == G`). **Unfixed.**
- **The kernel CANNOT RUN ml8 dense** (K=2560, 9216) **or either mlambaformer MoE expert**
  (K=768/1536 = **56% of that model's GEMM time**) — K must be a power-of-2 multiple of SEGK.
  Fix = magic-div in `DECODE_STI`: 2 SALU -> 3 SALU = **+0.06 instr/WMMA. Essentially free.**
  The machinery already exists in the `KMAJOR` path.
- **M=64 decode regime excluded** (M must be a multiple of `G*FM*16`).
- **REAL-SHAPE NUMBERS ARE OWED.** Everything above is a big synthetic. Real ml8 shapes are
  1.6-14 ms => **clock-suppressed** => do not quote them without fixing the >=1 s rule.
  Last measured (STALE, pre-J): DSWS 0.5-4.3 vs hipBLASLt 12.6-70.6.
- **`[dsws2 STARVATION]` % has a bad denominator** (printed **196.2%**). Direction fine, number not.
- **`CNT_FATFULL` wraps** (§4).

## 7. THE BAR (measured tonight, same card)

hipBLASLt fp8 on real ml8 MoE (M=512): **ffn_gate/up 14.8 | ffn_down 12.6 | attn_q 70.6 |
attn_kv 15.2 | attn_o 60.7** — at only **5.6%-23% of its OWN roofline**, and its fp8 path
**LOSES to bf16** on ffn_gate/up and ffn_down. **The 230 TF figure is a 4096^3 square nobody runs.**
The bar on real shapes is soft. We are at 36.9 on a synthetic; **real shapes not yet re-measured.**

## 8. TREE STATE — ALL UNCOMMITTED

| file | change |
|---|---|
| `occ_kernel_dsws_flow.s` | K-DEPTH `JDEPTH` + `STAGGER`/`MAXFAT`/`FATTOK`; `NOCFLUSH` wired; `OP_BASE` 256->512 + guard; PH_WMMA/PH_FLUSH/PH_SHRINK stamp fixes |
| `occ_dispatch.cpp` | **guard-page fix (`IsaMapBytes`) — COMMIT-WORTHY ALONE, zero bricks in ~200 dispatches**; `kOpBase` co-change + static_assert; fatal LDS>64KB check; geometry allow-list widened to `G in [2,16] x SEGK in {32,64,128,256}`; STARVATION + STAGGER prints |
| `build_flow.sh` | `KMAJOR` / `JDEPTH` / `STAGGER` / `MAXFAT` defsyms; `rm -f`s its bin on build failure |

**`OP_BASE` (kernel) MUST EQUAL `kOpBase` (host)** or the WG silently never launches. Both guard.
