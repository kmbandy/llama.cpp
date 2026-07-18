# DSWS strategy pickup — 2026-07-15 night

**Read this top-to-bottom on resume. It captures a full strategic conversation (kmbandy-led) that
reframed where DSWS stands and set the next direction. Backup copy of the `mneme_brief` of the same date.**

All work in `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`.

---

## 0. THE DECISION (kmbandy's call, end of night)

> **Keep the decentralized accounting (DECENTASN), but marry it to deep-J + "the next available wave
> carries the result to DRAM."**

i.e. do NOT throw away the emergent/lazy wave-role economy — but stop paying the split-K flush by holding
the accumulator deep in registers (J) and doing a single lazy carry-off, instead of WOFLUSH's per-slice
DRAM atomics. The night's exploration converged here from first principles.

**⚠️ The tension that must be resolved to do this (see §4):** the *current* DECENTASN code **requires
`WOFLUSH=1` and `JDEPTH=1`** (both are top-of-file `.error` guards). Those two flags **structurally forbid
deep-J and force the slowest flush.** So "DECENTASN + deep-J" is not a knob flip — it needs the pin/ring
protocol reworked to run on the banked path with J>1. That is the real next design problem.

---

## 1. CURRENT BUILD / TREE STATE (2026-07-15 EOD)

- **On-disk bin:** `7a4021426bfa7be36009030ede133b3b` = the **Site-J + claim-diagnostic** DECENTASN=1 build
  (FM=1 G=6 ACC_N=6 POOL_N=4 SEGK=64 WOFLUSH=1 STAGINSTR=1 TFPROBE=1). Diagnostic build, not a keeper.
- **Inertness intact:** `DECENTASN=0` → md5 `386dc28643ffb58568623ad6d89cfe62` (byte-identical baseline).
  All DECENTASN work is behind the default-off knob. Baseline is correct on all real shapes.
- **Latch:** CLEAR (`.gpu_last_hang` absent). No brick all session.
- **Git HEAD:** `6ce3be4d2`. Everything uncommitted.
- **Docs:** `DECENTASN_FIX_PLAN_2026-07-15.md` (the pin fix), `DSWS_TESTING_LOG.md` (every run logged),
  this file.
- **Tasks:** #40 (pin fix, in_progress), #41 (Site J, effectively done/refuted as the cause).

---

## 2. WHAT WE PROVED ABOUT DECENTASN TONIGHT (all by measurement)

The DECENTASN oracle goes ~95% BAD at FM=1 G=6 POOL_N=4. We ran four diagnostic builds and **refuted the
two convergent reviewer hypotheses**:

| build | key counters | verdict |
|---|---|---|
| (next,inflight) pin | bad=8712, canary occ[95]=65 | over-releases exist |
| release classifier | bucket B (same-gen pending)=823, A=C=0 | looked like producer-restamp-vs-live-pin |
| **claim-persistence diag** | **occ[95] execbad=0, occ[96] phantom=0, occ[97] rel-imbal=834** | **claims are FINE** |
| Site J (feeder decode-after-claim) | bad=8784 (UNCHANGED) | Site J refuted as the cause |

**Reviewers (sol gpt-5.6-sol + Fable, independent):** both proved the pin *cannot* over-release; both said
the seed is a phantom claim / underflow; **silicon refuted both** (occ[95]=occ[96]=0). Fable's sharp
bit-insight: a blind `-INFLIGHT_ONE` underflow *manufactures* `RB_PENDING` (`0x6+0xFFFFFF00=0xFFFFFF06`),
which explains the A=0/B≫0/C=0 signature — but the claim-diag then showed claims persist, so that seed
never fires. The residual is only a ~4% release race (occ[97]≈800), safely contained.

**THE REFRAME (the important finding):** across all runs `bad` stayed ~constant (8712–8808) while
`computed` swung 7× (4345→31008), and only ~307/384 tiles get *claimed* per rep at 99.97% coast. That is
**INCOMPLETENESS, not a per-segment race** — DECENTASN is so assign-bound it can't finish the work in the
window; most tiles hold partial/zero split-K sums → bad. The claim/pin/feeder machinery is essentially
CORRECT. The "95% wrong" is starvation.

**BIG caveat (KG-hardened rule):** "assign-bound / 99.97% coast" is *exactly* the cold-start artifact that
burned us twice before (assign-starved 76%→1.8% once fed ≥1s steady state; clock not committed below
~0.5s). Our runs are ~2s in 4 resetting chunks — **possibly under-fed**, so the "assign-bound" read is NOT
trustworthy until a long, single-chunk, fed run confirms it. Never quote TF <1s.

---

## 3. THE STRATEGIC REFRAME — WHY THE FLUSH IS THE REAL GAME (recap of the whole night)

kmbandy drove a Socratic walk that reconnected DECENTASN to the long-standing DSWS physics. The chain:

1. **The flush is the kernel.** Measured (fed, clock-committed): the C flush is ~97% of runtime under
   WOFLUSH, ~45–57% banked; the whole assign/pool/role economy is ~2.6% of the clock. DECENTASN optimizes
   the 2.6%.
2. **DECENTASN is on the wrong side of the flush:** it *requires* `WOFLUSH=1` — the eager per-slice DRAM
   atomic path, measured ~6.7× slower than the banked LDS reduction, ~1.2B `global_atomic_add_f32`.
3. **What the flush physically is:** WMMA leaves fp32 accumulators in registers; the flush writes them to C.
   Split-K makes each C element get written/summed **n_kseg times** (n_kseg = K/SEGK; K=2048,SEGK=64 → 32).
   It's **write VOLUME**, not atomic contention (measured: non-atomic slower, reorder no change).
4. **The only dials on the flush:** deeper K-slice (bigger SEGK → fewer slices, LDS-capped) OR register
   accumulation (J → a wave sums J slices before writing) OR banked LDS combine + one write-once carry.
   `flush:compute ≈ 128/SEGK`.
5. **The combine is unavoidable if you split.** n_kseg partials MUST be summed, and hardware gives exactly
   two places: **on-chip (LDS/registers → same WG only)** or **DRAM (cross-WG, expensive volume)**. There is
   **no cheap cross-wave register combine** — waves can't hand each other registers; the only relay is LDS
   (same WG) or DRAM. So "each wave streams its own slice to DRAM" = WOFLUSH = the slow path we measured.
6. **The elegant escape is NO combine:** one wave holds full-K in registers and writes C once. That's
   **deep-J at its limit** (J = n_kseg → split-K vanishes → banks are dead weight → LDS freed → bigger
   tile → higher roofline). It is *exactly* what **hipBLASLt does**: 612 gfx1201 fp8 kernels, **all GSU1,
   zero split-K, one C store**, and it's why they're 40–350× faster on real ml8 shapes.
7. **These shapes don't even need split-K.** attn_q (M=576, N=4096) at a 128×64 tile = ~320 output tiles
   for 64 CUs — 5× oversubscribed *before* splitting K. Split-K exists to manufacture parallelism when M is
   tiny; here we have plenty. So the ring/assign/flush apparatus is manufacturing parallelism we don't
   need and charging the 32× flush for it.

**Roofline the tile buys (once split-K is gone and LDS frees up):**

| tile | AI | roofline |
|---|---|---|
| 128×64 (today) | 85 | 54.8 TF |
| 128×128 | 128 | 82.5 TF |
| 256×128 | 170 | 110 TF |
| 256×256 | 256 | 165 TF |

---

## 4. HARDWARE FACTS ESTABLISHED (the constraints to design around)

- **Wave32 is fixed hardware** (RDNA4). Not derived from K/SEGK; the n_kseg=32 matching wave width is pure
  coincidence. Lanes are *spatial* (32 lanes cooperatively compute one 16×16 WMMA frag, 8 acc dwords/lane →
  `FM*FN*8`). K-slices are *temporal*, across waves. The hardware number that matters is **16** (WMMA
  16×16×16; SEGK is a multiple of 16, KSEG_STEPS=SEGK/16).
- **LDS is private per-WG.** Two co-resident WGs on one CU split the 64 KB into private halves and CANNOT
  see each other's LDS; barriers are per-WG. Co-resident WGs are as isolated as cross-CU WGs — only global
  memory connects them. **So the LDS-based lazy economy CANNOT span two WGs.** (kmbandy asked; this is why.)
- **One WG caps at 32 waves** (1024 work-items ÷ 32; we run 30). To get >32 waves/CU for latency hiding you
  MUST launch a 2nd WG, and it brings its OWN separate LDS economy (can't share the first's).
- **2 WGs/CU is feasible but untested:** needs each WG's LDS ≤ 32 KB (two fit in 64 KB). It's a launch/
  occupancy property (dispatch ~128 WGs, keep LDS small) — a real *latency-hiding* lever, not a flush lever.
- **The ring holds POOL_N super-tiles (K-slices) in flight, NOT tiles.** POOL_N≈2–4 (LDS-capped). A whole
  tile is n_kseg (32) slices — the ring holds a *fraction of one tile*; the WG streams a tile's 32 slices
  through its 2-slot window. Different WGs work different *tiles*. The global `occ[20]` atomic tile-claim is
  the only cross-WG shared state (coarse). Fine staging is per-WG LDS.
- **The round-trip is the cost when unfed:** ~100 µs frontier round-trip per work item for ~0.16 µs of math
  (~600×). POOL_N>1 pipelines it; shallow ring = no latency hiding.

---

## 5. DEEP-J — WHAT IT IS AND ITS LIMITS (already partly built)

- **J:** a wave holds ACC in registers across J consecutive ksi of the SAME rowblk and flushes ONCE. J-fold
  fewer flushes at ZERO extra LDS. Shipped once: 0.4→32.0 TF bit-exact (J=512, on a **synthetic 32K cube**).
- **Real-shape cap:** J must be a power of two AND divide n_kseg. Real ml8 K=2048/SEGK=64 → n_kseg=32 → J ∈
  {1,2,4,8,16,32}. Deep J (512) needs huge K. So on real shapes J only *partly* pays the flush; going deeper
  via bigger SEGK lengthens the VGPR peak (fights the dyn-VGPR trapezoid).
- **The invariant (learned from bad=64):** DRAIN MUST NEVER PASS AN UNFLUSHED SEGMENT (else zero_banks opens
  the next tile and carriers flush into the wrong tile). Mid-group slots retire early; the group's LAST slot
  is settled by the shared post-flush path.
- **The deadman trap (2026-07-14):** deadman must be STALL-scoped, not lifetime-scoped, or it force-retires
  fat J-carriers out of `.Lflow_jwait` and silently DROPS their unflushed ACC → short `computed` → flatters
  TF. ALWAYS check work-exactness: `computed == TOTAL_super × ACC_N`. DEADMAN_TICKS stays 0.5s (anti-brick).
- **The rule:** `MAXFAT < ACC_N` requires `JDEPTH ≤ POOL_N`. concurrent-fat == ACC_N == G. The stagger knob
  and the G knob are the same knob. `WAVES >= 2*ACC_N` guard (fat carriers need lean stagers).

---

## 6. THE PLAN (direction, not yet gated into steps)

**Goal:** DECENTASN's decentralized/emergent accounting + deep-J register accumulation + lazy single
carry-off — i.e. get the assign economy off `WOFLUSH`/`JDEPTH=1` and onto the banked path with J>1.

Two threads, and we owe a decision on order:

**Thread A — resolve the DECENTASN↔deep-J tension (the marriage kmbandy chose).**
- The pin/ring protocol currently assumes WOFLUSH (per-slice atomic, no ACC held across slices) and
  JDEPTH=1. Deep-J means a wave HOLDS a rowblk's ACC across J slices without re-claiming and flushes once
  to the banked LDS reduce, then the tile-scoped completer (whatever wave is free) carries the finished tile
  to DRAM once. The "next available wave carries it" = the existing `TILEDONE` tile-scoped completer
  (`BANKZERO=1`). Design question: can the decentralized claim (SL_RBNEXT pin) coexist with a carrier that
  holds ACC across J slices and with the banked completer? The J ownership model (poison non-lead slots
  SL_RBNEXT=ACC_N) already exists for JDEPTH>1 on the *coordinator* path — port it to DECENTASN.
- FIRST validate the premise cheaply: is DECENTASN's "95% bad / assign-bound" real or a cold-start artifact?
  Run the current build long + single-chunk + fed (or a smaller-N subset of the real shape) with
  work-exactness. If bad→~4%, correctness is basically closed and the wall is throughput.

**Thread B — the measured high-ground the whole night pointed at (may subsume A):**
- These shapes don't need split-K. The winning kernel is **full-K in registers, one tile per WG, one C
  write, big tile (256×128 → 110 TF, 256×256 → 165 TF), then 2 WGs/CU for latency.** That is close to the
  old "grind" path (6.9 TF, but that was small-tile/BW-bound) + the tile-size work. deep-J at J=n_kseg IS
  this. Question to answer with data: on a real ml8 shape, does banked + max-legal-J + biggest-tile-that-
  fits beat DECENTASN, and how close to hipBLASLt's 12.6–70.6 does it get?

**Standing measurement rules (do not violate):** feed to ≥1s steady state before ANY verdict; never quote TF
<1s; check `computed == TOTAL_super × ACC_N` every run; one greenlit dispatch at a time via `./gpu_run.sh`;
kernel change → one bring-up then STOP; hang/DMFAT/oracle-BAD = full stop; never raise DEADMAN_TICKS.

---

## 7. WHERE hipBLASLt ACTUALLY STANDS (the bar, and it's soft)

Real ml8 MoE fp8 (same card): ffn_gate/up 14.8, ffn_down 12.6, attn_q 70.6, attn_kv 15.2, attn_o 60.7 TF —
but at only **5.6%–23% of its own roofline**, and fp8 LOSES to bf16 on ffn_gate/up and ffn_down. The 230 TF
figure is a 4096³ square nobody runs. Our clock-committed peak was 8.8 TF banked (J=1), 32–37 on the
synthetic cube. Real-shape re-measurement on the banked+J path is STILL OWED.

## 8. KG pointers
DECENTASN debug: 7e92918f, a9cfc27e, 0afe9b7d, b21309db, b112c61b (this session's pin work).
Flush/J history: "THE FLUSH IS THE KERNEL" 2026-07-13 doc; K-DEPTH J shipped 2026-07-13; deadman lie
2026-07-14; hipBLASLt GSU1 teardown 2026-07-13. Testing rules: bb6bbe09. Codex thread: `codex resume
019f665b`.
