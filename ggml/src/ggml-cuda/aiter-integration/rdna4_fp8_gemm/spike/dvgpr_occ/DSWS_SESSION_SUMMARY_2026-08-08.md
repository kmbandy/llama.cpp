# DSWS — Session Summary: LDS round-trip batching (2026-08-08 night)

**Author:** fresh-eyes review pass. **Scope:** OFFLINE only — **no build was run** (RAM-constrained; the
build happens in the morning). All edits are **applied to the working tree, uncommitted**, and verified
*textually* (`.if/.endif` balance, register usage, offset ranges) — **not** by assembly.

> **Morning first step:** verify `BATCHLDS=0 WIDESLOT=0 ./build_flow.sh` reproduces the canonical
> `58e965a46f3e162d` **byte-identical**. If it does not, revert `occ_kernel_dsws_flow.s` and `build_flow.sh`
> to the backup at `/tmp/occ_kernel_dsws_flow.s.bak` (the pre-edit kernel) before anything else.

---

## 1. The diagnosis (why the 20.8 ns/event fixed cost exists)

The LEANGUARD result (§87) is conclusive: **removing 682 guard instructions moved per-event time 0.0%.** The
exec-guard ALU sits on the **SALU pipe**, which issues in parallel with the LDS/MEM pipe — it was already
fully hidden under the `s_wait_dscnt 0x0` of every LDS access. The cost is the **serialized LDS round-trips**:

```
lds_get:   v_mov_b32  v[RG_A], addr
           ds_load_b32 v[RG_D], v[RG_A]
           s_wait_dscnt 0x0            // FULL drain wait — serializes THIS access
           v_readfirstlane_b32 dst, v[RG_D]
```

A wave doing one event + one failed peek serializes **12–18 such round-trips** (claim, flush, `drain_advance`,
`da_peek`), each ~30–40 LDS-latency cycles with a hard wait. That is the fixed 20.8 ns. The math is ~free
(proven by NOWMMA/NODSADD/NOCFLUSH/NOBLOAD).

**Reframe:** `grow-fail = 0` in every run ever → the baton/traveling-peak is **runtime-inert**; waves coast
because there is no staged work, not because VGPR is full. The fixed cost is the **coordination fabric**, not
the wave economy.

---

## 2. What I implemented (5 fusion edits + build wiring)

All guarded behind `BATCHLDS` / `WIDESLOT` defsyms (default **0** → the `.else` arms keep the original byte-
identical). Full design rationale: `LDS_ROUNDTRIP_BATCHING_DESIGN.md`. Every edit is **pure wait-deferral /
wait-fusion** — no new atomics, no new writes, no CAS-semantics change → **correctness-preserving** (like
LEANGUARD), NOT ORACLE-INVALID (unlike NODSADD/NOBLOAD).

### 2.1 Defsyms added — `occ_kernel_dsws_flow.s` ~line 253 (levers block, after `NODSADD`)
```asm
    .set BATCHLDS, 0     // fuse independent LDS reads into ONE wait (see design doc §1)
    .set WIDESLOT, 0     // v2: single ds_load_b128 of the 32B slot ctrl block (design doc §2)
```

### 2.2 Claim path — `SL_RBNEXT` + `SL_STI` fused (`occ_kernel_dsws_flow.s` ~5332)
The post-CAS `SL_STI` re-read (line ~5340) was **fully independent** of the claim CAS. Pre-load it with the
`SL_RBNEXT` read → 2 serialized round-trips become 1.
- `BATCHLDS` arm: `ds_load_b32` STI→v11, RBNEXT→v12, one wait, readfirstlane into `s17`/`s33`.
- `WIDESLOT` arm: `ds_load_b128 v[11:14]` (STI/GEN/RBNEXT/RBDONE) via `v15`=`RM_A` addr, one wait.
- Dropped the post-CAS `lds_get_r s17, SL_STI` entirely (guarded `.if !BATCHLDS`).

### 2.3 Post-grow — `DRAIN_HEAD` + `STAGE_HEAD` fused (`occ_kernel_dsws_flow.s` ~5293)
Two independent `lds_get` (fresh dh / fresh sh) → one fused load pair.

### 2.4 `da_peek` — `DA_ZDONE` + `ASSIGN` + `DRAIN` fused (`occ_kernel_dsws_flow.s` ~6114)
**This is THE hottest path** (stage-5 da_peek = ~30% of wave-time). Three independent frontier reads →
3 serialized waits become 1. The `ZLOCK` check just moves after the single wait; on the boundary-bail branch
`s44/s45` are dead. Bonus: loading all three from the same instant is strictly more consistent (no torn
frontier read).

### 2.5 `drain_advance` — `DRAIN_HEAD` + `STAGE_HEAD` fused (macro, `occ_kernel_dsws_flow.s` ~1866)
Fused only the two independent frontier reads. The `SL_GEN`/`SL_RBDONE` gate is **left serialized** — it is
dependency-ordered and correctness-critical (DRAIN must never pass an unflushed segment).

### 2.6 `build_flow.sh` wiring
- defaults: `: ${BATCHLDS:=0}; : ${WIDESLOT:=0}`
- export list (line 86): added `BATCHLDS WIDESLOT`
- `mkflow` -Wa,-defsym line (line 143): added `-Wa,-defsym,BATCHLDS=${BATCHLDS:-0} -Wa,-defsym,WIDESLOT=${WIDESLOT:-0}`

### Verification done (textual only)
- All 5 `.if/.else/.elseif/.endif` blocks are balanced (read each block).
- Register usage safe: `v11`–`v15` are the pre-grow interior scratch (≤15), free in every edited region.
- `v15` (`RM_A`) used as WIDESLOT addr is not in the `v[11:14]` dest set → no address-conflict hazard.
- RBNEXT extracted from `v13` **before** the claim CAS clobbers `v13`.
- Offsets `ASSIGN_HEAD_OFF=0 STAGE_HEAD_OFF=4 DRAIN_HEAD_OFF=8 DA_ZDONE_OFF=508` all within `ds_load_b32`'s
  signed-16-bit offset range.

---

## 3. Measurement protocol for tomorrow (mirror §87 — same-session, fixed-rep)

1. **Static gate:** `BATCHLDS=0` reproduces `58e965a46f3e162d`. `BATCHLDS=1` / `WIDESLOT=1`: RGA 0-spill,
   `SGPR=72` unchanged. Record SHA-256s.
2. **Bring-up (one dispatch, STOP):** chunk=64, `DSWS2_REPS=1`, **full stride=1 oracle (320/320) bad=0**,
   WORK-EXACT, `occ[0]=0`, `occ[95]=0`, canary, no reset. The §2.2 STI pre-load is the correctness-critical
   bit — a wrong `s17` is a silent wrong C.
3. **Measure:** OFF vs BATCHLDS at config of record (`WAVES=6 FM=2 FN=4 G=8 ACC_N=4 SEGK=256`, `ML8_POOL=64`,
   `SSWIN=32`, fixed `DSWS2_REPS`), plus the `fm1` cell for the two-point fit. Re-fit
   `ns_per_event = b0 + 37.36×MFLOP`.
4. **Decision rule (pre-registered):** `b0` (now ~19.9–20.8 ns) dropping **≥ ~4 ns** → round-trip-batching is
   causal → pursue WIDESLOT + more sites. `b0` not moving → the fixed term is **polling cadence / contention**
   → pivot to §5.3 (right-size WAVES) — a *result*, not a failure.

---

## 4. The priority DESIGN change (bigger than the batching): re-enable deep-J (JDEPTH>1) under SELFSERVE

**This is the single highest-leverage event-count reduction on the board**, and it is the natural follow-on to
the batching work — both attack the same 20.8 ns fixed cost, one by making each round-trip cheaper, the other
by doing **J× more work per event**.

**Why it matters:** deep-J accumulates `JDEPTH` K-segments in the register file and flushes once. At J=4,
events drop 4×, and the flush/drain/claim coordination (the bulk of the 20.8 ns) is paid 4× less often. This
is potentially a **2–3× throughput** lever — far bigger than any single wait-fusion.

**The blocker:** `occ_kernel_dsws_flow.s:1390` — `.error "SELFSERVE requires JDEPTH=1."` Under SELFSERVE there
is no LDS operand-staging pool, so the deep-J model "a carrier walks J pre-staged consecutive ksi" has nothing
to walk through. Each segment is self-loaded from L2, and the J-window isn't atomically available → the
carrier stalls in `.Lflow_jwait` (the "invisible stall", CNT_JWAIT) waiting for a next segment that the
assigner hasn't made claimable.

**What a re-enable requires (the design checklist, all gated at assemble):**
1. **Re-establish the `ksi%J` lead-gate** in the SELFSERVE compute path (the `.Lflow_leadok` block, currently
   inside `.if JDEPTH > 1`), reading `SL_STI` correctly under the DECENTASN stamp.
2. **Guarantee the J-window is claimable before the carrier commits** — the J consecutive ksi must be
   self-loadable without deadlock. G7: `JDEPTH ≤ POOL_N` under *throttled* STAGGER, and **DRAIN must never
   pass an unflushed segment** (the measured `bad=64` bug at J=2/4/8).
3. **Respect the DUTY-CYCLE invariant** (`JDEPTH*SEGK ≤ DUTY_KMAX=256`, `occ_kernel_dsws_flow.s:629`): deep-J
   drives the VGPR duty cycle toward 100% — which is exactly the "dyn-VGPR moat" the whole design exists to
   keep. This is an **architectural** trade (`flush/WMMA = 128/SEGK` vs. square-wave duty) — **ask kmbandy
   before building**.
4. Assemble guards already present: `POOL_N % JDEPTH == 0` (line 986), `WAVES ≥ 2·ACC_N` (line 416) or
   STAGGER+`MAXFAT < ACC_N` (line 393).

**Recommended path:** don't build this blind. First get §3's batching measurement — it tells you whether the
fixed cost is round-trip latency (deep-J's J× amortization will compound the win) or polling cadence (deep-J
won't help; right-sizing WAVES will). Deep-J is a multi-day design; the batching is a same-day measurement.

---

## 5. Other levers (ranked, with what's already built vs. needs work)

### 5.1 `SEGK_STAYFAT` (C1) — BUILT, run the pre-registered matrix
Skip grow/shrink between consecutive bursts. Already gated + refused without `DSWS_ALLOW_NONSTD=1` (it's the
duty-deviation arm). Run the 12-cell SEGK_ATTRIBUTION matrix (3 baseline + 9 ablation) to price the
grow/shrink share of the per-segment slope. **ORACLE-INVALID** until separately gated — slope only.

### 5.2 `BATCH` cursor-batch — BUILT (`occ_kernel_dsws_flow.s:238`, default 1)
Claim N rowblks per reservation, amortizing the claim CAS + drain. Sweep it up (2, 4).

### 5.3 Right-size WAVES — no code, just measure
§85: `waves16` = real −14% with `coast/computed = 13.0`. Since `grow-fail=0`, extra waves are pure pollers.
Sweep 3–8 with the §81 clock-normalized instrument (raw TF is meaningless across cells).

### 5.4 `global_load_tr_b128` — design conversation
Would halve B-loads 124→62. Touches the wait pipeline (`KDBUF_LPT` watermark, `bcnt`). Proven on
`occ_kernel_btr128.s`. Defer until after §3 — it's a feed-path change, not a coordination change, and the
measurement says feed isn't the binding cost.

### 5.5 The `HEAD` cost (19.9% of wave-time, 1.5ms, contents unknown)
The largest *identified* cost. `TRACE=1` per-super-tile claimer timeline is the right instrument. This is the
MoE/adaptivity argument — small-MoE shapes (M=64, 0.09–0.61 TF) are dominated by per-dispatch ramp/drain.

---

## 6. Safety & standing gates (unchanged)

- **Offline-first**; no GPU until §3.2 passes.
- **One greenlit dispatch at a time**; one bring-up then STOP. `DEADMAN_TICKS` stays 0.5s.
- **Oracle + WORK-EXACT gate every TF claim** — the claim path edits are silent-wrong-C territory.
- **Stay at M=576 / 1 WG/CU** until the occupancy axis is re-justified.
- **Never PHASEPROBE** (unthrottled RTC read = the 07-14 brick vector). Use STAGINSTR.

---

## 7. Tree state at logoff

Modified (all **uncommitted**, shared tree): `occ_kernel_dsws_flow.s` (5 fusion edits + 2 defsyms),
`build_flow.sh` (BATCHLDS/WIDESLOT wiring), `LDS_ROUNDTRIP_BATCHING_DESIGN.md` (new), this file (new).
Backup: `/tmp/occ_kernel_dsws_flow.s.bak` = the exact pre-edit kernel. **Nothing staged. No GPU work done.**

## ERRATUM (2026-08-09 morning)

The byte-identity gate FAILED on the 2026-08-08 night edit. The `da_peek` BATCHLDS `.else` arm reordered
the three frontier reads, causing a 49-byte divergence at `.text` offsets `0x49F4-0x4A28` (byte offsets
18933-18984). Fixed by duplicating the ZLOCK mask/test/bail into both arms: the `.else` arm now reads
`DA_ZDONE`, checks ZLOCK, then reads ASSIGN and DRAIN; the BATCHLDS arm keeps its fused load sequence and
checks ZLOCK after the shared wait.

Static gates:

- Bare `./build_flow.sh`: `fail=0`, `LDS=34304`, `.text=28852`; SHA-256 `58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c`.
- `BATCHLDS=1 ./build_flow.sh`: `fail=0`, `LDS=34304`, `.text=28780`; SHA-256 `b813aa2a25a0e0463db76cc8494133bebdf9690fff7bc32a9b0e111c71d37dbf`.
- `BATCHLDS=1 WIDESLOT=1 ./build_flow.sh`: `fail=0`, `LDS=34304`, `.text=28772`; SHA-256 `66184f93e6d5031d1ed7fe034598d3903012feee895e5c0bb4413f51b64e31b9`.

Whole-object census versus bare baseline (`s_wait_dscnt=454`, `ds_load_b32=56`, `ds_load_b128=64`):
`BATCHLDS=1` -> `446/56/64` (`-8/0/0`); `BATCHLDS=1 WIDESLOT=1` -> `446/54/65`
(`-8/-2/+1`). The final bare rebuild is left in-tree.
