# PLAN — restore B sharing under self-serve: LDS as a CACHE, not a queue (2026-07-20)

## Goal

Recover the operand sharing that S1 removed, **without** reintroducing the ring handoff whose deletion
is what won S1 its 1.9x.

## What we know (measured, not assumed)

`B`'s global address depends on `tcol, ksi, frag, ks` — **it does not depend on `mblk` or the rowblk**
(`A` does). So all `G=6` rowblks of a tile load **byte-identical B**. Under carry-through every compute
wave self-loads its own operands, so that identical 16 KB slice is fetched 6 times per `(tile, ksi)`:

| | global B fetches per `(tile, ksi)` |
|---|---|
| today (self-serve) | 6 waves x 64 `global_load_tr_b64` = **384** |
| B staged once | **64**, then 6 waves read LDS |

Run #12 measured: B requests **768 GB** against a compulsory `N*K` of **0.50 GB**. Staging B once per
`(tile, ksi)` takes that to **128 GB** (6x). The residual 256x is a *tile-ordering* problem (the whole
B matrix re-streams once per tile row) and is explicitly **out of scope** here — see "Not in scope".

### The LDS accounting that makes this free

The retired operand pool staged **both** operands: `OPSTRIDE = 40960 = B 16384 + A 24576`. The A half
(60%) bought nothing — A is genuinely per-rowblk, there is no sharing to capture. The B half was the
only part earning, and S1 discarded it along with the rest.

```
B-only pool + ACC_N=6 + SSWIN=8 = 41728 B      (limit 65536, ~24 KB headroom)
```

So a B-only cache pays for itself **and** frees enough LDS to raise `ACC_N` to `G=6`, which makes
`GROUPS=1` — removing the second K-sweep per tile and doubling the concurrent-fat-wave count.

## The design principle (this is the whole plan)

**The ring was a QUEUE: consumers block until a producer publishes. This must be a CACHE: check, and
fill it yourself if it misses. Nobody ever waits on anybody.**

- Hit -> read B from LDS.
- Miss -> fetch from global exactly as today, and (if a slot is takeable) leave it behind for the others.
- Race -> two waves fetch the same slice. That is precisely today's behaviour, which is known survivable.

Blocking anywhere on this path re-creates the handoff latency S1 deleted. If the sharing cannot be had
without a wait, **report that instead of adding the wait.**

## Scope

One file: `occ_kernel_dsws_flow.s`, all under a new defsym (suggest `BCACHE`), **default 0 => the
current build is bit-identical**. `SELFSERVE`-path only; the ring path is untouched. No GPU execution —
offline assembly only. Host `occ_dispatch.cpp` needs no change *except* that LDS totals are published
via `.lds_total` and read by the host, so the size change propagates automatically (do not hand-mirror
it; that was run #10's hang).

## What has to happen

- A small, tagged B region in LDS. A WG works one tile at a time, so `tcol` is fixed per WG and the tag
  is effectively just `ksi` (plus a validity/ownership marker).
- The carry-through operand load consults it before issuing `global_load_tr_b64`, and falls back to the
  existing self-serve path on a miss. The existing `BSTAGE_R` (fill: `global_load_tr_b64` -> `ds_store`)
  and the ring's consume form (plain `ds_load_b64`, `:3279`) are both proven — reuse them rather than
  inventing a third layout. Note the transpose then happens **once at fill**, so staging amortizes the
  transpose as well as the traffic.
- Freeing the A half of the pool, and raising `ACC_N` to 6 (`GROUPS=1`). These may be split into a
  separate step if that keeps the first bring-up single-variable — see Gates.
- The coast/feed path must still have real work: today it stages into a pool nothing reads (4.1M wasted
  stages). Filling the B cache is the natural job for it. **Do not disable coast->feed to solve this.**

## The open design question (do not guess — decide it explicitly and say why)

How many B slices to hold, and what happens when `SSWIN=8` reservations span more distinct `ksi` than
there are slots. 8 x 16 KB = 128 KB does not fit; a full cache is not on the table. The sharing we
actually need is across the **6 rowblks of the same `(tile, ksi)`**, which are concurrent — a short
window. A small number of slots (2-4) with `ksi` tags plausibly captures most of it, with misses simply
self-serving. Whatever is chosen, state the eviction rule and what a miss costs.

## Caveats and nuances (each of these cost a real GPU run or a real day)

- **NEVER BLOCK. The cache must never make a wave wait for another wave.** Restated because it is the
  single thing that can turn this from a win into a regression.
- **THE DUTY-CYCLE INVARIANT (see the block at the top of the kernel).** Time at peak ~ `JDEPTH*SEGK`.
  This change must not extend the fat window — it should shorten it (LDS latency < DRAM). **Do not
  "while we're here" raise SEGK or JDEPTH; the assembler `DUTYGUARD` will refuse and it is right to.**
- **`grow-fail == 0` is not headroom, it is the symptom that the moat has not engaged.** Judge this by
  whether more waves cycle through brief peaks, not by whether one wave's segment got cheaper.
- **`BANKZERO=1` is mandatory and already guarded** (`:908`, `:920`): every ksi is a pure `ds_add_f32`,
  which is what makes concurrent-ksi reduction safe. Do not reintroduce a `ksi==0 ds_store` fast path.
- **`GSTORED`, not `DRAIN`, is the real bank-reuse barrier.**
- **`s49` is the reserved `exec_lo` save** for every `lds_*` macro. Never hold a live value across one.
- **THERE ARE ZERO FREE SGPRs.** A full audit (`.set` aliases AND literal `sNN` AND `s[a:b]` ranges,
  s0..s105) found none. `s91/s92/s93` look free in the `.set` table and are NOT (JDEPTH counter, fat
  scratch). Grep **both** spellings; either alone produces a false "free" and corrupts live state.
- **`s_alloc_vgpr` does NOT drain VMEM stores.** Drain before every reallocation.
- **Exec-masked atomics must target the first ACTIVE lane**, result read while that mask is installed.
- **Instrumentation cost scales with TIMES EXECUTED, not call sites.** A per-work-item eager gauge cost
  63s of a 66s run. PHIST measured **~220%** overhead, not the ~10% budgeted — it is FORENSICS-class.
- **Host/kernel LDS is now published by the artifact** (`.lds_total` -> `<tag>.lds` -> host reads it).
  Do not add a second hand-computed copy; that mismatch was run #10's hang and GPU reset.

## Not in scope (name them so they are not silently absorbed)

- The **256x tile-row re-stream** of B. That is a traversal-order problem (tile swizzle / K-blocking),
  it is the larger remaining factor, and it interacts with the C-flush count. Separate plan.
- `KMAJOR`. Still refused under `SELFSERVE`.
- Raising `SEGK` / `JDEPTH`. Retracted on architectural grounds.

## Gates (all offline, no GPU)

Config of record: `WAVES=30 G=6 FM=1 FN=4 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1
BANKZERO=1 STAGGER=1 SELFSERVE=1 SSWIN=8 PHIST=0 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1`.

- `BCACHE=0` is **bit-identical** to the current default bin, sha256 prefix `f36c06a0`. This is the
  proof it is inert until switched on.
- `SELFSERVE=0` still byte-identical to `43beb082`.
- `BCACHE=1` assembles with **zero** scratch/spill, and the published `.lds` matches the assembled
  `LDS_TOTAL_FLOW` (run `gate_lds.sh`).
- `gate_duty.sh` still 10/10 — this change must not move `J*SEGK`.
- `gate_phist.sh` still 6/6; `gate_sswin.sh` still 6/6.
- If `ACC_N=6` is included: it must assemble within 65536 and `GROUPS` must read 1.

**Bring-up is ONE dispatch, then STOP and report** (rule 2 — the kernel will have changed). Expected
`computed` at the config of record is **50331648 exactly**; a short count means work was dropped, and
dropped work makes TF look better.

## Falsification criterion (set in advance, on purpose)

If B request traffic does not fall materially below the measured 768 GB — or if TF regresses because the
cache lookup costs more than the fetch it saves — **the hypothesis is wrong; report it and stop.** Do not
widen the cache, add slots, or add a wait to rescue it.
