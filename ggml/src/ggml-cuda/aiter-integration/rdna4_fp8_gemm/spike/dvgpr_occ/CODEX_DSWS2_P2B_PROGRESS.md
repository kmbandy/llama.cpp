# DSWS2 Phase 2b — feed = prefetch the next super-tile, 2026-07-24 (Sonnet builder)

**Status: GATES PASS.** OFFLINE ONLY throughout: no `./gpu_run.sh` / `./occ_dispatch` invocation, no GPU
dispatch, no `test_oracle`. No `git add`/commit/stash. Files touched this session: `occ_kernel_dsws_flow.s`
(the mechanism) and `build_flow.sh` (wired `DSWS2_PREFETCH`/`PREFETCH_LINES` through to the assembler —
without this the env vars would be silently inert, per the same lesson P2a hit with `DSWS2_ROLEFLOW`).
`occ_dispatch.cpp` was read-only this session; its working-tree diff is pre-existing dirt from earlier
sessions (confirmed via `git status --porcelain`, which shows it `M` before I ever opened it and I never
called Edit/Write on it). `occ_kernel_coop.s` was never opened.

## New defsyms: `DSWS2_PREFETCH` (default 0), `PREFETCH_LINES` (default 4)

Declared at `occ_kernel_dsws_flow.s:921-946`, right after the existing `DSWS2_ROLEFLOW`/`DSWS2_ROLEFLOW_BACK_N`
guards. Two `.error` guards enforce the scope this was reasoned through:
- `DSWS2_PREFETCH` requires `DSWS2_OVERLAP=1` — the new code is inserted inside the SAME `.if DSWS2_OVERLAP`
  block that already branches `.Lflow_feed` away to `.Lflow_feed_empty` (so it is structurally unreachable
  at `DSWS2_OVERLAP=0` anyway), but the guard fails loud instead of silently compiling to nothing.
- `DSWS2_PREFETCH` requires `DECENTASN=1` — `DA_TILE_OFF`/`DA_BASE_OFF`/`ASSIGN_HEAD_OFF` are only
  initialized and meaningful under the DECENTASN coupled-cursor layout (`.Lflow_live` init, `:3263-3272` in
  the pre-this-session source, `.if DECENTASN`); under the old coordinator path these offsets carry
  different semantics (`COORD_KSI_OFF` aliasing) or aren't initialized at all.
- `PREFETCH_LINES < 1` is rejected.

## The gap this closes

Under `DSWS2_OVERLAP` (Phase 1), every compute path self-loads A+B from L2, so the old LDS-staging job at
`.Lflow_feed` (`occ_kernel_dsws_flow.s`, was ~:4223) is gone — the block just branches straight to
`.Lflow_feed_empty` (the same "reserve real work or yield" fallback a coasting compute wave already uses).
A wave labeled AFEED/BFEED therefore does the exact same thing a coasting compute wave does: try to win a
real DECENTASN/CFASSIGN reservation and become a compute wave. If it can't (window full, boundary contended,
already served this cohort), it does **nothing** — it has no job of its own. P2b gives it one: before falling
through to that existing reserve-or-yield fallback, do a small, bounded, read-only prefetch pass.

## Mechanism chosen: read-only peek of ASSIGN_HEAD/DA_TILE/DA_BASE → guess (tcol, ksi) → prefetch B only

**Why B, not A, not both:** the B operand's global address for a given (tcol, ksi) does not depend on
mblk, group, or row-block at all (`s_mul_i32 s20, s30, s14` / the ksi term — see the proven address algebra
at `.Lflow_da_ss_rowblk`, now `:5045-5052`). A's address additionally needs `group` (`within >> shift`,
which requires re-deriving the exact CFASSIGN field-stride math this task explicitly scoped OUT of P2b) and
the row-block index. Restricting to B keeps the mechanism minimal (no group derivation, fewer scratch
registers, smaller code) while still targeting the operand the design doc itself flags as the highest-value,
most-reused prefetch target ("B-reuse stays L2-warm-cache for all paths," §3). The task said "A and/or B" —
B alone satisfies that and is the conservative choice.

**Deriving "next":** read three already-live frontier fields, all via plain `lds_get` (no CAS, no atomic):
- `DA_TILE_OFF` → `t`, the frozen current-tile index. Always valid: the coordinator is the only writer, and
  it only ever publishes a validated `t_new < chunkHi <= TOTAL` (or the cold-start value 0, also valid).
- `DA_BASE_OFF` → `base`, the current field's base (tile-lifetime, frozen until the next tile/group boundary).
- `ASSIGN_HEAD_OFF` → the frontier of "next index to be reserved" — this **is** the design doc's own
  vocabulary for "the next super-tile about to be needed" (§2/§3).

`t` is decoded to `tcol` with the identical magic-div idiom `DECODE_STI` already uses (`s_mul_hi_u32` by
`s12`, `s_mul_i32`/`s_sub_u32` against `s13`=NTL) — applied directly to `t` since we already have it, with
no shift/mask step (that step exists in `DECODE_STI` only to split a combined `sti` back into `t`; we never
built a combined `sti` here). For each of `PREFETCH_LINES` unrolled guesses `i = 0..PREFETCH_LINES-1`:
`within = (ASSIGN_HEAD + i) - base`, `ksi = within & mask` (`s67`), then **clamped** `ksi = min(ksi,
n_kseg-1)` (`s66`) before it ever reaches an address computation. The B address is then built with the
exact same algebra as the real self-serve burst (`occ_kernel_dsws_flow.s:5045-5052`: `tcol*FN*256 +
ksi*KSEG_STEPS*(NT*256) + Bshuf_base`), and a `global_load_tr_b64` is issued into a fixed scratch VGPR pair
using the same per-lane B vaddr constant `v9` the real path uses (set once in the prologue, `:3125`). The
loaded value is never read by anything — it exists only to bring the line into L2. After the unrolled batch,
one `s_wait_loadcnt 0x0` drains exactly that visit's fixed small batch (this is the ordinary "loads I just
issued are done" idiom used throughout this file, e.g. `:5107` — not a spin, not a wait on any cross-wave
signal), then control falls through unchanged to `s_branch .Lflow_feed_empty`.

**Why the clamp is load-bearing, not decoration:** `ASSIGN_HEAD`/`DA_BASE` are read independently (three
separate `lds_get`s, no fence between them), so the "next" guess can be stale, torn-across-reads relative to
each other, or point past a phantom/field boundary. `within = idx - base` can therefore wrap to an enormous
unsigned value. The `s_and_b32 ... , mask` step already reduces that to `[0, mask]` regardless of the input
magnitude (a bitwise AND is insensitive to how a value wrapped), and `s_min_u32 ..., n_kseg-1` further
guarantees `ksi < n_kseg` — i.e. inside the real K range for this tile — no matter what. `tcol` needs no
such clamp because it is decoded from `t`, which is never garbage (see above). The consequence: **the
computed B address is always inside the Bshuf buffer's valid range for this dispatch, for every possible
value the three racy reads could produce.** Worst case is prefetching the wrong (but always in-bounds)
line — a perf miss, never an out-of-bounds access, never a fault.

## Register audit (lean path, 32 VGPR, no grow)

**SGPR scratch used:** `s16, s17, s18, s19, s20` (computed once per visit: t, base, idx0, mblk-temp,
tcol-term) and `s25, s26, s27, s28, s29` (per-unrolled-line: ksi, ksi-term, high-half-temp, final address
low, final address high). All ten are freshly written before any read in this block — none of them is read
"incoming." They are dead at this program point for the same reason P2a's audit gave for `s44/s45/s46`:
every role-entry point in this file (`.Lflow_compute`, `.Lflow_feed_empty` itself, immediately following)
clobbers this exact low-scratch range as its own very first action, with no dependency on whatever a prior
iteration or a sibling role left there. I traced forward from our block to `.Lflow_feed_empty`'s own entry
(`lds_get s44, STAGE_HEAD_OFF` under `!DSWS2_OVERLAP`, or straight into the DECENTASN peek under
`DSWS2_OVERLAP` — this profile's path) and confirmed nothing there reads `s16-s20` or `s25-s29` before
overwriting them fresh from LDS. **Persistent registers used are read-only:** `s4, s5` (Bshuf kernarg base),
`s10` (NT*256 kernarg), `s12, s13, s14` (magic-div/NTL/FN*256 kernarg), `s66` (n_kseg-1, live kernel-wide),
`s67` (mask, live kernel-wide). None of these is written by this block. `s24` (wid), `s34` (cur_role),
`s50`, `s69`, `s75`, `s15` — the documented persistent set — are untouched, confirmed by grep: none of those
symbols appear in the new block at all.

**VGPR scratch used:** `v16:v17` (one 64-bit-destination pair for `global_load_tr_b64`) and `v9` (read-only,
the per-lane B vaddr constant already live since the prologue). `v16:v17` sit inside the `BSTG` lean staging
window (`v[16..31]`, `BSTG=16`, `VLEAN=32`) that `ASTAGE_R`/`BSTAGE_R` use when they run — but those macros
are the OLD ring-feed staging calls, and they **never execute** once `.Lflow_feed`'s `DSWS2_OVERLAP` branch
has already fired (our new code sits strictly before the `s_branch .Lflow_feed_empty` inside that same `.if
DSWS2_OVERLAP` block; the `ASTAGE_R`/`BSTAGE_R` call sites live in the unconditional code AFTER that
`.endif`, which is provably unreachable whenever `DSWS2_OVERLAP=1` since the branch already left). Since
`DSWS2_PREFETCH` requires `DSWS2_OVERLAP=1` by its own guard, `v16:v17` is free in every buildable
configuration of this defsym. RGA's independent livereg count corroborates the SGPR side: peak stayed at
**55** (identical to the P2a rev2 ON build) — a clean 0-delta is the signature of scratch that never
overlapped anything live, not proof by itself but strong corroborating evidence, matching the same argument
P2a's own progress doc made for its `s75` addition. VGPR peak stayed at **48** (nothing here touches ACC/
FA/FB or grows).

**No `s_alloc_vgpr`, no growing:** the entire block is SALU (scalar) arithmetic plus one VALU-free vector
load; it never calls `s_alloc_vgpr`, never reads/writes anything in `[ACC, NFV)`, and runs entirely on the
lean 32-VGPR footprint, exactly as the task required ("a feed wave is lean, 32 VGPR — do NOT grow").

## The no-shared-writes argument (every memory-touching instruction in the new block)

Enumerating **every** instruction in the new block that touches any memory (LDS or global):

| Instruction | Direction | Target | Shared-state? |
|---|---|---|---|
| `lds_get s16, DA_TILE_OFF` | READ | LDS | No — read-only |
| `lds_get s17, DA_BASE_OFF` | READ | LDS | No — read-only |
| `lds_get s18, ASSIGN_HEAD_OFF` | READ | LDS | No — read-only |
| `global_load_tr_b64 v[16:17], v9, s[28:29]` (×`PREFETCH_LINES`) | READ | global (Bshuf buffer) | No — ordinary load, result discarded, never stored anywhere, never read by any later instruction |

That is the complete list. There is no `lds_put`, no `lds_put_r`, no `lds_cas_rtn`, no `lds_fetch_add`/
`lds_fetch_add_r`, no `global_atomic_*`, no `global_store_*`, no `s_sendmsg_rtn`, anywhere in the new block —
confirmed by grep over the exact lines added (the block contains only `lds_get` and `global_load_tr_b64`
among memory-touching mnemonics). Every other instruction is pure SALU register arithmetic
(`s_mul_i32`/`s_mul_hi_u32`/`s_add_u32`/`s_addc_u32`/`s_and_b32`/`s_min_u32`/`s_sub_u32`) or the single
`s_wait_loadcnt 0x0` drain. **Therefore this block cannot publish anything another wave can observe, race
on, or claim** — `ASSIGN_HEAD`, `DRAIN_HEAD`, `STAGE_HEAD`, `DA_ZDONE`, every `SL_*` slot field, every
counter, every mailbox: none of them is written. Combined with the clamp argument above (every possible
value the reads could produce still yields an in-bounds address), this block is read-only *and* safe
regardless of the actual values read — it cannot affect correctness by construction, not merely by
happenstance of the current profile. This is exactly the property the task asked me to STOP and report if
I couldn't establish; I didn't need to invent any synchronization to get it, which is itself the signal that
this is P2b territory and not accidentally P3.

## Offline gates

**Gate 1 — `DSWS2_PREFETCH=0` byte-identical to `cac3ff7c...`.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_PREFETCH=0 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (32324B .text)  LDS=54784B
sha256sum occ_dsws2_w30_flow_gd.bin
  cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553  occ_dsws2_w30_flow_gd.bin
```
**PASS** — exact match to the recorded baseline hash.

**Gate 2 — ON build (`DSWS2_PREFETCH=1 DSWS2_ROLEFLOW=1 DSWS2_OVERLAP=1 DSWS2_RCONV=1`, full A1 profile)
assembles + links 0-spill.**
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 G=6 ACC_N=3 \
  CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 DSWS2_OVERLAP=1 OVERLAP=2 DSWS2_RCONV=1 DSWS2_ROLEFLOW=1 \
  DSWS2_PREFETCH=1 ./build_flow.sh
  OK   occ_dsws2_w30_flow_gd.bin (34260B .text)  LDS=13824B
```
LDS still **13824B** (unaffected — nothing in this pass touches LDS layout; the new block only reads
existing LDS offsets and writes nothing). `.text` 34260B vs the P2a rev2 ON build's 33948B (+312B, consistent
with the added SALU/load instructions, `PREFETCH_LINES=4` unrolled lines).

RGA (`rga_check.sh p2b_on ...`, linked `.co`, same defsym profile as above plus `RGADESC=1`, via
`/home/kmbandy/Downloads/rdts/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s bin`, purely static, no GPU
dispatch):
```
DEVICE,...,AVAILABLE_SGPRs,USED_SGPRs,SGPR_SPILLS,AVAILABLE_VGPRs,USED_VGPRs,VGPR_SPILLS,...,ISA_SIZE
gfx1201,...,106,72,0,256,256,0,...,31428
Maximum # VGPR used  48, VGPRs allocated by HW:  96 (74 requested)
Maximum # SGPR used  55, SGPRs allocated : 106
```
**0 SGPR spills, 0 VGPR spills.** VGPR peak 48 — identical to every prior pass (the prefetch load's
destination `v16:v17` sits inside the already-provisioned lean 32-VGPR block, and nothing here grows).
SGPR peak 55 — identical to the P2a rev2 ON build (this pass's ten scratch registers, `s16-20`/`s25-29`, are
all reused/dead space, not a net increase in simultaneously-live registers — see the register audit above).
ISA_SIZE 31428 vs P2a rev2's 31132 (+296B, consistent with the `.text` delta). The `rga_out/p2b_on` scratch
directory was removed after this check.

**Gate 3 — host `occ_dispatch.cpp` compiles; guards hold.**
`./build.sh` completed (`OK -> ./occ_dispatch`), same pre-existing 23 `-Wformat` warnings, 0 errors.
`occ_dispatch.cpp` was **not touched this session** — `git status --porcelain` shows it `M` (pre-existing,
from earlier sessions) before this session ever opened it, and I never called Edit/Write on it. Nothing
about the LDS total or kernarg contract changed (LDS stayed 13824B), so no host-side guard was expected to
need updating, and none did.

**Gate 4 — `.if`/`.endif` nesting.** Full-file balance check (counting `.if`/`.ifdef`/`.ifndef` as openers,
`.endif` as closer, `.elseif` as neutral): depth reaches exactly 0 at EOF, no negative-depth point anywhere
in the file.

## Scope discipline

Only `occ_kernel_dsws_flow.s` (the mechanism) and `build_flow.sh` (defsym wiring, required for the mechanism
to be reachable at all from the environment — the same lesson P2a's own progress doc recorded for
`DSWS2_ROLEFLOW`) were edited this session. `occ_dispatch.cpp` was read-only (Gate 3 verification only; its
diff predates this session). `occ_kernel_coop.s` was never opened. Nothing staged (`git add`/commit/stash
never run) or dispatched (`./gpu_run.sh`/`./occ_dispatch` never invoked, not even `test_oracle`). The
baseline bin (`DSWS2_PREFETCH=0` A1 profile) was rebuilt as the LAST build of this session —
`occ_dsws2_w30_flow_gd.bin` on disk is
`cac3ff7c2338e73f8fe47c115b592ddcd5afa1014ea77c6da66a37eea4fd3553`, verified by a final `sha256sum` after the
Gate 2/RGA builds. The `rga_out/p2b_on` scratch directory used for Gate 2's static analysis was removed
afterward.

## STOP items

**None.** The mechanism never needed to write any shared state to be useful (it warms L2 for lines the
existing self-serve burst will read regardless of whether this wave or any other wave "did" the prefetch —
that is the whole point of a cache warm, it has no owner). I did not encounter a point where correctness
seemed to require publishing something claimable; if I had, per the task's own instruction, I would have
stopped here rather than building it. The one judgment call worth flagging for a second set of eyes (not a
blocker, just a design choice): I chose to prefetch B only, not A, to keep the mechanism minimal — the A
address needs `group` (`within >> shift`), which is derivable with the same clamp-everything-safe technique
but adds registers and complexity for a payoff the design doc weights lower ("B-reuse stays L2-warm-cache
for all paths" is the design doc's own framing, not A). If a future pass wants A too, the same pattern
extends directly: derive `group` the same way `.Lflow_da_bnd_tile`/`.Lflow_da_stamp` already do
(`within >> shift`), clamp it into `[0, GROUPS-1]` the same way ksi is clamped here, and add the A-address
term from `:5053-5063`.

## Bandwidth note for whoever runs this first on silicon (rule 7)

This raises HBM/L2 traffic relative to `DSWS2_PREFETCH=0` — every feed-role visit now issues
`PREFETCH_LINES` (default 4) additional 8-byte-per-lane loads before falling through to its existing
self-serve attempt. Per CLAUDE.md rule 7, the first GPU run of this build **must** use a small chunk
(`ML8_COOP_CHUNK<=1024` or equivalent) and a short shape, exactly like every other HBM-traffic-raising change
in this file's history. These loads are ordinary cached (non-NT) loads to lines every real compute path
already reads — never marked `th:TH_LOAD_NT` — so they should be strictly additive to L2 hit rate, not the
NTLOAD=1 class of mistake that killed the desktop on 2026-07-14, but "should be" is exactly why this gets a
small chunk first, not a full-scale run.
