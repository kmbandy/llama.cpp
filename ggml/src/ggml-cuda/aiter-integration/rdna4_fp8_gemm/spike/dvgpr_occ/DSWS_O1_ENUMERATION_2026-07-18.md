# O1 enumeration — coupled cursor + deep-J (DECENTASN), 2026-07-18

**Mandate:** `DECENTASN_BANKED_DEEPJ_DESIGN_2026-07-16.md` §5/§8 — before silicon, enumerate every
`SL_RBNEXT next++` and every `SL_RBDONE++`, show the pairing, prove DRAIN cannot pass an outstanding
claim, and prove the deep-J carrier bumps RBDONE for every segment it computes. This is the Gate-2 O1
re-run the doc explicitly defers to "when the J-carrier is wired."

## 1. Every `SL_RBNEXT next++` site (the only obligation-creating op)

Under `DECENTASN`, the ONLY place `next` (the low rowblk counter in `SL_RBNEXT`) is incremented is the
**claim CAS at line 2879**:

```
lds_cas_rtn s46, s45, s33, s47      // CAS(SL_RBNEXT, x, x+1); s46 = old   (.Lflow_claim_execok)
```

Every path that reaches the claim region but does NOT commit this CAS branches to `.Lflow_cmp_tryadv`
(coast — shrink + release, no obligation) BEFORE the CAS:

- pending bit set (unstaged) → `.Lflow_cmp_tryadv` (line ~2867)
- `next >= ACC_N` (exhausted) → `.Lflow_cmp_tryadv`
- **NEW post-grow lead re-check** (`DECENTASN && JDEPTH>1`): non-lead slot → `.Lflow_cmp_tryadv`
- lost CAS → `.Lflow_cmp_tryadv`

The other `SL_RBNEXT` writes are NOT claims: the stamp (line 3492, `RB_PENDING`), `side_final` (line 1042,
clears a pending bit), init (2511/2517), and the two DEAD sentinel blocks (`.Lflow_da_rollback` 3515 /
`.Lflow_da_termslot` 3550, now unreachable). None create a compute obligation.

⇒ **Sole `next++` = the won CAS at 2879. A coast owes no RBDONE.** (unchanged from the 2026-07-16 O1.)

## 2. Every `SL_RBDONE++` site and the pairing

Under `DECENTASN` the carrier bumps `SL_RBDONE` in exactly two places, both in the compute/carry path:

- **Line 2989–2990** (`.Lflow_jloop`, mid-group segment, `JDEPTH>1` only): after computing segment `j`
  (0..J-2), `lds_fetch_add SL_RBDONE[slot_j], 1`. The segment's operands are already consumed (WMMA done);
  its partial sum is in REGISTERS (not yet flushed). Freeing the *operand* slot here is safe (operands read);
  the *bank* is untouched until the flush.
- **Line 3089–3090** (`.Lflow_bankdn` / post-flush, the last segment `j=J-1`, or the only segment at `J=1`):
  after `s_wait_dscnt 0x0` on the banked `ds_add_f32` flush, `lds_fetch_add SL_RBDONE[last_slot], 1`.

**Pairing:** one won claim (line 2879) elects a carrier for rowblk `r` at a LEAD slot. That carrier walks its
J-window (`cursor++`, `slot = cursor mod POOL_N`, lines 2892/2997) and bumps RBDONE exactly once per window
slot: `J-1` times at 2989 (segments 0..J-2) + 1 time at 3089 (segment J-1 after the flush).

⇒ **one claim → exactly J `RBDONE++`, one on each of the J slots in the aligned window.** At `J=1` the window
is one slot and the single bump is at 3089 (2989 is inside the `JDEPTH>1` block, compiled out).

## 3. Per-slot accounting → the drain gate is sound

A pool slot at reservation position `p` holds the staged operands for `ksi = p − base`. All `ACC_N` rowblks
use that slot's operands (`A[r, ksi]` indexed by `r` within the slot's A-resident). Each rowblk `r`'s carrier
whose J-window contains `p` bumps `SL_RBDONE[p]` once. The coupled cursor makes position==ksi, so leads are
J-aligned and each position lies in **exactly one** aligned J-window; across the `ACC_N` rowblk-carriers that
is exactly `ACC_N` bumps of `SL_RBDONE[p]`.

⇒ `SL_RBDONE[p]` climbs `0 → ACC_N`, one per rowblk. The DECENTASN drain gate (`drain_advance`,
lines 963–996: `SL_GEN==DRAIN && SL_RBDONE==ACC_N`, head-walk) frees slot `p` exactly when the SLOWEST rowblk
carrier has passed it. Count check per group: `(n_kseg/J leads) × ACC_N rowblks × J slots = n_kseg × ACC_N`
= `n_kseg` slots × `ACC_N` each. **Exact — no slot over- or under-counted.**

## 4. DRAIN never passes an unflushed segment (the bad=64 invariant, preserved)

The carrier does NOT retire its LAST window slot until AFTER the flush (2989 is gated `s_cbranch .Lflow_jnext`
for non-last; the last segment falls through to `.Lflow_bankdn` → flush → 3089). The last slot is the highest
position in the window. DRAIN is an in-order head-walk gated on `RBDONE==ACC_N`, so it cannot advance past the
last slot until every rowblk carrier has flushed there. A group/tile boundary's `zero_banks` is gated on
`DRAIN < ASSIGN → bail` (`.Lflow_da_boundary`), i.e. it waits for `DRAIN==ASSIGN` (pool fully drained past the
last slot). ⇒ **banks cannot be zeroed while any carrier's sum is still in registers.** The coupled cursor
changed *which ksi* sits at each position but NOT the retire-last-after-flush discipline nor the boundary
barrier, so the invariant holds unchanged.

## 5. The one unpaired-`next` hazard (pre-existing, guarded, detectable — NOT wrong-C)

A carrier force-retired mid-`.Lflow_jloop` (deadman) drops the remaining `RBDONE++` for that rowblk → the
last slot's `RBDONE` never reaches `ACC_N` → DRAIN stalls at that slot → the run finishes SHORT (incomplete),
caught by the `computed == G*MTLsuper*NTL*n_kseg` work-exactness check. This is the 2026-07-14 class, mitigated
by `deadman_progress` at every computed segment (lines 2988/3088 — a WORKING carrier re-stamps and is not
killed) and the `WAVES ≥ 2*ACC_N` / `MAXFAT < ACC_N` guards (lean stagers always remain). It is a
liveness/completeness bug, never a silent wrong result, and the **coupled cursor introduces no new such path**
(the sole `next++` is still the single won CAS at 2879).

## Conclusion

The claim↔RBDONE pairing is 1→J (one per aligned window slot); per-slot RBDONE reaches ACC_N exactly once per
rowblk; the drain gate (`SL_GEN==DRAIN && RBDONE==ACC_N`) is sound; the retire-last-after-flush + boundary
`DRAIN==ASSIGN` barrier keep DRAIN behind every unflushed sum; the only unpaired-next path is the guarded,
work-exact-detectable deadman retire. **O1 clears for the coupled cursor + deep-J**, pending the independent
Codex adversarial pass on the position==ksi invariant and the boundary races (§(a)/(c) of that review).
