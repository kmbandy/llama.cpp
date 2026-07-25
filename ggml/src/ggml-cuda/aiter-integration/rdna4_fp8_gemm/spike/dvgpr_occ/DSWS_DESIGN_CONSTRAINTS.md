# DSWS design constraints — the checklist to gate EVERY design against (2026-07-16)

Pulled verbatim from the 2026-07-15 brief / `DSWS_STRATEGY_2026-07-15_NIGHT.md`. On 2026-07-16 AM I had this
content in context and did NOT design against it — built a banked kernel fed by a GLOBAL (cross-WG) claim,
which violates G1, and quoted a 0.5s "assign-bound" number, which violates G3. Two dispatches wasted. This
file exists so the constraints are a GATE, not a doc that scrolls past. Check a design against ALL gates
BEFORE writing code or dispatching.

| # | Gate (from the brief) | source |
|---|---|---|
| **G1** | **Banked LDS combine is SAME-WG ONLY.** Any banked-reduce design MUST keep a tile's `n_kseg` K-slices inside ONE WG. Cross-WG combine exists only in DRAM (= WOFLUSH, the slow path). "No cheap cross-wave register combine — waves can't hand registers." | strat §3.5, L86–89, L126 |
| **G2** | **No cross-wave register combine.** Deep-J holds ACC in ONE wave across J slices; never hand ACC between waves. | strat §3.5, L88 |
| **G3** | **Never trust assign-bound% or TF from a run < ~1s or chunked.** It's the cold-start artifact (assign 76%→1.8% once fed). Validate assign-bound / quote TF ONLY from a long, single-chunk, fed (≥1s steady state) run. | strat §2 caveat L64–67; §6 L176 |
| **G4** | **DECENTASN's global pin/ring assumes WOFLUSH=1 & JDEPTH=1** (top-of-file `.error`s). Banked deep-J needs both OFF → do NOT run the global pin/ring on banked; use a WG-local producer. | strat §0 L19–22; §6 L158 |
| **G5** | **The flush is the wall** — ~97% WOFLUSH, ~45–57% banked; the assign/pool/role economy is ~2.6%. Optimize the flush (deep-J / banked), not the economy. | strat §3.1 L75–76 |
| **G6** | **Thread B is the measured high-ground** — J=n_kseg → split-K vanishes → no combine at all → bigger tile → the hipBLASLt shape (612 gfx1201 fp8 kernels, all GSU1). If a design is still round-trip/assign-bound when FED, pivot to B rather than push harder on A. | strat §3.6 L90–92; §6 Thread B |
| **G7** | **J>1 config rules (per the KERNEL's assemble guards, lines 156–207 — authoritative):** J is pow2 (divides n_kseg); STAGGER=0 → `WAVES ≥ 2·ACC_N`; STAGGER=1 → `WAVES ≥ MAXFAT_EFF + STAGERS` AND `MAXFAT < ACC_N` AND **`JDEPTH ≤ POOL_N`**. DRAIN MUST NEVER PASS AN UNFLUSHED SEGMENT. ✅ **`JDEPTH ≤ POOL_N` IS REAL for the THROTTLED stagger case** (re-confirmed + now GUARDED 2026-07-16). A capped deep-J carrier reaches JDEPTH super-tiles ahead but the ASSIGN window is only POOL_N deep, and DRAIN can't advance until all ACC_N rowblks of a ksi finish (throttle prevents it) → the JDEPTH-th segment never stages → stage-starve deadlock. Proof (LDS-reconstructed): 0714 J=4/POOL_N=4 CLEAN, J=8/POOL_N=4 BROKE, and 2026-07-16 J=4/POOL_N=3 deadlocked fed (computed=0). ⚠️ It applies ONLY under STAGGER *throttled* — STAGGER=0 is unaffected (all waves fat, no window limit), which is why the deep-J sweep ran J=8/16/32 clean at POOL_N=3. (My 2026-07-16 "G7 is stale" note was WRONG — I'd tested STAGGER=0, the wrong regime.) | strat §5 L136–146; kernel L156–207 |
| **G8** | **Standing safety** (CLAUDE.md): one greenlit dispatch at a time; new/changed kernel = one bring-up then STOP; hang/DMFAT/BAD = full stop; DEADMAN_TICKS stays 0.5s; offline-first; `DECENTASN=0` stays md5 `386dc28`. | CLAUDE.md |

## A' checked against the gates (before building)

A' = coordinator's WG-local whole-tile claim + banked reduce + deep-J + TILEDONE lazy carry (all proven),
with the producer duty decentralized WITHIN the WG (ASSIGN_LOCK, any wave). Steps: (1) deep-J on the banked
coordinator, real shape, FED — validate + baseline; (2) add the WG-local producer, ablate vs (1).

- **G1 ✅** whole tiles per WG (the fix for the AM failure, which violated G1 via global claim).
- **G2 ✅** coordinator jloop holds ACC in one wave across J.
- **G3 ⚠️ enforce** every A' verdict must be fed ≥1s single-chunk (the rule I broke AM).
- **G4 ✅** abandons the global pin/ring; builds on the coordinator path.
- **G5 ✅** step-1 deep-J attacks the flush directly.
- **G6 🔭** if fed step-1 is still round-trip-bound, pivot to B before step-2.
- **G7 ⚙️** deep-J config validated at assemble (J pow2 | n_kseg; WAVES≥2·ACC_N; JDEPTH≤POOL_N if stagger).
- **G8 ✅** one dispatch, fed, then stop.
