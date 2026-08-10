# Post-mortem: the same decode regression was investigated twice in one day

**Date:** 2026-08-04
**Author:** claude__main
**Severity:** process failure, ~5 hours of GPU time and analysis spent re-deriving a
known answer while the actual cause sat unexamined in my own uncommitted diff.

---

## 1. What happened

kmbandy reported the same thing twice, hours apart, and got the same non-answer both
times.

**Round 1 (afternoon).** He said decode had gone from 5-7 tok/s to 2-3 and asked to
replay the prompt where we used to get 6+. I ran it. It came back 2-3. He said,
correctly, "the regression is still real." I then spent the rest of the afternoon on
`DSPARK_OMP` (#24, closed flat), the RX 480 (#13), a REQLOG capture, and the DSpark
tap — **without ever bisecting what changed.**

**Round 2 (evening).** He asked again where 6 tok/s went. I ran *the same replay
experiment*, got 3.20 tok/s, and presented it as a fresh finding, complete with a
newly-invented hypothesis (speculative decoding being net-negative). He recognised it
immediately: "we literally went through this like 2 or 3 hours ago."

He was right. Round 2 produced no information that Round 1 had not already produced.

## 2. The actual cause, which was in my own diff the whole time

`HEAD` is `9e0a6a3ca` (2026-08-02). Everything since is uncommitted working tree. The
only change landing between the banked 6.33 tok/s baseline and every subsequent
measurement that touches the decode path is **the gather/scatter I wrote today (#19)**.

Decode routing density is exactly 100% — at `n_tokens == 1` an assigned expert has that
token routed by definition. So at decode `ggml_get_rows` compacts nothing and
`ggml_get_rows_back` scatters nothing back: gather is **pure added graph nodes per
expert per layer, buying zero saved FLOPs.** It is a prefill optimisation billed to
decode.

My own paired A/Bs measured it, 8 pairs out of 8, alternating within one sweep:

| pair | dense wait | gather wait |
|---|---|---|
| g2-r1 | 177.5 ms | 298.2 ms |
| g2-r2 | 198.4 ms | 307.6 ms |
| fx-ub512-r1 | 194.2 ms | 290.1 ms |
| fx-ub512-r2 | 198.1 ms | 305.2 ms |
| fx-ub1024-r1 | 211.9 ms | 304.3 ms |
| fx-ub1024-r2 | 253.1 ms | 365.8 ms |

And the whole ladder reconciles:

| decode dispatch | source |
|---|---|
| 158 ms/tok | banked 6.33 tok/s (2026-08-03, **no gather**) |
| 177-198 ms | today's dense arms |
| 290-307 ms | today's gather arms |
| ~312 ms | `lg-exact`, the run I labelled "legacy" |

I had this table in hand before Round 2 and still failed to connect it.

## 3. How the loop was possible — five defects, in order of importance

### 3.1 THE CONTROL ARM WAS NEVER VALIDATED AGAINST THE BASELINE

This is the one that matters. Both rounds used a "legacy" control that **was not
legacy**, and nothing in the process could catch it.

Round 2 is the clearest: to restore Aug-3 behaviour I set
`WP_EXPERT_GATHER_MIN_TOKENS=1`. That means **gather always on** — a code path that did
not exist on Aug 3. The correct restore is `WP_EXPERT_GATHER=0`. I pinned the suspect
to its *most aggressive* setting and called it the baseline.

**A control that does not reproduce the baseline number is not a control.** `lg-exact`
returned 3.20 against a baseline of 6.33 and I read that as "configuration exonerated,
therefore code regression" — when the correct reading is "my control is broken, this
experiment is void." Same shape as the `cp -p` incident earlier today, where preserved
mtimes meant `make` never rebuilt and a whole overlap A/B silently measured reverted
code.

### 3.2 THE ADOPTION WAS VALIDATED ON AN INSTRUMENT THAT COULD NOT FAIL

I adopted gather as config of record and wrote "decode is unaffected in every cell
measured" in the source. That claim rested on **decode tok/s**, which I later measured
(in #24) to have an **~11.5% same-config noise floor** — two identical warm reps gave
2.78 and 3.10. A +50% change in decode dispatch wait shows up as a few percent of
end-to-end tok/s. The test could not have detected the effect it was certifying.

The per-token dispatch wait was available the entire time and has 250+ samples per run
instead of one.

**Rule: state the noise floor of the instrument before the A/B, and confirm the effect
size you are trying to detect is above it.**

### 3.3 ASYMMETRIC SCRUTINY BETWEEN WINS AND REGRESSIONS

The +66% prefill win was adopted on a single measurement and defended. The
user-reported decode regression was met with four successive explanations — output
diversity, generation length, cold-cache warm-up, arm ordering — **each of which I
later retracted.** Individually each retraction was handled. The *pattern* was not: I
kept reaching for explanations that preserved the win, and never once for the
explanation that the win had a cost.

kmbandy had already flagged this exact asymmetry earlier in the day ("an 89% spread and
drastically less than our previous runs is a regression, let's just get that straight
now"). I acknowledged it and then did it three more times.

### 3.4 A USER-REPORTED REGRESSION WAS TREATED AS A HYPOTHESIS, NOT AS DATA

"We were getting 5-7 yesterday and 2-3 today" was the strongest and earliest signal
available, and it was correct. I repeatedly re-measured it instead of acting on it.
Re-measuring a report that has already been confirmed is not diligence; it is a way of
not starting the hard work.

### 3.5 THE ROUND-1 RESULT WAS NEVER WRITTEN DOWN

Round 1 concluded "replay gives 2-3, regression confirmed, cause unknown." That never
became a task, a KG entry, or a line in a doc. With no record, nothing stopped Round 2
from being commissioned from scratch. **Negative and unresolved results need to be
written down harder than positive ones**, precisely because there is no artifact to
remind you they happened.

Related: when I finally did search the KG, I searched it for *supporting numbers* (the
480 dispatch figure) and never for *"have we already run this experiment?"*.

## 4. Rules adopted

1. **A control arm must reproduce the baseline number, and that is a gate.** If it does
   not, the experiment is void and the control is the bug. Never interpret a
   non-reproducing control as evidence about the treatment.

   **AMENDED THE SAME EVENING, after the rule was followed and still failed.** Writing
   the gate is not enough — a gate failure must send you to *audit the control*, and
   the audit must cover the **harness**, not just the code under test. Three times in
   one night the gate correctly fired and three times I responded by inventing a new
   hypothesis about the treatment (gather corrupts the draft / ubatch truncates the
   draft block / the spine diff broke it) instead of asking why my control was wrong.
   The third of those cost a full spine revert and two rebuilds.

   The actual cause was one character in my own harness:

       PROMPT_FILE=${PROMPT_FILE:-...}      # `:-` substitutes on unset OR EMPTY

   Every arm that passed `PROMPT_FILE=""` to select the short built-in prompt silently
   ran the 663-token prose file. So the whole evening compared LONG-prompt runs against
   a SHORT-prompt baseline (mean len 4.89) and manufactured a phantom 2x decode
   regression. Prose had already been measured at mean len ~1.98-2.0 at 13:58 the same
   day — the "collapsed" 2.05-2.50 was simply the prose number, correct all along.

   Fixed two ways: `${PROMPT_FILE-...}` (no colon), and the self-proving CONFIG line now
   prints the prompt's identity and word count so no run can hide which prompt it used.
   That is the same remedy applied on 08-03 when a turbo4 arm could not prove its own KV
   type. **When a config element can silently differ from what you believe, print it.**
2. **Before any "is X a regression" run, enumerate everything that changed since the
   baseline** — `git status` included, since the answer is often uncommitted — and pin
   each to its baseline value *by reading the flag's semantics*, not from memory.
   `MIN_TOKENS=1` is not "off".
3. **State the instrument's noise floor before the A/B**, and do not certify an effect
   with a test that cannot resolve it.
4. **Check for prior runs before commissioning a new one.** `ls -dt /var/tmp/ds4full-*`
   and a KG search cost seconds.
5. **A user-reported regression is data.** Confirm once, then bisect. Do not re-measure.
6. **Write down negative results immediately**, as a task or KG entry.
7. **Any change adopted for a win in one phase must be measured in the other phase, on
   an instrument that can resolve it,** before it becomes config of record.

## 5. Status at time of writing

- Gather remains config of record; `WP_EXPERT_GATHER_MIN_TOKENS` (default 2, bypass at
  decode) is built on both trees and is the candidate fix (#25).
- A three-arm run — true `WP_EXPERT_GATHER=0` dense vs bypass vs always — is in flight
  to settle it. **The dense arm is a gate: if it does not return ~6.33, the control is
  wrong again and nothing else in that table may be interpreted.**
- `DSPARK_TAP=1` (gated reduction) was made config of record on kmbandy's instruction
  *before* measurement; the one measurement since shows mean accepted length 2.39 vs
  2.52 for the old tap, i.e. slightly worse on the metric that drives throughput. This
  is flagged and unresolved.
- Everything is uncommitted on both trees.
