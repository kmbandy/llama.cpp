# DS4 prefetch — morning brief

**Written 2026-08-05 late, with arm 1 running.** Supersedes the "what to build"
half of `2026-08-05-prefetch-brief.md`; that document's *reasoning* (why previous
attempts failed, the kill criteria) still stands and is not repeated here.

---

## 1. The one-line state

**The whole prefetch path is built, wired, tested and shipped. Arm 1 says the
spine half works and costs nothing. Nothing has warmed a single page yet.**

Five commits on `master` (local, unpushed):

```
feba838ff  prefetch: hint from the draft block, where the lead time actually is
d4deac23d  prefetch: warm the pool from a hint, in the idle window
8f3b2a16e  prefetch: wire the hash oracle to the workers, hint frames only
53a317c82  prefetch: hash-layer expert oracle and the prefetch-hint frame
f3a780346  harness: host victim tier on main is 6 GiB   (+ f19bd6e5b before it)
```

---

## 2. Arm 1 result — the hint path works and is free

Interleaved `ctl / hint / ctl / hint`, config of record on rebuilt binaries,
`HOSTVICTIM=0` both machines, `EXPERT_WARM` off.

| arm | prefill tok/s | decode tok/s | acceptance | decode dispatch wait |
|---|---|---|---|---|
| ctl_1  | 22.96 | 3.34 | 0.84286 | 228.20 ms |
| hint_1 | 23.05 | 3.27 | 0.84286 | **227.62 ms** |
| ctl_2  | 24.86 | 3.28 | 0.84286 | 234.97 ms |
| hint_2 | 24.18 | 3.30 | 0.84286 | **227.54 ms** |
| **mean ctl** | **23.91** | **3.31** | 0.84286 | **231.59 ms** |
| **mean hint** | **23.62** | **3.29** | 0.84286 | **227.58 ms** |

Both hint arms' decode wait land *below* both controls, but the control spread
(228.20–234.97) is wider than the gap, so read this as "no cost", not "a win".
Raw: `/var/tmp/sweep_hint.tsv` and `/var/tmp/ds4-{ctl,hint}_{1,2}-run.log` on 2026.

Spine counters on the hint arms:

```
expert dispatch prefetch hint: layers=3 frames=5067 experts=17377
  send_failed=0 no_route=0 skip_dynamic=0 skip_in_flight=0
```

**What this establishes:**

- All three hash layers registered (`layers=3`), so `register_hash_oracle` found
  and copied the `tid2eid` tables outside the weight-pager gate. That extraction
  had never once run on this topology.
- 5067 frames / 17377 expert ids delivered with **zero** send failures, zero
  unroutable layers, zero dynamic-assignment declines, zero in-flight declines.
- The workers **accepted every frame**. PIPE_VERSION 5 negotiated cleanly and no
  session died — a pre-hint worker would have closed the connection on the first
  frame, so this also proves both sides are on the new protocol.
- **Determinism holds: acceptance 0.84286 on every arm**, controls and hints
  alike. The gate is intact.
- **The hint stream is bit-identical run to run** — `frames=5067
  experts=17377` on *both* hint arms, to the digit. The oracle, the dedup and the
  splitmix64 routing are all deterministic, which is what makes the counters
  usable as an A/B instrument rather than just a liveness check.
- **It costs nothing.** hint_1's decode wait (227.62 ms) sits *below* both
  controls; prefill sits between them. As predicted for warm-off, where frames
  are received, validated, counted and discarded.

That was the point of running hints-on/warm-off first: if the spine half were
broken we would have found it here, at zero changed page-ins.

---

## 3. The one thing arm 1 could NOT check — fix this first

**`foreign_layer` / `foreign_expert` were not observable — CONFIRMED, not
suspected.** Grepping every arm's log for `prefetch hints:` returns nothing. The
worker prints those counters from `report_prefetch_hints()` on *clean connection
close*, and the harness **SIGKILLs workers at teardown**, so the line never
fires. I flagged the risk in the code comment and then failed to act on it.

Why it matters: `foreign_expert` is the **routing-agreement check** — spine and
worker both resolving `(layer, expert)` through the same splitmix64. A nonzero
count means they disagree, and that bug would otherwise surface only much later,
disguised as "prefetch mysteriously doesn't help."

Mitigating evidence, not a substitute: `send_failed=0` with 5067 frames means
every frame was accepted by a live socket, and any layer/expert the worker could
not serve would have been counted rather than rejected — so nothing *crashed*.
But agreement is unproven.

**Morning task 1 (small):** make the worker's hint counters observable without a
clean close. Cheapest correct fix — emit them into `WP_REQ_LOG` as a distinct
line prefix, or write a `WP_HINT_LOG=path` that is `fflush`ed like
`WP_PAGEIN_LOG` already is (the pagein log solved exactly this SIGKILL problem
and the precedent is 40 lines away in the same file).

---

## 4. Morning task 2 — arm 2, the first run that can move `n_pagein`

```
EXPERT_WARM=1 PREFETCH_HINT=1 HOSTVICTIM_2026=0 HOSTVICTIM_MAIN=0 REQLOG=1
```

Still one variable (warm), tier still off. Interleave against `ctl`.

**Pre-registered, do not renegotiate after seeing numbers:**

- **Mechanism counter is `n_pagein`, not tok/s.** Decode only.
- **PREFILL IS JUDGED DIFFERENTLY** — page-ins **flat**, `ns_wait` **down**,
  device utilization **up**. At UBATCH=2048 the hash-layer union covers most of
  the 256 experts, so the same pages get read either way; the win is read
  *order*, not read *count*. Judging prefill by `n_pagein` scores a success as a
  failure.
- **Read-amplification gate:** `warm_bytes` up while demand `n_pagein` barely
  falls = pool pollution. **Stop.** Do not tune. The 2026-07 attempt proved a
  0.973-precision predictor does not rescue this.
- **Determinism gate: acceptance must stay exactly 0.84286.** Arm 1 held it.
- Realistic decode ceiling is small **by design**: hash layers were ~12.5% of
  page-ins and layer 0 gets almost no lead, so ~8% is the honest target. The
  point is that the oracle is exact and free — if it cannot win here it will not
  win anywhere.

---

## 5. Morning task 3 — the RAM tier, and a number to revisit

Wired in and committed, **off in arm 1 on purpose**: it changes reads by
construction (synchronous D2H demotes + host hits), which would have destroyed
arm 1's "zero changed reads" check.

Two facts to weigh before switching it on:

1. **The demote is synchronous and on the dispatch thread.** `ensure_batch` calls
   `demote_slot()` inline for every evicted slot *before* the first NVMe read, and
   `store_from_device` does a blocking D2H of the whole 12.75 MB page. The RX 480
   averages 31.4 page-ins per prefill request → ~400 MB of serialised D2H before
   the request starts reading.
2. **Prefill has nothing for it to catch** — 1381 references, 1352 page-ins,
   every page read *exactly once*. A victim cache only pays on a re-read.

The reason to have it is the *interaction with warm*: a page warmed too early and
evicted before its layer arrives is a new re-read class that does not exist today.
Without the tier that is a wasted NVMe read; with it, it returns over PCIe.

**`HOSTVICTIM_MAIN` is set to 6 GiB (kmbandy's number) and I think it is too big.**
main is **15 GB total with ~2 GB of swap already in use**, and also carries the
spine, the CPU DSpark worker (3.18 GiB of experts) and a desktop. Suggest 2 GiB.
Stated once; the knob is `HOSTVICTIM_MAIN` and the call is kmbandy's.

---

## 6. Fleet state — two things that were wrong and are now fixed

**mad-lab-2026 was five commits behind AND carrying a stale pre-fix DSpark
config.** It had `n_embd_nextn = n_embd_dec` with `s_dspark_mtp_embd` defaulting
OFF — the *one-stream* configuration that the 2026-08-04 evening comment block
calls "actively harmful", blaming it for acceptance 0.576–0.627 / mean len ~2.0
against a historical 0.799–0.988 / 3.5–5.9. main had the four-stream fix.

Resolved from the code, not by guessing: **our config-of-record acceptance
0.84286 sits in the four-stream band**, so main was live and 2026 was stale WIP
that never propagated. 2026 fast-forwarded to `f3a780346` and its 10 WIP files
replaced with main's. **Both trees now hash identically across all 13 relevant
sources.** Full backup of 2026's prior tree at
`/var/tmp/2026-wip-backup-20260805/` on that box.

**The "rebuilding kills the live router" risk was never real.** This build system
emits *versioned* libraries — the rebuild produced `libllama.so.0.0.11255` as a
new file while the running router keeps `11191` mapped by absolute path. Nothing
is overwritten in place. Router confirmed alive after both builds. Worth
remembering: **build-hip rebuilds do not endanger the router.**

---

## 7. Flags, and which side each lives on

| flag | side | default | effect |
|---|---|---|---|
| `WP_PREFETCH_HINT=1` | spine | off | compute + send hints; without it no tid2eid table is even copied |
| `WP_EXPERT_WARM=1` | worker | off | actually read hinted pages in the idle window |
| `WP_EXPERT_WARM_CHUNK=n` | worker | 1 | pages per idle step = worst-case latency a real request inherits |
| `HOSTVICTIM_2026 / _MAIN` | harness | 1 GiB / 6 GiB | host victim tier per machine |
| `PREFETCH_HINT / EXPERT_WARM / WARM_CHUNK` | harness | off | the above, plumbed |

Hint and warm are separate **on purpose**: hints-on/warm-off reads exactly what
the config of record reads while still reporting what was offered. That is what
made arm 1 a free test.

**PIPE_VERSION 4 → 5. Spine and all four workers must be rebuilt together** or
they refuse at HELLO. Both are built as of tonight; a `git pull` on either box
without a rebuild will produce a HELLO rejection that reads like a worker crash.

---

## 8. Where the lead time comes from (so it is not re-derived)

- **Layers 0–2 only.** `ggml_get_rows(ffn_gate_tid2eid, inp_tokens)` — a pure
  token-id lookup, zero prediction error. Layer 3+ is a data-dependent router and
  needs a predictor; that is the ground every previous attempt died on.
- **Three hint sites, very different value:**
  1. **top of `draft()`, before `llama_decode(ctx_dft)`** — `dp.id_last`, the last
     *accepted* token, hinted across the entire draft decode (~3 × 12.6 ms).
     **This is the one that matters.**
  2. post-draft, beside the old pager hook — drafted tokens, ~1 ms of lead.
  3. per-ubatch in `llama_context::decode` — covers prefill and the
     non-speculative path.
- Routing the *existing* `llama_wp_on_draft_tokens` hook would have bought
  nothing: it fires microseconds before verify, in the window (3) already covers.
- `graph_dispatcher` dedups an unchanged expert set per layer, so the three sites
  do not send three identical frames.

---

## 9. Open, lower priority

- Worker hint counters not observable under SIGKILL (§3) — **do this first**.
- `test-wp-expert-dispatcher` marginal-cost speed-split case fails, confirmed
  identical at HEAD. `WP_DISPATCH_SPEED_SPLIT` defaults off, not in the config of
  record. The slow worker is winning assignments; real, unrelated, unfixed.
- `test-routed-experts-external` does not compile at HEAD —
  `llama_model_params::use_mmap` became `load_mode` upstream; ctest was running a
  **Jul 30 binary**.
- Ten pre-existing WIP files on both boxes remain uncommitted and deliberately
  untouched. `src/llama-context.cpp` and `common/speculative.cpp` are among them —
  **`git add` on either sweeps someone else's work into your commit.** Stage
  hunks and verify by applying the staged diff to a clean HEAD worktree.
- Board claims `c6d3a592` (main) and `a0d03e0c` (2026) — release when done.
