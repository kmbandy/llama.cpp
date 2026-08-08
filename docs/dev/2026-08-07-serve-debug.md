# DS4 serving: how to start it, what failed tonight, and where to look next

2026-08-07 night. Five serve attempts; every failure symptom below is from
tonight's logs (`/var/tmp/ds4full-serve/` on main, `/var/tmp/serve.log` on
2026). Binaries: both boxes at `b207c160c` + the 9 pre-existing dirty files
(NOT DSWS work — provenance unconfirmed; the kv-cache hunks read like earlier
DS4/turbo4-session WIP).

## 1. How to start the server

From mad-lab-2026 (the harness's host — inversion to main is a pending
hygiene item):

```sh
cd ~/GitHub/llama.cpp
env ARM=serve SERVE=1 SPINE_HOST=0.0.0.0 CTX=163840 UBATCH=1024 \
  PROMPT_FILE=docs/dev/code700.txt COALESCE=1 \
  HOSTVICTIM_2026=3221225472 HOSTVICTIM_MAIN=3221225472 \
  PREFETCH_HINT=1 SPEC_PAGEIN=1 LEASE=256 SPEC_NMAX=7 SPEC_CONF=0.4 \
  SPEC_HOST=1 STRIPEPAR=1 \
  nohup setsid bash docs/dev/harness-2026-08-05-ds4_full.sh \
    > /var/tmp/serve.log 2>&1 < /dev/null &
```

- Wait for `############ SERVING ############` in `/var/tmp/serve.log`
  (~2.5 min: worker start + dense load + warmup).
- Endpoint: `http://100.86.191.92:8095/v1` (OpenAI-compatible llama-server,
  bound 0.0.0.0). Health: `curl http://127.0.0.1:8095/health` from main.
- `SERVE=1` skips the benchmark drive AND the teardown, and (as of
  `trap - EXIT` fix, this commit) disarms the cleanup trap. To stop the
  stack: rerun the harness (its startup sweep kills stale listeners) or kill
  the pids in `/var/tmp/ds4full-serve/*.pid` on main plus the worker pids on
  2026.
- The CTX/UBATCH above is the largest config that fit tonight. See §3 for
  why that number is under suspicion (it should NOT be the ceiling).

## 2. Tonight's failures, in order

| # | config | result |
|---|--------|--------|
| 1 | CTX=512000 UBATCH=2048 | KV cache alloc failed: **10500 MiB** requested on ROCm1 → OOM. Implies **21504 B/token**. |
| 2 | CTX=262144 UBATCH=2048 | KV (5.4 GB) allocated; **main compute pp buffer 2190 MiB** OOM. |
| 3 | CTX=262144 UBATCH=1024 | main compute **1220 MiB** OOM → free-after-KV was < 1.2 GB. |
| 4 | CTX=163840 UBATCH=2048 | main context fit; **draft (MTP/DSpark) context compute 1170 MiB** OOM — `failed to create MTP context`. The draft creates a SECOND context with its own KV + compute buffers; budget it. |
| 5 | CTX=163840 UBATCH=1024 | Came up, `/health` OK — then every process dead within ~a minute, no error lines, log ends after `listening on 0.0.0.0:8095`. **ROOT-CAUSED: not a crash.** The harness has `trap cleanup EXIT`; the SERVE block's `exit 0` fired it and the cleanup tore down the stack it had just started. Fixed by `trap - EXIT` in the SERVE block (this commit). |

## 3. THE KV CACHE ISSUE (open, the real problem)

**The reference measurement** (2026-08-03 decision memo, "turbo4 deferred,
DS4 runs f16"): f16 KV = **6880 B/token**; on the 16 GB 6900XT with model
7944 MiB + fixed 44 + overhead ~336 + compute 1520 (at 1M):

- 750K ctx → 14766 MiB total, **1602 MiB spare — IT FIT**
- 900K → 15749 MiB, 619 spare
- 1M → does not fit by 196 MiB

**Tonight**: the KV allocation at 512000 ctx was 10500 MiB =
**21504 B/token = 3.13× the measured rate**. Nothing about f16 KV physics
changed; something in the tree or the launch config now allocates ~3× the
KV bytes per token of context. This is a regression (or a config-of-launch
difference) introduced between 08-03 and tonight.

Where to look, in order:

1. **Server slot count / n_seq_max.** Every run today prints
   `slot print_timing: id  3` — i.e. FOUR server slots exist. If the 08-03
   budget was measured single-sequence (llama-cli or `-np 1`) and the dsv4
   cache allocates per-seq or pads per-seq (`n_seq_max` is a ctor arg to
   every `llama_kv_cache` in `llama-kv-cache-dsv4.cpp`), a slots mismatch
   could produce a clean integer-ish blowup. 3.13× is suspiciously close to
   π but also to "some caches ×4, some ×1". CHECK FIRST: what `-np` /
   `n_seq_max` the spine ran with tonight vs how the 08-03 numbers were
   taken, and whether `unified_compressed` covers all of the dsv4
   sub-caches.
2. **Per-cache breakdown from the init log.** The spine prints per-cache
   buffer sizes during init (before the OOM). Diff tonight's
   `/var/tmp/ds4full-serve/spine.log` cache-size lines against a
   reconstruction of the 08-03 budget: find WHICH cache (csa / hca /
   indexer / lid / swa) carries the 3×. That immediately narrows suspects.
3. **Committed changes to the dsv4 cache since 08-03**:
   `git log --oneline 2026-08-03.. -- src/llama-kv-cache-dsv4.* src/llama-kv-cache.*`
   — look for padding changes (`GGML_PAD(..., 256u)` appears in the ctor),
   new sub-caches, or type forcing.
4. **The uncommitted kv hunks in the working tree** (in every binary built
   today): `llama_kv_cache_indexer_type()` forces the lightning-indexer K
   cache to an unquantized type (its own comment says f16 indexer at 256K is
   336 MiB — small, so probably not the 3×, but it proves this area was
   being touched), plus an InnerQ scale forwarding hunk in
   `llama-kv-cache-dsv4.{cpp,h}`. Read them fully; identify their session of
   origin before committing or reverting.
5. **Cheap repro without the GPU rig**: run the spine model with
   `-ngl 0` (CPU KV) at two context sizes and diff the logged KV buffer
   sizes → B/token, no VRAM involved, ~minutes per point. Bisect commits
   against that number if (3) surfaces candidates.

## 4. The spine "crash" (closed)

Symptom: healthy server, log ends cleanly after `listening`, pid gone, port
unbound, nothing in dmesg/journal. Cause: the harness's own `trap cleanup
EXIT` fired when the SERVE block exited (§2 #5). Evidence: monitor health
check succeeded seconds after SERVING; the teardown's ssh kills landed
before the next request ~40 s later; the 2026 workers died in the same
window; no core, no OOM, no kernel log — because it was a plain SIGKILL
from our own cleanup.

If a FUTURE silent death occurs with the trap disarmed, then it's real:
run the spine in the foreground with stderr attached (nohup redirect
swallows the shell's segfault notice), enable coredumps, and check
`coredumpctl` — and remember the 2026-07-30 lesson: an ABI break in
libllama kills newly-spawned binaries while long-running ones survive;
check every binary's mtime against its libllama.so.

## 5. Open items out of tonight

- KV bytes/token regression (§3) — blocks every long-context serve config;
  at the 08-03 rate, 750K f16 fits and tonight's 160k cap is bogus.
- Draft-context VRAM budget (§2 #4) is real and was never in the 08-03
  ledger — even at 6880 B/token, re-derive the ceiling WITH the MTP
  context included.
- Harness host-role inversion (run on main, ssh to 2026) — hygiene item.
- SERVE mode needs a stop command (`SERVE_STOP=1` or similar) instead of
  "rerun the harness to kill it".
