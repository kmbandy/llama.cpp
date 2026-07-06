# Murmur Engine — Morning Triage (2026-06-26 night → pickup)

## TL;DR
Spent the night making the Murmur dynamic-workflow engine produce clean research
output on the junk-drawer fleet. **FOUR real bugs root-caused + TDD-fixed.** The
LAST fix (removing openai-python) introduced a **concurrency regression** that is
the open blocker: under real swarm load every model call fails `model_unreachable`
even though the llama-servers are healthy (HTTP 200). The v4 run is PAUSED.

## The four fixes that LANDED tonight (all TDD'd, all live)
1. **paged-KV seq_rm crash** — llama.cpp fork, **COMMITTED+PUSHED master `1d9d93906`**.
   seq_rm tail-truncate left freed blocks as holes → `compute_slot_mapping`
   GGML_ASSERT abort under swarm prompt-reuse. Fixed via `BlockTable::truncate`
   restoring the `num_blocks==ceil(live/block)` invariant. Validated on silicon
   (cards survived the load that bricked them). KG `b2ac012f`.
2. **MCP-bridge concurrency wedge** — `engine/mcp_bridge.py`. No per-call timeout +
   per-request session storm → scouts hung 19 min. Added 45s `wait_for` + 8-way
   semaphore. KG `6fdb764d`.
3. **forced-conclusion** — `engine/loop.py` + `types.py`. The agentic loop offered
   tools EVERY turn and never forced a final answer → tool-happy models burned all
   `max_turns`(8) searching, emitted no `content` → captured the raw reasoning
   ramble. Fix: reserve the FINAL turn, strip tools, inject `_FINAL_TURN_NUDGE`;
   `max_turns` 8→12. VALIDATED: v3 done scouts produced CLEAN FINDINGS (lens
   512–2970c, not 18k rambles). KG `a9ded2b5`.
4. **openai-python REMOVED** — `engine/model.py`. A prior session re-introduced
   `openai.AsyncOpenAI` into the engine (contradicts the engine's whole purpose).
   Rewrote `ModelClient` on raw httpx + explicit `timeout(read=300)`. 89/89 engine
   tests, validated single-request live. KG `c306326c`.
   ***** THIS IS THE ONE THAT BROKE UNDER LOAD *****

## THE OPEN BUG — morning task #1
v4 workflow `ab88955c-0e07-40c3-8432-dad92880e725` (PAUSED): ALL agents fail
`handoff run error: model_unreachable`:
- 6900xt: **18 failed** (it was the HEALTHY card that carried v3 — this is a
  REGRESSION), 8 in_progress (expired on pause)
- 480: 8 failed; 1070: 1 done (an 8996c ramble) + 7 failed
- **BUT `curl /health` on all three servers (6900xt:8092, 480:8097, 1070:8095)
  returns HTTP 200 — the servers are UP.**

**Diagnosis:** the httpx `ModelClient` swap regressed model calls UNDER CONCURRENCY.
v3 (openai client) had the 6900xt working (it carried the run solo). v4 (httpx) has
the 6900xt failing. Single-request httpx works (mock test + isolated 480 test
post-restart, both clean). Only fails under 42-scout × 12-turn concurrent load.
`engine/loop.py` maps ANY `model.stream` exception → `model_unreachable` (the
`except Exception as e: yield em.error(code="model_unreachable", message=str(e))`),
**MASKING the real error.** A 15-concurrent `ModelClient.stream` repro against the
6900xt HUNG (120s timeout, no output) — consistent with connection exhaustion/hang.

**PRIME SUSPECT:** `ModelClient._raw_chunks` creates a FRESH `httpx.AsyncClient`
PER `stream()` call (`owns = self._client is None`). 42 scouts × up to 12 turns =
hundreds of concurrent AsyncClient creations to the same few hosts → likely
ephemeral-port / fd exhaustion OR pool issues → httpx raises/hangs.
**SECONDARY:** the `_raw_chunks` async-generator pattern (`async with
client.stream` + `yield` + `finally aclose`) under concurrency / early-close.

## MORNING PLAN
1. **Unmask the real exception.** Either log `str(e)` at `engine/loop.py`'s
   `except Exception` (only "model_unreachable" reaches the handoff result), or
   re-run a concurrent `ModelClient.stream` repro that prints each task's exception
   class (add a per-task `asyncio.wait_for` so a hang surfaces as TimeoutError).
   Goal: see the actual httpx exception.
2. **LIKELY FIX: inject a SHARED app-level `httpx.AsyncClient`** with
   `httpx.Limits(max_connections=N, max_keepalive_connections=M)` instead of
   per-request creation. `ModelClient` ALREADY supports `client=` injection. The
   dashboard handler (`mad-dashboard.py` ~2306) should build ONE shared AsyncClient
   (module-level, with the explicit timeout + sane limits) and pass it to every
   `ModelClient`. TDD: a concurrency test (N concurrent `stream()` → assert all
   succeed).
3. **DO NOT revert to openai** (`model.py.pre-httpx.bak`). openai must stay OUT
   (explicit design goal). Fix the httpx version forward.
4. **Re-validate:** fire a fresh murmur, confirm 6900xt succeeds + clean FINDINGS +
   480/1070 contribute.

## ALSO OPEN (separate, lower priority)
- **480/1070 (Q5_K_M) tool-calling quality:** in isolated tests they give up
  WITHOUT searching ("unable to locate") while the 6900xt (Q6_K) searches+fetches.
  Possibly Q5-vs-Q6 tool-calling degradation. Address AFTER the concurrency fix.
- **2026 RAM pressure** (238 MiB free): the 480/1070 servers run `--ctx-size
  524288` (512k) — overkill for ~9k-context scouts. Drop to ~64k (frees RAM +
  speeds them). The semantic-index embedder model itself is fine (n_ctx capped at
  512, `mt-embed.cpp:48`) — NOT the RAM eater.
- **Murmur v2 polish (banked):** tool-call budget, redispatch load-rebalance,
  on-demand force-wrap control. (Note: the forced-conclusion fix likely supersedes
  the earlier "scouts must call handoff_complete" idea — content capture works now.)

## FILES TOUCHED (mad-lab-dash, NOT committed — backups alongside)
- `engine/model.py` (httpx rewrite) — `.pre-httpx.bak`
- `engine/loop.py` (forced-conclusion) — `.pre-forcedturn.bak`
- `engine/types.py` (max_turns 12) — `.pre-forcedturn.bak`
- `engine/mcp_bridge.py` (timeout+semaphore) — `.pre-timeout.bak`
- `engine/tests/{test_model,test_loop}.py` — new tests (+ test_mt-block-table,
  test-paged-lifecycle in the llama.cpp fork)
- `mad-dashboard.py` + `engine/tests/smoke_{live,image}.py` (openai removal) —
  `.pre-noopenai.bak`
- `~/.config/mad-lab-agents/config.json` (murmur prompts; tools trimmed by kmbandy)
  — `.pre-v2prompt.bak`
- llama.cpp fork: **COMMITTED + pushed** (master `1d9d93906`).

## RUN IDS
- v4 PAUSED (broken httpx-concurrency): `ab88955c-0e07-40c3-8432-dad92880e725`
- v3 (forced-conclusion validated; openai-timeout era; killed): `2b68541b-...`
- v1 (pipeline mechanically validated end-to-end; done): `1c2c3ef2-...`

## INFRA NOTES
- mad-dashboard.service = SYSTEM service → needs kmbandy `sudo systemctl restart`
  for engine code changes to load. (Tonight's restart "hung then came back" — worth
  confirming it loaded cleanly; though the isolated post-restart httpx test DID work,
  so the new code is live.)
- mad-lab-mcp.service (dispatcher + orchestrator + workflow tools) = SYSTEM service.
- Fire a murmur: `cd ~/GitHub/mad-lab-mcp && python3 fire_priorart.py` (on 2026).
- The whole point: prior-art scan for MAD-305/DSWS — does anyone do RUNTIME-ADAPTIVE
  producer:consumer wave-role rebalancing (vs CUTLASS static setmaxnreg)? Feeds the
  GPU kernel work (MAD305_DSWS_MASTER.md §6 on mad-lab-main).
