# FFN-Island Device Role — Implementation Plan

> **For agentic workers:** implement this plan one task at a time. Each task is
> dispatched to a fresh builder subagent and reviewed before the next begins.

> **Deliberate deviation from the plan template:** this plan contains **no
> implementation code**, by standing instruction from kmbandy — code written into
> a plan gets rewritten by the builder anyway, which doubles token usage. Tasks
> specify exact files, exact interface contracts, exact behaviour, and exact
> verification commands instead. Builders write the code.

**Goal:** Give the weight-pager router a third device role so the shared experts
and FFN island live on, and compute on, a second GPU while routed experts stay on
the paging device.

**Architecture:** A new optional "island" buffer type threaded through
`wp::build_router_overrides`, plus an island-device selector and VRAM preflight in
`llama-model.cpp`, plus one CLI flag. Default off; with the flag unset the emitted
override list must be byte-identical to today's.

**Spec:** `docs/superpowers/specs/2026-07-24-ffn-island-device-role-design.md`

**Tech stack:** C++17, CMake, the fork's hand-rolled test harness in
`tests/test-weight-pager.cpp` (no framework: each `TEST_FN` returns a failure
count, `main` sums them).

---

## Global Constraints

Every task inherits these. They are not optional.

1. **Never run a model, inference, or any GPU workload.** No `llama-cli`,
   `llama-server`, `llama-completion`, `llama-perplexity`, no `rocm-smi` state
   changes. Compilation is fine; execution against a GPU is not. If a task seems to
   require a GPU run, stop and report instead.
2. **Never run `systemctl --user daemon-reload`** or restart any service.
3. **The working tree has uncommitted work belonging to other efforts.** Namely
   `common/arg.cpp`, `tools/server/server-models.{cpp,h}`,
   `docs/examples/router-fleet-main.ini`, and deletions under
   `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`. All edits
   must be **strictly additive**. Never run `git checkout`, `git restore`,
   `git stash`, or `git reset` on any file. Never revert a hunk you did not write.
   Never `git add -A` or `git commit -a` — stage only the exact paths your task names.
4. **Default off, byte-identical.** With the new flag unset, behaviour and emitted
   data structures must be indistinguishable from current `master`.
5. **Stop and report rather than improvise.** If the code contradicts this plan — a
   signature differs, a file has moved, a test does not exist where stated — halt and
   report what you found. Do not invent a workaround, do not redesign, do not
   "fix" adjacent code. A halted task with a clear report is a success.
6. **Commit only your own task's files**, with a message describing only your change.
   Do not amend or rebase existing commits.
7. Repository is `~/GitHub/llama.cpp` on branch `master`. Do not push.

---

## File Map

| File | Responsibility | Task |
|---|---|---|
| `src/weight-pager/wp-router.h` | Declare the island parameter on the override builder | 1 |
| `src/weight-pager/wp-router.cpp` | Route shexp + island patterns to the island buft | 1 |
| `tests/test-weight-pager.cpp` | Unit-test both the off and on paths | 1 |
| `include/llama.h` | New model-params field | 2 |
| `common/arg.cpp` | New CLI flag | 2 |
| `src/llama-model.cpp` | Island device selection, VRAM preflight, wiring, logging | 3 |

---

## Task 1 — Island buft in the router override builder

**Files:** modify `src/weight-pager/wp-router.h`, `src/weight-pager/wp-router.cpp`,
`tests/test-weight-pager.cpp`.

**Interfaces produced** (Task 3 depends on this exact shape):
`wp::build_router_overrides` gains a **sixth** parameter, after the existing
`bool emit_dense_catch_all`, named `island_buft`, of type
`ggml_backend_buffer_type_t`, defaulting to `nullptr`. No existing parameter
changes name, type, order, or default.

**Behaviour required:**
- When `island_buft` is `nullptr`, the returned vector must be exactly what the
  function returns today — same entries, same order, same buft pointers.
- When `island_buft` is non-null, the two entries that currently use
  `ROUTER_SHEXP_PATTERN` and `ROUTER_FFN_ISLAND_PATTERN` must point at
  `island_buft` instead of `paging_buft`. Every other entry — the routed-expert
  pattern, the `token_embd` entry, the user overrides, the dense catch-all, and the
  null terminator — is unchanged in both content and position.
- Update the header comment to state what the island parameter does, in the style of
  the existing comments on that declaration.

**Steps:**
- [ ] Read the current `build_router_overrides` and the two existing tests
      `test_router_overrides_expert_only` and `test_router_overrides_preserve_user`
      in `tests/test-weight-pager.cpp` (they call the 4-argument form; the new
      default must keep them compiling untouched).
- [ ] Write two new failing tests in `tests/test-weight-pager.cpp`, following the
      existing file's conventions exactly (a `static int test_*()` returning a
      failure count, using the file's existing `EXPECT`/`EXPECT_EQ_INT` macros,
      registered in the table inside `main`). Use distinct fake non-null pointer
      values for the paging, resident, cpu and island bufts so mis-routing is
      detectable, exactly as the existing router tests do.
      - Test A: island is null → assert the emitted list is identical to the list
        emitted by the current 4-argument call, entry for entry.
      - Test B: island is non-null → assert the shexp entry and the FFN-island entry
        both carry the island buft, the routed-expert entry still carries the paging
        buft, the dense catch-all still carries the resident buft, and the entry
        count and ordering are unchanged from Test A.
- [ ] Build the test target and confirm the new tests **fail** before implementing.
- [ ] Implement the parameter and the routing change.
- [ ] Build and confirm all tests in the binary pass, including the two pre-existing
      router tests.
- [ ] Commit `src/weight-pager/wp-router.h`, `src/weight-pager/wp-router.cpp`,
      `tests/test-weight-pager.cpp` only.

**Build and test commands:** configure/build the CPU test target only — do **not**
start a HIP build. The test is registered at `tests/CMakeLists.txt:250` via
`llama_build_and_test(test-weight-pager.cpp)`. Use the existing `build-cpu`
directory. Build the `test-weight-pager` target and run the produced binary
directly; it takes no arguments and exits non-zero on failure. If `build-cpu` is not
configured or fails to configure, **stop and report** — do not create a new build
directory and do not fall back to a HIP build.

---

## Task 2 — Params field and CLI flag

**Files:** modify `include/llama.h`, `common/arg.cpp`.

**Interfaces produced** (Task 3 consumes these):
- A new field on `llama_model_params`, declared immediately after the existing
  `const char * weight_paging_resident_device;` at `include/llama.h:340`, named
  `weight_paging_ffn_island_device`, of type `const char *`, documented with a
  trailing comment in the same style as its neighbours.
- Its default is `nullptr`, set wherever the other `weight_paging_*` defaults are
  initialised in `llama_model_default_params`.
- A CLI flag `--weight-paging-ffn-island-device` taking one value, documented as
  `<dev|auto|off>`, which assigns to that field.

**Behaviour required:**
- The flag is registered for exactly the same set of examples as the neighbouring
  `--weight-paging-resident-device` flag at `common/arg.cpp:2793`. Match it; do not
  broaden the set. (Note `llama-completion` deliberately rejects `--weight-paging*`
  flags.)
- Help text should state: names the device that hosts the shared experts and FFN
  island; `auto` selects the first resident device; `off` or unset disables the role
  and keeps those tensors on the paging device.
- **`common/arg.cpp` contains kmbandy's uncommitted edits.** Add your flag block
  adjacent to the existing weight-paging flags without touching any other hunk. Do
  not reformat, do not reorder, do not revert.

**Steps:**
- [ ] Read the existing `--weight-paging-resident-device` registration and the
      surrounding block to match its structure and its example set exactly.
- [ ] Add the params field and its default.
- [ ] Add the flag registration.
- [ ] Verify the two files still compile. Compiling `common/arg.cpp` alone is
      sufficient; do **not** kick off a full HIP build. If you cannot compile it
      cheaply, say so in your report rather than starting a long build.
- [ ] Commit `include/llama.h` and `common/arg.cpp` only — and confirm in your
      report that `git status` shows the other pre-existing modified files still
      modified and **not** staged.

---

## Task 3 — Island device selection, VRAM preflight, wiring, logging

**Files:** modify `src/llama-model.cpp`.

**Interfaces consumed:** the six-parameter `wp::build_router_overrides` from Task 1
and the `weight_paging_ffn_island_device` params field from Task 2.

**Context:** the relevant block is the `WP_RESIDENT_DENSE` router setup in
`llama_model_base::load_tensors`, beginning around `src/llama-model.cpp:1412`, which
today resolves a paging device via `wp_select_paging_device_index` and resident
devices via `wp_select_resident_device_indices`, then calls
`wp::build_router_overrides`.

**Behaviour required:**

*Selection.* Add a static selector alongside the existing two, taking the model
params, the device list, the resolved paging index, and the resolved resident
indices, and returning a single device index or a sentinel meaning "no island".
Resolution rules, in order:
- If the params field is null or empty, fall back to the environment variable
  `WP_FFN_ISLAND_DEVICE`, read in the same style as the other `WP_*` reads in this
  file. An explicit params value always wins over the environment.
- Value is null, empty, `"off"`, or `"none"` → no island. This is the default.
- `"auto"` → the first resident device index; if there are no resident devices, no
  island.
- Any other value → look it up by device name using the same helper the resident
  selector uses. Not found → log a warning naming the value, and no island.
- If the resolved index equals the paging index → log a warning saying the island
  would be the paging device and the role is therefore a no-op, and no island.

*Preflight.* Before committing the placement, sum the byte sizes of the tensors that
would move — those matching the shared-expert and FFN-island patterns already
declared in `wp-router.h` — using the model loader's tensor metadata available in
`load_tensors`. Query free VRAM on the candidate device. If the island total exceeds
free VRAM minus a reserve, log a warning stating both numbers and fall back to no
island. The reserve is one named constant with a documented default of 1024 MiB,
overridable by the environment variable `WP_FFN_ISLAND_RESERVE_MB`, parsed in the
same defensive style as the other `WP_*` integer environment reads in this codebase.

*Wiring.* When an island device is resolved, obtain its buffer type and pass it as
the sixth argument to `wp::build_router_overrides`. When there is no island, pass
nothing (or `nullptr`) so the emitted list is byte-identical to today's.

*Logging.* Extend the existing `WP_RESIDENT_DENSE router:` log line, or add one line
directly after it, reporting the resolved island device name (or that there is none)
and the island byte total that was placed there. Keep it to a single line in the
style of the surrounding logging.

**Steps:**
- [ ] Read the existing block at `src/llama-model.cpp:1412` and the two existing
      selector functions to match their structure, naming, and warning style.
- [ ] Implement the selector.
- [ ] Implement the preflight sum, the free-VRAM query, and the reserve handling.
- [ ] Wire the island buft into the override builder call.
- [ ] Add the log line.
- [ ] Build. This file is part of the main library, so a HIP build is required —
      use the project's normal HIP build directory `build-hip` and build only what is
      needed to compile and link the library. Report the build command you used and
      its result. Do not run any produced binary.
- [ ] Confirm the pre-existing `test-weight-pager` tests still pass.
- [ ] Commit `src/llama-model.cpp` only.

---

## Verification not covered by these tasks

Tiers 2 and 3 of the spec — greedy-decode coherence, wikitext perplexity against the
**4.1524** paged baseline, and the decode-throughput A/B that answers the
`cpy_tensor_async`-over-TB3 overlap question in spec §7 — require both GPUs and are
**explicitly excluded** from every task above. They are run only with kmbandy's
go-ahead, after review.
