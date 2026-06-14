# Triton recon: RDNA4 (gfx1201) fp8 — bump or hold?

**Date:** 2026-06-14
**Scope:** Should we bump our vendored Triton (`~/GitHub/triton` @ `4768da5`, 2026-05-16) to pick up RDNA4 fp8 fixes, for an ml8/fp8 a8w8 blockscale GEMM substrate used both JIT (PyTorch trainer) and AOT (llama.cpp inference binary)?

**Targets we care about:** gfx1201 / gfx1200 (RDNA4, OCP e4m3/e5m2), gfx942 (CDNA3, fnuz fp8). **gfx1250 is a newer arch we do NOT have** — flagged below as "not for us."

---

## Range surveyed

- Our HEAD: `4768da5e8228dfbda8e0b7a61101f87d953341bd` (2026-05-16). Essentially `v3.7.0` (tagged 2026-05-07) + 1 commit; v3.7.0 is *not* a strict ancestor but they share lineage and our HEAD is post-3.7.0.
- `origin/main`: `007ef1530aa1c9d1a78d206417fb7721fbd53211` (2026-06-14), **150 commits ahead**. main has advanced to **3.8.0-dev** (version bump in #10583); there is **no 3.8.0 tag yet**.
- **Every commit in `4768da5..origin/main` is UNRELEASED** (post-v3.7.0, pre-3.8.0). Confirmed via `git tag --contains` on the key SHAs (no tag contains them).
- Only `origin/main` is fetched locally; no separate `release/*` branches are tracked in the clone. Latest tag overall is `v3.7.0`. (Note: `git fetch` itself was sandbox-denied this session, but FETCH_HEAD is dated 2026-06-14, so refs are current as of today.)

---

## Ranked table — most relevant Triton changes for gfx1201 fp8

Released? column: all are **unreleased** (in `4768da5..origin/main`, post-v3.7.0). "Rel" = relevance to *our* gfx1201/gfx1200 fp8 path.

| Rank | SHA | PR | Date | Rel to gfx1201 fp8 | One-line |
|------|-----|----|------|--------------------|----------|
| 1 | `bb5acbe59` | #10458 | 2026-06-04 | **HIGH — correctness** | Fix fp8 conversions on archs without native fnuz HW: stop using fnuz HW upcast paths unconditionally on gfx12 (which is OCP e4m3/e5m2). Directly fixes wrong fp8 results on RDNA4. |
| 2 | `150e02d74` | #10441 | 2026-06-01 | MED — enabler | NFC refactor: moves all AMD FpToFp (FP32/16/BF16/FP8, SW + HW intrinsics) into `ConvertFpCastOpToLLVM.cpp`. #10458 lands on top of this; needed if cherry-picking the conversion fixes cleanly. |
| 3 | `0418ee6c1` | #10390 | 2026-05-27 | MED — perf (LDS) | Extends `InThreadTranspose` to RDNA3/3.5. RDNA4 (gfx120x) was already enabled earlier in #10185 (before our HEAD), so RDNA4 already has wide `ds_load_b128` WMMA operand loads. Confirms our HEAD already has the RDNA4 LDS-transpose perf path. |
| 4 | `09500db9f` | #10439 | 2026-06-01 | LOW — hygiene | Replace raw LLVM intrinsic with ROCDL in warp-specialize lowering. |
| 5 | `3b5446d4a` | #10290 | 2026-06-03 | LOW for us | kWidth fix for MFMA with kPack>1 — **gfx942/CDNA3 only** (kPack is restricted to 1 on RDNA4, so this guard does not change gfx1201 codegen). Relevant to MI300X, not R9700. |
| 6 | `3023cc769` | #10422 | 2026-05-30 | LOW | `triton_kernels/reduce.py` Python warp-size heuristic to dodge an fp8 reduce regression. Host-side kernel lib, not WMMA codegen. |
| 7 | `709022caa` | #10202 | — | NONE for us | fp4/fp8 in TMEM for MMA LHS — **NVIDIA/Blackwell only.** Not AMD. |
| — | gfx1250 cluster | #10605, #10598, #10587, #10594, #10568, #10543, #10334, #10372 | various | **NOT FOR US** | TDM / partitioned-layout / scheduler work for gfx1250 (newer arch we do not have). Ignore. |

**Bottom of the table — supporting AMD commits in range (not fp8-specific but touch the AMD backend we ship):** #10498, #10535, #10528, #10510, #10496, #10529, #10541, #10450, #10369, #10414, #10362, #10383, #10392, #10367, #10340, #10184, #10332. These are buffer-op / pointer-canonicalization / fence / codegen-hygiene fixes; none are blockers but they ride along on a bump.

### Bottom line on the GEMM perf gap (pain point #1)

I found **no in-range commit that specifically closes the ~20% gfx1201 fp8 a8w8 GEMM perf gap.** The fp8 WMMA instruction-selection for gfx12 (OCP e4m3/e5m2 → `{fp8e4nv, fp8e5}` operand combos) already exists in our HEAD's `AccelerateAMDMatmul.cpp`. The perf-relevant RDNA4 lever (InThreadTranspose / wide LDS loads) was already enabled pre-HEAD in #10185. So a bump to current main does **not** obviously recover the 20%. **Unknown:** whether the perf delta is a scheduling/pipelining issue (cf. external reports of an AMD software-pipelining use-after-free at `num_stages>=2` on gfx12 around Triton 3.6) vs. a kWidth/layout issue. Worth a direct A/B benchmark, not assumed.

---

## AOT / `--target` driverless bug (pain point #2, issue #170)

**Status: NOT fixed upstream. No PR found, merged or in-flight.**

- `python/triton/tools/compile.py` on `origin/main` is **byte-identical** to our HEAD (last touched by our HEAD commit #10298, which is unrelated). No change in the 150-commit range.
- The bug is real and visible in the code: even when `--target hip:gfx1201:32` is passed, the script still calls `triton.runtime.driver.active.map_python_to_cpp_type` (line ~175) and `triton.runtime.driver.active` for `ty_to_cpp`. `driver.active` forces GPU/driver initialization at compile time. So `--target` correctly drives `triton.compile(...)` (lines 137-142) but the **C-stub emission path still demands a live GPU**. That matches "AOT ignores --target, demands a GPU at compile."
- Full-history search for `driverless` / `cross-compile` / `active.get_current_target` / `map_python_to_cpp` turned up **zero** commits.
- The long-standing upstream tracking issue is **#4219** ("cross-compile triton kernels with tools/compile.py") — still **OPEN**, no assignee, no linked fix.
- Open PRs touching AOT/fp8 cast we should watch (none fix the driverless demand): **#10534** (ravil-mobile, *draft*) continues the AMD FP-cast refactor on top of #10441/#10458; **#10595** (dot_scaled accept float8e8m0fnu scales); **#9603** (FNUZ fp8 abs NaN bug). None remove the `driver.active` dependency in `compile.py`.

**Implication:** Bumping Triton does **not** fix our AOT/driverless build for the llama.cpp binary. We still need our own local patch to `compile.py` (replace the two `triton.runtime.driver.active.*` uses with a target-derived backend/type mapping so no GPU is required). This is independent of the version bump.

---

## Bottom line: bump Triton for RDNA4 fp8 now?

**Yes — bump, but for correctness, not for the perf gap, and it does not solve AOT.**

- **Do bump** to at least **`bb5acbe59` (#10458, 2026-06-04)** — this is the one change that materially affects us: it fixes *wrong* fp8 results on gfx12/RDNA4 caused by fnuz HW conversion paths being used on an OCP-format arch. For a quant system this is a correctness must-have. Cherry-picking it cleanly also wants **`150e02d74` (#10441)** underneath it.
- **Minimum target:** if doing a full bump, target `origin/main @ 007ef1530` (2026-06-14) or any commit `>= bb5acbe59`. There is no tagged release containing these fixes yet (3.8.0 unreleased), so a bump means **pinning to a main SHA**, e.g. `007ef1530`. Recommend pinning to a specific SHA, not a moving branch, given the heavy in-flight gfx1250 churn.
- **Do not expect** the bump to recover the ~20% gfx1201 GEMM perf — no in-range change addresses it; benchmark before/after to confirm.
- **AOT/`--target` is orthogonal** — unfixed upstream (#4219 open). We must carry a local `compile.py` patch regardless of which Triton SHA we pin.
- **gfx942/MI300X note:** #10290 (kWidth kPack>1) is a real CDNA3 fix that comes along for free in a bump and benefits the datacenter path.

### Explicit unknowns
- Root cause of the 20% gfx1201 GEMM gap (scheduling/pipelining vs. layout) — not determined from commits; needs profiling.
- Whether #10458 fully covers the e4m3 *and* e5m2 down-convert (fp32→fp8) paths on gfx12 for our blockscale flow, or only up-convert correctness — diff covers both up and down paths (`Fp32ToFp8E*fnuz`, `Fp8E*fnuzToFp16/Bf16`) but our exact dtype mix should be unit-tested.
- Release branch landscape beyond main was not directly enumerable this session (only `origin/main` tracked locally; GitHub MCP list-branches/list-tags/releases were permission-denied). The v3.7.0 tag is the newest tag; 3.8.0 is dev-only.

---

## Method notes / caveats
- `Bash` network ops (`git fetch`) and all GitHub MCP tools were permission-denied this session. Local read-only git (`log`, `show`, `diff`, `tag --contains`, `merge-base`) ran via sandbox override. Remote/PR data came from WebFetch/WebSearch against github.com, which may lag or mis-summarize; PR numbers/titles above should be re-verified before action.
- All SHAs cited are short form from the local clone; expand before pinning in build scripts.
