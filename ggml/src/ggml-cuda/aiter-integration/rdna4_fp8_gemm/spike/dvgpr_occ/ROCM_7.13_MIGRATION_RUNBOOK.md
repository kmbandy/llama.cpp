# ROCm 7.2.3 → TheRock 7.13 multi-arch migration runbook

**Why (in priority order):**
1. **`s_alloc_vgpr` intrinsic** — installed 7.2.3 clang (`AMD clang 22.0.0git f58b06dce`) lacks `__builtin_amdgcn_s_alloc_vgpr` (tested). LLVM #163951 (2026-02-10) + GlobalISel #192937 (2026-04-20) add it → compiler-native dyn-VGPR from a compute kernel, the potential escape from the hand-asm ~64 TF ceiling toward 300 TF (MAD-305).
2. **gfx1201 SQTT/ATT trace decoder** — `rocprof-trace-decoder` (in 7.13, supports "9000 series" = RDNA4) → RGP-grade instruction-level profiling, which 7.2.3 cannot do at all.
3. **Our merged multi-arch fix** lands us back on a mainline ROCm release.

**Hard constraints (non-negotiable):**
- **NEVER two ROCm active at once** — that cost an entire Linux system. This is a *replacement*, and the old userspace ROCm must be inert/removed before the new one is wired in.
- **Both GPUs stay first-class:** gfx1201 (R9700) **and** gfx1030 (6900XT). Every rebuild uses both arches (`AMDGPU_TARGETS="gfx1201;gfx1030"`, `PYTORCH_ROCM_ARCH="...;gfx1030;...;gfx1201"`). Single-arch is itself a bug.
- **Rollback ready before the first destructive step.** Babysat session, watch a reboot.
- **Displays are NOT ROCm** — the monitors run on the amdgpu *kernel* module + Mesa, independent of the ROCm userspace we're swapping. We do **not** touch the kernel driver or Mesa. (Confirms: this migration is userspace-only; a bad ROCm swap should not, by itself, kill the displays — unlike the MES-hang crash earlier today.)

---

## Phase 0 — Pre-flight (NO destructive action)

- [ ] **0.1 Snapshot the system** — the real rollback. CachyOS is btrfs by default:
  `sudo snapper -c root create -d "pre-rocm713-migration"` (or `limine-snapper`/`timeshift` — **CONFIRM** which snapshot tool is configured: `snapper list-configs`). Verify the snapshot exists and is restorable before proceeding.
- [ ] **0.2 Capture exactly how 7.2.3 is installed** (drives the rollback):
  `pacman -Qo /opt/rocm/bin/clang` → which package owns it; `pacman -Qs rocm hip` → full ROCm package set; OR if installed via `amdgpu-install`, record the exact installer version/args. Save the list to `~/rocm-7.2.3-packages.txt`.
- [ ] **0.3 Baseline-capture the current good state:**
  - `clang --version`, `hipconfig --version`, `rocminfo | grep -E 'Name|gfx'`, `rocprofv3 --version`
  - `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` (record arch list — must regain `gfx1030`+`gfx1201` after)
  - which llama.cpp binary/commit is live, ml8 trainer venv, vllm version, Triton/aiter commits
  - Save to `~/rocm-7.2.3-baseline.txt`.
- [ ] **0.4 Quiesce production:** stop the ml8 trainer, any llama-server, vllm, and **the mneme daemon's GPU consumers** (mneme daemon itself is CPU/DuckDB — fine; but anything holding the GPU must be down so the ROCm swap is clean). Confirm `rocm-smi` shows both GPUs idle, no KFD procs.

## Phase 1 — Acquire + offline-verify TheRock 7.13 (NO destructive action)

- [x] **1.1 RESOLVED — exact 7.13 multi-arch artifact** (TheRock migrated to multi-arch packaging, issue #3323: builds all arches together, splits GPU `.kpack` from host code, select target at install). The `therock-7.13` GitHub release (2026-05-15) has NO GH assets; tarballs live on the CD index. Confirmed file (matches release date):
  `https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-multiarch-7.13.0a20260515.tar.gz` (~11.1 GB, all-arches `.kpack`, extracts to the `/opt/rocm` layout: `bin/ lib/ include/ .kpack/`). Per-family alt: `therock-dist-linux-gfx120X-all-*` + `gfx103X-all-*` (need BOTH → multiarch is simpler). Index: https://rocm.nightlies.amd.com/tarball-multi-arch/
- [ ] **1.2 Extract to a staging prefix** (NOT yet on PATH): `~/therock-7.13/` → `~/therock-7.13/install/`. Staging only — **inert** until Phase 2 (nothing on PATH/ldconfig, so NOT a second *active* ROCm).
- [ ] **1.3 Offline capability gates against the staged tree** (prove it's worth the swap BEFORE removing 7.2.3):
  - **`s_alloc_vgpr`:** `~/therock-7.13/install/lib/llvm/bin/clang -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c <(echo 'void k(){__builtin_amdgcn_s_alloc_vgpr(64);}') -o /tmp/sav.o` → **must compile** (the headline win). NOTE clang lives at `lib/llvm/bin/clang` (current 7.2.3 is `/opt/rocm/lib/llvm/bin/clang`, NOT `/opt/rocm/bin/clang` which does not exist). Confirmed absent on 7.2.3 ("use of undeclared identifier"). If it errors on the staged tree, STOP — reassess artifact.
  - **trace decoder:** `find ~/therock-7.13 -iname '*att*decod*' -o -iname '*trace*decod*'` → decoder lib present.
  - **rocprofv3 + gfx1201:** present in `~/therock-7.13/bin`.
  - **both arches:** the multi-arch build advertises gfx1201 + gfx1030 (check rocBLAS/hipBLASLt arch list or `<prefix>/lib/llvm/.../gfx*` device libs).

> Gate: do NOT proceed to Phase 2 unless 1.3 passes. This is the last fully-reversible checkpoint.

## Phase 2 — The swap (DESTRUCTIVE — snapshot must exist)

- [ ] **2.1 Remove the old userspace ROCm.** Install model CONFIRMED: pacman-managed (stock Arch binaries, Packager Christian Heusel) + ONE local build = `rocblas 7.2.3.r1.g51c8fc4-1` (PR #4781 multiarch, `~/GitHub/rocblas-pkg/PKGBUILD`). Removal: `sudo pacman -Rsc rocm-core` — VERIFIED blast radius (`pactree -r rocm-core`): cascades ONLY into ROCm-family packages, **zero** kernel/Mesa/firmware/display/Vulkan/libdrm. Displays survive. torch/triton/aiter/ml8/vllm are pip (untouched by pacman; rebuilt in Phase 3). Do NOT touch amdgpu kernel module / Mesa / linux-firmware. Confirm `/opt/rocm` is empty and nothing references it on PATH/ldconfig.
  - Rollback assets already staged: cached `.pkg.tar.zst` for 7.2.3 in `/var/cache/pacman/pkg/` (rocm-llvm, rocm-hip-sdk, rocminfo…) + `~/GitHub/rocblas-pkg/PKGBUILD` + the snapper snapshot + `~/rocm-7.2.3-packages.txt`.
- [ ] **2.2 Install TheRock 7.13 as the one ROCm.** Either relocate the staged tree to `/opt/rocm` (symlink `/opt/rocm -> ~/therock-7.13` is cleanest for hardcoded paths), OR set `ROCM_PATH=~/therock-7.13` globally. Pick ONE and make it authoritative.
- [ ] **2.3 Wire env:** `ROCM_PATH`, `PATH=$ROCM_PATH/bin:...`, `LD_LIBRARY_PATH=$ROCM_PATH/lib:...`, `HIP_PATH`. Add an ldconfig entry OR rely on `LD_LIBRARY_PATH` (be consistent — avoid a stale `/opt/rocm` ldconfig pointing at the removed tree). `sudo ldconfig`.
- [ ] **2.4 First sanity:** new shell → `clang --version` (expect 22.x with the intrinsic), `rocminfo | grep gfx` → **both gfx1201 and gfx1030 enumerate**, `rocm-smi` lists both cards. **If a GPU is missing → STOP, investigate before rebuilding anything.**

## Phase 3 — Rebuild the dependent stack (ordered; both arches always)

Rebuild in dependency order; gate each before the next.
- [ ] **3.1 llama.cpp / HIP** — `cmake -B build-hip -DAMDGPU_TARGETS="gfx1201;gfx1030" ...` (per the always-multi-arch principle); rebuild; the MAD-305 spike `build.sh` (gfx1201 asm) too.
- [ ] **3.2 torch** — source build, `PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1201"` (gfx1030 + gfx1201 mandatory — pytorch wheels still omit gfx10xx; source build is the only path), `ROCM_PATH=$ROCM_PATH`, `TMPDIR=~/.cache/pip-tmp` (/tmp is tmpfs). Note the prior flatbuffers-version conflict workaround if it recurs.
- [ ] **3.3 Triton / aiter** — rebuild against new ROCm/LLVM; re-verify the AOT `--target` patch still applies (carried patch). Confirm the fp8-conversion-on-non-fnuz fix is present in the Triton commit.
- [ ] **3.4 ml8 trainer** — reinstall its venv against the rebuilt torch; smoke a 0.8B fp8 step.
- [ ] **3.5 vllm** — rebuild against rebuilt torch (if still in the stack).

## Phase 4 — Verification gates (the migration is "done" only when all pass)

- [ ] **4.1 Capability (the reason we did this):**
  - `__builtin_amdgcn_s_alloc_vgpr` compiles for gfx1201 ✓
  - `rocprofv3 -L` lists gfx1201 counters (FETCH_SIZE etc.) ✓
  - `rocprofv3 --att` on a trivial HIP kernel **decodes** to an instruction view on gfx1201 ✓ (proves the decoder + RDNA4 support end-to-end)
- [ ] **4.2 Both-arch regression (the standing principle):**
  - `torch.cuda.get_arch_list()` contains gfx1030 **and** gfx1201
  - llama.cpp: a short gen on **each** GPU (gfx1201 and gfx1030) is correct
  - ml8: one fp8 train step + one inference pass, oracle/PPL sane vs the 7.2.3 baseline (0.3)
- [ ] **4.3 No-regression spot checks** vs `~/rocm-7.2.3-baseline.txt`: turbo4_fp8 PPL/NIAH still in range; the mixed-arch dual-GPU llama.cpp split still works.

## Phase 5 — Rollback (if any gate fails irrecoverably)

- [ ] **5.1 Boot into / restore the Phase-0 snapshot** (snapper rollback → reboot). This restores 7.2.3 + the entire working stack atomically. This is why 0.1 is non-negotiable.
- [ ] **5.2 If snapshot restore is unavailable:** reinstall the exact `~/rocm-7.2.3-packages.txt` set, restore env, rebuild stack against 7.2.3. (Slower; snapshot is strongly preferred.)

---

## CONFIRM-at-execution (unknowns to nail before/while running)

- Exact 7.13 multi-arch artifact name + URL + install mechanism (tarball vs pip).
- Which snapshot tool is live (snapper/limine/timeshift) and that a restore actually works.
- Exact current ROCm package ownership (pacman vs amdgpu-install) for clean removal + rollback.
- Whether to install at `/opt/rocm` (symlink) or a `ROCM_PATH` prefix — pick one, be consistent.
- POST-MIGRATION (the actual MAD-305 work, separate): does `s_alloc_vgpr` work from `amdgpu_kernel`/amdhsa on gfx1201, and does it self-arm `DYN_VGPR_EN` or still need the proven PM4 RSRC2-bit6 arming alongside? (The whole point of the upgrade — validate first thing after 4.1 passes.)
