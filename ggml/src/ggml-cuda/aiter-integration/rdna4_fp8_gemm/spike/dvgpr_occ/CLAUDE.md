# dvgpr_occ — GPU DISPATCH RULES (NON-NEGOTIABLE)

The R9700 drives kmbandy's displays AND hosts other agents' live sessions.
**A brick costs ~1M tokens of other people's context, not just a reboot.**
These rules held for two days. On 2026-07-14 I stopped following them and bricked the
box 6 times in one afternoon. Every brick was a rule I already knew and skipped.

## THE SIX RULES

1. **ONE DISPATCH PER GREENLIGHT. NEVER A BATCH.**
   A greenlight authorizes exactly one `./occ_dispatch` run. Sweeps/loops of 2+ runs
   need a greenlight *per run*. "yep go ahead" on a 3-run sweep is NOT 3 authorizations.

2. **A NEW OR CHANGED KERNEL GETS ONE RUN, THEN STOP AND REPORT.**
   If the kernel source changed at all since the last green run, the next dispatch is a
   *bring-up* run: one dispatch, known-good geometry, then stop. Do not chain a sweep
   onto code that has never executed.

3. **A HANG / TIMEOUT / INCOMPLETE IS A FULL STOP.**
   Not "try the next variant." Not "rerun with the probe on." STOP, report, go offline.

4. **NEVER RAISE `DEADMAN_TICKS`. IT IS THE ANTI-BRICK GUARD, NOT A TUNING KNOB.**
   0.5s. It converts a wedged WG into a clean retire before MES gives up on REMOVE_QUEUE
   and the driver falls back to MODE1. If the deadman is false-killing healthy waves, the
   bug is a MISSING `deadman_progress` SITE — fix that, never the threshold.
   (Raising it to 10s on 2026-07-14 = 3 bricks.)

5. **NOTHING NEW IN THE HOT PATH THAT TOUCHES THE MESSAGE BUS OR EMITS STORES.**
   - `s_sendmsg_rtn` (RTC read) spam from coast waves BRICKS. `deadman_check` throttles it
     1-in-64 for exactly this reason — the comment is right there. (DUTYPROBE ignored it = brick.)
   - A store in flight during `s_alloc_vgpr` corrupts the register file
     (S_ALLOC_VGPR = `WaitIdleExceptStoreCnt()` — it does NOT drain stores). Drain first.

6. **MAX WORK OFFLINE FIRST.** Assemble, disasm, static-check, compute the LDS/VGPR
   algebra, and predict the outcome IN WRITING before asking for silicon.

7. **A KERNEL CAN KILL THE DESKTOP WITHOUT BRICKING THE CARD.**
   The R9700 drives the displays off the same HBM bus. A kernel that SATURATES MEMORY
   BANDWIDTH starves the compositor -> Hyprland watchdog -> safe mode -> the session dies,
   even though the GPU never resets and the kernel is making progress. Every other rule here
   watches for MODE1/hung queues; this one passes all of them.
   (2026-07-14: NTLOAD=1 marked the A/B staging loads th:TH_LOAD_NT. That threw away their
   L2 reuse and turned a 1.14 GB *cached* stream into 1.14 GB of raw HBM traffic. Kernel only
   ~28% slower -- desktop died.)
   => ANY change that could raise HBM traffic (cache hints, bypassing L2/LDS, wider fetch,
      more WGs, NOFEED-style ablations) gets a SMALL chunk first (ML8_COOP_CHUNK<=1024) and a
      SHORT shape. Do not run it at full 64-WG / 34816-tile scale on the first try.

## BEFORE EVERY DISPATCH, SAY THIS OUT LOUD

- What changed in the kernel since the last GREEN run? (if anything → rule 2 applies)
- Which rule could this violate?
- What is the expected `computed` value, so a short count is caught? (work-exactness check)
- Is `DEADMAN=1` and `DEADMAN_TICKS` unset/0.5s?

## USE THE WRAPPER

Dispatch via `./gpu_run.sh <logname> -- <env...> ./occ_dispatch --dsws2`.
It enforces: single run, real-disk logging, post-run journal capture, and it **refuses to
run if the previous run hung** (clear with `rm .gpu_last_hang` only after a human says so).

## MEASUREMENT RULES (separate from safety, but they cost days)

- **FEED IT.** No throughput verdict from <1s of steady state.
- **Check work-exactness.** `computed` must equal `TOTAL_super * ACC_N`. A short count means
  work was silently DROPPED — and dropping work makes TF look BETTER. (J=64 read 31.5 TF while
  losing 34% of its work; the true number was 21.0.)
- **Probes lie.** PHASEPROBE = 44x slow. FORENSICS = 62% slow. Never quote TF from a probe build.
- **Grep for the CALL SITE, not the definition.** Five separate counters on this kernel have
  read 0 because they were never wired (STINSTR_FEED, NOCFLUSH, CSTORE, DIAG, SPIN[]).
- **`~/dsws_gpu_logs` is real disk** — it survives a brick. tmpfs does not.
