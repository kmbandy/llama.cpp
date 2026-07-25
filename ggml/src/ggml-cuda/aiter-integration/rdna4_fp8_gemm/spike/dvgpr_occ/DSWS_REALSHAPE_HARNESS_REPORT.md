# DSWS real-shape harness rebuild report

No GPU command was run while building or validating this harness. All acceptance work read the existing `rs_*.log` and `cfa_*.log` files under `/home/kmbandy/dsws_gpu_logs`.

## 1. TF-from-ticks derivation

The deleted sweep contributed only its 33-line shape inventory. The new timing relationship comes directly from `occ_dispatch.cpp`:

- `sumSpan` starts at zero and is documented as the sum of `occ[3]-occ[2]` over chunks (`occ_dispatch.cpp:2023`).
- Every chunk resets the entry slot to the `0xFFFFFFFF` sentinel (`occ_dispatch.cpp:2049`). After completion, the host adds `ge-gs`, with explicit 32-bit wrap handling, to `sumSpan` and increments `spanChunks` (`occ_dispatch.cpp:2162-2170`). Host-side yields are therefore outside the measured interval.
- A completed whole-GEMM repetition increments `repsDone`; its span is the difference between the new `sumSpan` and the value at repetition entry (`occ_dispatch.cpp:2178-2180`).
- The WORK-EXACT expected count is `G * TOTAL_super * repsDone` (`occ_dispatch.cpp:2255-2259`). The harness reverses this equality to derive the repetition count from full-precision integers, then cross-checks the optional SUSTAINED line. It does not accept a free-standing rendered repetition count.
- The dispatcher defines `workAll = 2 * Mo * No * Ko * repsDone` and `TF = workAll * freq_hz / sumSpan / 1e12` (`occ_dispatch.cpp:2435-2443`). Split K does not multiply work because its segments reduce the same K range.

Thus, for the emitted kernel geometry:

```text
padded_TF = 2 * padded_M * N * K * repetitions * tick_hz / summed_ticks / 1e12
real_TF   = padded_TF * real_M / padded_M
```

The second line removes work introduced by M padding. It is algebraically the same as replacing `padded_M` with `real_M` in the numerator. The harness matches the recovered real shape against the emitted super-tile M before applying this correction.

Concrete source-to-log check: `/home/kmbandy/dsws_gpu_logs/cfa_ml8_dense_ffn_gate_up_M2048_145316.log` emits padded geometry `2112x9216x2560`, `5194220` ticks, `7` chunks, and WORK-EXACT count `190080`. With `G=6` and `TOTAL_super=31680`, the derived repetition count is exactly one. At the emitted 100 MHz tick rate, the arithmetic result is 1.918606895 padded-M TFLOP/s, which correctly rounds to the dispatcher's `TF=1.9`. Scaling by `2048/2112` gives 1.860467292 real-FLOP-corrected TFLOP/s. Both results retain that exact log path in JSON and in the table.

The source measures `freq_hz` over a 200 ms host interval but prints it only to two decimals in the header and zero decimals on the throughput line (`occ_dispatch.cpp:3905-3919`, `occ_dispatch.cpp:2447-2448`). All 114 archived headers say `100.00 MHz`. The harness uses the two-decimal header, requires the zero-decimal line to agree, and records this limited archived precision. This is a remaining input-precision limitation, although it is far smaller than one-decimal TF rendering error.

## 2. Self-validation and demonstrated rejection

A PASS record is constructed only after all of these checks succeed:

1. Exactly one clock, config, oracle-shape, dispatch-geometry, completion, WORK-EXACT, throughput, oracle, and final CLEAN emission exists.
2. Header, dispatch, and timing geometries agree exactly; padded M is divisible by the emitted super-tile M.
3. Completion has `occ[0]=0`; the oracle has `ok>0` and `bad=0`; no abort, timeout, refuse, INCOMPLETE, WORK-INEXACT, dirty canary, or bad-oracle marker exists.
4. Repetitions are derived as `WORK_EXACT / (G*TOTAL_super)`. SUSTAINED repetitions, when present, must agree. Timed chunks must equal `chunks_per_rep * repetitions`.
5. The full-precision tick arithmetic must be within 0.05 TFLOP/s of the kernel's one-decimal `TF` rendering. The independently rendered percent-of-peak field must also agree within one-decimal rounding tolerance. A SUSTAINED mean is checked as a third rounded rendering when present.
6. The geometry must map to the recovered real-shape inventory so that a real-M correction is not guessed.

Failure records always contain `throughput_tflops: null`; the human table displays `-` in both throughput columns.

Two deliberately invalid fixtures prove the guards execute:

- `acceptance_corrupt_ticks.txt` changes the otherwise coherent fixture's span from 100000 to 10000 ticks while leaving the rendered TF unchanged.
- `acceptance_bad_oracle.txt` contains `bad=1` even though its final text claims CLEAN.

The actual offline command exited 1 and produced:

```text
inputs=2 pass=0 fail=2 other=0
FAIL ... acceptance_bad_oracle.txt :: ORACLE_FAILED: require ok>0 and bad=0
FAIL ... acceptance_corrupt_ticks.txt :: SELF_VALIDATION_MISMATCH: ticks/geometry/reps disagree with rendered TF
```

Neither rejection row contains a throughput number. The captured machine and human outputs are `acceptance_corruption_rejections.json` and `acceptance_corruption_rejections.txt`.

## 3. Offline acceptance test over all 114 logs

Command used:

```sh
python3 dsws_realshape_bench.py offline \
  --log-dir /home/kmbandy/dsws_gpu_logs \
  --glob 'rs_*.log' --glob 'cfa_*.log' \
  --json acceptance_114_logs.json \
  --table acceptance_114_logs.txt
```

The command exited 1 because a fail-closed batch returns nonzero when any input is rejected. It still emitted one record for every input:

```text
inputs=114 pass=91 fail=23 other=0
```

Failure breakdown:

| Count | Reason |
|---:|---|
| 9 | Explicit `WORK-INEXACT` |
| 8 | No WORK-EXACT emission, so work completion is unproved |
| 3 | Diagnostic geometry is not one of the recovered real shapes |
| 2 | Chunk exceeded its cap and the run aborted/was INCOMPLETE |
| 1 | No oracle emission |

All 91 accepted logs passed the tick-derived versus rounded-TF check. No unmodified archive log failed specifically on that check. This means the guard is compatible with the real format; the corruption run above proves it is not vacuous.

Notable accepted rows:

| Provenance | Reps | Spread | Padded-M TF | Real-corrected TF | Note |
|---|---:|---:|---:|---:|---|
| `/home/kmbandy/dsws_gpu_logs/cfa_ml8_dense_ffn_gate_up_M2048_145316.log` | 1 | n/a | 1.918607 | 1.860467 | Single shot; 2112 -> 2048 correction |
| `/home/kmbandy/dsws_gpu_logs/cfa_ml8_moe_ffn_gate_up_M64_145720.log` | 137 | 29.7% | 0.034636 | 0.023091 | Kernel printed `TF=0.0`; tick derivation retains useful precision and removes 96 -> 64 padding |
| `/home/kmbandy/dsws_gpu_logs/cfa_ml8_dense_ffn_down_M2048_145418.log` | 5 | 11.0% | 4.521655 | 4.384635 | Padded and real figures both reported |
| `/home/kmbandy/dsws_gpu_logs/rs_ml8_dense_ffn_gate_up_M2048_093331.log` | 12 | 97.3% | 0.705712 | 0.705712 | Arithmetic-valid but not a stable performance estimate |

The JSON also contains the full 33-shape inventory. Under the live config, 28 shapes are geometrically supported and five are explicitly UNSUPPORTED due to `N%64`. Among the archived logs, `mlmf_router_MLP 4096x256x256` has no accepted run; both matching logs are WORK-INEXACT. It is shown rather than omitted.

## 4. Suspicious or untrustworthy archived measurements

- Nine of the 91 accepted rows have `reps=1`, so they have no repeatability evidence. Accepted repetition counts range from 1 to 141.
- Three accepted rows have spread above 50%: 98.1% in `rs_ml8_dense_ffn_gate_up_M512_093353.log`, 97.3% in `rs_ml8_dense_ffn_gate_up_M2048_093331.log`, and 64.5% in `rs_mlmf_mamba_out_proj_M4096_100612.log`. These pass correctness and arithmetic checks but should not be quoted as stable benchmark results.
- Several current small-shape runs still have large spread: 33.8% for `cfa_ml8_moe_attn_q_M64_145823.log` and 29.7% for `cfa_ml8_moe_ffn_gate_up_M64_145720.log`.
- The archive mixes different G/FM/super-tile configurations, old and new chunking behavior, and diagnostic builds such as PHIST, PHASE, RESVPROBE, and BNDPROBE. Passing the parser means one log is internally coherent; it does not make unlike builds comparable.
- Fifty-six accepted logs execute padded M. The correction is material for small M: 64 -> 96 scales results by 2/3, while 512 -> 576 scales by 8/9. Padding must not be credited to the real workload.
- The oracle is sampled and usually uses the LOOSE split-K tolerance. For example, `cfa_ml8_moe_ffn_gate_up_M64_145720.log` checks only 1 of 8 tiles. WORK-EXACT closes the dropped-work hole but does not turn the sampled numerical oracle into a full output check.
- The 23 rejected logs show that archived stdout cannot be treated as a throughput table merely because a TF line exists or a filename resembles a real shape. Nine explicitly dropped or duplicated work, and eight older logs lack the gate needed to prove otherwise.

## Harness files and use

- `dsws_realshape_bench.py`: standalone offline/live harness and JSON/table writer.
- `test_dsws_realshape_bench.py`: offline-only unit tests.
- `acceptance_114_logs.json` and `acceptance_114_logs.txt`: complete 114-log acceptance result.
- `acceptance_corrupt_ticks.txt` and `acceptance_bad_oracle.txt`: deliberate rejection fixtures.
- `acceptance_corruption_rejections.json` and `acceptance_corruption_rejections.txt`: captured guard demonstration.

Live mode is present for a future human-approved run but was not executed. It dispatches each supported shape with one separate `gpu_run.sh` process. A nonzero return sets the halt condition immediately; all remaining inventory entries are emitted as NOT_RUN without further dispatch. Unsupported shapes are emitted with exact modular-geometry reasons.
