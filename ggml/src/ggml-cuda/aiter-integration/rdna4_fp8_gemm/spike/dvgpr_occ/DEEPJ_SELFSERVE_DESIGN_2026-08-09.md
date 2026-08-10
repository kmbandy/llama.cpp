# Deep-J SELFSERVE design - 2026-08-09

Status: offline implementation and static review only. No GPU dispatch was run.

## Design gate: POOL_N=1 cannot carry a live J-window

The live path is the `SELFSERVE=1 DECENTASN=1 ROLEFLOW=0` path from
`.Lflow_da_ss_decode` to `.Lflow_da_ss_rowblk` and `.Lflow_da_ss_rows_done`. The required
ownership choice is (a), a lead reservation that advances `ASSIGN_HEAD` by the whole window in
one CAS. Choice (b), reserving by `+1` and letting the carrier coast over followers, is rejected:
followers become visible to other waves and the single physical slot can be recycled before the
carrier has finished its private ACC window.

For a field with `n_kseg` real segments, the lead CAS must reserve `L = min(J, n_kseg - ksi)`
real indices and publish the corresponding slot generations. The required boundary arithmetic is:

```text
n_kseg=36, J=8: leads 0,8,16,24,32 -> lengths 8,8,8,8,4
n_kseg=18, J=8: leads 0,8,16    -> lengths 8,8,2
```

The current frontier has one `ASSIGN_HEAD`, one `STAGE_HEAD`, one `DRAIN_HEAD`, and one physical
control slot when `POOL_N=1`. A single CAS can advance the logical head from `r` to `r+L`, but
the slot can hold only one `SL_STI/SL_GEN/SL_RBNEXT/SL_RBDONE` record. Publishing only the lead
record leaves the skipped follower generations absent from the coupled stage/drain protocol;
publishing them requires either multiple physical slots or a new per-window ownership record and
new stage/drain completion rules. The phantom tail makes this stricter: `r+L` must not be
published as a real `ksi` when `r+L` reaches the field boundary, while the field stride must
advance to the next group. For example, the final `32 -> 36` lead in the first case is a four-item
window, not an eight-item window and not a phantom reservation.

Therefore this sweep stops at the design gate. Implementing (a) with `POOL_N=1` would weaken the
frontier protocol; implementing (b) would violate window ownership. No edit to
`occ_kernel_dsws_flow.s` is made, and no J=2/4/8 build is presented as live-path engagement.

## Scope

The deep-J path is enabled only for `SELFSERVE=1 DECENTASN=1 POOL_N=1`. A lead claim owns one
rowblk and its physical slot. The carrier loads the lead and the following ksi values directly from
L2, keeps all products in `ACC`, flushes once, then publishes one `SL_RBDONE` increment and shrinks.
The effective reservation window is one while this path is active. This is intentional: reserving a
later logical index would overwrite the only physical slot before the carrier flushes.

The carrier does not enter `.Lflow_jwait`. It advances its private logical cursor without waiting for
DRAIN or STAGE. `deadman_progress` is emitted after every WMMA segment, so the stall watchdog sees a
healthy carrier as making progress. The final window length is `min(JDEPTH, n_kseg - ksi)`; this
handles a short tail without requiring `JDEPTH | n_kseg`.

## Correctness requirements

### LEAD-GATE

The existing pre-grow gate reads the candidate `SL_STI` and admits only `ksi & (JDEPTH-1) == 0`.
The existing DECENTASN post-grow re-check repeats the test after the grow and after fresh DRAIN/STAGE
slot derivation. Both remain in place. The BATCHLDS=0 arms were not reordered by this change; their
original scalar-load sequence remains the default path.

### J-window claimability

With `POOL_N=1`, the carrier never asks the ring to stage or reserve its J-1 followers. Its cursor is
private, derived from the lead generation, and the self-load address uses that cursor's ksi. The next
claim cannot be made until the lead is flushed and DRAIN advances, because the effective reservation
window is one. Therefore the carrier never waits on DRAIN for an unflushed segment and cannot create a
head/frontier cycle. It also cannot spin fat-and-idle unboundedly: there is no J-wait loop; each segment
updates the forward-progress stamp and the bounded window ends at J or the runtime tail.

For a physical pool greater than one, this exact implementation refuses the combination at assembly
time. A separate multi-slot design would need a lead-slot pin and a distinct completion protocol.

### DRAIN never passes unflushed data

`SL_RBDONE` means flushed rowblocks, not computed rowblocks. In deep-J SELFSERVE it is not incremented
for a middle segment. The only increment is in the existing shared post-flush path, after the banked
`ds_add_f32` wait. Thus DRAIN cannot pass the lead slot while ACC still contains any part of its J-window.
`TILEDONE` is incremented by the actual window length, including a short tail, and its C-store gate
remains after the flush.

### Non-dividing tails

The self-serve path handles the remainder dynamically. For example, `n_kseg=18,J=8` executes windows
8, 8, and 2. The old divisible-K fail-safe remains for the non-self-serve deep-J path.

## Duty invariant

The standing invariant is `JDEPTH*SEGK <= DUTY_KMAX`, with `DUTY_KMAX=256`. It describes peak duty,
not LDS bytes: deep-J keeps ACC live across segments and can destroy the staggered-wave advantage.
The assembler guard and default remain unchanged. `DUTY_OVERRIDE=1` is required for the requested
measurement cells whose product exceeds 256.

## Static verification before the bring-up failure

The default build assembled, but its SHA was `2fbd3672d730047471ec3159035f0348a0ebf266c8c5d1f2dd1226f6931430ce`,
not the registered canonical `58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c`.
This is a hard gate failure; no silicon result should be attributed to this patch until the existing
working-tree byte-identity discrepancy is resolved.

Deep-J `SEGK=256`, `DUTY_OVERRIDE=1`, config-of-record geometry:

| J | .text bytes | SHA-256 | LDS | spills |
|---:|---:|---|---:|---|
| 2 | 29016 | `600c566123c7679d598af698f9faa68c20bb52fafe77b9e3f76dfc02e7a6cd0c` | 34304 | none observed |
| 4 | 29016 | `4eb38b64499e7fc04bb93357d071fa124bf3bc73be2019b5cbcc4d59344f6ab8` | 34304 | none observed |
| 8 | 29016 | `b9707a26621e57f654c18952c9e861c672bb3acd2230c41d8ae44804be474613` | 34304 | none observed |

The object reports `occ_kernel` with no scratch symbols. SGPR allocation and LDS are unchanged by
the deep-J source arms in this static pass; silicon occupancy remains unverified.

## Registered predictions

For the requested SEGK=256 comparison, the pre-registered model is

`rep_ms = 1.85 + 3.72/J`

This is a design prediction only: the POOL_N=1 ownership gate above prevents the live J>1 path
from being built or dispatched in this sweep.

The earlier exploratory model for `K=9216` was

`t_ms = 1.64 + 0.0057*n_kseg + 0.1034*n_kseg/J`.

I pre-register no correction to the coefficients for J=1. For J>1 I expect a positive serial-window
correction, because the pool-1 safety protocol removes reservation overlap; use
`+0.02*(J-1)` ms as an uncertainty band, not as a fitted prediction. At `SEGK=512`, `n_kseg=18` and
J=8 uses a two-segment tail, so the idealized formula should be treated as an upper-level comparison,
not an exact tail model.

## Silicon run matrix (future, supervised only)

Do not run until the byte gate is repaired. Use `SSWIN=32`, `DECENTASN=1 SELFSERVE=1 POOL_N=1
WAVES=6 G=8 FM=2 FN=4 ACC_N=4`, `DUTY_OVERRIDE=1` for J>1, and choose `DSWS2_REPS` per cell so
the expected busy interval is about 0.7 s. Every row gets stride-8 oracle output and perf output.

```sh
# First bring-up: full stride, J=2 only
JDEPTH=2 DUTY_OVERRIDE=1 SSWIN=32 DSWS2_REPS=<scaled> DSWS2_STRIDE=1 <normal DSWS2 host env>

# Matrix: each row is a separate supervised dispatch; use stride 8 for oracle/perf rows
for J in 1 2 4 8; do
  for SEGK in 64 128 256 512; do
    JDEPTH=$J SEGK=$SEGK DUTY_OVERRIDE=$([ "$J" -gt 1 ] && echo 1 || echo 0) \
      SSWIN=32 DSWS2_REPS=<scaled> DSWS2_STRIDE=8 <normal DSWS2 host env>
  done
done
```

Record the exact host command, oracle counts, TF, reset/deadman status, and stride-1 bring-up result per cell.

## Guard-condition fix and static gates

The three SELFSERVE/DECENTASN window-length sites in the shared flush/drain path were guarded only
by `SELFSERVE && DECENTASN`. That selected the `s91` arm for the default J=1 build, even though
`s91` is initialized only in the JDEPTH>1 carrier loop. The default could therefore read an
uninitialized counter, and its object differed from the registered canonical binary.

Each guard now also requires `JDEPTH > 1`. J=1 uses the existing inline `JDEPTH` operand, which
assembles as the constant 1 and restores byte identity. Deep-J SELFSERVE keeps the actual-window
length in `s91`. The other `s91` uses remain under their existing valid guards.

| Build | .text bytes | SHA-256 | fail | spills |
|---|---:|---|---:|---|
| default, SEGK=256, J=1 | 28852 | `58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c` | 0 | none |
| SELFSERVE, SEGK=256, J=2 | 29016 | `600c566123c7679d598af698f9faa68c20bb52fafe77b9e3f76dfc02e7a6cd0c` | 0 | none |
| SELFSERVE, SEGK=256, J=4 | 29016 | `4eb38b64499e7fc04bb93357d071fa124bf3bc73be2019b5cbcc4d59344f6ab8` | 0 | none |
| SELFSERVE, SEGK=256, J=8 | 29016 | `b9707a26621e57f654c18952c9e861c672bb3acd2230c41d8ae44804be474613` | 0 | none |
| SELFSERVE, SEGK=512, J=8 | 33948 | `8d7c21c3c7cd4254b728079752ee2aa266b0cbffe3466e4bd0c4d365a71dad72` | 0 | none |

## J=2 silicon failure diagnosis and fix

The J=2 bring-up failed in the completion choreography, not in scheduling, LDS zeroing, or
operand address generation. The decisive fact is the dispatch path selected by the recorded
build (`SELFSERVE=1 DECENTASN=1 DSWS2_ROLEFLOW=0`): a successful reservation branches at
`.Lflow_da_ss_decode` to `.Lflow_da_ss_rowblk`, computes exactly one `ksi`, and exits through
`.Lflow_da_ss_rows_done`. It does not enter `.Lflow_compute` or `.Lflow_jloop`.

The J=2 object confirms this. At `0x3458` it derives `s31 = s46 & s67`, then at `0x3460` and
`0x3494` uses that value for the B and A K offsets, but there is no cursor increment and no
second window iteration on this live path. The separate fallback J-loop contains the expected
`s_add_u32 s46, s46, 1` and re-derivation of `s31`, but that path is not reached by the bring-up.

The live carry-through path reaches the shared completer with `s91` uninitialized. The deep-J
completer then uses `s91` at the `TILEDONE` fetch-add and target arithmetic. Thus every row-block
segment can be computed, counted, and flushed into LDS while completion advances by an invalid
amount. This matches the log: WORK-EXACT, sound claims, no carrier wait, all tiles wrong, and no
boundary or canary fault. It is hypothesis 2: the completion target operand did not describe the
actual one-segment flush. Hypothesis 1 is ruled out because `zero_banks` still runs only at the
drained tile boundary; hypothesis 3 is ruled out because each reservation emits exactly one
distinct segment; hypothesis 4 is ruled out for the live path because it has no second iteration.

The minimal fix initializes `s91=1` at `.Lflow_da_ss_rows_done` for deep-J self-serve. That value
matches the actual carry-through flush cardinality and leaves the shared `TILEDONE`/C-store
protocol unchanged. The default J=1 arm is not assembled from this edit.

The J=2 retest must report `bad=0` and also verify `computed == G*TOTAL_super`,
`occ[96] == GROUPS*TOTAL_super*reps`, `occ[97] == 0`, clean live-wave completion, and a new
flush-cardinality check: for deep-J self-serve, the sum of the per-window completion increments
must equal the number of distinct claimed reservations (`TOTAL_super` per group), not
`JDEPTH*TOTAL_super`. A recurrence test should run J=2 twice over the same dispatch and confirm
that `TILEDONE` returns to zero at each boundary and never exceeds its target.

### Post-fix offline build matrix

All arms below used `WAVES=6 G=8 FM=2 FN=4 ACC_N=4 POOL_N=1`, with `DUTY_OVERRIDE=1` and
`DSWS_ALLOW_NONSTD=1` for J>1. Each had 0 scratch/spill references in `llvm-objdump`.

| J | SEGK | .text bytes | SHA-256 | LDS |
|---:|---:|---:|---|---:|
| 2 | 256 | 29020 | `35562efeefa69179b8e1f68e2560676af4861c8ba6382895b51bd8937dff0af1` | 34304 |
| 4 | 256 | 29020 | `c0b8151cccd4b51879ffb02960c4cd8e7fa9ed311ab3d908fccd3322215befce` | 34304 |
| 8 | 256 | 29020 | `c35003d642c246f50bf62ada9421e8e64b561c1be7a5922e8112d23e630220b3` | 34304 |
| 2 | 512 | 33952 | `81f1ca668168c77adf2fdaa116c70271c62890f7506ec73f9e343fe335d3a388` | 34304 |
| 4 | 512 | 33952 | `fed934a5e8b986225122f3e9269bb4ed0aedc25fa3b2f47458e1e6185604cf68` | 34304 |
| 8 | 512 | 33952 | `dffb5420ab76b9d3a9e0463503b522355824dcc3dc213fc34ccdf5b14eff1818` | 34304 |

The bare build was rebuilt last and restored the registered canonical artifact: 28852 bytes,
SHA-256 `58e965a46f3e162d870c86ecafbed5c4c25579dea12d173648b06fc163ef814c`, LDS 34304, zero
spill references. Its disassembly has zero `s91`, J-loop, and deep-J references.
