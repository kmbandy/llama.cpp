# Morning pickup — weight paging, 2026-07-25

Written at logoff 2026-07-24. **Read this instead of re-deriving anything.** Every
number below is measured, with the command that produced it. Everything marked
UNVERIFIED is genuinely unverified — do not promote it to fact.

---

## 0. TL;DR — where we got to

**DS4-Flash decode went 1.736 -> 3.570 tok/s (+106%) tonight**, with wikitext
perplexity **identical** (1.9007 +/- 0.07421). That is one machine, 2 GPUs, victim
RAM tier only, **no prefetch, no MTP speculation, no tuning**.

The gain came from two defects that were **not in the pager's logic**:

1. O_DIRECT was aligned to **512** (the NVMe's `logical_block_size`) when the
   authority is the **filesystem** — btrfs `f_bsize` = **4096**.
2. `/home` is mounted `compress=zstd:1` and the model had compressed extents;
   O_DIRECT cannot be served from one, so btrfs read+decompressed whole clusters.

Read amplification went **2.49x -> 1.011x**. The drive was never the bottleneck: it
was already delivering 3.88 GB/s while we asked for 2.5x more data than we needed.

**The next single best win is a ~50-line change**: see §3 Task 1.

---

## 1. State at logoff

- Repo `~/GitHub/llama.cpp` on **mad-lab-main**, branch `master`, tip **4fcc44dba**.
  **45 commits ahead of origin, NOT pushed.**
- **Board claims: all released.** Cards verified empty first (only kmbandy's
  `llama-router.service` remains, 0 VRAM). RAM 11 GB available.
- **kmbandy's uncommitted work is untouched** and must stay that way:
  `common/arg.cpp`, `tools/server/server-models.{cpp,h}`,
  `docs/examples/router-fleet-main.ini`, and the DSWS spike area under
  `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`.
  A DSWS session is active in that tree — its `DSWS_TESTING_LOG.md` changed under
  us mid-session, which is normal. **Never `git checkout/restore/stash/reset`,
  never `git add -A` or `commit -a`.**
- `~/models/ds4/*.gguf` were **rewritten uncompressed** (`chattr +C`, 0 encoded
  extents), each sha256-verified before its original was removed. Note `+C` also
  disables btrfs checksums on those files.
- No RPC servers running on mad-lab-2026 (stopped and verified).

## 2. Commits landed today

| commit | what |
|---|---|
| `2de864288` | RPC op-count tripwire pinned to `GGML_OP_COUNT == 105` (+ patch ver 4.0.2) |
| `6a1dcfe0d` | soft host-prefetch policy + HostTier on the P2P ensure path |
| `09c5636b2` | multi-device resident islands |
| `09221a91c` | `island_buft` param on `build_router_overrides` + 2 unit tests |
| `a0460ed53` | `--weight-paging-ffn-island-device` flag + params field |
| `447744bb5` | FFN-island device selection, VRAM preflight, wiring, logging |
| `851fc16f9` | HOST-path phase timings named for what they measure |
| `bf80ccff4` | transport identity line + storage-only `ensure_batch_gb_s` + counter de-contamination |
| `547f94bc8` | achieved-concurrency (in-flight) counter + 3 TDD tests |
| `775e48d62` | **HostTier LRU O(n) -> O(1)** (770 ms -> 1.5 ms regression test) |
| `9110d0ec6` | **direct promotion-vs-fresh H2D measurement** |
| `a9f2c139b` | **O_DIRECT alignment -> filesystem block size**, buffer sizing, EOF clamp |
| docs | specs/plans/roadmap + retractions (see §6) |

## 3. THE MORNING WORK — in order, with everything needed

### Task 1 — HostTier zero-copy promotion  ← DO THIS FIRST

**Why:** the biggest measured win available, and it is small.

`HostTier::lookup` at **`src/weight-pager/wp-host-tier.cpp:213`** does:

```
std::memcpy(dst_bytes, arena_ + it->second.offset, n);
```

a **full 4.25 MB page copy on the eval thread, inside the mutex**, per hit.
Measured: 3135 hits x 4.456 MB = **13.97 GB of RAM->RAM copy per run**, costing
**1.71 s** (= 8.2 GB/s, exactly single-threaded memcpy), reproducible to 1.2%.

The arena is **already pinned `hipHostMalloc` memory** — it can be the H2D source
directly, making the copy pure waste.

**Measured ledger for the 4 GB victim tier (settled, 3 rounds):**

| | today | with zero-copy |
|---|---|---|
| read-wait saved | −2387 ms | −2387 ms |
| lookup memcpy | **+1710 ms** | ~0 |
| h2d | +187 ms | +187 ms |
| **net** | **−490 ms (a wash)** | **~−2200 ms (~+6% end-to-end)** |

**THE CONSTRAINT — this is not a deletion.** The copy exists deliberately so the
worker can reclaim arena slots immediately after `lookup` returns. Removing it
means the arena slot must stay valid until the async H2D completes. So the design
question to settle first: **pin/borrow the slot across the H2D, or refcount it.**
Needs a short spec, then a builder task.

**Also relevant:** promotion H2D itself is **healthy** — measured directly at
**0.223 ms per 4.25 MB page (~20 GB/s)**, replicating to 0.16%. Do not go looking
for a promotion bandwidth problem; there isn't one.

### Task 2 — attribute the 2.49x amplification (cheap, 1 GPU run)

Alignment and btrfs compression were both fixed before any post-fix measurement,
so we do not know the split. Cheapest test: `chattr -C` a **copy** of one shard,
rewrite it so it is compressed again, and run one arm against it with the
alignment fix still in. Tells us whether compression alone accounted for much.
Worth doing because it tells us whether other machines/filesystems need the same
ops treatment or just the code fix.

### Task 3 — why does P2P not exist?

`file_io_->direct_to_device()` is **false** on this config, so with
`WP_ENSURE_BATCH_HOST` unset the ladder falls past io_uring/P2P all the way to the
per-page **serial** path. Consequences: the HostTier-on-P2P integration in
`6a1dcfe0d` **has never executed**, and `WP_IOWQ_MAX_WORKERS` is moot. Read
`create_file_io` / `create_host_file_io` in `wp-file-io.cpp` (~lines 736-824) and
find which rung fails and why. May be a quick win now that reads are 1x.

### Task 4 — the distributed split (roadmap P2)

Unchanged in design; see `docs/dev/2026-07-24-distributed-paging-roadmap.md` and
decision `cbb7417d`. **Sizing per kmbandy's ruling: 8 GB RAM tier per machine**
(he will kill superfluous services to make room — his call, do not re-litigate).
So: main 48 GB VRAM + 8 = **56 GB**, 2026 16 GB VRAM + 8 = **24 GB**, ≈ **70/30**,
about **13 of 44 DS4 layers** on 2026 → ~45 GB shard there, **~72 GB for GLM IQ2**.
Blocker remains: llama.cpp has no pipeline parallelism and RPC cannot express it
(the pager lives in the client; `ggml-rpc-server` has no model or catalog).

### Task 5 — GLM-5.2

Still blocked on NVMe space (kmbandy owns) and the mixed-quant size-class pool.
**Open question worth measuring early:** DS4 is 256 experts / 6 used with only
moderate routing skew (top 10% of instances = 41%), which is close to worst-case
for any cache — so tonight's tier numbers are plausibly **pessimistic** for GLM. A
coarser or more concentrated MoE should hit far more often. GLM's expert count is
read from GGUF metadata at load time (`src/models/glm4-moe.cpp:9-11`) and the
weights are not downloaded, so its concentration is **UNMEASURED**. Capture a
routing trace as soon as weights exist.

### Small loose ends (do opportunistically)

- `ensure_batch_host_odirect_cap_skips` never appears in any log — unconfirmed
  whether it is 0 or simply not printed. Check the print path.
- **Serial-path phase counters don't exist** — serial arms report `read_wait`/`h2d`
  as 0 because those are HOST-path fields. Fine while HOST wins, but it means that
  path is uninstrumented.
- EOF: post-fix runs show **0 pread failures** (was 3/run). The clamp worked better
  than predicted — §8 expected it might swap EIO for EINVAL since no shard size is
  4096-aligned. Consider it resolved but do not be shocked if EINVAL appears in an
  odd config.

## 4. CONFIG OF RECORD — the exact winning setup

```
build-hip/bin/llama-server \
  -m ~/models/ds4/DeepSeek-V4-Flash-Q8-MTP-00001-of-00004.gguf \
  --no-mmap --weight-paging --weight-paging-slots 5500 \
  --weight-paging-resident-device ROCm1 --device ROCm0,ROCm1 -ngl 99 \
  -c 4096 --parallel 1 --host 127.0.0.1 --port 8099
```
with env:
```
WP_ENSURE_BATCH_HOST=1 WP_RESIDENT_DENSE=1 WP_SIZE_CLASS_SLOTS=1 WP_PAGED_BATCH=1
WP_PREFETCH_DEPTH=16 WP_IOURING_DEPTH=16 WP_PIN_HOST=0
WP_PREFETCH_XLAYER=0 WP_SPEC_REAP=0 WP_DENSE_PREFETCH_N=0
WP_FADVISE_LOOKAHEAD=0 WP_SAMPLE_ORACLE=0 WP_DRAFT_PREFETCH=0 WP_STICKY_SPEC=0
# RAM victim tier (optional): WP_HOST_BUDGET_BYTES=8000000000
# prefetch worker (was OFF in all tonight's runs): WP_HOST_PREFETCH=1
```

- **ROCm0 = R9700 32 GB = paging device.  ROCm1 = RX 6900 XT 16 GB = resident.**
- **rocm-smi indices are REVERSED vs llama's**: llama ROCm0 = smi `GPU[1]`;
  llama ROCm1 = smi `GPU[0]`. Mixing these up mislabels every VRAM reading.

**Builds** (library + tests are CPU-only and fast; only GPU runs need HIP):
```
cd ~/GitHub/llama.cpp/build-cpu && cmake --build . --target test-weight-pager -j 8 && ./bin/test-weight-pager
cd ~/GitHub/llama.cpp/build-hip && cmake --build . --target llama-server llama-perplexity -j 12
```
The library target is **`llama-common`** — there is **no** target called `common`,
and asking for one **exits 0 having done nothing** (a silent no-op that cost time).

**Harnesses on mad-lab-main** (`~/wp_logs/<name>/` for logs):
- `/tmp/unit3b_ab.sh` — 4-arm x 3-round decode A/B (h0/h4/p0/p4), order rotated.
  `TAG`, `HOSTGB`, `PREFETCH`, `RESIDENT`, `ROUNDS` env knobs.
- `/tmp/ppl_validate.sh` — single-arm wikitext PPL + amplification + errno check.
- `/tmp/nvme_probe.py` — O_DIRECT page-size read bandwidth probe.
- `~/host_cache.sh` — kmbandy's original sweep. **Its `killsweep.sh llama-server`
  WOULD KILL THE LIVE ROUTER.** Do not run it as-is.

## 5. Reference numbers (do not re-measure)

- **Model:** DS4-Flash-Q8-MTP, 4 shards, **160.0 GB** on disk, 44 layers,
  256 experts, 6 used, n_embd 4096, shared experts 1.03 GiB, FFN island 0.06 GiB.
- **Page = 4456448 B**; 33792 expert sub-pages (44 x 256 x 3); **expert = 12.75 MiB**.
- **Drives** (pager's page size, O_DIRECT, random offsets, fresh file):
  main SN850X QD1 0.74-0.91 / QD16 2.84-2.95 GB/s;
  2026 SN750 250GB QD1 **2.13-2.20** / QD16 2.82-2.89 GB/s.
  **These are FLOORS, not ceilings** — post-fix prefill sustains **6.206 GB/s** and
  the SN850X is rated ~7. Short-burst probes under-measure.
  *Probe gotcha:* fixed seeds mean a re-run re-reads the same offsets and the drive
  cache serves them (once produced an impossible 17 GB/s). Use a fresh file.
- **Decode, settled over 3 rounds** (round-1 of each arm is a cold outlier — drop
  it or add a warmup):

| arm | tok/s | NVMe | read_wait | h2d |
|---|---|---|---|---|
| h0 HOST, no tier | **3.570** | 82.72 GB | 15865 ms | 3736 ms |
| h4 HOST, 4 GB tier | 3.354 | 68.69 GB | 13478 ms | 3923 ms |
| p0 serial, no tier | 2.287 | 82.28 GB | n/a | n/a |
| p4 serial, 4 GB tier | 1.913 | 72.77 GB | n/a | n/a |

- Pre-fix, for contrast: h0 1.736, h8 0.960, p0 1.911, p8 1.186.
- **Prefill:** `ensure_batch_gb_s` 2.248 -> **6.206** on an exact control (same
  `page_ins` 264150, same `io_gb_read` 1177.171 GB). Wall 818 s -> 580 s.
- **PPL:** 1.9007 +/- 0.07421 pre- and post-fix. (8 chunks at n_ctx 512 — this is
  NOT comparable to the 4.1524 full-corpus figure; the running estimate falls
  3.37 -> 1.90 across those chunks. Arm-vs-arm only.)

## 6. Retractions from today — do not resurrect these

1. **"The RAM tier is a net loss."** RETRACTED (`7c6229a62`). n=2 with a 2x spread
   in one arm. It is a **wash**, and only because of the §3-Task-1 memcpy.
2. **"~5.7 ms per promotion, ~30x off link speed."** WRONG BY ~25x. Direct
   measurement: **0.223 ms/page**. It came from differencing two noisy aggregates.
3. **"Per-expert size is 25.2 MiB, the design was off by 2x."** RETRACTED
   (`41a34142e`). The 283.5 GB routed-expert figure **exceeds the whole 160 GB
   file**. Truth is **12.75 MiB**; the 2026-07-21 design's ~13.4 MB was right and
   its hot-set coverage table stands (its single-prompt caveat does not).
4. **"~2.9 GB/s drive ceiling, ~2x headroom."** Too low — see §5.
5. **"Submission is the bottleneck (submit >> wait)."** Artefact of a mislabeled
   counter; enqueue is ~60-120 ms out of ~16-60 s.
6. **"38% of the model is compressed."** That was shard 4 alone (961/2529); the
   real distribution was 111 / 0 / 0 / 961.

## 7. Traps that bit us today — read before working

- **O_DIRECT alignment comes from `statfs f_bsize`, NOT the block device.** I
  cleared alignment as a suspect by checking `logical_block_size` (512) and was
  wrong; btrfs wanted 4096. Same wrong-authority mistake twice in one night.
- **Never infer a per-event cost by differencing two aggregate phases.** Instrument
  the event. That single habit produced retraction #2.
- **`pgrep -f <pattern>` self-matches your own shell** — its command line contains
  the pattern. Bit us **three times**; once it SIGINT'd our own wrapper. Use
  explicit PIDs or `ps -eo args | grep '[l]lama-server'`.
- **A DS4 server can ignore SIGINT mid-model-load** and need SIGKILL — while
  holding 35 GB VRAM. Always verify the card is actually empty **before** releasing
  a board claim.
- **Claim as soon as you are ready to use the card.** The queue handles contention
  and notifies you on your turn. Do not sit on a claim through CPU builds — a DSWS
  session sat queued behind us for over an hour. Set `vram_alert_pct=98` for
  paging workloads (the pool pre-allocates ~95% by design).
- **Sanity-check subagent numbers against physical reality.** "283 GB of experts"
  in a "160 GB file" should stop you instantly.
- **A/B method:** alternate arm order between rounds (a fixed order made the
  second arm faster in all 3 rounds of an earlier A/B regardless of which arm it
  was); drop round-1 as cold; read each arm against **its own** replicates because
  variance is arm-dependent (`read_wait` 0.08% on h0 while its tok/s varies 8%,
  yet p0's tok/s replicates to 0.1%); and **don't mix buffered and O_DIRECT arms**
  in one sweep — the buffered arms pollute the page cache for later arms.
- **Build the HIP path before trusting HIP-guarded code.** A builder could only
  compile the CPU side and said so; the `GGML_USE_HIP` block went to hardware
  uncompiled until we built it.
