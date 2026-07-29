# Weight pager — 2026-07-29 brief (compaction point, work continues this evening)

Repo: `~/GitHub/llama.cpp`, `master` at **`68ef96df2`**, converged on
mad-lab-main, mad-lab-2026, and origin (0 ahead / 0 behind on both boxes).
All GPUs idle, all board claims released, `llama-router.service` up on both.

---

## 1. The headline

**GLM-5.2 decode went 0.3004 -> 0.7189 tok/s today. 2.39x, entirely from
scheduling.** Same bytes, same pages, same evictions, same model, same hardware.
`io_bytes = 801,125,105,664` in every arm of every A/B.

```
0.3004 t/s   SERIAL transport at QD4   <- what EVERY prior GLM number was taken on
0.6106 t/s   + WP_ENSURE_BATCH_HOST=1              (+103%)
0.7189 t/s   + WP_ENSURE_BATCH_H2D_OVERLAP=1       (+18%)
```

**Use these settings for GLM-class paging:**
```
WP_ENSURE_BATCH_HOST=1
WP_IOURING_DEPTH=16
WP_ENSURE_BATCH_H2D_OVERLAP=1
LLAMA_WP_TRANSPORT unset          # p2p is SLOWER, do not set it
```

**And bytes/token was measured directly for the first time: 6.26 GB.**
801 GB / 128 tokens. Effective I/O went 4.53 -> 5.51 GB/s. The drive does
6.25 GB/s standalone, so ~13% remains before NVMe is genuinely the wall. After
that, transport is finished as a lever and only bytes/token is left.

---

## 2. Commits (all pushed)

```
68ef96df2  vulkan: declare the weight-pager host register/unregister entry points
d4f2e06a7  wp: fix the Vulkan H2D overlap -- import the O_DIRECT arena via external_memory_host
11fb5c38b  wp: the Vulkan overlap was silently falling back, and the stats hid it
26f406d92  docs: spec for Vulkan paging instrumentation + the GGML_USE_* macro sweep
dfad47e27  Merge origin/master into mad-lab-2026's master
159368422  wp: route the H2D overlap through GpuTransport (Vulkan parity, and faster)
3a7a6c0a5  pipeline: fix the loopback harness (wrong binary, tmpfs, n_layer parse)
63dfe38d0  wp: overlap the H2D copies with the O_DIRECT reads in ensure_batch
86035f84e  pipeline: Phase 2 protocol + stage driver, and unbreak the CPU-only build
542e723fa  wp: per-class pin floor for size-class slots (fixes the GLM abort)
```

Also: 132 commits pushed to origin (first push in a long while, at kmbandy's
instruction), then 2026's 9 unique commits merged in -- including
`fdb441155 wp: fix gate 4 -- prefetch batch cannibalized its own slots (silent
wrong weights)`, a correctness fix any reset-and-pull shortcut would have lost.

---

## 3. Things that are now DEAD, with evidence

### Static expert pinning -- killed three independent ways
GLM's routing concentration is REAL and new: **max/mean 11.32** (laguna measured
2.1), gini 0.596, top-10% of experts = 44.5% of activations, only 65.5% of the
57,825 expert pages touched in a run (laguna touched 98.8%). The long-standing
"MoE routing is near-uniform" belief came from laguna and does NOT hold for GLM.

But the concentration is **entirely prompt-specific**, so nothing static exploits it:
1. **Top-40 page overlap between two prompts = ZERO.** Prompt A's hot pages sit
   in blocks 9-28, prompt B's in blocks 21-77 -- different prompts light up
   different DEPTHS.
2. **The hot pages are already cached.** Top-40 hit rate under plain LRU is
   94.0% / 93.4%. Pinning them recovers ~0.4% of page_ins.
3. **Out-of-sample recall is ~chance.** A's per-block top-K scored on B, vs the
   K/256 random baseline: K=8 in-sample 26.6% -> OUT 4.9% (random 3.1%);
   K=128 in-sample 93.8% -> OUT 59.2% (random 50%). Jaccard at K=8 is 4.3%.

Also: per-expert pinning is impossible at the ggml level anyway (expert weights
are fused 3D tensors indexed by `mul_mat_id`; one expert is a SLICE, and ggml
places whole tensors). It IS possible at the pager level via a permanent-pin
flag on sub-pages -- about a day's work -- but the data says don't.

### P2P NVMe->VRAM is not real peer-to-peer
Standalone probe (`/tmp/wpx/p2p_probe.cpp`, no llama.cpp), random offsets over a
45.8 GiB file, O_DIRECT:
```
NVMe -> RAM   4 MiB:  QD1 4.77  QD4 6.20  QD16 6.25  QD64 6.00 GB/s
NVMe -> VRAM  4 MiB:  QD1 2.00  QD4 2.00  QD16 2.02  QD64 1.99 GB/s   <- FLAT
RAM  -> VRAM       :  4 MiB 25.0, 16 MiB 27.4 GB/s
```
Flat across queue depth = serialized. `io_uring_register_buffers` on an exported
dma_buf returns **EFAULT** -- the kernel cannot pin BAR memory for DMA, so there
is no userspace path to real P2P on this stack (`CONFIG_PCI_P2PDMA=y` means the
kernel supports the mechanism, not that amdgpu exposes VRAM through it).
The two-hop path beats "direct" 2.5x. PCIe is NOT the limit -- H2D does 25 GB/s.

Prior "+33%/+59% for p2p" results all compared against SERIAL sync-pread, never
against the O_DIRECT worker pool. Against the pool, p2p LOSES (0.5155 vs 0.6106).
The 142-batch P2P death is gone, incidentally cured by the window-cache raise
to 256.

---

## 4. Vulkan: made measurable, then made to work

Four handoff rounds. Final state at `68ef96df2`, RX 480 / LFM2.5-8B-A1B:
```
ctl       1.7338 t/s   read_wait 31,422 ms   h2d 12,882 ms   backend-barrier
overlap   2.5652 t/s   read_wait 11,682 ms   h2d    190 ms   transport-overlap
ctl2      1.7314 t/s   read_wait 31,400 ms   h2d 12,908 ms   backend-barrier
stage_submissions=35568  stage_completions=35568  sync_fallback_pages=0
copies_before_last_read=27954/35568 (78.6%)
```
**+48.2% over the warm control.** Zero fallbacks, h2d collapsed 98.5%.

**Two caveats that matter more than the number:**

1. **The `external_memory_host` import FAILS on Polaris** ("Vulkan host pointer
   import failed, ptr=..., size=3018752"), even though the extension is present
   (vulkaninfo, revision 1). So `src_is_pinned` stays false and every transfer is
   still fenced inline. **The +48% is interleaving, not async DMA** -- each copy
   is synchronous, but issuing copy N as read N lands while later reads continue
   on worker threads is enough. Fixing the import should add more. Likely
   `minImportedHostPointerAlignment` or the hipHostMalloc'd pages not being
   importable.
2. **Every arm is SLOWER than the previously-broken run** (ctl 3.29 -> 1.73).
   The bug had been routing everything to `page_in_sync_`, which is BUFFERED and
   cached a 5.6 GB model on a 15 GB box. Fixing it engaged O_DIRECT, which
   bypasses the cache by design (read_wait 84.6 ms -> 31.4 s -- the reads are
   finally real). **LFM2.5 is the wrong size to judge this feature.**

How the four rounds went, because the sequence is the lesson: round 1 built the
overlap and silently broke Vulkan; round 2 added instrumentation that exposed it;
round 3 found round 1 had CAUSED it and applied the real fix; round 4 was two
missing header declarations, because I told Terra to verify Vulkan code with a
CPU-only build. Twice. It reported "Vulkan compilation unverified" both times,
accurately, and I pushed anyway.

Also landed: the **55-site `GGML_USE_*` macro sweep** -- 43 legitimately
compile-time (now commented with why), 10 converted to runtime checks, 2 flagged
ambiguous rather than guessed (`GpuTransport`'s outer HIP/CUDA guard and the
routed-expert block both also ENCLOSE Vulkan; removing them needs a pure-Vulkan
build). That bug class had produced seven live bugs in this one directory.

---

## 5. Corrections I made to my own claims today

- **"The drive tops out at 2.9 GB/s, so transport can only buy 1.5x."** Wrong
  constant -- that was a QD16 io_uring figure. O_DIRECT with a thread pool does
  5.78 GB/s on the batch path. All arithmetic built on 2.9 was wrong.
- **"P2P is limited by the PCIe/BAR aperture."** Wrong mechanism -- the H2D leg
  does 25 GB/s over the same link.
- **"GLM has no exploitable reuse."** Generalized from a PREFILL workload (2
  perplexity chunks, where the per-layer expert union is nearly all experts).
  Decode gets ~55% hit rate.
- **"Pinning top-8 buys 26.6% of demand."** In-sample -- hot set computed from
  one prompt and scored on that same prompt. Out-of-sample it is 4.9% against a
  3.1% random baseline.
- **`ensure_batch_gb_s` is not comparable across transports.** On HOST it is
  bytes/read-leg-only (the 38 s H2D is excluded); on P2P it is bytes/full-path.
  Comparing the two reported numbers (5.780 vs 3.876) is invalid; apples-to-apples
  is 4.53 vs 3.88. **Derive transport rates from total bytes / total wall time.**
- **Recommended striking "raise the slot count"** on the strength of an LRU-only
  result, then had to reinstate capacity as necessary once concentration was
  measured. Capacity is necessary but not sufficient; policy matters too.

---

## 6. Method rules earned today (apply to every future A/B)

1. **A discarded warm-up pass before arm 1.** A cold page cache faked a +3.7%
   Vulkan gain (ctl prefill 1.53 vs 3.74/3.70 on warm arms) that would have
   shipped as "parity confirmed."
2. **A mechanism counter, not just an outcome.** `copies_before_last_read` is the
   only reason that fake gain didn't stick. A path that "ran but did nothing"
   looks identical to success in a throughput number.
3. **Use a model LARGER than host RAM for paging A/Bs**, or the OS page cache
   silently becomes the thing under test.
4. **Read the `TRANSPORT: active=` line before quoting any throughput number.**
   Three times now I have measured a baseline on the serial fallback.
5. **State the random baseline alongside every recall figure**, and use a
   held-out prompt.
6. **Never verify a backend-specific change with a build that excludes that
   backend.** If the implementer cannot compile the file, dispatch it to a
   machine where it can.
7. **Harness facts** (all cost real time today): `llama-cli` rejects `-no-cnv` on
   this tree and spins on an interactive prompt; `llama-completion` does not
   register `--weight-paging`; `llama-perplexity` registers paging but cannot
   decode. **`llama-server` + curl is the only combination that does both.**
   Always `--no-mmap`. Launch servers as a simple command, not inside `( ... ) &`
   -- `$!` of a subshell means `kill` misses the server, the next arm fails to
   bind, `/health` answers from the OLD server, and every arm silently measures
   arm 1.

---

## 7. Open items, ranked for this evening

1. **PREFETCH.** The remaining lever, and kmbandy's call from the start. The
   concentration data says the structure exists: 11.32 max/mean, and cross-layer
   chains P(e'@L+1 | e@L) with **1,621 links >= 0.90** (vs 58 that Speedwagon
   found on Qwen3.5). **That 1,621 is UNVERIFIED** -- it counts links with >=5
   observations and a 5/5 link scores 1.00 by luck. Re-cut at a higher
   observation threshold before building on it. Captures already on disk:
   `/var/tmp/routing_capture.A.bin` (prompt A, 462 MB, 256 steps) and
   `/home/kmbandy/wp_logs/accounting/routing_capture.bin` (prompt B, 128 steps).
   Analysis scripts: `/tmp/wpx/router_analysis.py`, `/tmp/wpx/xprompt.py`.
2. **HIP non-regression run.** Owed all evening, blocked by the DSWS campaign on
   the R9700. Reference to reproduce at `68ef96df2`: overlap OFF 0.6054 t/s, ON
   0.7189, `h2d_ms` 39,180 -> 2,580, `copies_before_last_read` 186,556/196,576,
   `page_ins` 196,580, `io_bytes` 801,125,105,664. Not about defending a past
   number -- those counters are the instrument for everything after this, and the
   sweep touched 43 compile-time guards.
3. **The remaining ~13% to the drive ceiling** (5.51 -> 6.25 GB/s achieved).
4. **Why the Polaris host-pointer import fails.** Would convert Vulkan's
   interleaving into genuine async. One look at
   `minImportedHostPointerAlignment` vs the arena's alignment.
5. **Loopback correctness gate -- STILL NEVER RUN.** Phase 2 builds and passes 52
   assertions, but the composed pipeline has never been shown to equal the
   monolithic model. Needs GLM itself: the band-capable archs are
   `DEEPSEEK2 / DEEPSEEK2OCR / GLM_DSA / MISTRAL4` and GLM-5.2 is the only one on
   disk. smollm3 was tried and correctly refused.
6. **Kimi K3 prerequisites.** K3 is NOT in this tree (only `LLM_ARCH_KIMI_LINEAR`,
   a different model) and will not be band-capable when it arrives -- the
   whitelist needs a case AND the graph builder must skip unowned layers.
   Both are prerequisites before 620 GB of download is worth anything.
7. **Per-device `--weight-paging-slots`**; `GpuTransport::init` 4-vs-3 (fixed);
   the two ambiguous macro guards pending a pure-Vulkan build.

---

## 8. The arithmetic on Kimi at ~5 tok/s

At 6.26 GB/token and a 5.51 GB/s achieved rate, the transport lever is nearly
spent. Reaching ~5 tok/s needs a ~10x cut in bytes/token, and the stack is
multiplicative:

| lever | factor | status |
|---|---|---|
| prefetch -> saturate the batch path | ~1.13x | 5.51 -> 6.25 GB/s |
| Q1 instead of Q2 | ~2x | halves bytes/token |
| second drive streaming concurrently | up to 2x | needs cross-machine |
| speculation (3x accept) | 2-3x | amortizes the stream across tokens |

**Speculation is the highest-leverage single item** and it is also what UNLOCKS
the cross-machine 2x: pipeline stages are sequential per token, so both drives
only stream at once when multiple tokens are in flight. Phase 2 explicitly scoped
microbatch pipelining out.

Measured constraint to keep in view: I/O, host work and GPU compute are FULLY
SERIALIZED (32.5 s + 10.8 s + 30.0 s = 73.3 s vs 73.6 s wall on DS4), and the two
GPUs never compute concurrently. Today's fix removed ONE serialization
(read vs H2D). I/O vs GPU compute is the next and larger one -- and overlapping
those IS prefetch.
