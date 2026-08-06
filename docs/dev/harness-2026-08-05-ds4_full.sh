#!/usr/bin/env bash
# DS4-Flash-0731-DSpark on the FULL cross-machine expert-dispatch layout.
# Derived from stage7.sh (which drove GLM-5.2); only the model paths, the 2026
# build dir, and the slot budgets change.
#
#   mad-lab-main  RX 6900 XT  ROCm1    SPINE   ds4-dense.gguf (9.8 GB), fully resident
#   mad-lab-main  R9700       ROCm0    worker  experts 85..255  (~98 GB, local SN850X)
#   mad-lab-2026  GTX 1070    CUDA0    worker  experts  0..84   (~49 GB, /mnt/nvme SN750)
#   mad-lab-2026  RX 480      Vulkan0  worker  experts  0..84   (same shard, load-shared)
#
# 2026 binaries come from build-army-cachy (the post-CachyOS build dir).
# Vulkan is forced to posix_memalign staging: its host buffer type returns
# 4096-aligned BAR memory that passes the alignment check and then fails O_DIRECT
# read() -- see stage7.sh's note.
#
# DO NOT TOUCH on mad-lab-2026: the nemotron embedder and llama-router are LIVE
# FLEET SERVICES. Only ever kill by the PIDs this script itself started.
set -uo pipefail

MAIN_REPO=/home/kmbandy/GitHub/llama.cpp
ES_MAIN=/home/kmbandy/models/DS4-eshard-main
ES_2026=/mnt/nvme/models/DS4-eshard
# 2026-08-02: the experts-0..84 shard carved by layer, so the DSpark stages
# (blk.43/44/45) can be served from mad-lab-MAIN instead of 2026. Made by
# make_layer_shard.py -- manifest/descriptor only, the .wpb blobs are reused
# (trunk by symlink, dspark by a 3.18 GiB copy to main). See DSPARK_HOST below.
ES_2026_TRUNK=/mnt/nvme/models/DS4-eshard-trunk        # layers 0..42, experts 0..84
ES_DSPARK=/home/kmbandy/models/DS4-eshard-dspark       # layers 43..45, experts 0..84 (ON MAIN)
# The mirror-image carve, for DSPARK_SPLIT=cpu: each machine keeps the expert
# range it already owns and serves ITS OWN DSpark half. All symlinks, 404 KB real.
ES_2026_DSPARK=/mnt/nvme/models/DS4-eshard-dspark          # layers 43..45, experts 0..84   (ON 2026)
ES_MAIN_TRUNK=/home/kmbandy/models/DS4-eshard-main-trunk   # layers 0..42,  experts 85..255
ES_MAIN_DSPARK=/home/kmbandy/models/DS4-eshard-main-dspark # layers 43..45, experts 85..255
DENSE=/home/kmbandy/models/DS4-Flash-dense/ds4-dense.gguf
# 2026-08-01: 2026's Tailscale IP CHANGED in the CachyOS reinstall.
# Was 100.102.191.30 (still in the SOP guide and four repo design docs).
# The stale address does not refuse -- it DROPs, so the spine hangs 2m20s and
# then reports "failed to connect to worker", which reads like a worker fault.
IP2026=100.124.155.84
IPMAIN=100.86.191.92
# Per-arm output dir. A fixed OUT silently DESTROYS the previous arm's worker
# logs, which is exactly what happened comparing CUDA0 vs Vulkan1 on 2026-08-01
# -- the baseline survived only in a saved task transcript. ARM defaults to the
# 1070's backend so the common A/B needs no extra argument.
ARM=${ARM:-${DEV_1070:-CUDA0}}
OUT=/var/tmp/ds4full-$ARM
# Overridable: every trace we have is from this one prompt, so any claim about
# which experts get touched is really a claim about this single trajectory.
PROMPT=${PROMPT:-"The capital of France is"}

# ===================== CONFIG OF RECORD =====================
# These defaults ARE the measured best config. Do NOT run the harness bare and
# then compare the result against a banked number -- that is exactly how the
# 2026-08-02 night A/B got a 3.180 control against a 4.231 baseline and wasted
# four runs. Every value below was measured, with the win it carries:
#
# *** PART OF THE CONFIG OF RECORD BUT NOT SETTABLE HERE (2026-08-05). ***
# Five defaults now live in the C++ and this harness cannot see them, so they do
# not appear in the CONFIG echo below as harness knobs -- but a run is only the
# config of record if the BINARIES carry them. An incomplete checklist reads as
# a complete one, which is how the SPEC-default trap bit us on 2026-08-03, so
# they are enumerated here:
#   WP_DISPATCH_GATHER=1            src/pipeline/pipe-expert-dispatcher.cpp
#       Spine sends each worker only the token rows its experts need. issue
#       8236 -> 5245 ms (-36.3%). Numerically exact. =0 disables.
#   WP_DISPATCH_GATHER_MAX_FRAC=0.90  same file
#       Skip the gather when it would save <10% of rows -- the R9700 needs 658
#       of 659 and was gathering to drop ONE row. Measured INERT on issue; kept
#       because it removes strictly-wasted work that scales with any rebalance.
#   WP_EXPERT_COMPUTE_CHUNKS=4      tools/wp-expert-worker/wp-expert-worker.cpp
#       Compute expert chunk k while chunk k+1 is still being read. Gated on
#       n_pagein > 0 so all-resident decode/verify requests do not pay a second
#       graph submit for overlap that does not exist. Prefill wait -5.6%.
#   WP_EXPERT_READ_STRIPES=4        same file
#       Read each page in 4096-aligned stripes so the dispatch thread uploads
#       stripe k while the reader reads k+1. Decode-side wait -9.1%.
#   WP_EXPERT_STRIPE_MAX_PAGEINS=4  same file
#       Only stripe read-SPARSE batches. Ungated striping COST prefill +10.7%
#       (31.4 page-ins/request already overlap each other); gated it recovers.
# VERIFY BEFORE TRUSTING A NUMBER -- a stale worker or spine binary silently
# runs a different config:
#   grep -c WP_EXPERT_READ_STRIPES  ~/GitHub/llama.cpp/tools/wp-expert-worker/wp-expert-worker.cpp
#   grep -c WP_DISPATCH_GATHER_MAX_FRAC ~/GitHub/llama.cpp/src/pipeline/pipe-expert-dispatcher.cpp
# and check both build dirs are newer than those sources.
#
#   slots 500/500/2200   +10.9% over 400/400/1600   (2026-08-01 brief)
#   VKSPLIT=1048576      +7.9% end to end, 1.88x on RX 480 expert compute.
#                        The 480 has a 256 MB BAR and no ReBAR; amdgpu
#                        OVERSUBSCRIBES it into GTT rather than failing, so
#                        without this ~95% of the slot pool physically sits in
#                        system RAM. Boundary is exact at 255 MB.
#   KEEPALIVE=100        part of the +17.5% RX 480 idle-recovery fix; the other
#                        half is power_dpm_force_performance_level=high, which
#                        is a MACHINE setting, not a harness one -- verify it
#                        separately, it does not travel with this file.
#                        *** 200 -> 100 on 2026-08-05. *** A 6-run interleaved
#                        A/B/AB sweep separated decode dispatch wait with ZERO
#                        OVERLAP: keepalive=100 gave 27.52/26.52/26.63/25.50 s
#                        (max 27.52), keepalive=200 gave 29.78/29.44/34.84 s
#                        (min 29.44). Means 26.54 vs 31.35 = -15.3%. It held
#                        across mad-lab-main load 6.4-17.9, which is what makes
#                        it credible -- host load inflates every phase at once,
#                        so a clean group separation under varying load is hard
#                        to fake. Cannot affect numerics: it is a keepalive
#                        kernel submitted between requests. Confirms the earlier
#                        standalone sweep (100 -> 350.9 us/expert, 200 -> 404-432).
#                        NOTE end-to-end tok/s could NOT resolve this at n=2
#                        (within-arm decode spread 0.17-0.38, as large as the
#                        between-arm difference). The per-leg dispatch numbers
#                        did. Prefer legs over tok/s on this rig.
#
#   REJECTED the same sweep, do not retry without new evidence:
#   WP_EXPERT_GATHER_MIN_TOKENS=2   Bypassing the gather at n_tokens==1 was
#                        banked at +9.3% decode with "byte-identical output".
#                        BOTH HALVES FAILED. It BROKE DETERMINISM -- one rep
#                        returned draft acceptance 0.80952 where the identical
#                        config returned 0.84286, a value that had held across
#                        ~10 runs. At n_tokens==1 gather and dense are
#                        mathematically equivalent but NOT bit-identical (gather
#                        threads the FFN through get_rows/get_rows_back, a
#                        different op sequence and reassociation), and with the
#                        timing-dependent worker assignment it intermittently
#                        tips a marginal routing decision. It also gave NO
#                        speedup (decode wait 29.44/34.84 vs 29.78 baseline) --
#                        the +9.3% predates the chunking and striping changes
#                        that already cut the path those nodes sat on.
#   CTX=1024, NPRED=512  64-token runs produced THREE false findings in one
#                        session (2026-08-01). Never measure at 64. 256 minimum.
#
# The banked figure on this config is 4.231 tok/s (ARM=sp1 and ARM=wire,
# reproduced exactly twice, 2026-08-02), without DSpark. With DSpark: 6.535
# baseline / 6.738 on DSPARK_HOST=CPU (9 runs, 2026-08-02).
#
# *** ALL OF THOSE ARE COMBINED WALL FIGURES AT CTX=1024 WITH A 6-TOKEN PROMPT. ***
# They are therefore ~pure decode, and they are NOT comparable to the split
# PREFILL/DECODE numbers this harness now prints, nor to any run with a real
# prompt or a larger CTX. Re-baseline before A/B-ing against them.
# *** RULE (kmbandy, 2026-08-03): ALWAYS RUN THE CONFIG OF RECORD, CHANGING ONLY
# *** THE ONE ELEMENT UNDER TEST. Anything else and you are comparing two
# *** variables against a one-variable baseline.
# This bit me the same day it was written: I ran a whole 13-run prefill/decode
# sweep with SPEC unset and then compared the decode numbers against last night's
# 6.738, which had DSpark ON. Two variables, one conclusion, wrong. SPEC now
# DEFAULTS ON so a bare run IS the config of record; set SPEC= explicitly (empty)
# only when DSpark itself is the element under test.
SPEC=${SPEC-1}
# *** CONFIG OF RECORD 2026-08-04: CTX=8192, NPRED=256. ***
# CTX=1024 is now ACTIVELY WRONG: the default prompt is 663 tokens and prose1500.txt
# is 1526, which does not fit. 8192 also leaves room -- at UBATCH=2048 the spine sits
# at 13.47 GB of 17.16, so ~3.7 GB is free for KV = roughly 500K tokens of f16 at
# ~6.9 KB/token. NPRED=256 is what every 2026-08-04 measurement used; 64-token runs
# produced THREE false findings in one session (2026-08-01), so 256 is the floor.
NPRED=${NPRED:-256}
CTX=${CTX:-8192}
# UBATCH=<n> sets the spine's --ubatch-size. Prefill is chunked at n_ubatch and
# the DS4 expert set is re-swept PER UBATCH, so prefill page-ins scale with
# ceil(n_prompt / n_ubatch). Unset = llama-server's 512 default, which is what
# every banked prefill number used.
#
# *** RAISING THIS WITHOUT RAISING THE WORKERS' io-buffer PREALLOC IS A TRAP. ***
# The worker sizes its staging buffer as 2*(n_embd * tokens * 4 + 64K) and grows
# it on demand; a 1024-token ubatch against a 512-token prealloc reallocates
# DURING SERVING, which lands entirely on the arm under test. So UBATCH also
# drives WP_IO_PREALLOC_TOKENS on every worker below -- they move together, on
# purpose. n_batch is deliberately LEFT ALONE at its 2048 default: one variable.
# *** CONFIG OF RECORD 2026-08-04: UBATCH=2048. *** Measured ladder, all with the
# worker's gather/scatter ON, varied-prose prompts, f16, CTX=8192, warm-up discarded:
#    663 tok  ub512  dense 13.61 | ub512 gather 16.93 | ub1024 dense 15.89 | ub1024 gather 22.77
#   1526 tok  ub1024 gather 25.48 | ub2048 gather 33.06   (+29.7%)
#   3044 tok  ub2048 gather 31.13 | ub4096 gather 33.46   (+7.5% only, and 96% VRAM)
# ubatch x gather at 512->1024 is SUPER-ADDITIVE: +67.3% combined vs +45.3% predicted
# multiplicatively -- a wider ubatch MANUFACTURES redundant compute (dense computes
# 935,985 token-expert pairs at ub1024 vs 827,904 at ub512 for identical output) and
# gather then strips it, so the two levers feed each other. NEVER quote them separately.
# *** STOP AT 2048. *** 4096 needs -b 4096 (n_ubatch cannot exceed n_batch) and puts the
# spine at 16.43 GB of 17.16 = 96%, leaving ~106K tokens of f16 KV with no fragmentation
# margin: n_ubatch=4096 and long context are MUTUALLY EXCLUSIVE on the 6900XT.
# Compute buffer measured near-linear: ub1024 ~10.0 | ub2048 13.47 | ub4096 16.43 GB.
# That is BATCH-scaled, not context-scaled, so KV quantisation cannot relieve it.
# NOTE the gain depends on n_prompt mod n_ubatch, not n_ubatch alone (at 3044 tokens
# ub2048's second sweep is only 996 tokens), so 2048 is the right default for MIXED
# prompt lengths, not a universal constant.
UBATCH=${UBATCH:-2048}
UBARGS=""
WPRE=""
[ -n "$UBATCH" ] && UBARGS="--ubatch-size $UBATCH" && WPRE="WP_IO_PREALLOC_TOKENS=$UBATCH"
# Explicit override, independent of UBATCH, for BISECTING the io-buffer prealloc
# change (2026-08-03 evening) as a throughput suspect. WP_PREALLOC=0 restores the
# ORIGINAL 1 MiB floor that predates that change; unset = current behaviour.
[ -n "${WP_PREALLOC:-}" ] && WPRE="WP_IO_PREALLOC_TOKENS=$WP_PREALLOC"
# BATCH=<n> sets --batch-size. n_ubatch CANNOT EXCEED n_batch (llama-server default
# 2048), so testing n_ubatch=4096 requires raising this too. That makes -b a SECOND
# variable: hold it CONSTANT across both arms of any n_ubatch comparison, or the
# result is uninterpretable.
BATCH=${BATCH:-}
[ -n "$BATCH" ] && UBARGS="$UBARGS --batch-size $BATCH"
# WEXTRA passes arbitrary env to ALL FOUR workers (WPRE is on every worker launch).
# Added to A/B WP_EXPERT_OVERLAP, whose own comment promises a measured cost that
# was never recorded.
[ -n "${WEXTRA:-}" ] && WPRE="$WPRE $WEXTRA"
# Applied to EVERY worker via WPRE (see the HOST VICTIM TIER and PREFETCH blocks
# below for what these do and what they cost). Appended after WEXTRA so an
# explicit WEXTRA override still wins on the command line.
WPOST=""
VKSPLIT=${VKSPLIT:-1048576}
KEEPALIVE=${KEEPALIVE:-100}
# ---------------------------------------------------------------------------
# HOST VICTIM TIER (RAM L2 between VRAM and NVMe). NOT part of the config of
# record -- added 2026-08-05 for the prefetch work. Set to 0 to disable.
#
# WHAT IT DOES. On eviction a slot is copied D2H into a host arena instead of
# being dropped; a later page-in of the same page borrows it from RAM at PCIe
# speed instead of re-reading NVMe (counted as n_host_hit, not n_pagein).
#
# *** IT IS NOT FREE, AND THE COST IS ON THE DISPATCH THREAD. ***
# ensure_batch calls demote_slot() INLINE for every evicted slot, before the
# first NVMe read is issued, and HostTier::store_from_device does a SYNCHRONOUS
# D2H of the whole 12.75 MB page. The RX 480 averages 31.4 page-ins per prefill
# request, so a full-eviction prefill request pays ~400 MB of serialised D2H
# BEFORE it starts reading. Expect prefill to get worse.
#
# *** AND ON PREFILL IT HAS NOTHING TO CATCH. *** Measured 2026-08-05: the 480
# does 1381 expert references, 1352 page-ins, 16.83 GiB = exactly 12.75 MB each.
# EVERY PAGE IS READ EXACTLY ONCE. A victim cache can only pay off on a re-read,
# and prefill has none, so on the CURRENT prefill workload this can only cost.
#
# WHY TURN IT ON ANYWAY (kmbandy, 2026-08-05): decode DOES re-reference experts
# across tokens (72-73% resident on pure demand LRU), and the prefetch work
# creates a NEW class of re-read that does not exist today -- a page speculated too
# early and evicted before its layer arrives. Without the tier that is a wasted
# NVMe read; with it, it comes back over PCIe. That is the specific interaction
# to measure, and n_host_hit vs n_host_demote reports it directly.
#
# SIZING. mad-lab-2026 has 15 GB TOTAL / ~8 GB available and also runs the
# nemotron embedder and llama-router (LIVE FLEET SERVICES) plus THREE workers,
# so its budget is deliberately small: 3 x 1 GB = 3 GB of ~8 GB available.
# Raise HOSTVICTIM_2026 only after checking `free -g` on that box.
# mad-lab-main gets 6 GiB (kmbandy, 2026-08-05). That box also runs the spine
# (~13.5 GB of compute buffer at UBATCH=2048), the CPU DSpark worker and a
# desktop, so this is a budget set by the owner rather than derived from a
# reading -- `free -g` was not reachable from the session that wrote this.
HOSTVICTIM_2026=${HOSTVICTIM_2026:-1073741824}   # 1 GiB per 2026 worker
HOSTVICTIM_MAIN=${HOSTVICTIM_MAIN:-6442450944}   # 6 GiB, R9700
# ---------------------------------------------------------------------------
# PREFETCH (2026-08-05). Both default OFF: a bare run must stay the config of
# record. See docs/dev/2026-08-05-prefetch-brief.md.
#   PREFETCH_HINT=1  SPINE computes hash-layer (0..2) expert ids from the token
#                    id and sends them ahead of the dispatch. Costs no reads.
#   SPEC_PAGEIN=1    WORKERS actually read hinted pages in their idle window.
#   HINTLOG=1        the worker-side event stream, fflushed per line. SET THIS ON
#                    ANY HINT ARM. Two reasons, both learned the hard way:
#                    (1) the counters otherwise exist only in a stderr line printed
#                        on clean close, which this harness's SIGKILL teardown
#                        destroys -- arm 1 lost foreign_expert that way;
#                    (2) it carries the hinted ids, the speculative page-ins, the
#                        reference stream and the demand page-ins IN ORDER, which
#                        is the only way to separate MISPREDICT (never selected)
#                        from LATE (selected, but evicted before its layer came).
#                        Arm 2 reported one lumped bucket and could answer neither.
#                    Read it with docs/dev/analyze-hint-log.py.
# DELIBERATELY SEPARATE. Hints ON + speculation OFF reads exactly what the config of
# record reads while still reporting everything the spine offered, so a broken
# spine side is found with ZERO changed page-ins before any extra I/O is risked.
# Run that arm FIRST.
#
# *** PIPE_VERSION 4 -> 5. REBUILD THE SPINE AND ALL FOUR WORKERS TOGETHER *** or
# they refuse at HELLO (deliberate: an unknown frame type closes the session,
# which mid-run is indistinguishable from a worker crash).
PREFETCH_HINT=${PREFETCH_HINT:-}
SPEC_PAGEIN=${SPEC_PAGEIN:-}
SPEC_CHUNK=${SPEC_CHUNK:-}
[ -n "$SPEC_PAGEIN" ] && WPOST="$WPOST WP_EXPERT_SPEC_PAGEIN=$SPEC_PAGEIN"
# LFU=0 reverts the slot pool to pure LRU. Ranking by use count is ON in the
# binary by default, so a bare run is the NEW config of record and the control
# arm is the one that has to say so.
[ -n "${LFU:-}" ] && WPOST="$WPOST WP_EXPERT_LFU=$LFU"
# LEASE=<n> -- evictions a speculative page survives before it becomes an
# ordinary victim. 0 is the original first-victim behaviour. Anything above 0 is
# DELIBERATE POOL POLLUTION: read the amplification gate on any arm using it.
[ -n "${LEASE:-}" ] && WPOST="$WPOST WP_EXPERT_SPEC_LEASE=$LEASE"
# PREDICT=0 drops the previous-block half of the top-of-draft hint, leaving only
# id_last, which is ground truth and cannot mispredict. SPINE-side: the hint is
# computed in common/speculative.cpp, which runs in the spine process.
[ -n "${PREDICT:-}" ] && SPINEENV="${SPINEENV:-} WP_SPEC_PREDICT_PREV=$PREDICT"
[ -n "$SPEC_CHUNK" ]  && WPOST="$WPOST WP_EXPERT_SPEC_CHUNK=$SPEC_CHUNK"
[ -n "$PREFETCH_HINT" ] && SPINEENV="${SPINEENV:-} WP_PREFETCH_HINT=$PREFETCH_HINT"
# Warm without hints is a no-op, and hints without a rebuilt spine is a HELLO
# rejection. Say so at launch rather than after a wasted run.
if [ -n "$SPEC_PAGEIN" ] && [ -z "$PREFETCH_HINT" ]; then
    echo "*** SPEC_PAGEIN=1 with PREFETCH_HINT unset: the workers will never be" \
         "sent anything to page in. Set PREFETCH_HINT=1 too. ***"
fi
# The spine's own counters always survive (it exits cleanly); the WORKERS' do not.
# So a hint arm without HINTLOG can still report what was OFFERED and never what
# was RECEIVED -- which is exactly how arm 1 ran and exactly the number it lost.
if [ -n "$PREFETCH_HINT" ] && [ -z "${HINTLOG:-}" ]; then
    echo "*** PREFETCH_HINT=1 with HINTLOG unset: the workers' hint counters" \
         "(foreign_layer/foreign_expert -- the routing-agreement check) will be" \
         "LOST to the SIGKILL teardown. Set HINTLOG=1. ***"
fi
# One append, at the end, rather than $WPOST at four launch sites -- a knob that
# reaches three of four workers is worse than one that reaches none, because the
# run still looks complete.
WPRE="$WPRE $WPOST"
# *** CONFIG OF RECORD 2026-08-05: DSPARK_OMP=8. ***
# Caps the CPU DSpark worker's thread count. `-` not `:-` so DSPARK_OMP=
# (explicitly empty) still means "leave it at ggml's default", which is the
# arm this replaced.
#
# WHY. The CPU DSpark worker (:8802, device=CPU, layers 43-45) sets the cost of
# every 3-layer NextN draft block -- there are 196 of them and they were the
# worst per-layer cost in the system, 12.6 ms/layer against 5.45 for a 43-layer
# trunk block. It is COMPUTE bound, not I/O bound: its shard fits its 255 slots
# so only 4.3% of expert references page in, and 81% of its time is ns_submit,
# which on a CPU device is the raw matmul.
#
# MEASURED, 8-run interleaved sweep 4/8/16/default x2, load sampled per rep:
#   DSPARK_OMP   L3 block wait      dspark ns_submit
#      4         5.26 / 5.43 s      3.51 / 3.56 s
#      8         4.90 / 4.98 s      2.93 / 3.15 s   <- adopted
#     16         5.33 / 5.67 s      3.49 / 3.89 s
#   default(24)  8.02 / 9.08 s      6.29 / 7.28 s
# The default is ~2x slower than ANY capped value on raw matmul time, and no
# pair of adjacent arms overlaps on either metric, so the optimum at 8 is real
# and not noise. L3 wait 8.55 -> 4.94 s = -42%, roughly 5% off decode-side
# dispatch. Cause is oversubscription: 24 threads on a 12-core/24-thread box
# that also runs the spine, the R9700 worker and a desktop.
#
# NOTE end-to-end tok/s ranked 4 above 8, because the omp8 rep caught an
# unrelated DEC_wait outlier (32.21 s against 25.7-26.8 in every other run).
# The per-leg numbers are the ones to trust here, as elsewhere on this rig.
DSPARK_OMP=${DSPARK_OMP-8}

# ---------------- KV CACHE: turbo4 IS THE STANDARD (2026-08-03) ----------------
# Measured on the 6900XT, spine alone, np=1, load-only. Model 7944.21 MiB on GPU.
#              KV total        compute buf   VRAM total
#   256K       712 MiB          620.76 MiB    10.09 GB
#   1M        2823 MiB         1520.21 MiB    13.25 GB  of 17.16 GB  <- 1M FITS
# All-f16 at 1M would be ~6.6 GiB of KV and does NOT fit. turbo4 on the three big
# caches is what makes full 1M context possible on a 16 GB card.
#
# The lightning-indexer (lid) cache is NOT quantized -- llama_kv_cache_indexer_type()
# forces it to f16 regardless of what is asked for here, and it is meant to.
# ggml_cuda_lightning_indexer_supported() accepts no turbo type, so a turbo4 lid
# SILENTLY disables the fused indexer and the unfused fallback tries to allocate
# 24875 MiB -- reported only as "failed to allocate compute pp buffers", which
# reads like "context too big". That fix is UNCOMMITTED in main's tree; a build
# without it CANNOT run this config above ~128K. Verify before trusting a number:
#   grep -c llama_kv_cache_indexer_type ~/GitHub/llama.cpp/src/llama-kv-cache.cpp
#
# turbo4_64_ol* remain UNUSABLE on DS4 for an unrelated reason: SET_ROWS has no
# TURBO4_64 dst type (ggml-cuda.cu:6030-6057), so the graph aborts at reserve.
# turbo4 IS THE CONFIG OF RECORD. f16 is a diagnostic control ONLY, never the target.
# turbo4 is what makes full 1M context possible on a 16 GB card (1M f16 KV is
# ~6.7 GiB and does not fit alongside the 7944 MiB dense model; turbo4 is 1.8 GiB).
# Abandoning it means abandoning the context goal, so the corruption below is a
# BUG TO FIX, not a reason to switch back.
#
# KNOWN OPEN DEFECT (2026-08-03): turbo4 currently corrupts DS4 output.
#   f16     decode 7.10/7.11   draft acceptance 0.810   coherent
#   turbo4  decode 4.80/4.85   draft acceptance 0.225   DEGENERATE
# *** DRAFT ACCEPTANCE IS THE FIDELITY GATE. *** It collapses when the target
# model's distribution is disturbed and needs no golden output. Any turbo4 fix is
# proven when acceptance returns to ~0.81 at the config of record, NOT when the
# text merely looks better.
#
# DO NOT READ A TURBO4 THROUGHPUT WIN AS A WIN WHILE ACCEPTANCE IS LOW. A SPEC-off
# run made turbo4 look 1.96x FASTER than f16; that was the corruption flattering
# itself -- degenerate repetitive output routes to the same few experts, so the
# expert cache stops missing. A corrupted config can benchmark faster than a
# correct one. Always gate a throughput number on fidelity from the SAME run.
# *** CONFIG OF RECORD 2026-08-04: f16. TURBO4 IS DEFERRED (decision 251231a7). ***
# The block above documents turbo4 as "the standard" -- that is SUPERSEDED. f16 is
# faster (decode 6.52 vs 5.29), higher fidelity (acceptance 0.953 vs 0.897), and DS4
# is structurally a bad fit for KV quantisation (csa is a /4 compressor output, hca
# /128, and the lightning-indexer cache refuses quantisation outright). The context
# turbo4 buys is unusable while prefill is the binding constraint. Keep the turbo4
# code, disabled by configuration. Set CACHE_TYPE_K/V=turbo4 explicitly to test it.
CACHE_TYPE_K=${CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${CACHE_TYPE_V:-f16}

# PROMPT_FILE overrides PROMPT with the contents of a file, for prefill work --
# the default 6-token prompt makes every prefill measurement meaningless.
# *** CONFIG OF RECORD 2026-08-04: prose739.txt. *** q1024.txt is DISQUALIFIED as a
# benchmark prompt -- it is a repeated "Structure: / Detection:" list template and at
# temp 0 under ignore_eos the model correctly continues the template and loops, tripping
# the coherence gate. Every historical figure taken on it is suspect, including the
# "3.61-3.67 tok/s f16 long-prompt decode" baseline, which was itself degenerate
# (distinct 0.18). Prose prompts, cut at sentence boundaries, live in the scratchpad:
#   prose739.txt   663 tok   (default; distinct 0.53)
#   prose1500.txt 1526 tok   (distinct 0.42) -- needed to exercise ubatch 1024 vs 2048
#   prose3000.txt 3044 tok   (distinct 0.31) -- needed to exercise 2048 vs 4096
# n_ubatch is CAPPED BY PROMPT LENGTH: testing a ubatch >= n_prompt is a no-op, because
# both settings collapse to one sweep. Match the prompt to the ubatch under test.
# (Type-token ratio falls with length by construction, so the lower distinct values on
# the longer prompts are an artefact of length, not a quality drop.)
# *** ${PROMPT_FILE-...} -- NO COLON. THIS IS LOAD-BEARING. ***
# It was `${PROMPT_FILE:-...}` until 2026-08-04, and `:-` substitutes the default when
# the variable is unset OR EMPTY. So every arm that passed PROMPT_FILE="" to select the
# short built-in PROMPT silently ran the 663-token prose file instead. That cost an
# entire evening: four "legacy replay" arms compared a LONG-prompt run against a
# SHORT-prompt baseline (mean len 4.89) and produced a phantom 2x decode regression
# that was chased through gather, ubatch, spec, and a full spine revert before the
# harness itself turned out to be the bug. Without the colon, PROMPT_FILE="" means
# exactly what it says: no file, use $PROMPT.
PROMPT_FILE=${PROMPT_FILE-/tmp/claude-1000/-home-kmbandy/87d16c2e-6d13-4480-bcdf-d27bcd4d9c55/scratchpad/prose739.txt}

# PIN=<blocks> holds those blocks' routed experts RESIDENT in worker VRAM instead
# of paging them (--weight-paging-resident-experts / WP_EXPERT_RESIDENT_EXPERTS,
# added 8adc35f3a). Syntax is the router's range form: "43-45" or "0-6,20-22".
# Per-worker overrides let you pin on one card and leave the others as controls.
#   PIN=43-45              pin the DSpark stages everywhere
#   PIN_MAIN=43-45         R9700 only (it holds experts 85-255, so 513 pages = 6.39 GiB)
#   PIN_1070= / PIN_480=   the 8 GiB cards (85 experts each = 255 pages = 3.18 GiB)
# COST: pinned bytes come out of the slot budget. On an 8 GiB card, pinning 43-45
# drops slots from 550 to roughly 295, and slots have measured ~+10.9% per 100.
# This may well be net NEGATIVE -- that is what it exists to measure.
PIN=${PIN:-}
PIN_MAIN=${PIN_MAIN:-$PIN}
PIN_1070=${PIN_1070:-$PIN}
PIN_480=${PIN_480:-$PIN}

# RESERVE=<blocks> + RESERVE_BYTES=<size> gives those blocks a slot partition
# nothing else may evict (9e0a6a3ca). Pages still arrive by demand paging and LRU
# still runs INSIDE the partition, so the experts the current prompt actually
# routes to settle in on their own -- no offline hot set, and prompt-independent.
# Prefer this over PIN: PIN holds whole blocks (9.56 GiB for 43-45), a reservation
# holds only what gets used, bounded by RESERVE_BYTES.
#   RESERVE=43-45 RESERVE_BYTES=2GiB     ~157 pages per worker
# Watch n_pagein_reserved in the worker stats: it should warm and then go quiet.
RESERVE=${RESERVE:-}
RESERVE_BYTES=${RESERVE_BYTES:-2GiB}
RESERVE_MAIN=${RESERVE_MAIN:-$RESERVE}
RESERVE_1070=${RESERVE_1070:-$RESERVE}
RESERVE_480=${RESERVE_480:-$RESERVE}

WSTATS=${WSTATS:-1}
SLOTS_R9700=${SLOTS_R9700:-2200}
# 2026-08-02: kmbandy raised the two 8 GB cards 500 -> 550. Measured headroom at
# 500 was RX480 vram_used 6397 MiB and GTX1070 5721 MiB at 400, i.e. ~620 MiB of
# per-worker overhead on top of slots*12.75 MiB. At 550 that projects to
# ~7634 MiB of 8192 on the 1070 -- roughly 560 MiB spare. TIGHT. If either
# worker OOMs, drop straight back to 500 rather than trimming anything else;
# the 500/500/2200 figure is the one with a banked result behind it.
SLOTS_1070=${SLOTS_1070:-550}
SLOTS_480=${SLOTS_480:-550}

# DSPARK_HOST -- where the experts-0..84 half of the DSpark stages is served.
#   ""        current topology: the 1070/480 serve ALL 46 layers (full manifest)
#   ROCm1     mad-lab-main 6900 XT -- the card that ALSO runs the dense spine
#   ROCm0     mad-lab-main R9700   -- shares the box with the 85..255 worker
# Setting it moves blk.43/44/45 to main and hands the 1070/480 the TRUNK manifest.
# Both halves of every DSpark expert are then on main, so the speculative draft
# pass stops crossing Tailscale -- that is the point, the slot test rides along.
#
# THIS SWAP IS NOT OPTIONAL, it is forced by the dispatcher: build_routes()
# (pipe-expert-dispatcher.cpp:412-441) throws "advertised by more than one
# machine" if an expert on a layer is served from both boxes. Withdrawing
# layers 43-45 from 2026 is what makes room for main to claim them.
# *** CONFIG OF RECORD: DSPARK_HOST=CPU. *** Banked 6.738 vs 6.535 baseline (9 runs,
# 2026-08-02) and EVERY 2026-08-04 measurement used it. Leaving this empty gives
# DSPARK_HOST=none, i.e. a run that looks like the config of record and is not --
# the same class of trap as SPEC defaulting off (see the rule above).
DSPARK_HOST=${DSPARK_HOST-CPU}

# DSPARK_SPLIT=cpu -- BOTH machines serve their OWN DSpark half on their OWN CPU,
# mirroring the baseline's expert-index split instead of relocating it:
#   2026  i7-6700K  layers 43-45, experts 0..84    (the 1070+480's range)
#   main  3900X     layers 43-45, experts 85..255  (the R9700's range)
# Both machines fall back to trunk manifests, so every GPU worker serves 0..42 only.
# Routing stays legal: per DSpark layer, 0..84 resolves to 2026 and 85..255 to main,
# one machine each, no coverage gap.
# NOTE this REINTRODUCES the Tailscale hop for the 0..84 half of the draft pass,
# which DSPARK_HOST=CPU avoids by keeping both halves on main. That is the point of
# the arm -- it isolates CPU-vs-GPU at CONSTANT topology.
DSPARK_SPLIT=${DSPARK_SPLIT:-}
# 85..255 is 513 pages = 6.39 GiB fully resident, and main has ~10 GB available with
# ~2.6 GB already swapped. Cap it and let LRU work: the 0..84 half touched only
# 109 of 255 pages, so expect ~220 live here. Watch n_pagein flatline.
SLOTS_DSPARK_MAIN=${SLOTS_DSPARK_MAIN:-256}
SLOTS_DSPARK_2026=${SLOTS_DSPARK_2026:-256}

# NO_480=1 drops the RX 480 (Vulkan) worker entirely. The 1070 already advertises
# the SAME expert range 0-84, so coverage is preserved and build_routes stays happy
# (both are on 2026, so the one-machine-per-expert rule is unaffected). The 1070
# just pages harder with the whole range to itself.
# WHY IT EXISTS: 2026-08-03 ablation. The pipeline has a GLOBAL intermittent
# last-bit non-determinism (f16 diverged in 1 of 6 reps, turbo4 in 3 of 3 pairs).
# Every turbo4-specific path was audited clean, so the suspect moved to the expert
# workers, which span three backends -- and the RX 480's Vulkan path is the least
# mature and entirely unexamined. Dropping it is the direct test.
NO_480=${NO_480:-}
if [ -n "$NO_480" ]; then
    DISPATCH_ENDPOINTS="${IPMAIN}:8801,${IP2026}:8803"
    echo "  NO_480: RX 480 dropped; 1070 serves all of 0-84"
else
    DISPATCH_ENDPOINTS="${IPMAIN}:8801,${IP2026}:8803,${IP2026}:8804"
fi
# The whole 0..84 DSpark set is 255 pages x 12.75 MiB = 3.18 GiB, so 256 slots
# holds ALL of it resident and the reserve knob becomes irrelevant on this
# worker -- there is nothing to evict. Lower it to force paging on purpose.
SLOTS_DSPARK=${SLOTS_DSPARK:-256}

if [ -n "$DSPARK_HOST" ] && [ -n "$DSPARK_SPLIT" ]; then
    echo "*** DSPARK_HOST and DSPARK_SPLIT are mutually exclusive -- pick one ***"; exit 1
fi

if [ -n "$DSPARK_HOST" ]; then
    ES_2026=$ES_2026_TRUNK
    DISPATCH_ENDPOINTS="$DISPATCH_ENDPOINTS,${IPMAIN}:8802"
    echo "=== DSpark experts 0..84 -> mad-lab-main $DSPARK_HOST ($SLOTS_DSPARK slots) ==="
    echo "    2026 workers fall back to layers 0..42 (trunk manifest)"
fi

if [ -n "$DSPARK_SPLIT" ]; then
    ES_2026=$ES_2026_TRUNK
    ES_MAIN=$ES_MAIN_TRUNK
    DISPATCH_ENDPOINTS="$DISPATCH_ENDPOINTS,${IPMAIN}:8802,${IP2026}:8805"
    echo "=== DSpark split across BOTH CPUs (mirrors the baseline expert split) ==="
    echo "    2026 i7-6700K : blk.43-45 experts 0..84    ($SLOTS_DSPARK_2026 slots)"
    echo "    main  3900X   : blk.43-45 experts 85..255  ($SLOTS_DSPARK_MAIN slots)"
    echo "    every GPU worker falls back to layers 0..42 (trunk manifests)"
fi

mkdir -p "$OUT"; ssh mad-lab-main "mkdir -p $OUT"
LOCAL_PIDS=""

remote_kill() {   # remote_kill <pid> -- SIGINT, then SIGKILL. Never by pattern.
    [ -z "${1:-}" ] && return
    ssh mad-lab-main "kill -INT $1" 2>/dev/null
    sleep 6
    ssh mad-lab-main "kill -0 $1" 2>/dev/null && ssh mad-lab-main "kill -9 $1" 2>/dev/null
}

cleanup() {
    echo; echo "=== stopping ==="
    for p in $LOCAL_PIDS; do kill -INT "$p" 2>/dev/null; done
    remote_kill "${SPINE_PID:-}"
    remote_kill "${MAIN_PID:-}"
    remote_kill "${DSPARK_PID:-}"
    sleep 4
    for p in $LOCAL_PIDS; do kill -9 "$p" 2>/dev/null; done
    echo "  verifying nothing leaked:"
    ssh mad-lab-main "ss -ltn | grep -E ':8095 |:8801 |:8802 '" 2>/dev/null && echo "  *** MAIN PORTS STILL BOUND ***" || echo "  main ports free"
}
trap cleanup EXIT

# ---------- preflight: refuse to run into a stale process ----------
MAIN_PORTS="8095 8801"
[ -n "$DSPARK_HOST" ] && MAIN_PORTS="$MAIN_PORTS 8802"
[ -n "$DSPARK_SPLIT" ] && MAIN_PORTS="$MAIN_PORTS 8802"
LOCAL_PORTS="8803 8804"
[ -n "$DSPARK_SPLIT" ] && LOCAL_PORTS="$LOCAL_PORTS 8805"
for port in $MAIN_PORTS; do
    if [ "$(ssh mad-lab-main "ss -ltn | grep -c ':$port '" 2>/dev/null | tail -1)" -ge 1 ] 2>/dev/null; then
        echo "*** mad-lab-main port $port ALREADY BOUND -- refusing to run ***"; exit 1
    fi
done
for port in $LOCAL_PORTS; do
    if [ "$(ss -ltn 2>/dev/null | grep -c ":$port ")" -ge 1 ]; then
        echo "*** local port $port ALREADY BOUND -- refusing to run ***"; exit 1
    fi
done

# ---------- WORKERS FIRST -- NOT a habit, a hard constraint.
# The expert dispatcher opens its worker connections inside llama_init_from_model,
# not lazily on first use. Starting the spine first (2026-08-01 attempt) aborts
# with "expert dispatcher failed to connect to worker :8801" ~8 s in, AFTER the
# dense model has loaded. Verified: 1270 tensors landed on ROCm1 including the
# DSpark heads before the connect failed, so the reader is fine -- the ordering
# is simply not negotiable.
echo "=== worker: R9700 (ROCm0) experts 85..255, $SLOTS_R9700 slots ==="
# The R9700 was the ONLY worker with no phase instrumentation: REQLOG/REFLOG/
# PAGEINLOG are carried in VKENV, which is built in the 2026 loop below and never
# reaches this launch. That is why every breakdown so far has had a blank column
# for this card -- we could see what the spine WAITED for it, never what it SPENT.
# Logs land in $OUT on main (created above) and are collected after the run.
MAINENV=""
# Host victim tier: per MACHINE, not via WPRE, because the two boxes have very
# different RAM. The DSpark workers are deliberately excluded -- their shard fits
# their slots (4.3% of references page in), so there is nothing for a victim
# cache to catch and the arena would be pure RAM cost on an already tight box.
[ "${HOSTVICTIM_MAIN:-0}" != "0" ] && MAINENV="$MAINENV WP_EXPERT_HOST_VICTIM_BYTES=$HOSTVICTIM_MAIN"
[ -n "${REQLOG:-}"  ] && MAINENV="$MAINENV WP_REQ_LOG=$OUT/req-w-r9700.txt"
[ -n "${REFLOG:-}"  ] && MAINENV="$MAINENV WP_REF_LOG=$OUT/ref-w-r9700.txt"
[ -n "${PAGEINLOG:-}" ] && MAINENV="$MAINENV WP_PAGEIN_LOG=$OUT/pagein-w-r9700.txt"
# HINTLOG=1 is the ONLY durable record of the hint counters -- the worker prints
# them to stderr on a clean close and this harness SIGKILLs workers, so arm 1
# lost foreign_expert (the spine-vs-worker routing-agreement check) entirely.
# `tail -1` of each of these four files is that worker's final count.
[ -n "${HINTLOG:-}" ] && MAINENV="$MAINENV WP_HINT_LOG=$OUT/hint-w-r9700.txt"
[ -n "$PIN_MAIN" ] && MAINENV="$MAINENV WP_EXPERT_RESIDENT_EXPERTS=$PIN_MAIN" && echo "  PIN(r9700)=$PIN_MAIN"
[ -n "$RESERVE_MAIN" ] && MAINENV="$MAINENV WP_EXPERT_RESERVE_BLOCKS=$RESERVE_MAIN WP_EXPERT_RESERVE_BYTES=$RESERVE_BYTES" && echo "  RESERVE(r9700)=$RESERVE_MAIN @ $RESERVE_BYTES"
ssh mad-lab-main "cd $MAIN_REPO; and env WP_WORKER_STATS=$WSTATS $WPRE $MAINENV setsid nohup stdbuf -o0 -e0 ./build-hip/bin/llama-wp-expert-worker \
    --shard-manifest $ES_MAIN/ds4-e085-255-experts-experts-manifest.json \
    --descriptor $ES_MAIN/ds4-e085-255-experts.expert-descriptor.json \
    --device ROCm0 --listen 0.0.0.0:8801 --slots $SLOTS_R9700 > $OUT/w-r9700.log 2>&1 & echo \$last_pid > $OUT/w-r9700.pid"
sleep 3
MAIN_PID=$(ssh mad-lab-main "cat $OUT/w-r9700.pid" 2>/dev/null)
echo "  w-r9700 pid ${MAIN_PID:-?}"

# ---------- optional 4th worker: DSpark stages 43..45, experts 0..84, ON MAIN ----------
# Deliberately NOT given the reserve knob by default: at SLOTS_DSPARK=256 the
# entire 3.18 GiB set is resident, so there is nothing for a reserve partition
# to protect. Drop SLOTS_DSPARK below 255 first if you want to exercise it.
if [ -n "$DSPARK_HOST" ]; then
    echo "=== worker: DSpark blk.43-45 ($DSPARK_HOST) experts 0..84, $SLOTS_DSPARK slots ==="
    DSENV=""
    # DSPARK_OMP=<n> caps the CPU DSpark worker's thread count. ggml is built with
    # GGML_OPENMP=ON and links libomp, so OMP_NUM_THREADS is honoured without a
    # rebuild; the worker otherwise runs hardware_concurrency() = 24 on the 3900X
    # (wp-expert-worker.cpp:954).
    # WHY: 2026-08-03. The CPU DSpark worker is the confirmed source of the
    # pipeline's intermittent non-determinism (12/12 identical without it, 3/3
    # pairs divergent with it). DSPARK_OMP=1 is the one-variable test of whether
    # the cause is CPU THREADING -- if determinism returns at 1 thread, it is.
    # OMP_THREAD_LIMIT, NOT OMP_NUM_THREADS. ggml launches its parallel region as
    #   #pragma omp parallel num_threads(n_threads)      (ggml-cpu.c:3476)
    # and an explicit num_threads clause OVERRIDES OMP_NUM_THREADS -- so setting
    # OMP_NUM_THREADS alone is a NO-OP here and silently leaves the worker at 24
    # threads. OMP_THREAD_LIMIT is a hard program-wide ceiling that num_threads
    # cannot exceed, and ggml re-reads omp_get_num_threads() inside the region
    # (:3481) so it adapts correctly. Both are set; the LIMIT is the load-bearing one.
    [ -n "${DSPARK_OMP:-}" ] && DSENV="$DSENV OMP_THREAD_LIMIT=$DSPARK_OMP OMP_NUM_THREADS=$DSPARK_OMP" && echo "  DSPARK_OMP=$DSPARK_OMP (CPU DSpark worker thread cap, via OMP_THREAD_LIMIT)"
    [ -n "${PAGEINLOG:-}" ] && DSENV="$DSENV WP_PAGEIN_LOG=$OUT/pagein-w-dspark.txt"
    [ -n "${HINTLOG:-}" ] && DSENV="$DSENV WP_HINT_LOG=$OUT/hint-w-dspark.txt"
    [ -n "${RESERVE_DSPARK:-}" ] && DSENV="$DSENV WP_EXPERT_RESERVE_BLOCKS=$RESERVE_DSPARK WP_EXPERT_RESERVE_BYTES=$RESERVE_BYTES" && echo "  RESERVE(dspark)=$RESERVE_DSPARK @ $RESERVE_BYTES"
    ssh mad-lab-main "cd $MAIN_REPO; and env WP_WORKER_STATS=$WSTATS $WPRE $DSENV setsid nohup stdbuf -o0 -e0 ./build-hip/bin/llama-wp-expert-worker \
        --shard-manifest $ES_DSPARK/ds4-e000-084-experts-experts-manifest.json \
        --descriptor $ES_DSPARK/ds4-e000-084-experts.expert-descriptor.json \
        --device $DSPARK_HOST --listen 0.0.0.0:8802 --slots $SLOTS_DSPARK > $OUT/w-dspark.log 2>&1 & echo \$last_pid > $OUT/w-dspark.pid"
    sleep 3
    DSPARK_PID=$(ssh mad-lab-main "cat $OUT/w-dspark.pid" 2>/dev/null)
    echo "  w-dspark pid ${DSPARK_PID:-?}"
fi

# ---------- DSPARK_SPLIT=cpu: one CPU worker per machine, each on its own half ----------
if [ -n "$DSPARK_SPLIT" ]; then
    echo "=== worker: DSpark blk.43-45 experts 85..255 on main CPU, $SLOTS_DSPARK_MAIN slots ==="
    ssh mad-lab-main "cd $MAIN_REPO; and env WP_WORKER_STATS=$WSTATS $WPRE setsid nohup stdbuf -o0 -e0 ./build-hip/bin/llama-wp-expert-worker \
        --shard-manifest $ES_MAIN_DSPARK/ds4-e085-255-experts-experts-manifest.json \
        --descriptor $ES_MAIN_DSPARK/ds4-e085-255-experts.expert-descriptor.json \
        --device CPU --listen 0.0.0.0:8802 --slots $SLOTS_DSPARK_MAIN > $OUT/w-dspark.log 2>&1 & echo \$last_pid > $OUT/w-dspark.pid"
    sleep 3
    DSPARK_PID=$(ssh mad-lab-main "cat $OUT/w-dspark.pid" 2>/dev/null)
    echo "  w-dspark(main CPU) pid ${DSPARK_PID:-?}"

    echo "=== worker: DSpark blk.43-45 experts 0..84 on 2026 CPU, $SLOTS_DSPARK_2026 slots ==="
    env WP_WORKER_STATS=$WSTATS $WPRE setsid nohup stdbuf -o0 -e0 "$MAIN_REPO"/build-army-cachy/bin/llama-wp-expert-worker \
        --shard-manifest $ES_2026_DSPARK/ds4-e000-084-experts-experts-manifest.json \
        --descriptor $ES_2026_DSPARK/ds4-e000-084-experts.expert-descriptor.json \
        --device CPU --listen 0.0.0.0:8805 --slots $SLOTS_DSPARK_2026 > "$OUT/w-dspark-2026.log" 2>&1 &
    LOCAL_PIDS="$LOCAL_PIDS $!"
    echo "  w-dspark(2026 CPU) pid $!"
fi

echo "=== workers on 2026: 1070 (CUDA0), RX 480 (Vulkan0), experts 0..84 ==="
cd "$MAIN_REPO" || exit 1
# DEV_1070 selects the 1070's BACKEND. CUDA0 (default) vs Vulkan1 is the arm that
# separates "Vulkan backend is slow" from "Polaris has no integer dot product":
# the 1070 reports int dot:1 under Vulkan, the RX 480 reports int dot:0, so
# Vulkan-on-1070 vs CUDA-on-1070 is pure backend cost on identical silicon, and
# Vulkan-on-1070 vs Vulkan-on-480 is pure hardware cost on an identical backend.
DEV_1070=${DEV_1070:-CUDA0}
# NOTE: must stay a plain `for` in THIS shell. A `while read` over a pipe runs in a
# subshell and LOCAL_PIDS (set in the body, used by the cleanup trap) would be lost,
# leaking workers that hold ~7 GB each.
SPEC_480="Vulkan0 8804 w-480 $SLOTS_480"
[ -n "$NO_480" ] && SPEC_480=""
for spec in "$DEV_1070 8803 w-1070 $SLOTS_1070" ${SPEC_480:+"$SPEC_480"}; do
    set -- $spec
    # ANY Vulkan device must use posix_memalign staging, not just Vulkan0: the
    # host buffer type returns 4096-aligned BAR memory that passes every check
    # and then fails O_DIRECT read() at layer 3. Matching "Vulkan0" literally
    # would silently take the broken path for the Vulkan1 (1070) arm.
    case "$1" in Vulkan*) PIN=0 ;; *) PIN=1 ;; esac
    # GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM: the RX 480 has a 256 MB BAR (no
    # ReBAR). ggml_vk_create_buffer_device prefers DeviceLocal|HostVisible,
    # which is that BAR window -- and on AMD it OVERSUBSCRIBES into system RAM
    # rather than failing, so every slot buffer past ~19 (255 MB) physically
    # lives in GTT and every matmul against it streams over PCIe. Measured
    # 2026-08-01: reading the 400th buffer costs 1413 us by default and 193 us
    # with this set; the boundary is exactly at 255 MB. The worker allocates
    # 400 slots = 5.1 GiB, so ~95% of its experts were in system RAM.
    VKENV=""
    # Host victim tier, 2026 side. Small on purpose: this box has 15 GB TOTAL,
    # ~8 GB available, and also runs the nemotron embedder and llama-router
    # (LIVE FLEET SERVICES) plus three workers. Two GPU workers x this budget.
    [ "${HOSTVICTIM_2026:-0}" != "0" ] && VKENV="WP_EXPERT_HOST_VICTIM_BYTES=$HOSTVICTIM_2026"
    # VKFIX=0 disables it, for a controlled A/B against the pre-fix behaviour.
    # VKFIX=1 (default) forces ALL Vulkan buffers device-local -- fixes the bulk
    # weight spill but costs 1.69 ms/req on the small activation upload.
    # VKSPLIT=<bytes> instead keeps the BAR for buffers <= N and forces larger
    # ones device-local, so small frequently-written buffers keep memcpy writes.
    case "$1" in Vulkan*)
        if [ -n "${VKSPLIT:-}" ]; then
            VKENV="GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES=$VKSPLIT"
        elif [ "${VKFIX:-1}" = "1" ]; then
            VKENV="GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1"
        fi ;;
    esac
    # PROBE=N re-times a STATIC pre-built graph every N requests while serving.
    # Static stays fast while real requests are slow => per-request graph content.
    # Static degrades too => process/backend state under load.
    [ -n "${PROBE:-}" ] && case "$1" in Vulkan*) VKENV="$VKENV WP_SELF_BENCH=1 WP_SELF_BENCH_EVERY=$PROBE" ;; esac
    # PAGEINLOG=1 records every page each 2026 worker READS, so the two logs can be
    # intersected to measure whether residency-affinity routing keeps their caches
    # disjoint or whether they duplicate fetches of the same pages.
    # $ARM in the path: a fixed filename here silently overwrote a previous arm's
    # miss log, the same way a fixed OUT= destroyed a previous arm's run dir.
    [ -n "${PAGEINLOG:-}" ] && VKENV="$VKENV WP_PAGEIN_LOG=/tmp/claude-1000/pagein-$ARM-$3.txt"
    # REQLOG=1 dumps one line per dispatch request (layer + every phase timer).
    # Unlike PAGEINLOG these are self-segmenting into tokens via the layer index,
    # so no cross-machine clock join is needed to get a per-token breakdown.
    # Requires WP_WORKER_STATS=1 -- the phase timers are gated on it.
    [ -n "${REQLOG:-}" ] && VKENV="$VKENV WP_REQ_LOG=/tmp/claude-1000/req-$ARM-$3.txt"
    # REFLOG=1 captures the policy-INDEPENDENT reference stream, so cache
    # replacement policies can be simulated offline instead of measured on GPUs.
    [ -n "${REFLOG:-}" ] && VKENV="$VKENV WP_REF_LOG=/tmp/claude-1000/ref-$ARM-$3.txt"
    # HINTLOG=1: hint counters, per frame, fflushed -- see the R9700 block. $ARM
    # in the path for the same reason PAGEINLOG carries it.
    [ -n "${HINTLOG:-}" ] && VKENV="$VKENV WP_HINT_LOG=/tmp/claude-1000/hint-$ARM-$3.txt"
    [ -n "${ALLOCLOG:-}" ] && case "$1" in Vulkan*) VKENV="$VKENV GGML_VK_ALLOC_LOG=1" ;; esac
    # VKEXTRA passes arbitrary GGML_VK_* env to the Vulkan workers only, so the
    # CUDA arm stays a clean control. Quote it: it may contain several settings.
    [ -n "${VKEXTRA:-}" ] && case "$1" in Vulkan*) VKENV="$VKENV $VKEXTRA" ;; esac
    # KEEPALIVE=<us>: occupy the GPU between requests instead of letting it idle.
    # Vulkan-only on purpose -- it targets the RX 480's idle-recovery cost, and
    # leaving the CUDA 1070 without it keeps a clean control in the same run.
    [ -n "${KEEPALIVE:-}" ] && case "$1" in Vulkan*) VKENV="$VKENV WP_KEEPALIVE_US=$KEEPALIVE" ;; esac
    # PIN is per-worker so one card can be pinned while the other stays a control
    # in the SAME run -- the pattern that made the keepalive result credible.
    case "$3" in w-1070) WPIN="$PIN_1070" ;; w-480) WPIN="$PIN_480" ;; *) WPIN="" ;; esac
    [ -n "$WPIN" ] && VKENV="$VKENV WP_EXPERT_RESIDENT_EXPERTS=$WPIN" && echo "  PIN($3)=$WPIN"
    case "$3" in w-1070) WRES="$RESERVE_1070" ;; w-480) WRES="$RESERVE_480" ;; *) WRES="" ;; esac
    [ -n "$WRES" ] && VKENV="$VKENV WP_EXPERT_RESERVE_BLOCKS=$WRES WP_EXPERT_RESERVE_BYTES=$RESERVE_BYTES" && echo "  RESERVE($3)=$WRES @ $RESERVE_BYTES"
    env $VKENV $WPRE WP_STAGING_PINNED=$PIN WP_WORKER_STATS=$WSTATS setsid nohup stdbuf -o0 -e0 ./build-army-cachy/bin/llama-wp-expert-worker \
        --shard-manifest $ES_2026/ds4-e000-084-experts-experts-manifest.json \
        --descriptor $ES_2026/ds4-e000-084-experts.expert-descriptor.json \
        --device "$1" --listen 0.0.0.0:"$2" --slots "$4" > "$OUT/$3.log" 2>&1 &
    LOCAL_PIDS="$LOCAL_PIDS $!"
    echo "  $3 on $1:$2 slots=$4 (pid $!)"
done

echo "=== waiting for all workers to listen ==="
for _ in $(seq 1 900); do
    a=$(ssh mad-lab-main "ss -ltn | grep -c ':8801 '" 2>/dev/null | tail -1)
    b=$(ss -ltn 2>/dev/null | grep -c ":8803 ")
    if [ -n "$NO_480" ]; then c=1; else c=$(ss -ltn 2>/dev/null | grep -c ":8804 "); fi
    if [ -n "$DSPARK_HOST" ] || [ -n "$DSPARK_SPLIT" ]; then
        d=$(ssh mad-lab-main "ss -ltn | grep -c ':8802 '" 2>/dev/null | tail -1)
    else
        d=1
    fi
    if [ -n "$DSPARK_SPLIT" ]; then
        e=$(ss -ltn 2>/dev/null | grep -c ":8805 ")
    else
        e=1
    fi
    [ "${a:-0}" -ge 1 ] && [ "$b" -ge 1 ] && [ "$c" -ge 1 ] && [ "${d:-0}" -ge 1 ] && [ "${e:-0}" -ge 1 ] && break
    sleep 2
done
echo "  r9700=${a:-0} 1070=$b 480=$c dspark=${d:-0} dspark2026=${e:-0}"
[ "${a:-0}" -ge 1 ] && [ "$b" -ge 1 ] && [ "$c" -ge 1 ] && [ "${d:-0}" -ge 1 ] && [ "${e:-0}" -ge 1 ] || {
    echo "NOT ALL WORKERS LISTENING"; tail -20 "$OUT/w-1070.log" "$OUT/w-480.log"
    ssh mad-lab-main "tail -20 $OUT/w-r9700.log"
    [ -n "$DSPARK_HOST" ] && ssh mad-lab-main "tail -20 $OUT/w-dspark.log"; exit 1; }

# SPEC=1 turns on DSpark speculative decoding. The DSpark stages live INSIDE the
# DS4 model (blk.43/44/45 + the nextn heads + fc/enc.output_norm/markov/conf_proj),
# so there is no draft model file -- it follows the MTP pattern of a second context
# on the same model. They load with the rest of the dense path onto ROCm1 (6900 XT).
#
# RUN IT STOCK -- BUT NOT ON THE FIRST RUN. DSpark carries its OWN trained
# confidence head, gated by conf_min, which defaults to 0.9 (common.h:339;
# speculative.cpp:1214-1219 reads llama_get_embeddings_nextn only when conf_min > 0
# and truncates the block at the first position below it).
#
# UNTIL 2026-08-02 THAT DEFAULT HAD NEVER ONCE BEEN EXERCISED -- ctx_dft was always
# null, so the whole DSpark draft path was a silent no-op. e83ba2f17 is the first
# build where it can actually run. Two consequences:
#   1. Any earlier note calling conf_min "active" described the code, not a run.
#   2. Our gate compares the RAW PER-POSITION CONDITIONAL c_k against the threshold.
#      The DSpark paper (arXiv 2607.05147 eq. 7 + alg. 1) specifies the CUMULATIVE
#      PRODUCT a_j = prod_{i<=j} c_i for prefix survival. Ours is therefore a
#      different, more permissive criterion than the number suggests.
# For BRING-UP set SPEC_CONF=0 so the gate cannot truncate every block to zero and
# make "drafting works" indistinguishable from "drafting is gated off". Turn it back
# on only once a draft token has been observed.
#
# *** CONFIG OF RECORD 2026-08-05: SPEC_CONF=0.99 (was the model default, 0.9). ***
# Chosen on the 13-point sweep (0.0-0.99 at 0.1 resolution + 0.95/0.99 + SPEC-off).
# THE BASIS IS THE DETERMINISTIC METRICS, NOT tok/s:
#     conf   acceptance  mean_len   decode(n=1)
#     0.6      0.767       1.77        2.95
#     0.9      0.750       1.76        2.77     <- previous default
#     0.99     0.843       1.86        2.81     <- highest acceptance AND mean_len
#                                                  of the entire sweep
# acceptance and mean_len are DETERMINISTIC per config (all six baseline reps
# returned exactly 0.75000 / 1.76), so they reproduce. decode tok/s at n=1 does
# NOT: the whole 0.1-0.99 range spans 2.77-2.95, entirely inside the measured
# decode envelope of 2.54-2.88, so 0.6's "win" over 0.99 is 0.14 tok/s of noise.
# mean accepted length is what sets decode throughput (decode ~= 1.28 * mean_len
# across every run on record), which is why the tie-break goes to 0.99.
# NOT YET a measured throughput win -- 0.99 vs 0.9 at 6 reps has not been run.
#
# *** THIS INVALIDATES THE 2026-08-05 BASELINE AS A COMPARATOR. *** SHA
# 2e6e3c5985, prefill [20.00, 20.80] and decode [2.54, 2.88] were all measured at
# conf_min=0.9. Re-baseline at 0.99 before gating anything else against them.
# NO COLON. `:-` substitutes on unset OR EMPTY, so SPEC_CONF= would silently
# become 0.99 instead of disabling the flag -- the exact ${PROMPT_FILE:-...} trap
# recorded above, which made every "legacy replay" arm run the wrong prompt.
# With `-`, SPEC_CONF= explicitly means "do not pass --spec-draft-conf-min".
SPEC_CONF=${SPEC_CONF-0.99}
#
# DO NOT reflexively set --spec-draft-p-min here. That is a SEPARATE, generic gate
# on the drafted token's own sampled probability (:1254). The standing "p_min must
# never be 0" rule was written for the generic draft-model path, where p_min is the
# ONLY admission gate -- it does not transfer to DSpark, and setting it stacks a
# non-native gate on top of the trained one, so the run stops being stock DSpark.
#
# Every value in play is printed at WARN by speculative.cpp:985 on startup, so the
# log records what actually ran rather than what we assumed.
SPECARGS=""
if [ -n "${SPEC:-}" ]; then
    SPECARGS="--spec-type draft-dspark"
    # Only for deliberate sweeps -- unset means DSpark's own defaults ride.
    [ -n "${SPEC_PMIN:-}" ] && SPECARGS="$SPECARGS --spec-draft-p-min $SPEC_PMIN"
    [ -n "${SPEC_CONF:-}" ] && SPECARGS="$SPECARGS --spec-draft-conf-min $SPEC_CONF"
    [ -n "${SPEC_NMAX:-}" ] && SPECARGS="$SPECARGS --spec-draft-n-max $SPEC_NMAX"
    echo "  SPEC ON: $SPECARGS  (unset knobs = DSpark defaults, conf_min=0.9 active)"
fi

# SELF-PROVING CONFIG LINE. 2026-08-03: an A/B produced a turbo4 arm whose
# acceptance nearly matched the f16 arm, and NOTHING IN ANY LOG could confirm
# which KV type that run actually used -- llama-server suppresses the KV cache
# INFO lines without -v, and the harness never echoed its own flags. A result
# that cannot prove its own configuration is not a result.
# DSPARK_TAP -- how the m parallel mHC residual streams are collapsed at the two
# DSpark layer taps (src/models/deepseek4.cpp:1766, :1849).
#   1 = GATED reduction, alpha_t = sigmoid(W_f RMSNorm(vec(H_t))) (CONFIG OF RECORD)
#   0 = unweighted mean, 1/m per stream -- the SHIPPED DEFECT, control arm only
# WHY GATED IS THE RECORD: DS4-Flash uses Manifold-Constrained Hyper-Connections at
# expansion factor 4, and HyperDFlash (arXiv 2606.26744) specifies an INPUT-DEPENDENT
# gate mirroring the target's own hc_head aggregation -- explicitly "rather than using
# a generic dense projection". An unweighted mean is alpha_j = 1/m for all j, i.e. the
# input-independent special case the paper argues against. Researched 2026-08-04 from
# the primary sources at kmbandy's direction; llama.cpp's implementation appears to
# have consulted only the DSpark paper (2607.05147), which predates hyper-connections
# and assumes ONE consolidated hidden state per layer.
# *** THIS IS A FAITHFULNESS FIX, NOT A TUNING KNOB. *** The mean tap is wrong against
# the architecture regardless of what it measures. Adopted 2026-08-04 on kmbandy's
# instruction. It affects only the DRAFT head -- speculative decoding verifies against
# the unchanged target, so it can move ACCEPTANCE but cannot alter output text.
# HISTORY THIS CORRECTS: the flag shipped this morning defaulting OFF and the harness
# never set it, so it NEVER EXECUTED. Every run of 2026-08-04 reported acceptance
# 0.67213 / mean len 2.52, byte-identical to 5 decimals across ~12 runs -- the
# fingerprint of a code path that never varied, which went unnoticed all day.
DSPARK_TAP=${DSPARK_TAP:-1}
[ "$DSPARK_TAP" = "1" ] && SPINEENV="${SPINEENV:-} WP_DSPARK_TAP_GATED=1"
echo "=== CONFIG: KV=${CACHE_TYPE_K}/${CACHE_TYPE_V} CTX=$CTX NPRED=$NPRED SPEC=${SPEC:-off}" \
     "DSPARK_HOST=${DSPARK_HOST:-none} DSPARK_OMP=${DSPARK_OMP:-default} NO_480=${NO_480:-off}" \
     "IGNORE_EOS=${IGNORE_EOS:-1} slots=${SLOTS_480}/${SLOTS_1070}/${SLOTS_R9700}" \
     "UBATCH=$UBATCH BATCH=${BATCH:-2048-default} GATHER=${WEXTRA:-default-ON}" \
     "DSPARK_TAP=${DSPARK_TAP}(1=gated,0=mean) KEEPALIVE=${KEEPALIVE:-off}" \
     "code-defaults[dispatch_gather/gather_max_frac/compute_chunks/read_stripes/stripe_max_pageins]=${WP_CODE_DEFAULTS:-1/0.90/4/4/4}" \
     "hostvictim[2026/main]=${HOSTVICTIM_2026:-0}/${HOSTVICTIM_MAIN:-0}" \
     "prefetch[hint/spec/chunk/hintlog]=${PREFETCH_HINT:-off}/${SPEC_PAGEIN:-off}/${SPEC_CHUNK:-default-1}/${HINTLOG:-off}" \
     "evict=${LFU:-1-usecount}(0=pure-LRU)" \
     "PROMPT=$( [ -n "$PROMPT_FILE" ] && echo "$(basename "$PROMPT_FILE")/$(wc -w < "$PROMPT_FILE" 2>/dev/null) words" || echo "inline/${#PROMPT} chars" )" \
     "ARM=$ARM ==="
echo "=== spine: 6900 XT (ROCm1), dense fully resident ==="
ssh mad-lab-main "cd $MAIN_REPO; and env WP_DISPATCH_STATS=1 ${SPINEENV:-} stdbuf -o0 -e0 nohup ./build-hip/bin/llama-server \
    -m $DENSE --device ROCm1 --fit off --no-mmap -ngl 99 -c $CTX $SPECARGS $UBARGS \
    --cache-type-k $CACHE_TYPE_K --cache-type-v $CACHE_TYPE_V \
    --expert-dispatch ${DISPATCH_ENDPOINTS} \
    --port 8095 --host 127.0.0.1 > $OUT/spine.log 2>&1 & echo \$last_pid > $OUT/spine.pid"
sleep 3
SPINE_PID=$(ssh mad-lab-main "cat $OUT/spine.pid" 2>/dev/null)
echo "  spine pid ${SPINE_PID:-?}"

echo "=== waiting for spine /health ==="
ready=0
for _ in $(seq 1 900); do
    ssh mad-lab-main "curl -s -m 2 http://127.0.0.1:8095/health" 2>/dev/null | grep -q '"status"' && { ready=1; break; }
    if [ -n "${SPINE_PID:-}" ] && ! ssh mad-lab-main "kill -0 $SPINE_PID" 2>/dev/null; then
        echo "  *** SPINE DIED during init ***"; ssh mad-lab-main "tail -40 $OUT/spine.log"; exit 1
    fi
    sleep 2
done
[ "$ready" -eq 1 ] || { echo "SPINE NOT READY"; ssh mad-lab-main "tail -40 $OUT/spine.log"; exit 1; }
echo "  spine up"

echo
echo "--- VRAM on all four cards, with the spine loaded and workers warm ---"
ssh mad-lab-main 'rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[|Used"'
nvidia-smi --query-gpu=name,memory.used --format=csv 2>/dev/null

# =====================================================================
echo
echo "############ GENERATION ############"
# DEVICE-LEVEL NVMe COUNTERS. The worker's bytes_read is what we ASKED the
# filesystem for; /proc/diskstats is what the DEVICE actually delivered. The two
# diverged 2.49x once before (O_DIRECT aligned to 512 against a 4096-byte btrfs
# f_bsize, plus zstd-compressed extents) and nothing in our own instrumentation
# could see it. Fields: 6 = sectors read (x512 = bytes), 7 = ms spent reading,
# 13 = io_ticks (ms the device was busy at all) -> duty cycle.
DS_2026_0=$(grep -w nvme0n1 /proc/diskstats)
DS_MAIN_0=$(ssh mad-lab-main "bash -lc 'grep -w nvme0n1 /proc/diskstats'" 2>/dev/null)
# Build the request body with json.dumps rather than interpolating into an
# ssh-escaped string: a multi-KB prompt breaks that quoting outright.
PROMPT="$PROMPT" PROMPT_FILE="$PROMPT_FILE" NPRED="$NPRED" python3 - "$OUT/payload.json" <<'PY'
import json, os, sys
p = os.environ["PROMPT"]
if os.environ.get("PROMPT_FILE"):
    p = open(os.environ["PROMPT_FILE"]).read()
# ignore_eos is REQUIRED for a decode measurement. Without it the run decodes
# however many tokens the model felt like: a prompt that does not invite
# continuation (e.g. a doc chunk truncated mid-word) emits EOS immediately and
# you get content:'' and "eval time = 0.00 ms / 1 tokens" -- a VOID decode
# number next to a perfectly valid prefill one. Cost us run 1 of the 2026-08-03
# sweep. With it, decode is always exactly n_predict tokens and runs compare.
# IGNORE_EOS=0 turns it OFF. Needed whenever the run is being judged on FIDELITY
# rather than on a fixed token count, because forcing generation past a natural
# stop produces a repetition loop that DISTORTS EVERY METRIC AT ONCE:
#   - draft acceptance INFLATES (a loop is trivially predictable: 0.988 measured
#     on a run whose output was one sentence repeated)
#   - the coherence ratio reports DEGENERATE for what is really just ignore_eos
#   - decode tok/s INFLATES, because repetitive output routes to a tiny expert
#     set and the weight pager stops missing
# All three fired at once on 2026-08-03 and nearly produced a false "fix validated".
# *** 2026-08-04 NIGHT: THE ABOVE WARNING FIRED FOUR MORE TIMES AND I MISSED IT. ***
# df-ov (4.29 tok/s), ov3-r2, ov3-r3 and cs-05 (3.67 tok/s) were all reported as wins
# before their text was read. Every one was the ignore_eos repetition loop: acceptance
# inflated to 0.863, decode inflated, output collapsed to a wall of repeated CJK. The
# tell was in the FIRST LINE of every prose run all day -- the completion opens with
# "**Chapter 1**", i.e. the model considered prose739 finished, started a new document,
# ran ~70 tokens, hit EOS, and was forced to produce 256 anyway. Everything past that
# is a model with nothing left to say, and WHICH degenerate trap it falls into is set
# by batch-shape FP noise -- which is why it appeared to track whichever knob was last
# touched (conf_min, WP_EXPERT_OVERLAP, n_ubatch). It tracked none of them.
#
# THE STRUCTURAL PROBLEM: ignore_eos is REQUIRED for a comparable fixed-token decode
# number, and it DESTROYS the fidelity signal, and the harness could not tell the two
# apart. NATURAL_LEN closes that: a fidelity-judged run must know how many tokens the
# prompt actually sustains. Below n_natural the output is the model's; above it, the
# output is an artefact and NO quality claim may be made from it.
# Set NATURAL_LEN=<n> once per prompt (measure with IGNORE_EOS=0), or 0 = unknown.
NATURAL_LEN = int(os.environ.get("NATURAL_LEN", "0"))
npred = int(os.environ["NPRED"])
ignore_eos = os.environ.get("IGNORE_EOS", "1") != "0"
if ignore_eos and NATURAL_LEN and npred > NATURAL_LEN:
    print("  *** FIDELITY WARNING: NPRED=%d exceeds this prompt's natural length %d." % (npred, NATURAL_LEN))
    print("  *** Tokens %d..%d are ignore_eos filler. THROUGHPUT IS VALID; the TEXT IS NOT."
          % (NATURAL_LEN, npred))
    print("  *** Do not read acceptance, mean len, or coherence from this run.")
elif ignore_eos and not NATURAL_LEN:
    print("  NOTE: NATURAL_LEN unset -- cannot tell model degeneracy from ignore_eos filler.")
    print("        Measure it once with IGNORE_EOS=0 NPRED=512 before judging any output.")
# RETURN_TOKENS=1 asks the server to echo the generated token IDs into gen.json's
# "tokens" array. Costs NO compute -- it only changes what the response carries --
# so a RETURN_TOKENS=1 arm is still directly comparable to every arm without it.
# Added 2026-08-05 for #27: the artifact ("grieving|cars", "container|cars") is a
# SPURIOUS INSERTED TOKEN, and the text alone cannot say whether the insert is one
# token or several, nor whether it carries a leading space. Default off = the
# payload is byte-identical to every run before this line existed.
json.dump({"prompt": p, "n_predict": npred,
           "temperature": 0, "seed": 0, "cache_prompt": False,
           "ignore_eos": ignore_eos,
           "return_tokens": os.environ.get("RETURN_TOKENS", "0") == "1"},
          open(sys.argv[1], "w"))
print("  payload: %d chars of prompt" % len(p))
PY
scp -q "$OUT/payload.json" mad-lab-main:/tmp/ds4-payload.json

# *** PROBE MODE (#27, 2026-08-05). PROBE_FROM_ARM=<arm> PROBE_N=<k>. ***
# Replays an EXACT prefix -- the prompt's own token IDs plus the first PROBE_N
# generated token IDs of a previous arm -- and asks for the top-PROBE_NPROBS
# distribution at the very next position.
#
# WHY TOKEN IDS AND NOT TEXT: re-tokenising detokenised text can differ at the
# boundary, and a boundary difference is exactly the class of artefact under
# investigation. Passing IDs makes the prefix bit-exact.
#
# WHY IT MUST RUN WITH SPEC OFF: the speculative path never populates probs
# (server-context.cpp: `result.prob = 1.0f; // set later`, "TODO: set
# result.probs"), so probs requested under SPEC come back meaningless. SPEC off
# is also the ground truth we want: the target's own distribution at M=1.
#
# WHAT IT DECIDES: whether the artefact token is REACHABLE. If the intended
# token dominates and the artefact token is ~0, batch-width FP noise (~1e-3)
# cannot flip it and #27 is a real logic bug. If the two are within a factor of
# a few, it is noise and #27 is not a bug at all.
if [ -n "${PROBE_FROM_ARM:-}" ]; then
    echo "  PROBE: exact-prefix top-k from arm $PROBE_FROM_ARM, first ${PROBE_N:-64} generated tokens"
    python3 - "$OUT/payload.json" "$OUT/tokreq.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
json.dump({"content": d["prompt"], "add_special": True}, open(sys.argv[2], "w"))
PY
    scp -q "$OUT/tokreq.json" mad-lab-main:/tmp/ds4-tokreq.json
    ssh mad-lab-main "curl -s -m 300 http://127.0.0.1:8095/tokenize \
      -H 'Content-Type: application/json' --data-binary @/tmp/ds4-tokreq.json" \
      > "$OUT/tokens.json" 2>/dev/null
    PROBE_N="${PROBE_N:-64}" PROBE_NPROBS="${PROBE_NPROBS:-20}" \
    PROBE_SRC="/var/tmp/ds4full-$PROBE_FROM_ARM/gen.json" \
    python3 - "$OUT/tokens.json" "$OUT/payload.json" <<'PY'
import json, os, sys
ptoks = json.load(open(sys.argv[1])).get("tokens")
if not ptoks:
    sys.exit("PROBE: /tokenize returned no tokens -- is the spine up?")
gen = json.load(open(os.environ["PROBE_SRC"])).get("tokens") or []
n = int(os.environ["PROBE_N"])
if len(gen) < n:
    sys.exit("PROBE: source arm has %d tokens, need %d (RETURN_TOKENS=1 on that arm?)" % (len(gen), n))
prefix = list(ptoks) + list(gen[:n])
json.dump({"prompt": prefix, "n_predict": 1,
           "n_probs": int(os.environ["PROBE_NPROBS"]),
           "temperature": 0, "seed": 0, "cache_prompt": False,
           "post_sampling_probs": False,
           "return_tokens": True},
          open(sys.argv[2], "w"))
print("  PROBE: prompt %d tok (%d prompt + %d generated), asking top-%s"
      % (len(prefix), len(ptoks), n, os.environ["PROBE_NPROBS"]))
PY
    scp -q "$OUT/payload.json" mad-lab-main:/tmp/ds4-payload.json
fi

GEN_T0=$(date +%s)
ssh mad-lab-main "curl -s -m 3600 http://127.0.0.1:8095/completion \
  -H 'Content-Type: application/json' --data-binary @/tmp/ds4-payload.json" \
  > "$OUT/gen.json" 2>/dev/null
GEN_EL=$(( $(date +%s) - GEN_T0 ))
DS_2026_1=$(grep -w nvme0n1 /proc/diskstats)
DS_MAIN_1=$(ssh mad-lab-main "bash -lc 'grep -w nvme0n1 /proc/diskstats'" 2>/dev/null)
echo
echo "--- DEVICE-LEVEL NVMe over the generation window (${GEN_EL}s) ---"
DS_ELAPSED="$GEN_EL" DS_A2026="$DS_2026_0" DS_B2026="$DS_2026_1" \
DS_AMAIN="$DS_MAIN_0" DS_BMAIN="$DS_MAIN_1" python3 <<'PY'
import os
el = float(os.environ.get("DS_ELAPSED") or 0) or 1.0
# diskstats: 1-indexed field 6 = sectors read, 7 = ms reading, 13 = io_ticks.
# Index by name, not position, so a kernel that appends fields cannot silently
# shift them (the extended fields 15-20 are newer additions already).
for host, a, b in (("2026", os.environ.get("DS_A2026"), os.environ.get("DS_B2026")),
                   ("main", os.environ.get("DS_AMAIN"), os.environ.get("DS_BMAIN"))):
    if not a or not b:
        print("  %-5s diskstats unavailable" % host); continue
    fa, fb = a.split(), b.split()
    if len(fa) < 13 or len(fb) < 13:
        print("  %-5s diskstats too short (%d fields)" % (host, len(fa))); continue
    gb   = (int(fb[5]) - int(fa[5])) * 512 / 1e9
    read = (int(fb[6]) - int(fa[6])) / 1000.0
    busy = (int(fb[12]) - int(fa[12])) / 1000.0
    # 2026-08-03: reads-completed (field 4) and weighted-io-time (field 14) added.
    # WHY: the profiled run measured 1.68 GB/s on 2026 while an O_DIRECT probe of
    # the SAME files with the SAME 12.75 MiB page size hits 2.84 GB/s on ONE thread
    # and 3.14 GB/s at peak (nvme_probe.py). The drive is at spec, so the deficit is
    # in HOW we ask. Bytes and busy-time alone cannot tell "few large reads at low
    # concurrency" from "many small reads" -- these two fields can:
    #   avg request size : probe measures 501.5 KB (max_sectors_kb=512 caps it).
    #                      Much smaller here => the pager is fragmenting its reads.
    #   avg queue depth  : probe measures 14.85 at ONE thread, because a single
    #                      12.75 MiB preadv already splits into ~25 requests.
    #                      Much lower here => we are serialising, not saturating.
    nreads = int(fb[3]) - int(fa[3])
    weighted = (int(fb[13]) - int(fa[13])) / 1000.0 if len(fb) > 13 else 0.0
    avg_req_kb = (gb * 1e9 / nreads / 1024.0) if nreads else 0.0
    qd = (weighted / busy) if busy > 0 else 0.0
    print("  %-5s device read %6.2f GB   device-busy %6.1f s of %.0fs (%.0f%% duty)"
          % (host, gb, busy, el, 100*busy/el))
    print("        %d requests, avg %.1f KB/req, avg queue depth %.2f"
          % (nreads, avg_req_kb, qd))
    # Rate MUST come from io_ticks (wall time the device was busy), not from
    # field 7: that one sums per-request service time, so at queue depth > 1 it
    # exceeds wall and understates the rate. Validated 2026-08-02 against a
    # 512 MiB O_DIRECT read -- bytes matched 1.000x, field 7 gave 0.81 GB/s
    # against io_ticks' 1.38 and dd's 1.1.
    if busy > 0:
        print("        %.2f GB/s while the device was busy   (summed request time %.1f s)"
              % (gb/busy, read))
PY
python3 - "$OUT/gen.json" "$GEN_EL" "$NPRED" <<'PY'
# *** HARD VALIDITY GATE (kmbandy, 2026-08-04). ***
# This block used to print timings first and flag problems afterwards, as advice.
# That is how a run with DEGENERATE output and a CRASHED second arm got reported
# as a "-38% win" on 2026-08-04. kmbandy: "no degenerate text ever. If it's
# degenerate, it's a problem" and "So you were ready to claim victory on
# degenerate text, a crashed run, and with no prefill number? No. Unacceptable"
#
# THE RULE NOW: validity is decided BEFORE any number is printed, and an invalid
# run PRINTS NO THROUGHPUT AT ALL. A number that is never printed cannot be
# quoted, mined, or "partially" cited later. Exit 3 marks the arm invalid.
import json,sys
def invalid(reason, extra=""):
    print("  *** RUN INVALID: %s ***" % reason)
    if extra:
        print("  %s" % extra)
    print("  NO THROUGHPUT REPORTED FOR THIS ARM -- it is not a result, do not cite it.")
    sys.exit(3)
try:
    raw = open(sys.argv[1]).read()
except Exception as e:
    invalid("could not read the response file (%s)" % e)
if not raw.strip():
    invalid("EMPTY RESPONSE BODY -- the server closed the connection without replying",
            "Check for a coredump: ssh mad-lab-main 'coredumpctl list | tail'")
try:
    d=json.loads(raw)
except Exception as e:
    invalid("response was not JSON (%s)" % e, "first 800 bytes: %r" % raw[:800])
try:
    el=int(sys.argv[2]); n=int(sys.argv[3])
    print("  content: %r" % d.get("content","")[:400])

    # GATE 1 -- COHERENCE, evaluated BEFORE any timing is printed.
    c = d.get("content","") or ""
    w = c.split()
    if len(w) >= 20:
        ratio = len(set(w))/len(w)
        top = max((w.count(x) for x in set(w)), default=0)/len(w)
        print("  COHERENCE: distinct=%.2f  top-token=%.2f" % (ratio, top))
        if ratio < 0.35 or top > 0.25:
            invalid("DEGENERATE OUTPUT (distinct=%.2f, top-token=%.2f)" % (ratio, top),
                    "Degenerate text is a BUG to fix, never a caveat to annotate. "
                    "Check the PROMPT first: a repeated/list-structured prompt (e.g. q1024.txt) "
                    "induces a repetition loop at temp 0 under ignore_eos. Use real varied prose.")
    else:
        invalid("only %d words emitted -- cannot judge coherence" % len(w))

    # GATE 2 -- the timings object must exist, or there is no prefill/decode split.
    t=d.get("timings") or {}
    if not t:
        invalid("no timings object in response -- cannot split prefill from decode")
    # PREFILL AND DECODE ARE REPORTED SEPARATELY, ALWAYS. A combined wall t/s
    # cannot say which half is lagging: at a short prompt it is ~pure decode, at
    # a long one prefill swamps it, so the same number means different things run
    # to run. Quote these two. Do NOT quote wall as throughput.
    # Reached ONLY after both gates pass, so every number below is from a run
    # that produced coherent output and a complete timings object.
    print("  PREFILL: %5s tok  %9.1f ms  ->  %7.2f tok/s   (%6.2f ms/tok)" % (
        t.get("prompt_n"), t.get("prompt_ms",0), t.get("prompt_per_second") or 0,
        t.get("prompt_per_token_ms",0)))
    print("  DECODE : %5s tok  %9.1f ms  ->  %7.2f tok/s   (%6.2f ms/tok)" % (
        t.get("predicted_n"), t.get("predicted_ms",0), t.get("predicted_per_second") or 0,
        t.get("predicted_per_token_ms",0)))
    # time-to-first-token is what "conversational" actually means to a user
    print("  TTFT   : %.2f s  (prefill only; decode adds %.2f s for %s tok)" % (
        t.get("prompt_ms",0)/1000.0, t.get("predicted_ms",0)/1000.0, t.get("predicted_n")))
    print("  [sanity] wall %ds end-to-end incl. request overhead -- not a throughput figure" % el)
    print("  RUN VALID: coherence passed, timings present.")
except SystemExit:
    raise
except Exception as e:
    invalid("unexpected error while validating the response: %r" % e)
PY
GEN_RC=$?
if [ "$GEN_RC" -ne 0 ]; then
    echo "  !! ARM $ARM PRODUCED NO USABLE RESULT (validator exit $GEN_RC)"
fi

echo
echo "--- VRAM after generation ---"
ssh mad-lab-main 'rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[|Used"'
nvidia-smi --query-gpu=name,memory.used --format=csv 2>/dev/null
echo
echo "--- spine timings ---"
ssh mad-lab-main "grep -E 'print_timing|eval time' $OUT/spine.log | tail -6"
echo "--- per-leg dispatch breakdown ---"
ssh mad-lab-main "grep -E 'expert dispatch' $OUT/spine.log | tail -8"
echo "--- worker stats: R9700 / 1070 / 480 ---"
ssh mad-lab-main "grep -iE 'worker stats|requests=' $OUT/w-r9700.log | tail -3"
grep -iE 'worker stats|requests=' "$OUT/w-1070.log" | tail -3
grep -iE 'worker stats|requests=' "$OUT/w-480.log" | tail -3
