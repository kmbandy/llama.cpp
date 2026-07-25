#!/bin/bash
# P0 Unit 2 — HOST vs P2P transport A/B, with the RAM tier off and on.
#
# 2x2 arms:
#   h0 = HOST pthread pool  (WP_ENSURE_BATCH_HOST=1), RAM tier OFF
#   h8 = HOST pthread pool  (WP_ENSURE_BATCH_HOST=1), RAM tier 8 GB
#   p0 = P2P io_uring       (WP_ENSURE_BATCH_HOST unset), RAM tier OFF
#   p8 = P2P io_uring       (WP_ENSURE_BATCH_HOST unset), RAM tier 8 GB
#
# ORDER IS ALTERNATED between rounds: tonight's FFN-island A/B showed the arm
# that ran SECOND was faster in all three rounds regardless of which arm it
# was, so a fixed order manufactures a spurious winner.
#
# Reads the new honest instrumentation (commits 851fc16f9 / bf80ccff4 /
# 547f94bc8): the TRANSPORT line proves which path actually served reads, the
# ACHIEVED CONCURRENCY line separates reads-in-flight from queued jobs, and
# ensure_batch_gb_s is now storage-read-only on both paths.
#
# SAFETY: llama-router.service is LIVE on 8090. We bind 8099 and reap ONLY our
# own PID -- never killsweep/pkill by name.
set -u
BIN=/home/kmbandy/GitHub/llama.cpp/build-hip/bin/llama-server
MODEL=/home/kmbandy/models/ds4/DeepSeek-V4-Flash-Q8-MTP-00001-of-00004.gguf
OUT=/home/kmbandy/wp_logs/p2p
NPRED=${NPRED:-128}
SLOTS=${SLOTS:-5500}
ROUNDS=${ROUNDS:-3}
mkdir -p "$OUT"
SUM=$OUT/summary.tsv
[ -s "$SUM" ] || printf 'tag\thtenv\tram_gb\thost_inflight\teb_gb_s\tio_gb_s\tNVMe_GB\ttok_s\tpages\thits\tp2p_ipeak\tp2p_iavg\tp2p_readwait\tp2p_h2d\tqd\twin\tsync_zc\thjobs\threadwait\thh2d\tpromo_n\tpromo_ms\thost_zc\tstatus\n' > "$SUM"

# rocm-smi indices are REVERSED vs llama's: llama ROCm0 = R9700 = smi GPU[1].
vram_used(){ local smi; case "$1" in 0) smi=1 ;; 1) smi=0 ;; *) echo NA; return ;; esac
  rocm-smi --showmeminfo vram 2>/dev/null | awk -v g="GPU[$smi]" 'index($0,g) && /Total Used Memory/ {print $NF}' | tail -1; }

run(){ # $1=tag  $2=host_transport(1|0)  $3=ram_gb  $4=extra env (optional)
  local tag="$1" ht="$2" gb="$3" extra="${4:-}"
  local LOG=$OUT/$tag.log; : > "$LOG"
  local envs=()
  [ "$ht" = "1" ] && envs+=("WP_ENSURE_BATCH_HOST=1")
  if [ "$extra" = "__MULTI__" ]; then
    # shellcheck disable=SC2206
    for kv in ${ARM_EXTRA:-}; do envs+=("$kv"); done
  elif [ -n "$extra" ]; then
    envs+=("$extra")
  fi
  if [ "$gb" != "0" ]; then
    envs+=("WP_HOST_BUDGET_BYTES=$(python3 -c "print(int($gb*1e9))")")
  fi
  # HARD RAM GUARD. An 8GB pinned HostTier arena on this 16GB box left only
  # 1.1GB available and 7GB swap -- the shape that has OOM'd this machine and
  # killed the user's session manager before. Refuse to launch rather than
  # trusting a human to be watching.
  local need_gb=$(( gb + 3 ))
  local avail_gb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
  if [ "$avail_gb" -lt "$need_gb" ]; then
    echo "######## [$tag] WARNING: MemAvailable ${avail_gb}GB < ${need_gb}GB (tier ${gb}GB + 3GB margin)."
    echo "########          kmbandy's ruling: 8GB/machine stands; free RAM before running rather than shrinking the tier."
    echo "########          Proceeding. Set WP_AB_HARD_GUARD=1 to make this a skip instead."
    if [ "${WP_AB_HARD_GUARD:-0}" = "1" ]; then echo "######## [$tag] SKIPPED (hard guard) ########"; echo; return 1; fi
  fi

  echo "######## [$tag] host_transport=$ht ram_tier=${gb}GB extra=${ARM_EXTRA:-${extra:-none}} (r9700_used_before=$(vram_used 0)) ########"

  env "${envs[@]}" WP_PIN_HOST=0 \
      WP_PREFETCH_XLAYER=0 WP_SPEC_REAP=0 WP_PREFETCH_DEPTH=16 WP_IOURING_DEPTH=16 \
      WP_RESIDENT_DENSE=1 WP_SIZE_CLASS_SLOTS=1 WP_PAGED_BATCH=1 WP_DENSE_PREFETCH_N=0 \
      WP_FADVISE_LOOKAHEAD=0 WP_SAMPLE_ORACLE=0 WP_DRAFT_PREFETCH=0 WP_STICKY_SPEC=0 \
    setsid "$BIN" -m "$MODEL" --no-mmap --weight-paging --weight-paging-slots "$SLOTS" \
      --weight-paging-resident-device ROCm1 --device ROCm0,ROCm1 -ngl 99 \
      -c 4096 --parallel 1 --host 127.0.0.1 --port 8099 >>"$LOG" 2>&1 &
  local SRV=$!

  local ok=0 i
  for i in $(seq 1 1500); do
    curl -s http://127.0.0.1:8099/health 2>/dev/null | grep -q '"status":"ok"' && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || { echo "[$tag] DIED"; grep -aiE "error|assert|abort" "$LOG" | grep -v common_fit_params | tail -6; return 1; }
    sleep 1
  done
  [ $ok = 1 ] || { echo "[$tag] LOAD TIMEOUT"; kill -INT $SRV; sleep 15; kill -9 $SRV 2>/dev/null; return 1; }

  sect(){ awk -v d=nvme0n1p2 '$3==d{print $6}' /proc/diskstats; }
  local s0 s1; s0=$(sect)
  curl -s http://127.0.0.1:8099/completion -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"The capital of France is\",\"n_predict\":$NPRED,\"temperature\":0,\"seed\":0,\"cache_prompt\":false}" \
    > "$OUT/$tag.json" 2>/dev/null
  s1=$(sect)

  kill -INT $SRV 2>/dev/null                     # our PID only
  local w; for w in $(seq 1 240); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
  kill -0 $SRV 2>/dev/null && kill -9 $SRV
  sleep 4

  local GB; GB=$(python3 -c "print(f'{($s1-$s0)*512/1e9:.2f}')")
  python3 - "$OUT/$tag.json" "$LOG" "$tag" "$ht" "$gb" "$GB" "$SUM" "$NPRED" <<'PY'
import json, re, sys
js, log, tag, ht, gb, nvme, sumf, npred = sys.argv[1:9]
try:
    d = json.load(open(js)); t = d.get("timings", {})
    txt = (d.get("content","") or "").replace("\n"," ")
    toks = t.get("predicted_per_second", 0.0); pn = int(t.get("predicted_n", 0) or 0)
except Exception:
    toks, txt, pn = 0.0, "<no output / error>", 0
raw = open(log, errors="ignore").read()
def g(pat, d="0"):
    m = re.search(pat, raw); return m.group(1) if m else d
active   = g(r"TRANSPORT: active=([^\s]+(?: \([^)]*\))?)", "ABSENT")
host_b   = g(r"host_batches=(\d+)");  p2p_b = g(r"p2p_batches=(\d+)")
ser_b    = g(r"serial_batches=(\d+)")
peak     = g(r"inflight_peak=(\d+)"); avg   = g(r"inflight_avg_at_read_start=([0-9.]+)")
eb_gbs   = g(r"ensure_batch_gb_s:\s*([0-9.]+)")
io_gbs   = g(r"io_effective_gb_s:\s*([0-9.]+)")
pages    = g(r"ensure_batch_pages:\s*(\d+)")
hits     = g(r"host_tier_hits:\s*(\d+)")
rw_ms    = g(r"ensure_batch_host_read_wait_ms:\s*([0-9.]+)")
promo_n  = g(r"ensure_batch_host_promotion[a-z_]*count[a-z_]*:\s*(\d+)")
promo_ms = g(r"ensure_batch_host_promotion_h2d_ms:\s*([0-9.]+)")
fresh_n  = g(r"ensure_batch_host_fresh[a-z_]*count[a-z_]*:\s*(\d+)")
fresh_ms = g(r"ensure_batch_host_fresh_h2d_ms:\s*([0-9.]+)")
# SERIAL arms never enter ensure_batch's HOST path; they promote via
# page_in_sync_, so fold those counters in when the ensure_batch ones are 0.
sp_pn = g(r"page_in_sync_promotion_count:\s*(\d+)")
sp_pms= g(r"page_in_sync_promotion_h2d_ms:\s*([0-9.]+)")
sp_fn = g(r"page_in_sync_fresh_count:\s*(\d+)")
sp_fms= g(r"page_in_sync_fresh_h2d_ms:\s*([0-9.]+)")
if promo_n == "0" and sp_pn != "0": promo_n, promo_ms = sp_pn+"(sync)", sp_pms
if fresh_n == "0" and sp_fn != "0": fresh_n, fresh_ms = sp_fn+"(sync)", sp_fms
h2d_ms   = g(r"ensure_batch_host_h2d_ms:\s*([0-9.]+)")
zc_n     = g(r"ensure_batch_host_zerocopy_promotions:\s*(\d+)", "-")
# New in 04ecae824. "ABSENT" here means the counter did not print at all,
# which is itself the finding -- do not silently read it as zero.
p2p_rw   = g(r"ensure_batch_p2p_read_wait_ms:\s*([0-9.]+)", "ABSENT")
p2p_jobs = g(r"ensure_batch_p2p_jobs_ms:\s*([0-9.]+)", "ABSENT")
p2p_prep = g(r"ensure_batch_p2p_prep_ms:\s*([0-9.]+)", "ABSENT")
p2p_enq  = g(r"ensure_batch_p2p_enqueue_ms:\s*([0-9.]+)", "ABSENT")
p2p_h2d  = g(r"ensure_batch_p2p_h2d_ms:\s*([0-9.]+)", "ABSENT")
p2p_fr   = g(r"ensure_batch_p2p_fresh_count:\s*(\d+)", "ABSENT")
p2p_ipk  = g(r"ensure_batch_p2p_inflight_peak:\s*(\d+)", "ABSENT")
p2p_iav  = g(r"ensure_batch_p2p_inflight_avg_at_read_start:\s*([0-9.]+)", "ABSENT")
sync_zc  = g(r"page_in_sync_zerocopy_promotions:\s*(\d+)", "-")
p2p_qd   = g(r"queue_depth=(\d+)", "-")
p2p_win  = g(r"window cache max=(\d+)", "-")
iowq     = g(r"iowq[_ ]max[_ ]workers=(\d+)", "-")
jobs_ms  = g(r"ensure_batch_host_jobs_ms:\s*([0-9.]+)")
enq_ms   = g(r"ensure_batch_host_enqueue_ms:\s*([0-9.]+)")
status = "DEGENERATE" if txt.count("1.1.1") > 3 or "<<<<" in txt else ("SHORT" if pn < int(npred)-8 else "COHERENT")
print("  transport=%s host_b=%s p2p_b=%s serial_b=%s | inflight peak=%s avg=%s | eb_gb_s=%s io_gb_s=%s NVMe=%sGB tok/s=%.3f %s"
      % (active, host_b, p2p_b, ser_b, peak, avg, eb_gbs, io_gbs, nvme, toks, status))
print("    HOST phases: jobs=%sms read_wait=%sms h2d=%sms | promo n=%s %sms (zc=%s) | fresh n=%s %sms | hits=%s"
      % (jobs_ms, rw_ms, h2d_ms, promo_n, promo_ms, zc_n, fresh_n, fresh_ms, hits))
print("    P2P  phases: jobs=%s prep=%s enq=%s read_wait=%s h2d=%s ms | fresh=%s | INFLIGHT peak=%s avg=%s"
      % (p2p_jobs, p2p_prep, p2p_enq, p2p_rw, p2p_h2d, p2p_fr, p2p_ipk, p2p_iav))
print("    P2P  config: queue_depth=%s window_cache=%s iowq=%s | page_in_sync zerocopy=%s"
      % (p2p_qd, p2p_win, iowq, sync_zc))
print("    txt: %s" % txt[:70])
open(sumf,"a").write("\t".join([tag,ht,gb,avg,eb_gbs,io_gbs,nvme,"%.3f"%toks,pages,hits,
                                p2p_ipk,p2p_iav,p2p_rw,p2p_h2d,p2p_qd,p2p_win,sync_zc,
                                jobs_ms,rw_ms,h2d_ms,promo_n,promo_ms,zc_n,status])+"\n")
PY
  echo
}

echo "########## P2P tuning — target 5-6 GB/s with the RAM tier co-resident — $(date) ##########"
echo "commit: $(cd ~/GitHub/llama.cpp && git log --oneline -1)"
echo "NOTE: llama-router.service is live; port 8099, reap own PID only."
echo
# ARMS is a ;-separated list of "tag|ht|ram_gb|extra_env_space_separated".
# Default = the single diagnostic arm: P2P + 4GB tier at stock settings, which is
# the configuration kmbandy actually wants working (P2P AND the RAM tier).
: "${ARMS:=Pdiag|0|4|LLAMA_WP_TRANSPORT=p2p}"

IFS=';' read -ra ARM_LIST <<< "$ARMS"
for spec in "${ARM_LIST[@]}"; do
  IFS='|' read -r tag ht gb extra <<< "$spec"
  # run() takes ONE extra env assignment; pass several by pre-exporting them via
  # a wrapper assignment string. Split on spaces and hand them all to env(1).
  run_multi(){
    local t="$1" h="$2" g="$3"; shift 3
    local saved="$*"
    ARM_EXTRA="$saved" run "$t" "$h" "$g" "__MULTI__"
  }
  run_multi "$tag" "$ht" "$gb" $extra
done

echo "########## SUMMARY ##########"
column -t "$SUM" 2>/dev/null || cat "$SUM"
echo
echo "########## P2P config actually resolved (grep, do not infer) ##########"
grep -ahH "IoUringP2PFileIO: P2P enabled\|create_file_io: active transport\|TRANSPORT: active" $OUT/*.log 2>/dev/null | sed 's/^/  /'
