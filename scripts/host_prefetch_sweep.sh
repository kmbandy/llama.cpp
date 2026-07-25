#!/bin/bash
# GPU integration harness for the HostTier prefetch worker. Do not run without
# a board claim and explicit approval: this starts llama-server on ROCm devices.

set -u

BIN=${BIN:-/home/kmbandy/GitHub/llama.cpp/build-hip/bin/llama-server}
MODEL=${MODEL:-/home/kmbandy/models/ds4/DeepSeek-V4-Flash-Q8-MTP-00001-of-00004.gguf}
OUT=${OUT:-/home/kmbandy/wp_logs/accounting/host-prefetch}
NPRED=${NPRED:-384}
ROUNDS=${ROUNDS:-2}
SLOTS=5500
PORT=${PORT:-8099}
DISK=${DISK:-nvme0n1p2}

mkdir -p "$OUT"
SUM="$OUT/summary.tsv"
: > "$SUM"
printf 'round\tarm\tnvme_gb\twall_s\ttok_s\tpred_n\thost_hits\tenqueued\tread\tskipped\tdropped\tstatus\n' >> "$SUM"

disk_sectors() {
    awk -v disk="$DISK" '$3 == disk { print $6; exit }' /proc/diskstats
}

run_arm() {
    local round="$1" arm="$2" prefetch="$3" minconf="${4:-0}" lookahead="${5:-2}"
    local tag="${round}_${arm}"
    local log="$OUT/${tag}.log"
    local json="$OUT/${tag}.json"
    : > "$log"

    echo "######## [$tag] WP_HOST_PREFETCH=$prefetch MIN_CONF=$minconf K=$lookahead slots=$SLOTS NPRED=$NPRED ########"
    env WP_ENSURE_BATCH_HOST=1 WP_HOST_BUDGET_BYTES=8000000000 WP_PIN_HOST=0 \
        WP_HOST_PREFETCH="$prefetch" WP_HOST_PREFETCH_MIN_CONF="$minconf" WP_HOST_PREFETCH_LOOKAHEAD="$lookahead" WP_PREFETCH_XLAYER=0 WP_SPEC_REAP=0 \
        WP_PREFETCH_DEPTH=16 WP_IOURING_DEPTH=16 WP_RESIDENT_DENSE=1 \
        WP_SIZE_CLASS_SLOTS=1 WP_PAGED_BATCH=1 WP_DENSE_PREFETCH_N=0 \
        WP_FADVISE_LOOKAHEAD=0 WP_SAMPLE_ORACLE=0 WP_DRAFT_PREFETCH=0 \
        WP_STICKY_SPEC=0 \
        setsid "$BIN" -m "$MODEL" --no-mmap --weight-paging \
            --weight-paging-slots "$SLOTS" --weight-paging-resident-device ROCm1 \
            --device ROCm0,ROCm1 -ngl 99 -c 4096 --parallel 1 \
            --host 127.0.0.1 --port "$PORT" >>"$log" 2>&1 &
    local server_pid=$!

    local ready=0 i
    for i in $(seq 1 900); do
        curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"' && {
            ready=1
            break
        }
        kill -0 "$server_pid" 2>/dev/null || {
            echo "[$tag] server exited"
            grep -iE 'error|assert|abort' "$log" | tail -6
            return 1
        }
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "[$tag] health timeout"
        kill -INT "$server_pid" 2>/dev/null || true
        return 1
    fi

    local sectors_before sectors_after time_before time_after
    sectors_before=$(disk_sectors)
    time_before=$(date +%s.%N)
    curl -s "http://127.0.0.1:${PORT}/completion" -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"The history of Rome\",\"n_predict\":$NPRED,\"temperature\":0,\"seed\":0,\"cache_prompt\":false}" \
        > "$json" 2>/dev/null
    sectors_after=$(disk_sectors)
    time_after=$(date +%s.%N)

    kill -INT "$server_pid" 2>/dev/null || true
    for i in $(seq 1 180); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" 2>/dev/null || true
    sleep 4

    python3 - "$json" "$log" "$round" "$arm" "$sectors_before" "$sectors_after" \
        "$time_before" "$time_after" "$SUM" "$NPRED" <<'PY'
import json
import re
import sys

js, log, rnd, arm, s0, s1, t0, t1, summary, npred = sys.argv[1:11]
try:
    response = json.load(open(js))
    timings = response.get("timings", {})
    text = (response.get("content", "") or "").replace("\n", " ")
    tok_s = float(timings.get("predicted_per_second", 0.0) or 0.0)
    pred_n = int(timings.get("predicted_n", 0) or 0)
except Exception:
    text, tok_s, pred_n = "<parse fail>", 0.0, 0
raw = open(log, errors="ignore").read()
def stat(name):
    match = re.search(r"%s:\s*(\d+)" % re.escape(name), raw)
    return match.group(1) if match else "0"
nvme_gb = (int(s1) - int(s0)) * 512 / 1e9
wall_s = float(t1) - float(t0)
status = "DEGENERATE" if text.count("1.1.1") > 3 else (
    "SHORT" if pred_n < int(npred) - 8 else "COHERENT")
values = [stat("host_tier_hits"), stat("host_prefetch_enqueued"),
          stat("host_prefetch_read"), stat("host_prefetch_skipped"),
          stat("host_prefetch_dropped")]
print("  NVMe=%.2f GB wall=%.1fs tok/s=%.3f pred_n=%d | hits=%s enq=%s read=%s skip=%s drop=%s %s"
      % (nvme_gb, wall_s, tok_s, pred_n, *values, status))
print("    txt: %s" % text[:70])
with open(summary, "a") as out:
    out.write("%s\t%s\t%.2f\t%.1f\t%.3f\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" %
              (rnd, arm, nvme_gb, wall_s, tok_s, pred_n, *values, status))
PY
}

echo "Host-prefetch sweep: victim vs prefetch, $ROUNDS interleaved rounds"
for round in $(seq 1 "$ROUNDS"); do
    run_arm "$round" victim 0 0 2
    run_arm "$round" k1c20 1 0.20 1
    run_arm "$round" k1c35 1 0.35 1
    run_arm "$round" k2c35 1 0.35 2
    run_arm "$round" k2c50 1 0.50 2
done

echo "########## SUMMARY ##########"
column -t "$SUM" 2>/dev/null || cat "$SUM"
echo "Reading: prefetch should increase host_tier_hits and reduce NVMe GB."
echo "Watch tok/s for NVMe-bandwidth contention; dropped indicates queue pressure."
echo "Every arm must be COHERENT; DEGENERATE or SHORT invalidates the sweep."
