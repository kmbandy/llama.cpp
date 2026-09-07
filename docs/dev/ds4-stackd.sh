#!/usr/bin/env bash
# DS4 elastic worker supervisor for mad-lab-main.  It intentionally manages
# only the WORKERS, never llama-router.
#
# ---- CONFIG OF RECORD: d3, 2026-09-04 (DS4-Flash two-slice rig) -------------
# This script launches the two d3 worker LAUNCHERS verbatim, with the ds4-gates
# worker env on top.  It no longer runs docs/dev/harness-2026-08-05-ds4_full.sh
# and no longer sources ~/ds4-runs/stackd-worker.env: that file still carries the
# four-worker A/B cells (8802/8804, old ds4-eslice/ds4-eslice4 slice dirs, spec
# caps, HOSTVICTIM_*) and sourcing it with `set -a` injected WP_* vars that are
# NOT part of d3.  It is kept on disk as the historical record; to source it
# again for an experiment, set STACKD_WORKER_ENV=/path explicitly.
#
#   main :8801  slice0 /home/kmbandy/models/ds4-eslice-v2   ROCm0(R9700)+CPU
#               --slots 3350,300   ~/ds4-runs/ds4-eslice-w8801.sh
#   2026 :8803  slice1 /mnt/nvme/ds4-eslice-v2  CUDA0(1070)+Vulkan0(RX480)+CPU
#               --slots 1800,1850,400  ~/ds4-runs/ds4-eslice-w8803.sh (on 2026)
#
# TWO WORKERS ONLY.  There is no 8802, no 8804, and no CPU DSpark worker.
# Slot sets and device lists live in the LAUNCHERS, not here — do not re-encode
# them in this file or in an env file.
#
# Lifecycle:
#   router spawn -> stackd raises workers -> spine retries connect -> serving
#   router eviction -> grace -> stackd stops both workers by port-derived pid
#
# Knobs: STACKD_GRACE_S=300 is the continuous no-spine grace period;
# STACKD_POLL_S=2 is the poll interval; STACKD_ONESHOT=1 prints one cycle's
# proposed action without launching or terminating anything; STACKD_NC names
# an absolute netcat path (default /usr/bin/nc); STACKD_WORKER_ENV names an
# optional env file to source into the worker environment (default: none).
# The spine gets WP_DISPATCH_CONNECT_RETRY_S=180 from the [ds4-flash] preset's
# per-model `env =` line (per-model env exists since 2026-08-27), so it waits
# for these workers instead of failing fast at --expert-dispatch connect.
#
# v1 does not manage board claims for the worker GPUs.  Those claims are
# advisory and remain a human/session concern.

set -u

readonly REPO=/home/kmbandy/GitHub/llama.cpp
readonly RUN_ROOT=/home/kmbandy/ds4-runs
readonly STACKD_LOG="$RUN_ROOT/stackd.log"
readonly POLL_S="${STACKD_POLL_S:-2}"
readonly GRACE_S="${STACKD_GRACE_S:-300}"
readonly ONESHOT="${STACKD_ONESHOT:-0}"
readonly NC="${STACKD_NC:-/usr/bin/nc}"

# ---- d3 topology -----------------------------------------------------------
# Probe the SAME addresses the spine dispatches to (--expert-dispatch
# 127.0.0.1:8801,192.168.1.33:8803).  Probing a tailscale address instead would
# report "ready" for a leg the spine cannot actually reach.
readonly WORKER_MAIN_SH="$RUN_ROOT/ds4-eslice-w8801.sh"
readonly WORKER_2026_SH="\$HOME/ds4-runs/ds4-eslice-w8803.sh"   # expanded ON 2026
readonly MAIN_ADDR=127.0.0.1
readonly MAIN_PORT=8801
readonly R2026_SSH=mad-lab-2026
readonly R2026_ADDR=192.168.1.33
readonly R2026_PORT=8803
readonly SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=8"

# ---- ds4-gates worker env (dethash/ds4-gates.sh, 2026-09-04) ---------------
# The gate adds exactly this on top of each launcher's own env block:
#   WP_WORKER_PIPELINE=1            reader/writer worker threads
#   WP_LOCAL_SHM=1                  MAIN ONLY (loopback shm ring advertised)
#   WP_EXPERT_LFU_PLACEMENT=0       LFU placement off  (kmbandy 2026-09-01)
#   WP_EXPERT_LFU_MIGRATION_CAP=0   no migrations
#   GGML_MUL_MAT_ID_FORCE_MM=1      prefill mul_mat_id pinned to the matrix
#                                   kernel for >= 64 tokens, so per-expert row
#                                   count cannot flip kernels
# The gate's WP_EXPERT_COUNTS_DUMP / WP_REQ_LOG overrides are deliberately NOT
# carried here: those point at gate scratch files.  On the serve path the
# launchers' own defaults keep the living pin files (ds4-pins-main.txt /
# ds4-pins-2026.txt) learning across serves.  No pin files are forced.
readonly GATE_COMMON="WP_WORKER_PIPELINE=1 WP_EXPERT_LFU_PLACEMENT=0 WP_EXPERT_LFU_MIGRATION_CAP=0 GGML_MUL_MAT_ID_FORCE_MM=1"
readonly GATE_MAIN="$GATE_COMMON WP_LOCAL_SHM=1"
readonly GATE_2026="$GATE_COMMON"

valid_nonnegative_integer() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

valid_positive_integer() {
    valid_nonnegative_integer "$1" && [ "$1" -gt 0 ]
}

valid_nonnegative_integer "$GRACE_S" || {
    echo "ds4-stackd: STACKD_GRACE_S must be a non-negative integer" >&2
    exit 2
}
valid_positive_integer "$POLL_S" || {
    echo "ds4-stackd: STACKD_POLL_S must be a positive integer" >&2
    exit 2
}
case "$NC" in
    /*) ;;
    *) echo "ds4-stackd: STACKD_NC must be an absolute path" >&2; exit 2 ;;
esac
# netcat is OPTIONAL: the default probe is bash /dev/tcp (see tcp_open).
# Only enforce an executable when the operator explicitly chose STACKD_NC.
if [ -n "${STACKD_NC:-}" ] && [ ! -x "$NC" ]; then
    echo "ds4-stackd: STACKD_NC=$NC is not executable" >&2
    exit 2
fi

log() {
    local message="$*"
    if [ "$ONESHOT" = 1 ]; then
        printf 'ds4-stackd: WOULD: %s\n' "$message"
        return
    fi
    local line
    line="$(/usr/bin/date '+%F %T') ds4-stackd: $message"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$STACKD_LOG"
}

last_state=''
log_state() {
    local state="$1"
    shift
    if [ "$ONESHOT" != 1 ] && [ "$state" = "$last_state" ]; then
        return
    fi
    last_state=$state
    log "$*"
}

spine_pids() {
    # [l] avoids matching this pgrep invocation.  The router itself has
    # --models-preset, never --expert-dispatch; only its DS4 child matches.
    /usr/bin/pgrep -f '[l]lama-server.*--expert-dispatch' 2>/dev/null || true
}

# Port probe via bash /dev/tcp -- no netcat dependency (nc is absent on
# mad-lab-main). Falls back to $NC only if STACKD_NC was explicitly set.
tcp_open() {
    if [ -n "${STACKD_NC:-}" ] && [ -x "$NC" ]; then
        "$NC" -z -w 1 "$1" "$2"
    else
        timeout 2 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null && exec 3>&- 3<&-
    fi
}

all_worker_ports_open() {
    tcp_open "$MAIN_ADDR"  "$MAIN_PORT" &&
    tcp_open "$R2026_ADDR" "$R2026_PORT"
}

workers_running() {
    /usr/bin/pgrep -f '[l]lama-wp-expert-worker' >/dev/null 2>&1 && all_worker_ports_open
}

# Present = the local worker process exists OR either worker port answers
# (covers the window where local pgrep misses but the remote worker still holds).
workers_present() {
    /usr/bin/pgrep -f '[l]lama-wp-expert-worker' >/dev/null 2>&1 && return 0
    tcp_open "$MAIN_ADDR"  "$MAIN_PORT"  && return 0
    tcp_open "$R2026_ADDR" "$R2026_PORT" && return 0
    return 1
}

# Optional worker-env override file, OFF by default (see the header).  When
# STACKD_WORKER_ENV names a file, its VAR=VALUE lines are exported into the
# launcher environment.  Everything in d3 that matters lives in the launchers.
readonly WORKER_ENV_FILE="${STACKD_WORKER_ENV:-}"

last_launch=0
launch_workers() {
    local run_dir launch_log
    run_dir="$RUN_ROOT/stackd-$(/usr/bin/date +%Y%m%d-%H%M%S)"
    launch_log="$run_dir/launch.log"
    if [ -n "$WORKER_ENV_FILE" ] && [ -f "$WORKER_ENV_FILE" ]; then
        set -a; . "$WORKER_ENV_FILE"; set +a
    fi
    if [ "$ONESHOT" = 1 ]; then
        log "launch 2026 worker: ssh $R2026_SSH env $GATE_2026 bash $WORKER_2026_SH"
        log "launch main worker: env $GATE_MAIN bash $WORKER_MAIN_SH"
        log "(env-file=$( [ -n "$WORKER_ENV_FILE" ] && echo "$WORKER_ENV_FILE" || echo none ), log $launch_log)"
        return
    fi
    /usr/bin/mkdir -p "$run_dir"
    last_launch=$(/usr/bin/date +%s)
    log "launching d3 workers (log $launch_log, env-file $( [ -n "$WORKER_ENV_FILE" ] && echo "$WORKER_ENV_FILE" || echo none ))"
    # 2026 first, then main -- same order as the gate's workers_up().  Both
    # launchers setsid the worker and return immediately; slots take 2-4 min
    # to fill, which is why the poll loop below has a launch cooldown.
    (
        echo "=== ds4-stackd launch $(/usr/bin/date -Is) (d3 2026-09-04) ==="
        echo "--- 2026 :$R2026_PORT"
        ssh $SSH_OPTS "$R2026_SSH" "env $GATE_2026 bash $WORKER_2026_SH" 2>&1
        echo "--- main :$MAIN_PORT"
        env $GATE_MAIN bash "$WORKER_MAIN_SH" 2>&1
    ) > "$launch_log" 2>&1 &
}

pid_on_port() { ss -ltnp 2>/dev/null | sed -n "s/.*:${1} .*pid=\([0-9]*\).*/\1/p" | head -1; }

terminate_workers() {
    local pid deadline now
    pid=$(pid_on_port "$MAIN_PORT")
    if [ -n "$pid" ]; then
        if [ "$ONESHOT" = 1 ]; then
            log "SIGTERM main worker :$MAIN_PORT (pid $pid)"
        else
            log "SIGTERM main worker :$MAIN_PORT (pid $pid)"
            /usr/bin/kill -TERM "$pid" 2>/dev/null || log "main worker $pid was already gone"
        fi
    else
        log_state no-main-worker "no main worker on :$MAIN_PORT to terminate"
    fi
    if [ "$ONESHOT" = 1 ]; then
        log "SIGTERM 2026 worker :$R2026_PORT over ssh $R2026_SSH"
        return
    fi
    # NEVER pkill on either box: on main it also matches the ssh client cmdline
    # that carries the remote launch string.  Kill by port-derived pid only.
    ssh $SSH_OPTS "$R2026_SSH" \
        "pid=\$(ss -ltnp 2>/dev/null | sed -n 's/.*:$R2026_PORT .*pid=\([0-9]*\).*/\1/p' | head -1); \
         [ -n \"\$pid\" ] && kill -TERM \"\$pid\" || true" >/dev/null 2>&1 \
        || log "WARNING: could not reach $R2026_SSH to stop the :$R2026_PORT worker"

    deadline=$(( $(/usr/bin/date +%s) + 30 ))
    while workers_present; do
        now=$(/usr/bin/date +%s)
        [ "$now" -ge "$deadline" ] && break
        /usr/bin/sleep 2
    done
    if workers_present; then
        log "WARNING: worker process or endpoint remains after termination"
    else
        log "verified workers gone"
    fi
}

absent_since=''
teardown_attempted=0

if [ "$ONESHOT" != 1 ]; then
    /usr/bin/mkdir -p "$RUN_ROOT"
fi

while :; do
    spine="$(spine_pids)"
    if [ -n "$spine" ]; then
        absent_since=''
        teardown_attempted=0
        if workers_running; then
            log_state spine-ready "spine present; workers ready"
        elif workers_present; then
            log_state spine-starting "spine present; workers are still starting (slots fill in 2-4 min)"
        elif [ $(( $(/usr/bin/date +%s) - last_launch )) -lt 180 ]; then
            # The old script used the harness process-group leader as the
            # "already starting" guard.  The d3 launchers exit as soon as they
            # setsid the worker, so there is no leader to find -- a cooldown is
            # what keeps a 2 s poll from relaunching the rig every tick while
            # the worker is still opening its manifest.
            log_state spine-launching "spine present; launch issued <180s ago, waiting"
        else
            launch_workers
        fi
    else
        now=$(/usr/bin/date +%s)
        [ -n "$absent_since" ] || absent_since=$now
        elapsed=$(( now - absent_since ))
        if [ "$elapsed" -lt "$GRACE_S" ]; then
            log_state no-spine-grace "no spine for ${elapsed}s/${GRACE_S}s grace; keeping workers"
        elif [ "$teardown_attempted" -eq 0 ]; then
            teardown_attempted=1
            last_state=''
            if workers_present; then
                terminate_workers
            else
                log "no workers present; nothing to terminate"
            fi
        fi
    fi

    [ "$ONESHOT" = 1 ] && exit 0
    /usr/bin/sleep "$POLL_S"
done
