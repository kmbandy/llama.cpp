#!/usr/bin/env bash
# DS4 elastic worker supervisor for mad-lab-main.  It intentionally manages
# only the WORKERS_ONLY harness, never llama-router.
#
# Lifecycle:
#   router spawn -> stackd raises workers -> spine retries connect -> serving
#   router eviction -> grace -> harness SIGTERM/EXIT trap -> worker teardown
#
# Knobs: STACKD_GRACE_S=300 is the continuous no-spine grace period;
# STACKD_POLL_S=2 is the poll interval; STACKD_ONESHOT=1 prints one cycle's
# proposed action without launching or terminating anything; STACKD_NC names
# an absolute netcat path (default /usr/bin/nc).  The spine needs
# WP_DISPATCH_CONNECT_RETRY_S=180 in the *llama-router.service* Environment=
# so router-spawned children inherit it (the preset INI has no child-env key).
#
# v1 does not manage board claims for the worker GPUs.  Those claims are
# advisory and remain a human/session concern.

set -u

readonly REPO=/home/kmbandy/GitHub/llama.cpp
readonly HARNESS="$REPO/docs/dev/harness-2026-08-05-ds4_full.sh"
readonly RUN_ROOT=/home/kmbandy/ds4-runs
readonly STACKD_LOG="$RUN_ROOT/stackd.log"
readonly POLL_S="${STACKD_POLL_S:-2}"
readonly GRACE_S="${STACKD_GRACE_S:-300}"
readonly ONESHOT="${STACKD_ONESHOT:-0}"
readonly NC="${STACKD_NC:-/usr/bin/nc}"

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

harness_leaders() {
    # The supervisor launches this bash as a setsid leader.  Do not signal a
    # similarly named process that is not the leader of its own process group.
    local pid pgid
    for pid in $(/usr/bin/pgrep -f '[b]ash /home/kmbandy/GitHub/llama.cpp/docs/dev/harness-2026-08-05-ds4_full.sh' 2>/dev/null || true); do
        pgid=$(/usr/bin/ps -o pgid= -p "$pid" 2>/dev/null | /usr/bin/tr -d ' ')
        [ "$pgid" = "$pid" ] && printf '%s\n' "$pid"
    done
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

# Expected ports are DERIVED from the worker env file (2026-08-23) so the env
# file is the single cutover point. DSPARK_ON_GPU=1 (config of record since
# 08-20) drops the CPU dspark workers 8807/8808 — probing them kept this
# permanently false and stackd never reached "spine-ready". W6900=1 (rung 2)
# adds the 6900XT worker on main:8802.
env_flag_on() {  # $1=VAR — LAST uncommented VAR= assignment wins (sourced file)
    [ -f "$WORKER_ENV_FILE" ] || return 1
    [ "$(/usr/bin/grep -E "^${1}=" "$WORKER_ENV_FILE" | tail -1 | tr -d '[:space:]')" = "${1}=1" ]
}

main_worker_ports() {
    printf '8801\n'
    env_flag_on W6900 && printf '8802\n'
    env_flag_on DSPARK_ON_GPU || printf '8807\n8808\n'
}

all_worker_ports_open() {
    local p
    for p in $(main_worker_ports); do
        tcp_open 100.86.191.92 "$p" || return 1
    done
    tcp_open 100.124.155.84 8803 &&
    tcp_open 100.124.155.84 8804
}

workers_running() {
    /usr/bin/pgrep -f '[l]lama-wp-expert-worker' >/dev/null 2>&1 && all_worker_ports_open
}

# Present = the local worker process exists OR any worker port answers
# (covers the window where local pgrep misses but remote workers still hold).
workers_present() {
    /usr/bin/pgrep -f '[l]lama-wp-expert-worker' >/dev/null 2>&1 && return 0
    # Any expected port answering => a worker still present.
    local p
    for p in $(main_worker_ports); do
        tcp_open 100.86.191.92 "$p" && return 0
    done
    tcp_open 100.124.155.84 8803 && return 0
    tcp_open 100.124.155.84 8804 && return 0
    return 1
}

# Optional worker-env override file. When present, its VAR=VALUE lines are
# exported into the worker harness environment (STAGING, PREEMPT_BORROW,
# FILL_HOST, REQLOG, ARM, HOSTVICTIM_*, ...). Absent file = the defaults
# below, which are the config of record. This is how A/B experiments vary
# worker knobs on the ROUTER serve path without touching this script.
readonly WORKER_ENV_FILE="${STACKD_WORKER_ENV:-$RUN_ROOT/stackd-worker.env}"

launch_workers() {
    local run_dir launch_log
    run_dir="$RUN_ROOT/stackd-$(/usr/bin/date +%Y%m%d-%H%M%S)"
    launch_log="$run_dir/launch.log"
    if [ -f "$WORKER_ENV_FILE" ]; then
        set -a; . "$WORKER_ENV_FILE"; set +a
    fi
    if [ "$ONESHOT" = 1 ]; then
        log "launch workers: WORKERS_ONLY=1 FILL_HOST=${FILL_HOST:-1} STAGING=${STAGING:-32} PREEMPT_BORROW=${PREEMPT_BORROW:-1} env-file=$( [ -f "$WORKER_ENV_FILE" ] && echo "$WORKER_ENV_FILE" || echo none ) setsid nohup bash $HARNESS (log $launch_log)"
        return
    fi
    /usr/bin/mkdir -p "$run_dir"
    log "launching worker harness (log $launch_log, env-file $( [ -f "$WORKER_ENV_FILE" ] && echo "$WORKER_ENV_FILE" || echo none ))"
    WORKERS_ONLY=1 FILL_HOST="${FILL_HOST:-1}" STAGING="${STAGING:-32}" PREEMPT_BORROW="${PREEMPT_BORROW:-1}" \
        /usr/bin/setsid /usr/bin/nohup /usr/bin/bash "$HARNESS" \
        > "$launch_log" 2>&1 < /dev/null &
}

terminate_workers() {
    local pid deadline now
    for pid in $(harness_leaders); do
        if [ "$ONESHOT" = 1 ]; then
            log "SIGTERM harness process-group leader $pid; its EXIT trap will clean workers"
            continue
        fi
        log "SIGTERM harness process-group leader $pid; waiting for its EXIT cleanup"
        /usr/bin/kill -TERM "$pid" 2>/dev/null || log "harness leader $pid was already gone"
    done

    [ "$ONESHOT" = 1 ] && return
    deadline=$(( $(/usr/bin/date +%s) + 30 ))
    while workers_present; do
        now=$(/usr/bin/date +%s)
        [ "$now" -ge "$deadline" ] && break
        /usr/bin/sleep 2
    done
    if workers_present; then
        log "WARNING: worker process or endpoint remains after harness cleanup"
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
        elif [ -n "$(harness_leaders)" ]; then
            log_state spine-starting "spine present; worker harness is still starting"
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
            if [ -n "$(harness_leaders)" ]; then
                terminate_workers
            else
                log "no stackd-style WORKERS_ONLY harness leader found; nothing to terminate"
            fi
        fi
    fi

    [ "$ONESHOT" = 1 ] && exit 0
    /usr/bin/sleep "$POLL_S"
done
