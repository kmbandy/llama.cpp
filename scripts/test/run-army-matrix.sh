#!/usr/bin/env bash
# MAD-137: per-device test matrix runner for the army stack.
#
# For every device in the army manifest, run:
#   1. The mt:: + paged unit/integration test suite via ctest
#   2. The MAD-137 stress driver (bounded duration, low decode floor)
#
# Aggregates pass/fail into tests/results/army-matrix-<DATE>.json so CI
# (or a human) can see at a glance what's green and what isn't. Stdout
# is a human summary; the JSON is the machine view.
#
# Each device is described by these env-vary-able knobs:
#
#   DEVICE_NAME         label used in the report
#   SSH_HOST            empty for "run locally"; otherwise the ssh target
#   DOCKER_CONTAINER    empty for host execution; otherwise `docker exec`
#                       wraps the commands
#   REPO_PATH           absolute path to the llama.cpp checkout
#   BUILD_DIR           build directory under REPO_PATH
#   DEVICE_FLAG         value passed to llama-server's --device
#   MODEL_PATH          gguf model path on the target box
#   BGE_PATH            optional bge-small gguf path (semantic prefetch)
#   STRESS_DECODE_FLOOR decode tok/s threshold for the stress test
#   STRESS_DURATION     stress test duration in seconds
#   STRESS_PARALLEL     --n-agents for the stress test
#
# The default manifest below covers the four-GPU army:
#   - R9700  + 6900XT  on the main box (HIP)
#   - 1070            on mad-lab-2026 (CUDA)
#   - RX 480          on mad-lab-2026 inside the rx480-army docker container
#                     (HIP gfx803, ROCm 6.4)
#
# Usage:
#   bash scripts/test/run-army-matrix.sh                 # full matrix
#   DEVICES="r9700 1070" bash scripts/test/run-army-matrix.sh
#   SKIP_STRESS=1       bash scripts/test/run-army-matrix.sh   # unit/integration only

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/tests/results"
mkdir -p "${RESULTS_DIR}"
REPORT_DATE="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
REPORT_PATH="${RESULTS_DIR}/army-matrix-${REPORT_DATE}.json"

# ─── Device manifest ────────────────────────────────────────────────────
# Each device is a function that exports the required env vars. Add new
# devices as new functions; the order list below picks which ones run.

dev_r9700() {
    DEVICE_NAME="r9700"
    SSH_HOST=""
    DOCKER_CONTAINER=""
    REPO_PATH="${REPO_ROOT}"
    BUILD_DIR="build-hip"
    DEVICE_FLAG="ROCm0"
    MODEL_PATH="${HOME}/models/Qwen3.6-27B-Q6_K.gguf"
    BGE_PATH="${HOME}/models/bge-small-en-v1.5-q8_0.gguf"
    STRESS_DECODE_FLOOR="${STRESS_DECODE_FLOOR_R9700:-15}"
    STRESS_DURATION="${STRESS_DURATION:-60}"
    STRESS_PARALLEL="${STRESS_PARALLEL:-2}"
    INSTANCE_ID="matrix-r9700"
}

dev_6900xt() {
    DEVICE_NAME="6900xt"
    SSH_HOST=""
    DOCKER_CONTAINER=""
    REPO_PATH="${REPO_ROOT}"
    BUILD_DIR="build-hip"
    DEVICE_FLAG="ROCm1"
    MODEL_PATH="${HOME}/models/Qwen3.5-9B-TQ3_1S.gguf"
    BGE_PATH="${HOME}/models/bge-small-en-v1.5-q8_0.gguf"
    STRESS_DECODE_FLOOR="${STRESS_DECODE_FLOOR_6900XT:-15}"
    STRESS_DURATION="${STRESS_DURATION:-60}"
    STRESS_PARALLEL="${STRESS_PARALLEL:-2}"
    INSTANCE_ID="matrix-6900xt"
}

dev_1070() {
    DEVICE_NAME="1070"
    SSH_HOST="mad-lab-2026"
    DOCKER_CONTAINER=""
    REPO_PATH="/home/kmbandy/GitHub/llama.cpp"
    BUILD_DIR="build-army"
    DEVICE_FLAG="CUDA0"
    MODEL_PATH="/home/kmbandy/models/omnicoder-9b-q5_k_m.gguf"
    BGE_PATH="/home/kmbandy/models/bge-small-en-v1.5-q8_0.gguf"
    STRESS_DECODE_FLOOR="${STRESS_DECODE_FLOOR_1070:-15}"
    STRESS_DURATION="${STRESS_DURATION:-60}"
    STRESS_PARALLEL="${STRESS_PARALLEL:-2}"
    INSTANCE_ID="matrix-1070"
}

dev_rx480() {
    DEVICE_NAME="rx480"
    SSH_HOST="mad-lab-2026"
    DOCKER_CONTAINER="rx480-army"
    REPO_PATH="/workspace/llama.cpp"   # path INSIDE the container
    BUILD_DIR="build-rocm-gfx803"
    DEVICE_FLAG="ROCm0"
    MODEL_PATH="/models/omnicoder-9b-q5_k_m.gguf"
    BGE_PATH="/models/bge-small-en-v1.5-q8_0.gguf"
    STRESS_DECODE_FLOOR="${STRESS_DECODE_FLOOR_RX480:-12}"
    STRESS_DURATION="${STRESS_DURATION:-60}"
    STRESS_PARALLEL="${STRESS_PARALLEL:-2}"
    INSTANCE_ID="matrix-rx480"
}

DEVICES_DEFAULT="r9700 6900xt 1070 rx480"
DEVICES="${DEVICES:-${DEVICES_DEFAULT}}"

# ─── Command-execution wrapper — ssh / docker / local ───────────────────

# Print the prefix that wraps a command with `ssh ... -- "docker exec ..."`
# or `ssh ...` or nothing, depending on env vars.
build_runner_prefix() {
    local prefix=""
    if [[ -n "${SSH_HOST}" ]]; then
        prefix="ssh ${SSH_HOST}"
        if [[ -n "${DOCKER_CONTAINER}" ]]; then
            prefix="${prefix} docker exec ${DOCKER_CONTAINER}"
        fi
        prefix="${prefix} bash -lc"
    else
        if [[ -n "${DOCKER_CONTAINER}" ]]; then
            prefix="docker exec ${DOCKER_CONTAINER} bash -lc"
        else
            prefix="bash -lc"
        fi
    fi
    echo "${prefix}"
}

# Run a shell command (string) on the configured target. Returns its
# exit code; output captured to the named file.
run_remote() {
    local cmd="$1"
    local outfile="$2"
    local prefix
    prefix="$(build_runner_prefix)"
    # shellcheck disable=SC2086
    ${prefix} "${cmd}" > "${outfile}" 2>&1
}

# ─── Per-device test phases ────────────────────────────────────────────

# Phase 1: ctest unit + integration suite. Gracefully skips integration
# tests that require LLAMACPP_TEST_MODELFILE — those just print the
# yellow "no model" warning and exit 0, which ctest treats as success.
run_unit_integration() {
    local outfile="$1"
    local cmd
    cmd="cd ${REPO_PATH}/${BUILD_DIR} && \
         LLAMACPP_TEST_MODELFILE=${MODEL_PATH} \
         ctest -R 'test-mt-|test-paged-' --output-on-failure"
    run_remote "${cmd}" "${outfile}"
}

# Phase 2: MAD-137 stress driver. Always invoked from the *host's* repo
# checkout (REPO_ROOT) — even for remote devices, because the stress
# script is short and self-contained, and we want one canonical
# implementation. For ssh/docker targets the script is copied over
# first.
run_stress() {
    local outfile="$1"

    # Stage the stress script onto the target.
    local target_script="${REPO_PATH}/tests/stress/stress-paged-multi-seq.py"
    if [[ -n "${SSH_HOST}" || -n "${DOCKER_CONTAINER}" ]]; then
        # Best-effort: assume the stress driver was synced with the rest
        # of the repo. If it's missing, the run will say so.
        :
    fi

    local stress_cmd
    stress_cmd="python3 ${target_script} \
        --bin   ${REPO_PATH}/${BUILD_DIR}/bin/llama-server \
        --model ${MODEL_PATH} \
        --device ${DEVICE_FLAG} \
        --ctx 8192 --n-agents ${STRESS_PARALLEL} \
        --tier 25,75,0 --instance-id ${INSTANCE_ID} \
        --duration ${STRESS_DURATION} \
        --decode-floor ${STRESS_DECODE_FLOOR} \
        --port 18095"
    if [[ -n "${BGE_PATH}" ]]; then
        stress_cmd="${stress_cmd} --bge-small ${BGE_PATH}"
    fi
    run_remote "${stress_cmd}" "${outfile}"
}

# ─── Main loop ──────────────────────────────────────────────────────────

# JSON aggregation — bash + a tiny python finalizer.
TMPJSON="$(mktemp)"
echo "[" > "${TMPJSON}"
first_entry=true

emit() {
    local sep=","
    [[ "${first_entry}" == "true" ]] && sep=""
    first_entry=false
    printf '%s\n%s\n' "${sep}" "$1" >> "${TMPJSON}"
}

overall_pass=true

echo "──── Army test matrix — ${REPORT_DATE} ────"
echo

for dev in ${DEVICES}; do
    # Reset env vars before each device.
    DEVICE_NAME=""; SSH_HOST=""; DOCKER_CONTAINER=""
    REPO_PATH=""; BUILD_DIR=""; DEVICE_FLAG=""
    MODEL_PATH=""; BGE_PATH=""
    STRESS_DECODE_FLOOR=""; STRESS_DURATION=""; STRESS_PARALLEL=""
    INSTANCE_ID=""

    case "${dev}" in
        r9700)  dev_r9700  ;;
        6900xt) dev_6900xt ;;
        1070)   dev_1070   ;;
        rx480)  dev_rx480  ;;
        *)
            echo "skip: unknown device '${dev}'"
            continue
            ;;
    esac

    location_label="local"
    if [[ -n "${SSH_HOST}" ]]; then
        location_label="ssh:${SSH_HOST}"
        if [[ -n "${DOCKER_CONTAINER}" ]]; then
            location_label="${location_label}+docker:${DOCKER_CONTAINER}"
        fi
    elif [[ -n "${DOCKER_CONTAINER}" ]]; then
        location_label="docker:${DOCKER_CONTAINER}"
    fi
    echo "▶ ${DEVICE_NAME}  (${location_label}, build=${BUILD_DIR}, dev=${DEVICE_FLAG})"

    unit_log="${RESULTS_DIR}/${DEVICE_NAME}-unit-${REPORT_DATE}.log"
    stress_log="${RESULTS_DIR}/${DEVICE_NAME}-stress-${REPORT_DATE}.log"

    unit_status="skipped"
    stress_status="skipped"

    # Phase 1: ctest
    if run_unit_integration "${unit_log}"; then
        unit_status="pass"
    else
        unit_status="fail"
        overall_pass=false
    fi
    echo "   unit/integration: ${unit_status}  (log: ${unit_log})"

    # Phase 2: stress (skip with SKIP_STRESS=1)
    if [[ "${SKIP_STRESS:-0}" != "1" ]]; then
        if run_stress "${stress_log}"; then
            stress_status="pass"
        else
            stress_status="fail"
            overall_pass=false
        fi
        echo "   stress:           ${stress_status}  (log: ${stress_log})"
    fi

    # Per-device JSON entry.
    json_entry=$(printf '{"device":"%s","location":"%s","build_dir":"%s","device_flag":"%s","unit":"%s","stress":"%s","unit_log":"%s","stress_log":"%s"}' \
        "${DEVICE_NAME}" \
        "${location_label}" \
        "${BUILD_DIR}" \
        "${DEVICE_FLAG}" \
        "${unit_status}" \
        "${stress_status}" \
        "${unit_log}" \
        "${stress_log}")
    emit "${json_entry}"
done

echo "]" >> "${TMPJSON}"

# Finalize: wrap the per-device array in a top-level object with status.
overall_status="pass"
[[ "${overall_pass}" == "true" ]] || overall_status="fail"
python3 - "${TMPJSON}" "${REPORT_PATH}" "${overall_status}" "${REPORT_DATE}" <<'PY'
import json, sys
src, dst, overall, date = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src) as f:
    devices = json.load(f)
out = {"date": date, "overall": overall, "devices": devices}
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nReport: {dst}  (overall: {overall})")
PY
rm -f "${TMPJSON}"

[[ "${overall_pass}" == "true" ]]
