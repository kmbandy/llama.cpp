#!/usr/bin/env bash
# turbo_fp8_validate_recipe.sh — re-run a winning calibration recipe across
# multiple ctx sizes to confirm the win holds at production-long context.
#
# Default recipe: the MAD-214 matrix-sweep winner as of 2026-05-22:
#   --fit-loss mag_weighted --mag-weight-p 3.0
#   --granularity per_layer_dir --snap-strategy distinct
#   --corpus bigger (16k tokens captured)
#
# Pipeline:
#   1. (Re)capture bigger corpus dumps if not already present.
#   2. Fit the recipe → writes 16-byte LUTs to ~/.cache/llama.cpp/turbo-fp8/<fp>/.
#   3. Run llama-perplexity at each ctx in CTXS — record PPL per ctx.
#   4. Wipe LUT cache, run llama-perplexity at each ctx with FALLBACK LUTs.
#   5. Write a comparison JSON.
#
# Output: tests/perf-baseline/calibration-sweep/<short_commit>-<ts>-validate.json

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
FULL_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

LLAMA_PERPLEXITY="${LLAMA_PERPLEXITY:-${REPO_ROOT}/build-hip/bin/llama-perplexity}"
MODEL_PATH="${MODEL_PATH:-/home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
WIKI_TEST="${WIKI_TEST:-${REPO_ROOT}/wikitext-2-raw/wiki.test.raw}"

MODEL_FINGERPRINT="${MODEL_FINGERPRINT:-1d25d29f9a7093e3}"
LUT_CACHE_DIR="${LUT_CACHE_DIR:-${HOME}/.cache/llama.cpp/turbo-fp8/${MODEL_FINGERPRINT}}"
HEAD_SIZE="${HEAD_SIZE:-256}"
N_KV_HEADS="${N_KV_HEADS:-2}"

# Recipe knobs (override via env to validate a different config).
FIT_LOSS="${FIT_LOSS:-mag_weighted}"
MAG_WEIGHT_P="${MAG_WEIGHT_P:-3.0}"
GRANULARITY="${GRANULARITY:-per_layer_dir}"
SNAP_STRATEGY="${SNAP_STRATEGY:-distinct}"
FORCED_ANCHORS="${FORCED_ANCHORS:-0x00,0x38}"
CORPUS_LABEL="${CORPUS_LABEL:-bigger}"
N_CAPTURE_CHUNKS="${N_CAPTURE_CHUNKS:-4}"

# Contexts to validate at. Default covers the practical production range.
CTXS="${CTXS:-4096,8192,16384}"

WORK_DIR="${WORK_DIR:-/tmp/claude-1000/turbo_fp8_calibration_sweep}"
DUMP_DIR="${WORK_DIR}/dump_${CORPUS_LABEL}"
OUT_DIR="${REPO_ROOT}/tests/perf-baseline/calibration-sweep"
OUT_PATH="${OUT_DIR}/${SHORT_COMMIT}-${TIMESTAMP}-validate.json"
mkdir -p "${WORK_DIR}" "${OUT_DIR}" "${LUT_CACHE_DIR}"

for f in "${LLAMA_PERPLEXITY}" "${MODEL_PATH}" "${WIKI_TEST}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
if pgrep -f "llama-server\|llama-perplexity" >/dev/null; then
    echo "ERROR: a llama process is already running. Kill it first." >&2
    pgrep -af "llama-server\|llama-perplexity" >&2
    exit 1
fi

# ─────────────────────── Capture phase (if needed) ───────────────────────
if compgen -G "${DUMP_DIR}/l*_k.fp16" >/dev/null; then
    echo "[capture ${CORPUS_LABEL}] reusing existing dumps at ${DUMP_DIR}" >&2
else
    rm -rf "${DUMP_DIR}"; mkdir -p "${DUMP_DIR}"
    rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"  # fallback during capture
    log="${WORK_DIR}/capture_${CORPUS_LABEL}.log"
    echo "[capture ${CORPUS_LABEL}] capturing ${N_CAPTURE_CHUNKS} chunks..." >&2
    MAD_USE_AITER=1 MT_TURBO_FP8_DUMP_DIR="${DUMP_DIR}" \
        "${LLAMA_PERPLEXITY}" \
            --model "${MODEL_PATH}" \
            --device ROCm0 --n-gpu-layers 999 \
            --ctx-size 4096 \
            --kv-tiered 100,0,0 \
            --cache-type-k turbo4_fp8_bs256 --cache-type-v turbo4_fp8_bs256 \
            --flash-attn on --no-mmap \
            --chunks "${N_CAPTURE_CHUNKS}" \
            -f "${WIKI_TEST}" \
            > "${log}" 2>&1 || {
        echo "ERROR [capture]: failed; tail:" >&2; tail -20 "${log}" >&2; exit 1
    }
    echo "[capture ${CORPUS_LABEL}] $(find "${DUMP_DIR}" -name 'l*_*.fp16' | wc -l) dump files" >&2
fi

# ─────────────────────── Fit the recipe ───────────────────────
fit_recipe() {
    rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"
    local args=( --dump-dir "${DUMP_DIR}"
                 --head-size "${HEAD_SIZE}" --n-kv-heads "${N_KV_HEADS}"
                 --out-dir "${LUT_CACHE_DIR}"
                 --fit-loss "${FIT_LOSS}"
                 --granularity "${GRANULARITY}"
                 --snap-strategy "${SNAP_STRATEGY}" )
    [[ "${FIT_LOSS}" == "mag_weighted" ]] && args+=( --mag-weight-p "${MAG_WEIGHT_P}" )
    [[ "${SNAP_STRATEGY}" == "forced_anchors" ]] && args+=( --forced-anchors "${FORCED_ANCHORS}" )
    python3 "${REPO_ROOT}/scripts/calibration/fit_centroids_from_dump.py" "${args[@]}" >&2
}

# ─────────────────────── PPL runner at a given ctx ───────────────────────
# Echoes "ppl ppl_err n_chunks elapsed" to stdout; status to stderr.
run_ppl_at_ctx() {
    local label="$1" cache_type="$2" ctx="$3"
    local log="${WORK_DIR}/validate_${label}_ctx${ctx}.log"
    echo "[ppl ${label} ctx=${ctx}] running..." >&2
    local start; start=$(date +%s)
    MAD_USE_AITER=1 \
        "${LLAMA_PERPLEXITY}" \
            --model "${MODEL_PATH}" \
            --device ROCm0 --n-gpu-layers 999 \
            --ctx-size "${ctx}" \
            --kv-tiered 100,0,0 \
            --cache-type-k "${cache_type}" --cache-type-v "${cache_type}" \
            --flash-attn on --no-mmap \
            -f "${WIKI_TEST}" \
            > "${log}" 2>&1 || {
        echo "ERROR [ppl ${label} ctx=${ctx}]: failed; tail:" >&2
        tail -20 "${log}" >&2; return 1
    }
    local elapsed=$(( $(date +%s) - start ))
    local line; line="$(grep -aE 'Final estimate.*PPL' "${log}" | tail -1)"
    [[ -z "${line}" ]] && { echo "ERROR [ppl ${label} ctx=${ctx}]: no Final estimate" >&2; return 1; }
    local ppl ppl_err n_chunks
    ppl=$(echo "${line}"     | sed -nE 's/.*PPL = ([0-9.]+).*/\1/p')
    ppl_err=$(echo "${line}" | sed -nE 's/.*\+\/- ([0-9.]+).*/\1/p')
    n_chunks=$(grep -aoE '\[[0-9]+\][0-9.]+' "${log}" | wc -l)
    echo "${ppl} ${ppl_err} ${n_chunks} ${elapsed}"
}

# ─────────────────────── Phase 1: recipe LUTs ───────────────────────
echo "=== fitting recipe: ${FIT_LOSS} p=${MAG_WEIGHT_P} ${GRANULARITY} ${SNAP_STRATEGY} corpus=${CORPUS_LABEL} ===" >&2
fit_recipe
first_k_file=$(ls -v "${LUT_CACHE_DIR}"/l*_k.bin 2>/dev/null | head -1)
sample_k_hex=$([ -n "${first_k_file}" ] && xxd -c 16 -p "${first_k_file}" || echo "")
n_luts=$(ls "${LUT_CACHE_DIR}"/l*_k.bin 2>/dev/null | wc -l)
echo "  fit complete: ${n_luts} K LUTs, sample $(basename "${first_k_file}" .bin)=${sample_k_hex}" >&2

echo "=== phase 1: recipe LUTs at each ctx ===" >&2
recipe_cells=()
IFS=',' read -r -a CTX_ARR <<< "${CTXS}"
for ctx in "${CTX_ARR[@]}"; do
    read -r ppl ppl_err nc el < <(run_ppl_at_ctx "recipe_ctx${ctx}" "turbo4_fp8_bs256" "${ctx}")
    cell=$(python3 -c "import json,sys; print(json.dumps({'phase':'recipe','ctx_size':int(sys.argv[1]),'ppl':float(sys.argv[2]),'ppl_err':float(sys.argv[3]),'n_chunks':int(sys.argv[4]),'elapsed_s':int(sys.argv[5])}))" \
        "${ctx}" "${ppl}" "${ppl_err}" "${nc}" "${el}")
    recipe_cells+=("${cell}")
    echo "    -> ${cell}" >&2
done

# ─────────────────────── Phase 2: fallback ───────────────────────
echo "=== phase 2: fallback LUTs (wipe + re-run) at each ctx ===" >&2
rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"
fallback_cells=()
for ctx in "${CTX_ARR[@]}"; do
    read -r ppl ppl_err nc el < <(run_ppl_at_ctx "fallback_ctx${ctx}" "turbo4_fp8_bs256" "${ctx}")
    cell=$(python3 -c "import json,sys; print(json.dumps({'phase':'fallback','ctx_size':int(sys.argv[1]),'ppl':float(sys.argv[2]),'ppl_err':float(sys.argv[3]),'n_chunks':int(sys.argv[4]),'elapsed_s':int(sys.argv[5])}))" \
        "${ctx}" "${ppl}" "${ppl_err}" "${nc}" "${el}")
    fallback_cells+=("${cell}")
    echo "    -> ${cell}" >&2
done

# ─────────────────────── Write JSON + summary ───────────────────────
# Re-fit the recipe so the cache dir is left in winning state for the user.
echo "=== leaving cache dir in winning state (re-fit) ===" >&2
fit_recipe >/dev/null

python3 - "${OUT_PATH}" "${SHORT_COMMIT}" "${FULL_COMMIT}" "${TIMESTAMP}" \
            "${MODEL_PATH}" "${WIKI_TEST}" "${MODEL_FINGERPRINT}" \
            "${FIT_LOSS}" "${MAG_WEIGHT_P}" "${GRANULARITY}" "${SNAP_STRATEGY}" \
            "${FORCED_ANCHORS}" "${CORPUS_LABEL}" \
            "${sample_k_hex}" \
            "${recipe_cells[@]}" "--" "${fallback_cells[@]}" <<'PY'
import json, sys
args = sys.argv[1:]
(out_path, short, full, ts, model, wiki, fp,
 fit_loss, mag_p, gran, snap, anchors, corpus, sample_lut) = args[:14]
rest = args[14:]
sep_idx = rest.index("--")
recipe = [json.loads(c) for c in rest[:sep_idx]]
fallback = [json.loads(c) for c in rest[sep_idx+1:]]

# Build a compact comparison table.
by_ctx = {}
for c in recipe:   by_ctx.setdefault(c["ctx_size"], {})["recipe"]   = c
for c in fallback: by_ctx.setdefault(c["ctx_size"], {})["fallback"] = c

comparison = []
for ctx in sorted(by_ctx):
    r = by_ctx[ctx].get("recipe");   f = by_ctx[ctx].get("fallback")
    if not (r and f): continue
    comparison.append({
        "ctx_size":         ctx,
        "recipe_ppl":       r["ppl"],
        "recipe_ppl_err":   r["ppl_err"],
        "fallback_ppl":     f["ppl"],
        "fallback_ppl_err": f["ppl_err"],
        "recipe_minus_fallback":  round(r["ppl"] - f["ppl"], 5),
        "recipe_elapsed_s":       r["elapsed_s"],
        "fallback_elapsed_s":     f["elapsed_s"],
    })

doc = {
    "commit":             full,
    "short_commit":       short,
    "timestamp_utc":      ts,
    "phase":              "validate-recipe",
    "model":              model,
    "model_fingerprint":  fp,
    "corpus_ppl":         wiki,
    "recipe": {
        "fit_loss":       fit_loss,
        "mag_weight_p":   float(mag_p),
        "granularity":    gran,
        "snap_strategy":  snap,
        "forced_anchors": anchors,
        "capture_corpus": corpus,
        "sample_k_lut_hex": sample_lut.strip(),
    },
    "comparison":         comparison,
    "raw_cells":          recipe + fallback,
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))

# Summary table on stderr.
print("", file=sys.stderr)
print(f"=== Recipe validation summary ===", file=sys.stderr)
print(f"recipe: {fit_loss} p={mag_p} {gran} {snap} corpus={corpus}", file=sys.stderr)
print(f"{'ctx':>6s} {'recipe':>10s} {'fallback':>10s} {'Δ(R-F)':>9s} {'recipe_s':>9s} {'fallback_s':>11s}", file=sys.stderr)
for row in comparison:
    print(f"{row['ctx_size']:6d} {row['recipe_ppl']:10.4f} {row['fallback_ppl']:10.4f} "
          f"{row['recipe_minus_fallback']:+9.4f} {row['recipe_elapsed_s']:9d} {row['fallback_elapsed_s']:11d}",
          file=sys.stderr)
PY

echo "" >&2
echo "wrote ${OUT_PATH}" >&2
