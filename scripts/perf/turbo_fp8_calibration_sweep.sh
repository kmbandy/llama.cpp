#!/usr/bin/env bash
# turbo_fp8_calibration_sweep.sh — MAD-214 calibration matrix sweep.
#
# Measures PPL across configurations of the three calibration levers:
#   • fit-loss       {mse, mag_weighted, log_space}
#   • granularity    {per_layer_dir, per_dir, global}
#   • snap-strategy  {first_fit, distinct, greedy_coverage, forced_anchors}
# plus corpus-source variation (default / bigger / mixed).
#
# Pipeline per cell:
#   1. Capture phase (shared across cells with same corpus): run llama-perplexity
#      with MT_TURBO_FP8_DUMP_DIR=<corpus dump dir> so the AITER scatter path
#      writes K_cur/V_cur fp16 tensors for each layer.
#   2. Clear LUT cache dir; run fit_centroids_from_dump.py with this cell's
#      flags → writes 16-byte LUTs to ~/.cache/llama.cpp/turbo-fp8/<fp>/.
#   3. Run llama-perplexity at ctx=4096 with --cache-type-{k,v} turbo4_fp8_bs256
#      on the AITER FP8 WMMA path; parse final PPL.
#   4. Append cell row to results JSON.
#
# Output: tests/perf-baseline/calibration-sweep/<short_commit>-<UTC_ts>.json.
# Idempotent across reruns when TIER env var is set: skips cells whose label
# already appears in the latest output for the same commit. Pass TIER=1, 2,
# or 3 to run a tier; default runs only Tier 1.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
SHORT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
FULL_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

LLAMA_PERPLEXITY="${LLAMA_PERPLEXITY:-${REPO_ROOT}/build-hip/bin/llama-perplexity}"
MODEL_PATH="${MODEL_PATH:-/home/kmbandy/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
WIKI_TEST="${WIKI_TEST:-${REPO_ROOT}/wikitext-2-raw/wiki.test.raw}"
WIKI_VALID="${WIKI_VALID:-${REPO_ROOT}/wikitext-2-raw/wiki.valid.raw}"

# Qwen3.6-35B-A3B registry fingerprint (computed from arch+shape). Hardcoded
# because we've already verified the dir on disk; if the registry hash function
# changes, update this.
MODEL_FINGERPRINT="${MODEL_FINGERPRINT:-1d25d29f9a7093e3}"
LUT_CACHE_DIR="${LUT_CACHE_DIR:-${HOME}/.cache/llama.cpp/turbo-fp8/${MODEL_FINGERPRINT}}"

HEAD_SIZE="${HEAD_SIZE:-256}"
N_KV_HEADS="${N_KV_HEADS:-2}"
CTX="${CTX:-4096}"
TIER="${TIER:-1}"

WORK_DIR="${WORK_DIR:-/tmp/claude-1000/turbo_fp8_calibration_sweep}"
OUT_DIR="${REPO_ROOT}/tests/perf-baseline/calibration-sweep"
OUT_PATH="${OUT_DIR}/${SHORT_COMMIT}-${TIMESTAMP}.json"
mkdir -p "${WORK_DIR}" "${OUT_DIR}" "${LUT_CACHE_DIR}"

# Capture phase dump dirs (one per corpus variant), reused across cells.
DUMP_DEFAULT="${WORK_DIR}/dump_default"
DUMP_BIGGER="${WORK_DIR}/dump_bigger"
DUMP_MIXED="${WORK_DIR}/dump_mixed"

# Built mixed corpus path (prose + C++ + Python, ~30KB).
MIXED_CORPUS="${WORK_DIR}/mixed_corpus.txt"

# Preflight.
for f in "${LLAMA_PERPLEXITY}" "${MODEL_PATH}" "${WIKI_TEST}"; do
    [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
if pgrep -f "llama-server\|llama-perplexity" >/dev/null; then
    echo "ERROR: a llama process is already running. Kill it first." >&2
    pgrep -af "llama-server\|llama-perplexity" >&2
    exit 1
fi

# ─────────────────────── Mixed corpus builder ───────────────────────
# Concatenates ~10KB each of: wikitext prose, llama.cpp C++ source, and our
# Python calibration scripts. Gives content diversity in a small footprint.
build_mixed_corpus() {
    if [[ -f "${MIXED_CORPUS}" ]] && [[ $(wc -c < "${MIXED_CORPUS}") -gt 20000 ]]; then
        echo "  mixed corpus already at ${MIXED_CORPUS}" >&2
        return 0
    fi
    echo "  building mixed corpus at ${MIXED_CORPUS}..." >&2
    # `head -c N` closes stdin early → upstream cat hits SIGPIPE → with
    # `set -o pipefail` the whole pipeline returns non-zero. Run in a subshell
    # with pipefail disabled so the SIGPIPE is benign as intended.
    ( set +o pipefail
      {
          head -c 10240 "${WIKI_TEST}"
          printf '\n\n'
          find "${REPO_ROOT}/src" -name '*.cpp' -size +5k 2>/dev/null \
              | head -3 | xargs cat 2>/dev/null | head -c 10240
          printf '\n\n'
          find "${REPO_ROOT}/scripts" -name '*.py' -size +2k 2>/dev/null \
              | head -8 | xargs cat 2>/dev/null | head -c 10240
      } > "${MIXED_CORPUS}"
    )
    echo "  mixed corpus size: $(wc -c < "${MIXED_CORPUS}") bytes" >&2
}

# ─────────────────────── Capture phase ───────────────────────
# Runs llama-perplexity on a given corpus with MT_TURBO_FP8_DUMP_DIR set,
# stops after N chunks. Captures K_cur/V_cur fp16 dumps to the dump dir.
# Reuses an existing populated dump dir if present (skips re-capture).
do_capture() {
    local label="$1" corpus="$2" dump_dir="$3" n_chunks="$4"
    if [[ -d "${dump_dir}" ]] && compgen -G "${dump_dir}/l*_k.fp16" >/dev/null; then
        echo "  [capture ${label}] already populated at ${dump_dir}, skipping" >&2
        return 0
    fi
    rm -rf "${dump_dir}"; mkdir -p "${dump_dir}"
    rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"  # force fallback during capture
    local log="${WORK_DIR}/capture_${label}.log"
    echo "  [capture ${label}] running (corpus=$(basename "${corpus}"), chunks=${n_chunks})..." >&2
    local start; start=$(date +%s)
    MAD_USE_AITER=1 MT_TURBO_FP8_DUMP_DIR="${dump_dir}" \
        "${LLAMA_PERPLEXITY}" \
            --model "${MODEL_PATH}" \
            --device ROCm0 --n-gpu-layers 999 \
            --ctx-size "${CTX}" \
            --kv-tiered 100,0,0 \
            --cache-type-k turbo4_fp8_bs256 --cache-type-v turbo4_fp8_bs256 \
            --flash-attn on --no-mmap \
            --chunks "${n_chunks}" \
            -f "${corpus}" \
            > "${log}" 2>&1 || {
        echo "ERROR [capture ${label}]: perplexity failed; tail:" >&2
        tail -20 "${log}" >&2; return 1
    }
    local elapsed=$(( $(date +%s) - start ))
    local n_files; n_files=$(find "${dump_dir}" -name 'l*_*.fp16' | wc -l)
    echo "  [capture ${label}] ${n_files} dump files in ${elapsed}s" >&2
}

# ─────────────────────── Per-cell PPL ───────────────────────
# Runs llama-perplexity at full corpus, no dump dir, with the cell's LUTs
# already in place. Returns elapsed_s and parsed PPL via stdout (JSON line).
run_ppl_cell() {
    local label="$1" cache_type="$2" extra_flags="$3"
    local log="${WORK_DIR}/ppl_${label}.log"
    echo "  [ppl ${label}] running (cache=${cache_type})..." >&2
    local start; start=$(date +%s)
    # shellcheck disable=SC2086
    MAD_USE_AITER=1 \
        "${LLAMA_PERPLEXITY}" \
            --model "${MODEL_PATH}" \
            --device ROCm0 --n-gpu-layers 999 \
            --ctx-size "${CTX}" \
            ${extra_flags} \
            --cache-type-k "${cache_type}" --cache-type-v "${cache_type}" \
            --flash-attn on --no-mmap \
            -f "${WIKI_TEST}" \
            > "${log}" 2>&1 || {
        echo "ERROR [ppl ${label}]: perplexity failed; tail:" >&2
        tail -20 "${log}" >&2; return 1
    }
    local elapsed=$(( $(date +%s) - start ))
    local ppl_line; ppl_line="$(grep -aE 'Final estimate.*PPL' "${log}" | tail -1)"
    [[ -z "${ppl_line}" ]] && { echo "ERROR [ppl ${label}]: no Final estimate line" >&2; tail -20 "${log}" >&2; return 1; }
    local ppl ppl_err n_chunks
    ppl=$(echo "${ppl_line}"     | sed -nE 's/.*PPL = ([0-9.]+).*/\1/p')
    ppl_err=$(echo "${ppl_line}" | sed -nE 's/.*\+\/- ([0-9.]+).*/\1/p')
    n_chunks=$(grep -aoE '\[[0-9]+\][0-9.]+' "${log}" | wc -l)
    echo "${ppl} ${ppl_err} ${n_chunks} ${elapsed}"
}

# ─────────────────────── Fit a cell ───────────────────────
fit_cell() {
    local dump_dir="$1" fit_loss="$2" granularity="$3" snap_strategy="$4" extra="$5"
    rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"
    # shellcheck disable=SC2086
    python3 "${REPO_ROOT}/scripts/calibration/fit_centroids_from_dump.py" \
        --dump-dir "${dump_dir}" \
        --head-size "${HEAD_SIZE}" --n-kv-heads "${N_KV_HEADS}" \
        --out-dir "${LUT_CACHE_DIR}" \
        --fit-loss "${fit_loss}" \
        --granularity "${granularity}" \
        --snap-strategy "${snap_strategy}" \
        ${extra} \
        >&2
}

# ─────────────────────── Cell runner: emit JSON row ───────────────────────
# Args: label corpus_dump fit_loss granularity snap_strategy fitter_extra notes
emit_cell() {
    local label="$1" dump_dir="$2" fit_loss="$3" granularity="$4" \
          snap_strategy="$5" fitter_extra="$6" notes="$7"
    fit_cell "${dump_dir}" "${fit_loss}" "${granularity}" "${snap_strategy}" "${fitter_extra}"
    # Qwen3.6-A3B is hybrid attention/DeltaNet — only every Nth layer is an
    # attention layer and gets a LUT file. Grab the first available K/V file
    # rather than assuming l0 exists. Also report counts and unique-LUT counts
    # so we can see at a glance whether per_dir/global pooling collapsed the
    # set as expected.
    local first_k_file first_v_file sample_k_lut sample_v_lut
    local n_k_luts n_v_luts uniq_k_luts uniq_v_luts first_k_layer first_v_layer
    first_k_file=$(ls -v "${LUT_CACHE_DIR}"/l*_k.bin 2>/dev/null | head -1)
    first_v_file=$(ls -v "${LUT_CACHE_DIR}"/l*_v.bin 2>/dev/null | head -1)
    sample_k_lut=$([ -n "${first_k_file}" ] && xxd -c 16 -p "${first_k_file}" || echo "")
    sample_v_lut=$([ -n "${first_v_file}" ] && xxd -c 16 -p "${first_v_file}" || echo "")
    first_k_layer=$([ -n "${first_k_file}" ] && basename "${first_k_file}" .bin || echo "")
    first_v_layer=$([ -n "${first_v_file}" ] && basename "${first_v_file}" .bin || echo "")
    n_k_luts=$(ls "${LUT_CACHE_DIR}"/l*_k.bin 2>/dev/null | wc -l)
    n_v_luts=$(ls "${LUT_CACHE_DIR}"/l*_v.bin 2>/dev/null | wc -l)
    uniq_k_luts=$(find "${LUT_CACHE_DIR}" -name 'l*_k.bin' -exec md5sum {} + 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
    uniq_v_luts=$(find "${LUT_CACHE_DIR}" -name 'l*_v.bin' -exec md5sum {} + 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
    read -r ppl ppl_err n_chunks elapsed < <(run_ppl_cell "${label}" "turbo4_fp8_bs256" "--kv-tiered 100,0,0")
    python3 - "${label}" "${fit_loss}" "${granularity}" "${snap_strategy}" \
                "$(basename "${dump_dir}")" "${fitter_extra}" \
                "${first_k_layer}" "${sample_k_lut}" "${first_v_layer}" "${sample_v_lut}" \
                "${n_k_luts}" "${n_v_luts}" "${uniq_k_luts}" "${uniq_v_luts}" \
                "${ppl}" "${ppl_err}" "${n_chunks}" "${elapsed}" "${notes}" <<'PY'
import json, sys
(label, fit, gran, snap, dump, extra,
 first_k, k_hex, first_v, v_hex,
 n_k, n_v, uk, uv,
 ppl, ppl_err, nc, el, notes) = sys.argv[1:]
print(json.dumps({
    "label":            label,
    "fit_loss":         fit,
    "granularity":      gran,
    "snap_strategy":    snap,
    "corpus":           dump,
    "fitter_extra":     extra.strip(),
    "sample_k_layer":   first_k,
    "sample_k_lut_hex": k_hex.strip(),
    "sample_v_layer":   first_v,
    "sample_v_lut_hex": v_hex.strip(),
    "n_k_luts":         int(n_k),
    "n_v_luts":         int(n_v),
    "uniq_k_luts":      int(uk),
    "uniq_v_luts":      int(uv),
    "ppl":              float(ppl),
    "ppl_err":          float(ppl_err),
    "n_chunks":         int(nc),
    "elapsed_s":        int(el),
    "notes":            notes,
}))
PY
}

# Control cells: skip fitter; either run with empty LUT dir (fallback) or
# different cache type (f16). Same JSON shape, with sentinel fitter fields.
emit_control() {
    local label="$1" cache_type="$2" extra_flags="$3" notes="$4"
    if [[ "${cache_type}" == "turbo4_fp8_bs256" ]]; then
        # Wipe the LUT dir to force registry fallback (Qwen3.5-4B canonical).
        rm -rf "${LUT_CACHE_DIR}"; mkdir -p "${LUT_CACHE_DIR}"
    fi
    read -r ppl ppl_err n_chunks elapsed < <(run_ppl_cell "${label}" "${cache_type}" "${extra_flags}")
    python3 - "${label}" "${cache_type}" "${ppl}" "${ppl_err}" "${n_chunks}" "${elapsed}" "${notes}" <<'PY'
import json, sys
label, ct, ppl, ppl_err, nc, el, notes = sys.argv[1:]
print(json.dumps({
    "label":         label,
    "cache_type":    ct,
    "ppl":           float(ppl),
    "ppl_err":       float(ppl_err),
    "n_chunks":      int(nc),
    "elapsed_s":     int(el),
    "control":       True,
    "notes":         notes,
}))
PY
}

# ─────────────────────── Build manifest ───────────────────────
build_mixed_corpus

# Capture phase: 3 dump dirs, ~1-4 chunks each.
echo "=== capture phase ===" >&2
do_capture "default" "${WIKI_TEST}"     "${DUMP_DEFAULT}" 1
do_capture "bigger"  "${WIKI_TEST}"     "${DUMP_BIGGER}"  4
do_capture "mixed"   "${MIXED_CORPUS}"  "${DUMP_MIXED}"   1

# Tier 1 manifest. Each row: label dump fit_loss granularity snap_strategy fitter_extra notes
#   Cells 1-8: fitter variations; cells 9-10: controls.
declare -a TIER1=(
    "T1_baseline|${DUMP_DEFAULT}|mse|per_layer_dir|distinct||baseline (current Option F distinct-snap)"
    "T1_mag_weighted|${DUMP_DEFAULT}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 1.0|magnitude-weighted MSE, p=1.0"
    "T1_log_space|${DUMP_DEFAULT}|log_space|per_layer_dir|distinct||Lloyd-Max in log domain (geometric spacing)"
    "T1_bigger_corpus|${DUMP_BIGGER}|mse|per_layer_dir|distinct||4x captured tokens, same fitter"
    "T1_mixed_corpus|${DUMP_MIXED}|mse|per_layer_dir|distinct||prose+C+++Python mixed corpus, same fitter"
    "T1_per_dir_only|${DUMP_DEFAULT}|mse|per_dir|distinct||pool across layers, one LUT per dir"
    "T1_forced_anchors|${DUMP_DEFAULT}|mse|per_layer_dir|forced_anchors|--forced-anchors 0x00,0x38|always include 0x00 (zero) and 0x38 (1.0)"
    "T1_mag_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 1.0|combo: mag_weighted on bigger corpus (Tier 2 seed)"
)
declare -a CONTROLS=(
    "T1_ctrl_fallback|turbo4_fp8_bs256|--kv-tiered 100,0,0|fallback LUT (no fit; registry uses embedded Qwen3.5-4B canonical)"
    "T1_ctrl_f16|f16|--kv-tiered 100,0,0|f16 reference, AITER path"
)

# Tier 2 — interaction-hunt. 6 cells, no controls (reuse Tier 1 controls).
# Each row: label dump fit_loss granularity snap_strategy fitter_extra notes
declare -a TIER2=(
    "T2_mag_x_per_dir|${DUMP_DEFAULT}|mag_weighted|per_dir|distinct|--mag-weight-p 1.0|combo: mag_weighted (p=1) + per_dir"
    "T2_mag_x_anchors|${DUMP_DEFAULT}|mag_weighted|per_layer_dir|forced_anchors|--mag-weight-p 1.0 --forced-anchors 0x00,0x38|combo: mag_weighted (p=1) + forced_anchors"
    "T2_mag_x_bigger_x_per_dir|${DUMP_BIGGER}|mag_weighted|per_dir|distinct|--mag-weight-p 1.0|triple: mag (p=1) + bigger + per_dir"
    "T2_mag_x_bigger_x_anchors|${DUMP_BIGGER}|mag_weighted|per_layer_dir|forced_anchors|--mag-weight-p 1.0 --forced-anchors 0x00,0x38|triple: mag (p=1) + bigger + forced_anchors"
    "T2_mag_p2|${DUMP_DEFAULT}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 2.0|fine-tune: isolate p=2.0 effect"
    "T2_mag_p2_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 2.0|combo: mag p=2 + bigger"
)

# Tier 3 — fine-tune the winning region (mag_weighted family). 5 cells.
# Scans the p exponent curve at p ∈ {1.5, 2.5, 3.0} × bigger to find where
# the p curve peaks, plus tests whether p=2 also benefits from the
# forced_anchors synergy that p=1 showed (+0.0081 interaction). All cells
# use the same bigger corpus where applicable, since bigger added a tiny
# but consistent benefit and is essentially free.
declare -a TIER3=(
    "T3_mag_p1_5_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 1.5|p-curve: mag p=1.5 + bigger"
    "T3_mag_p2_5_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 2.5|p-curve: mag p=2.5 + bigger"
    "T3_mag_p3_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 3.0|p-curve: mag p=3.0 + bigger"
    "T3_mag_p2_x_anchors|${DUMP_DEFAULT}|mag_weighted|per_layer_dir|forced_anchors|--mag-weight-p 2.0 --forced-anchors 0x00,0x38|combo: mag p=2 + forced_anchors (does p=2 also synergize?)"
    "T3_mag_p2_x_bigger_x_anchors|${DUMP_BIGGER}|mag_weighted|per_layer_dir|forced_anchors|--mag-weight-p 2.0 --forced-anchors 0x00,0x38|triple: best-of-each"
)

# Tier 4 — extend the p curve beyond the Tier 3 winner (p=3). 3 cells.
# Tier 3 found p=3 > p=2 > p=2.5 (non-monotone), so probe further to either
# find the actual peak or confirm p=3 is the plateau.
declare -a TIER4=(
    "T4_mag_p3_5_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 3.5|p-curve: mag p=3.5 + bigger"
    "T4_mag_p4_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 4.0|p-curve: mag p=4.0 + bigger"
    "T4_mag_p5_x_bigger|${DUMP_BIGGER}|mag_weighted|per_layer_dir|distinct|--mag-weight-p 5.0|p-curve: mag p=5.0 + bigger"
)

# ─────────────────────── Run ───────────────────────
echo "=== Tier ${TIER} cells ===" >&2
cells=()
case "${TIER}" in
    1)
        for row in "${TIER1[@]}"; do
            IFS='|' read -r label dump fit gran snap extra notes <<< "${row}"
            cell_json="$(emit_cell "${label}" "${dump}" "${fit}" "${gran}" "${snap}" "${extra}" "${notes}")"
            cells+=("${cell_json}")
            echo "    -> ${cell_json}" >&2
        done
        for row in "${CONTROLS[@]}"; do
            IFS='|' read -r label ct extra notes <<< "${row}"
            cell_json="$(emit_control "${label}" "${ct}" "${extra}" "${notes}")"
            cells+=("${cell_json}")
            echo "    -> ${cell_json}" >&2
        done
        ;;
    2)
        for row in "${TIER2[@]}"; do
            IFS='|' read -r label dump fit gran snap extra notes <<< "${row}"
            cell_json="$(emit_cell "${label}" "${dump}" "${fit}" "${gran}" "${snap}" "${extra}" "${notes}")"
            cells+=("${cell_json}")
            echo "    -> ${cell_json}" >&2
        done
        ;;
    3)
        for row in "${TIER3[@]}"; do
            IFS='|' read -r label dump fit gran snap extra notes <<< "${row}"
            cell_json="$(emit_cell "${label}" "${dump}" "${fit}" "${gran}" "${snap}" "${extra}" "${notes}")"
            cells+=("${cell_json}")
            echo "    -> ${cell_json}" >&2
        done
        ;;
    4)
        for row in "${TIER4[@]}"; do
            IFS='|' read -r label dump fit gran snap extra notes <<< "${row}"
            cell_json="$(emit_cell "${label}" "${dump}" "${fit}" "${gran}" "${snap}" "${extra}" "${notes}")"
            cells+=("${cell_json}")
            echo "    -> ${cell_json}" >&2
        done
        ;;
    *)
        echo "ERROR: Tier ${TIER} not yet implemented." >&2
        exit 2
        ;;
esac

# ─────────────────────── Write results JSON ───────────────────────
python3 - "${OUT_PATH}" "${SHORT_COMMIT}" "${FULL_COMMIT}" "${TIMESTAMP}" \
            "${MODEL_PATH}" "${WIKI_TEST}" "${TIER}" "${CTX}" "${MODEL_FINGERPRINT}" \
            "${OUT_DIR}" \
            "${cells[@]}" <<'PY'
import glob, json, os, sys

(out_path, short, full, ts, model, wiki, tier, ctx, fp, sweep_dir, *cells) = sys.argv[1:]
tier = int(tier)

# Lever taxonomy: maps each known cell label to the set of orthogonal levers
# it activates relative to the T1_baseline (mse + per_layer_dir + distinct +
# default-corpus). The interaction_term for a combo cell is
#   combo_delta - Σ(single_lever_delta)
# where each delta is (baseline_ppl - cell_ppl). Positive interaction_term
# means the combo beat the sum-of-singles prediction = real synergy.
LEVERS: dict[str, tuple[str, ...]] = {
    "T1_baseline":                (),
    "T1_mag_weighted":            ("mag_weighted_p1",),
    "T1_log_space":               ("log_space",),
    "T1_bigger_corpus":           ("bigger",),
    "T1_mixed_corpus":            ("mixed",),
    "T1_per_dir_only":            ("per_dir",),
    "T1_forced_anchors":          ("forced_anchors",),
    "T1_mag_x_bigger":            ("mag_weighted_p1", "bigger"),
    "T2_mag_x_per_dir":           ("mag_weighted_p1", "per_dir"),
    "T2_mag_x_anchors":           ("mag_weighted_p1", "forced_anchors"),
    "T2_mag_x_bigger_x_per_dir":  ("mag_weighted_p1", "bigger", "per_dir"),
    "T2_mag_x_bigger_x_anchors":  ("mag_weighted_p1", "bigger", "forced_anchors"),
    "T2_mag_p2":                  ("mag_weighted_p2",),
    "T2_mag_p2_x_bigger":         ("mag_weighted_p2", "bigger"),
    "T3_mag_p1_5_x_bigger":       ("mag_weighted_p1.5", "bigger"),
    "T3_mag_p2_5_x_bigger":       ("mag_weighted_p2.5", "bigger"),
    "T3_mag_p3_x_bigger":         ("mag_weighted_p3", "bigger"),
    "T3_mag_p2_x_anchors":        ("mag_weighted_p2", "forced_anchors"),
    "T3_mag_p2_x_bigger_x_anchors": ("mag_weighted_p2", "bigger", "forced_anchors"),
    "T4_mag_p3_5_x_bigger":        ("mag_weighted_p3.5", "bigger"),
    "T4_mag_p4_x_bigger":          ("mag_weighted_p4", "bigger"),
    "T4_mag_p5_x_bigger":          ("mag_weighted_p5", "bigger"),
}

parsed_cells = [json.loads(c) for c in cells]

# Collect single-lever PPLs from Tier 1 (and any singles already in this run).
single_ppl: dict[str, float] = {}
baseline_ppl: float | None = None

def absorb_cell(cell: dict) -> None:
    """Record single-lever PPLs from any cell whose lever-set has length 0 (baseline) or 1."""
    global baseline_ppl
    label = cell.get("label", "")
    if label not in LEVERS:
        return
    levers = LEVERS[label]
    if len(levers) == 0:
        baseline_ppl = cell.get("ppl")
    elif len(levers) == 1:
        single_ppl[levers[0]] = cell.get("ppl")

for c in parsed_cells:
    absorb_cell(c)

# For Tier 2+, pull single-lever PPLs from all earlier-tier results. Tier 3
# needs Tier 1 (baseline, forced_anchors, bigger) AND Tier 2 (mag_weighted_p2)
# to decompose its combos.
if tier >= 2:
    prior = sorted(glob.glob(os.path.join(sweep_dir, "*.json")))
    # Drop the file we're about to write (it doesn't exist yet, but be safe).
    prior = [p for p in prior if os.path.abspath(p) != os.path.abspath(out_path)]
    for path in prior:
        try:
            with open(path) as f:
                d = json.load(f)
            if isinstance(d.get("tier"), int) and d["tier"] < tier:
                for c in d.get("cells", []):
                    absorb_cell(c)
        except (OSError, json.JSONDecodeError):
            continue

# Now annotate each cell with levers + interaction_term (where computable).
for c in parsed_cells:
    label = c.get("label", "")
    if label not in LEVERS:
        continue
    levers = LEVERS[label]
    c["levers"] = list(levers)
    if len(levers) < 2 or baseline_ppl is None:
        continue
    if not all(lv in single_ppl for lv in levers):
        c["interaction_term"] = None
        c["interaction_note"] = "missing singles: " + ",".join(
            lv for lv in levers if lv not in single_ppl)
        continue
    combo_ppl = c["ppl"]
    combo_delta = baseline_ppl - combo_ppl
    single_deltas = {lv: baseline_ppl - single_ppl[lv] for lv in levers}
    sum_singles = sum(single_deltas.values())
    c["combo_delta_vs_baseline"] = round(combo_delta, 5)
    c["sum_single_deltas"]       = round(sum_singles, 5)
    c["interaction_term"]        = round(combo_delta - sum_singles, 5)
    c["single_lever_ppls"]       = {lv: round(single_ppl[lv], 5) for lv in levers}

doc = {
    "commit":             full,
    "short_commit":       short,
    "timestamp_utc":      ts,
    "tier":               tier,
    "model":              model,
    "model_fingerprint":  fp,
    "corpus_ppl":         wiki,
    "ctx_size":           int(ctx),
    "notes":              f"MAD-214 calibration matrix sweep — Tier {tier} of 3. "
                          "All cells run on AITER FP8 WMMA path with "
                          "--kv-tiered 100,0,0 except f16 control. "
                          "PPL measured at single ctx (4096) to keep cell cost ~3 min. "
                          "Tier 2+ cells annotate `interaction_term` = combo_delta - Σ(single deltas) "
                          "where positive = combo beat the sum-of-singles prediction.",
    "cells":              parsed_cells,
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
print(json.dumps(doc, indent=2))

# Tier 2 summary table on stderr for at-a-glance review.
if tier >= 2 and baseline_ppl is not None:
    print("", file=sys.stderr)
    print(f"=== Tier {tier} interaction summary (baseline PPL = {baseline_ppl:.4f}) ===", file=sys.stderr)
    print(f"{'cell':30s} {'PPL':>8s} {'Δbase':>8s} {'Σsngl':>8s} {'inter':>8s}", file=sys.stderr)
    for c in parsed_cells:
        label = c.get("label", "")
        if "interaction_term" not in c or c.get("interaction_term") is None:
            continue
        print(f"{label:30s} {c['ppl']:8.4f} {c['combo_delta_vs_baseline']:+8.4f} "
              f"{c['sum_single_deltas']:+8.4f} {c['interaction_term']:+8.4f}",
              file=sys.stderr)
PY

echo "" >&2
echo "wrote ${OUT_PATH}" >&2
