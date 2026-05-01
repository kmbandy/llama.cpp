#!/usr/bin/env bash
# regress-memory-tier.sh — regression matrix for the memory-tier rewrite.
#
# Companion to /home/kmbandy/.claude/plans/proud-stargazing-engelbart.md and
# docs/dev/memory-tier-bug-catalog.md. Each row of the matrix is one model
# in one mode with one acceptance check.
#
# Usage:
#   scripts/regress-memory-tier.sh --row 3              # run gate row only
#   scripts/regress-memory-tier.sh --all                # run every runnable row
#   scripts/regress-memory-tier.sh --list               # list rows + status
#
# Environment:
#   LLAMA_BIN     path to llama-cli (default: build-feature/bin/llama-cli)
#   LLAMA_SERVER  path to llama-server (default: build-feature/bin/llama-server)
#   MODEL_DIR     where ggufs live (default: $HOME/models)
#   OUT_DIR       per-run logs (default: /tmp/regress-memory-tier)
#   SEED          sampler seed (default: 42)
#
# Exit codes: 0 = pass; 1 = fail; 2 = row not yet implemented (gated by phase).

set -u
set -o pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || echo /home/kmbandy/GitHub/llama.cpp)"
LLAMA_BIN="${LLAMA_BIN:-$REPO_ROOT/build-feature/bin/llama-cli}"
LLAMA_SERVER="${LLAMA_SERVER:-$REPO_ROOT/build-feature/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
OUT_DIR="${OUT_DIR:-/tmp/regress-memory-tier}"
SEED="${SEED:-42}"
PROMPT_FILE="$REPO_ROOT/tests/prompts/regress-coherence-prompt.txt"

mkdir -p "$OUT_DIR"

# ---------- helpers ----------

# coherence_check FILE TOKENS_DECODED MIN_UNIQUE_RATIO
# Reads decoded text from FILE, asserts the output is not a repetition of a
# single character/token. MIN_UNIQUE_RATIO is in tenths (e.g. 3 = 30%).
coherence_check() {
    local file="$1"
    local min_unique_ratio="${2:-3}"
    local total bytes_uniq

    if [[ ! -s "$file" ]]; then
        echo "FAIL: output file empty ($file)"
        return 1
    fi
    total=$(wc -c < "$file")
    bytes_uniq=$(fold -w1 < "$file" | sort -u | wc -l)
    if (( total < 50 )); then
        echo "FAIL: output too short ($total bytes); expected >= 50"
        return 1
    fi
    # Detect the '/' degenerate case: any single character that occupies > 70%
    # of the output is considered a coherence failure.
    awk -v total="$total" '
        { for (i=1; i<=length($0); i++) c[substr($0,i,1)]++ }
        END {
            for (k in c) if (c[k]/total > 0.70) {
                printf "FAIL: char %q occupies %.0f%% of output (>70%% threshold)\n", k, c[k]*100/total
                exit 1
            }
        }
    ' "$file" || return 1
    echo "PASS: $total bytes, $bytes_uniq distinct bytes"
    return 0
}

require_bin() {
    if [[ ! -x "$1" ]]; then
        echo "ERROR: missing binary: $1" >&2
        echo "       hint: build with cmake --build $REPO_ROOT/build-feature -j" >&2
        return 1
    fi
}

require_model() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: missing model: $1" >&2
        return 1
    fi
}

ts() { date +%Y%m%d-%H%M%S; }

# ---------- rows ----------

# Row 3 — Phase 0.5 HARD GATE.
# 97B at ctx=131072, no tier, no pager. Decode 50 tokens. Acceptance: not all '/'.
row3_97b_no_tier() {
    local model="$MODEL_DIR/Qwen3.5-REAP-97B-A10B-IQ3_XS-00001-of-00002.gguf"
    local out="$OUT_DIR/row3-$(ts).txt"
    require_bin "$LLAMA_BIN" || return 1
    require_model "$model"   || return 1

    echo "[row 3] 97B at ctx=131072, no tier, no pager → $out"
    "$LLAMA_BIN" \
        --model "$model" \
        --device ROCm0,ROCm1 \
        --tensor-split 2,1 \
        --n-gpu-layers 999 \
        --ctx-size 131072 \
        --cache-type-k turbo4 \
        --cache-type-v turbo4 \
        --flash-attn on \
        --no-mmap \
        --seed "$SEED" \
        --temp 0.0 \
        --n-predict 64 \
        --file "$PROMPT_FILE" \
        --simple-io \
        2>"$out.stderr" | tee "$out.stdout" >/dev/null

    # llama-cli echoes the prompt back; isolate the assistant's continuation.
    # The simplest invariant for the gate is: the output stream is not '/'-only.
    coherence_check "$out.stdout" 3 || { echo "[row 3] FAIL"; return 1; }
    echo "[row 3] PASS"
}

# Row 2 — 35B baseline, no tier, no pager. Sanity check that the no-regression
# reference still works.
row2_35b_baseline() {
    local model="$MODEL_DIR/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-I-Balanced.gguf"
    local out="$OUT_DIR/row2-$(ts).txt"
    require_bin "$LLAMA_BIN" || return 1
    require_model "$model"   || return 1

    echo "[row 2] 35B baseline → $out"
    "$LLAMA_BIN" \
        --model "$model" \
        --device ROCm0,ROCm1 \
        --tensor-split 2,1 \
        --n-gpu-layers 999 \
        --ctx-size 32768 \
        --flash-attn on \
        --no-mmap \
        --seed "$SEED" \
        --temp 0.0 \
        --n-predict 64 \
        --file "$PROMPT_FILE" \
        --simple-io \
        2>"$out.stderr" | tee "$out.stdout" >/dev/null

    coherence_check "$out.stdout" 3 || { echo "[row 2] FAIL"; return 1; }
    echo "[row 2] PASS"
}

# Row 1 — 27B Q6_K with tiered KV (hot+warm). Requires Phase 2 exit.
row1_27b_tiered() {
    echo "[row 1] not yet implemented — requires Phase 2 (tiered KV rewrite) exit"
    return 2
}

# Row 4 — 97B with tiered KV (hot+warm+cold). Requires Phase 2 exit including cold tier.
row4_97b_tiered() {
    echo "[row 4] not yet implemented — requires Phase 2 (tier rewrite + cold path) exit"
    return 2
}

# Row 5 — MiniMax-M2.7 with weight paging. Requires Phase 1 exit.
row5_minimax_paging() {
    echo "[row 5] not yet implemented — requires Phase 1 (weight pager rewrite) exit"
    return 2
}

# ---------- driver ----------

list_rows() {
    cat <<EOF
row 1: 27B Q6_K, tiered KV (hot+warm)            — gated by Phase 2
row 2: 35B baseline, no tier, no pager            — runnable now
row 3: 97B no tier, no pager (PHASE 0.5 GATE)     — runnable now
row 4: 97B + tier (hot+warm+cold)                 — gated by Phase 2 (incl. cold)
row 5: MiniMax-M2.7 + weight paging               — gated by Phase 1
EOF
}

run_one() {
    case "$1" in
        1) row1_27b_tiered ;;
        2) row2_35b_baseline ;;
        3) row3_97b_no_tier ;;
        4) row4_97b_tiered ;;
        5) row5_minimax_paging ;;
        *) echo "unknown row: $1" >&2; return 1 ;;
    esac
}

main() {
    local mode="row"
    local row=3   # default to gate
    local fail_count=0
    local skip_count=0
    local pass_count=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --row)   mode="row"; row="$2"; shift 2 ;;
            --all)   mode="all"; shift ;;
            --list)  list_rows; exit 0 ;;
            -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
            *) echo "unknown arg: $1" >&2; exit 64 ;;
        esac
    done

    if [[ "$mode" == "row" ]]; then
        run_one "$row"
        exit $?
    fi

    for r in 1 2 3 4 5; do
        echo
        echo "===== row $r ====="
        run_one "$r"
        case $? in
            0) ((pass_count++)) ;;
            2) ((skip_count++)) ;;
            *) ((fail_count++)) ;;
        esac
    done

    echo
    echo "===== summary ====="
    echo "pass: $pass_count   fail: $fail_count   skipped (gated): $skip_count"
    [[ $fail_count -eq 0 ]]
}

main "$@"
