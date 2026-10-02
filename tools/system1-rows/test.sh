#!/usr/bin/env bash
# Tests for system1-rows that require NO model inference (task-1-brief.md hard constraint:
# never run this tool against a real GGUF). Every malformed-input case uses a
# deliberately non-existent --model path to prove the resulting exit 2 is about the
# input, not about the model.
#
# Usage: tools/system1-rows/test.sh [build-dir]   (default: build-system1)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${1:-build-system1}"
BIN="$ROOT/$BUILD_DIR/bin/system1-rows"
LIBDIR="$ROOT/$BUILD_DIR/bin"
NONEXISTENT_MODEL="$ROOT/$BUILD_DIR/DOES-NOT-EXIST.gguf"

TMPDIR_T="$(mktemp -d "$ROOT/$BUILD_DIR/system1-rows-test.XXXXXX")"
trap 'rm -rf "$TMPDIR_T"' EXIT

n_pass=0
n_fail=0

pass() { n_pass=$((n_pass+1)); echo "PASS: $1"; }
fail() { n_fail=$((n_fail+1)); echo "FAIL: $1"; }

# Runs the binary, writes its combined stdout+stderr to $LAST_OUT (readable by the
# caller for content checks), and reports pass/fail for the exit code -- all to the
# terminal directly, never through a command substitution (which would swallow the
# pass/fail lines themselves).
LAST_OUT=""
expect_exit() {
    local desc="$1" want="$2" stdin_data="$3"; shift 3
    local rc
    LAST_OUT="$(printf '%s' "$stdin_data" | "$BIN" "$@" 2>&1)"
    rc=$?
    if [ "$rc" -eq "$want" ]; then
        pass "$desc (exit $rc)"
    else
        fail "$desc (expected exit $want, got $rc; output: $LAST_OUT)"
    fi
}

# --- binary builds -----------------------------------------------------------------
if [ -x "$BIN" ]; then
    pass "binary builds ($BIN exists and is executable)"
else
    fail "binary builds ($BIN missing -- run: cmake --build $BUILD_DIR --target system1-rows -j6)"
    echo "$n_pass passed, $n_fail failed"
    exit 1
fi

# --- --help exits 0 ------------------------------------------------------------------
"$BIN" --help >/dev/null 2>&1
rc=$?
[ "$rc" -eq 0 ] && pass "--help exits 0" || fail "--help exits 0 (got $rc)"

# --- usage errors (exit 2, no stdin needed) ------------------------------------------
"$BIN" --out "$TMPDIR_T/out.bin" </dev/null >/dev/null 2>&1
rc=$?
[ "$rc" -eq 2 ] && pass "missing --model is a usage error (exit 2)" || fail "missing --model (got $rc)"

"$BIN" --model "$NONEXISTENT_MODEL" </dev/null >/dev/null 2>&1
rc=$?
[ "$rc" -eq 2 ] && pass "missing --out is a usage error (exit 2)" || fail "missing --out (got $rc)"

"$BIN" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin" --ctx notanumber </dev/null >/dev/null 2>&1
rc=$?
[ "$rc" -eq 2 ] && pass "bad --ctx value is a usage error (exit 2)" || fail "bad --ctx (got $rc)"

# --- malformed input cases: exit 2, naming the model must never be attempted --------
# (a non-existent --model is used throughout; the tool must fail on the INPUT before
# ever trying to open the model file)

VALID_STATE_TOKS="[1,2,3]"

expect_exit "malformed JSON exits 2" 2 'not json at all
' --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "malformed-JSON error message mentions 'model' (should be about input)"

expect_exit "empty state exits 2" 2 "{\"id\":\"a\",\"state\":[],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "empty-state error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "state" || fail "empty-state error message doesn't mention state"

expect_exit "outs offset outside its branch exits 2" 2 \
  "{\"id\":\"a\",\"state\":$VALID_STATE_TOKS,\"branches\":[{\"ids\":[10,11],\"outs\":[5]}]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "outs-offset error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "outs" || fail "outs-offset error message doesn't mention outs"

expect_exit "duplicate ids exits 2" 2 \
  "{\"id\":\"dup\",\"state\":$VALID_STATE_TOKS,\"branches\":[]}
{\"id\":\"dup\",\"state\":$VALID_STATE_TOKS,\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "duplicate-id error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "duplicate" || fail "duplicate-id error message doesn't say duplicate"

expect_exit "missing id field exits 2" 2 \
  "{\"state\":$VALID_STATE_TOKS,\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "missing-id error message mentions 'model' (should be about input)"

expect_exit "non-integer state element exits 2" 2 \
  "{\"id\":\"a\",\"state\":[1,\"x\",3],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "non-integer-state error message mentions 'model' (should be about input)"

expect_exit "negative state token id exits 2" 2 \
  "{\"id\":\"a\",\"state\":[1,-2,3],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "negative-state-id error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "out of range" || fail "negative-state-id error message doesn't say out of range"

expect_exit "state token id > INT32_MAX exits 2" 2 \
  "{\"id\":\"a\",\"state\":[1,9999999999,3],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "too-large-state-id error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "out of range" || fail "too-large-state-id error message doesn't say out of range"

expect_exit "negative branch id exits 2" 2 \
  "{\"id\":\"a\",\"state\":$VALID_STATE_TOKS,\"branches\":[{\"ids\":[10,-11],\"outs\":[1]}]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "negative-branch-id error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "out of range" || fail "negative-branch-id error message doesn't say out of range"

expect_exit "branch id > INT32_MAX exits 2" 2 \
  "{\"id\":\"a\",\"state\":$VALID_STATE_TOKS,\"branches\":[{\"ids\":[10,9999999999],\"outs\":[1]}]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "too-large-branch-id error message mentions 'model' (should be about input)"
echo "$LAST_OUT" | grep -qi "out of range" || fail "too-large-branch-id error message doesn't say out of range"

expect_exit "duplicate outs offset exits 2" 2 \
  "{\"id\":\"a\",\"state\":$VALID_STATE_TOKS,\"branches\":[{\"ids\":[10,11],\"outs\":[1,1]}]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "model" && fail "duplicate-outs error message mentions 'model' (should be about input)"

expect_exit "state_outs offset outside the state exits 2" 2 \
  "{\"id\":\"a\",\"state\":[1,2,3],\"state_outs\":[0,3],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"

expect_exit "duplicate state_outs offset exits 2" 2 \
  "{\"id\":\"a\",\"state\":[1,2,3],\"state_outs\":[1,1],\"branches\":[]}
" --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"
echo "$LAST_OUT" | grep -qi "duplicate" || fail "duplicate-outs error message doesn't say duplicate"

# --- missing --model file with VALID input: exit 1 (model load failure) -------------
VALID_INPUT="{\"id\":\"a\",\"state\":$VALID_STATE_TOKS,\"branches\":[{\"ids\":[10,11],\"outs\":[1]}]}
"
expect_exit "missing --model file with valid input exits 1" 1 "$VALID_INPUT" \
  --model "$NONEXISTENT_MODEL" --out "$TMPDIR_T/out.bin"

# --- default-params check (compiled+run ad hoc against the built libllama) ----------
DP_SRC="$ROOT/tools/system1-rows/test-default-params.cpp"
DP_BIN="$TMPDIR_T/test-default-params"
CXX="${CXX:-/opt/rocm/llvm/bin/clang++}"
if "$CXX" -O0 -std=c++17 \
    -I"$ROOT/include" -I"$ROOT/ggml/include" \
    "$DP_SRC" -L"$LIBDIR" -lllama -Wl,-rpath,"$LIBDIR" \
    -o "$DP_BIN" 2>"$TMPDIR_T/dp-build.log"; then
    if LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" "$DP_BIN"; then
        pass "llama_context_default_params().embd_sparse_outputs == false"
    else
        fail "llama_context_default_params().embd_sparse_outputs == false (binary ran and reported failure)"
    fi
else
    fail "default-params test failed to compile: $(cat "$TMPDIR_T/dp-build.log")"
fi

echo
echo "$n_pass passed, $n_fail failed"
[ "$n_fail" -eq 0 ]
