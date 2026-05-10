#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
MODEL_CONFIG="${MODEL_CONFIG:-$TOKENIZER_NAME}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/inputs/musique_s.json}"
KV_STORAGE_SIZE="${KV_STORAGE_SIZE:-100GB}"
KV_CHUNK_SIZE="${KV_CHUNK_SIZE:-256}"
QPS="${QPS:-0.25}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
END_INDEX="${END_INDEX:-}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-5400}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-30000}"
BLEND_SEPARATOR="${BLEND_SEPARATOR:- # # }"
WARMUP="${WARMUP:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR}"

check_endpoint() {
    if ! "$PYTHON_BIN" - <<'PY' "$BASE_URL"; then
import sys
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
try:
    with urlopen(f"{base_url}/models", timeout=5) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")
except (URLError, RuntimeError) as exc:
    print(
        f"Serving endpoint is not reachable at {base_url}/models: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
        exit 1
    fi
}

infer_prompt_build_method() {
    local dataset_basename
    dataset_basename=$(basename "$1" | tr '[:upper:]' '[:lower:]')
    case "$dataset_basename" in
        *samsum*|*multinews*)
            echo "FEW_SHOT"
            ;;
        *)
            echo "QA"
            ;;
    esac
}

PROMPT_BUILD_METHOD="${PROMPT_BUILD_METHOD:-$(infer_prompt_build_method "$DATASET_PATH")}"
DATASET_NAME=$(basename "$DATASET_PATH")
DATASET_NAME="${DATASET_NAME%.*}"
if [[ -n "$EXPERIMENT_NAME" ]]; then
    DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/${EXPERIMENT_NAME}_${DATASET_NAME}_lmcache_qps_${QPS}.csv"
else
    DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/${DATASET_NAME}_lmcache_qps_${QPS}.csv"
fi
OUTPUT_FILE="${OUTPUT_FILE:-$DEFAULT_OUTPUT_FILE}"
DEBUG_OUTPUT_FILE="${DEBUG_OUTPUT_FILE:-${OUTPUT_FILE}.debug.csv}"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$DEBUG_OUTPUT_FILE")"

check_endpoint

PRECOMPUTE_ARGS=(
    "$SCRIPT_DIR/precompute.py"
    --model "$MODEL_NAME"
    --tokenizer "$TOKENIZER_NAME"
    --model-config "$MODEL_CONFIG"
    --dataset "$DATASET_PATH"
    --prompt-build-method "$PROMPT_BUILD_METHOD"
    --kv-chunk-size "$KV_CHUNK_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --base-url "$BASE_URL"
    --api-key "$API_KEY"
)

if [[ -z "$END_INDEX" && -n "$KV_STORAGE_SIZE" ]]; then
    PRECOMPUTE_ARGS+=(--kv-storage-size "$KV_STORAGE_SIZE")
fi

if [[ -n "$END_INDEX" ]]; then
    PRECOMPUTE_ARGS+=(--end-index "$END_INDEX")
fi

log_str=$("$PYTHON_BIN" "${PRECOMPUTE_ARGS[@]}")
echo "$log_str"
RETURNED_END_INDEX=$(awk '/^Precompute from /{print $5}' <<<"$log_str" | tail -n1)

if [[ -z "$RETURNED_END_INDEX" ]]; then
    echo "Precompute returned an empty end index"
    exit 1
fi

if [[ "$WARMUP" == "1" ]]; then
    set -- "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
        --qps "$QPS" \
        --model "$MODEL_NAME" \
        --tokenizer "$TOKENIZER_NAME" \
        --dataset "$DATASET_PATH" \
        --end-index "$RETURNED_END_INDEX" \
        --separator "$BLEND_SEPARATOR" \
        --prompt-build-method "$PROMPT_BUILD_METHOD" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --max-tokens "$MAX_TOKENS" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --output "$OUTPUT_FILE" \
        --debug-output "$DEBUG_OUTPUT_FILE" \
        --warmup
else
    set -- "$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
        --qps "$QPS" \
        --model "$MODEL_NAME" \
        --tokenizer "$TOKENIZER_NAME" \
        --dataset "$DATASET_PATH" \
        --end-index "$RETURNED_END_INDEX" \
        --separator "$BLEND_SEPARATOR" \
        --prompt-build-method "$PROMPT_BUILD_METHOD" \
        --base-url "$BASE_URL" \
        --api-key "$API_KEY" \
        --max-tokens "$MAX_TOKENS" \
        --request-timeout "$REQUEST_TIMEOUT" \
        --output "$OUTPUT_FILE" \
        --debug-output "$DEBUG_OUTPUT_FILE"
fi

"$@"
