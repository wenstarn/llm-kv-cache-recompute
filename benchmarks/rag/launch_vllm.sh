#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.2}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/inputs/musique_s.json}"
QPS="${QPS:-0.25}"
END_INDEX="${END_INDEX:-}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-5400}"
BLEND_SEPARATOR="${BLEND_SEPARATOR:- # # }"
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
    DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/${EXPERIMENT_NAME}_${DATASET_NAME}_vllm_qps_${QPS}.csv"
else
    DEFAULT_OUTPUT_FILE="$OUTPUT_DIR/${DATASET_NAME}_vllm_qps_${QPS}.csv"
fi
OUTPUT_FILE="${OUTPUT_FILE:-$DEFAULT_OUTPUT_FILE}"
DEBUG_OUTPUT_FILE="${DEBUG_OUTPUT_FILE:-${OUTPUT_FILE}.debug.csv}"
mkdir -p "$(dirname "$OUTPUT_FILE")" "$(dirname "$DEBUG_OUTPUT_FILE")"

if [[ -z "$END_INDEX" ]]; then
    END_INDEX=$(
        "$PYTHON_BIN" - <<'PY' "$DATASET_PATH"
import json
import sys

with open(sys.argv[1]) as f:
    print(len(json.load(f)))
PY
    )
fi

check_endpoint

"$PYTHON_BIN" "$SCRIPT_DIR/rag.py" \
    --qps "$QPS" \
    --model "$MODEL_NAME" \
    --tokenizer "$TOKENIZER_NAME" \
    --dataset "$DATASET_PATH" \
    --end-index "$END_INDEX" \
    --warmup \
    --separator "$BLEND_SEPARATOR" \
    --prompt-build-method "$PROMPT_BUILD_METHOD" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-tokens "$MAX_TOKENS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --output "$OUTPUT_FILE" \
    --debug-output "$DEBUG_OUTPUT_FILE"
