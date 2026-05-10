#!/bin/bash -l
#SBATCH --job-name=vllm_rag
#SBATCH --partition=rocky
#SBATCH -C type_h
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --time=10:00:00
#SBATCH --output=/home/a.tomat/work/LMCache/benchmarks/rag/logs/%x_%j.out
#SBATCH --error=/home/a.tomat/work/LMCache/benchmarks/rag/logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/a.tomat/work/LMCache}"
RAG_DIR="$REPO_DIR/benchmarks/rag"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/a.tomat/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_SNAPSHOT}"
DATASET_PATH="${DATASET_PATH:-$RAG_DIR/inputs/musique_s.json}"
QPS="${QPS:-0.25}"
END_INDEX="${END_INDEX:-150}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-5400}"
DEFAULT_PORT=$((20000 + SLURM_JOB_ID % 20000))
PORT="${PORT:-$DEFAULT_PORT}"
API_KEY="${API_KEY:-EMPTY}"
BLEND_SEPARATOR="${BLEND_SEPARATOR:- # # }"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
SERVER_START_ATTEMPTS="${SERVER_START_ATTEMPTS:-2}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/Miniconda}"
COMPILER_MODULE="${COMPILER_MODULE:-gnu12/12.1}"
CUDA_MODULE="${CUDA_MODULE:-CUDA/12.9.1}"
GIT_MODULE="${GIT_MODULE:-tools/git/v2.33}"

DATASET_STEM=$(basename "$DATASET_PATH")
DATASET_STEM="${DATASET_STEM%.*}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-full_recompute_${DATASET_STEM}}"
RUN_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
LOG_DIR="${LOG_DIR:-$RAG_DIR/logs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RAG_DIR/results/$RUN_NAME}"
SERVER_LOG="$LOG_DIR/server.log"
BENCH_LOG="$LOG_DIR/benchmark.log"
OUTPUT_FILE="$OUTPUT_DIR/${RUN_NAME}.csv"
DEBUG_OUTPUT_FILE="$OUTPUT_DIR/${RUN_NAME}.debug.csv"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

load_required_module() {
  local module_name="$1"
  if ! module load "$module_name"; then
    echo "Failed to load module: $module_name"
    echo "Current loaded modules:"
    module list || true
    echo "module spider output for $module_name:"
    module spider "$module_name" || true
    exit 1
  fi
}

module purge
load_required_module "$PYTHON_MODULE"
load_required_module "$COMPILER_MODULE"
load_required_module "$CUDA_MODULE"
load_required_module "$GIT_MODULE"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lmcache

export TMPDIR="${TMPDIR:-/tmp/$USER-tmp}"
mkdir -p "$TMPDIR"

cd "$RAG_DIR"

echo "run_name=$RUN_NAME"
echo "node=$(hostname)"
echo "dataset_path=$DATASET_PATH"
echo "end_index=$END_INDEX"
echo "output_file=$OUTPUT_FILE"
echo "debug_output_file=$DEBUG_OUTPUT_FILE"
echo "server_log=$SERVER_LOG"
echo "benchmark_log=$BENCH_LOG"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_server() {
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "Port ${PORT} already has a healthy server before this job started."
    echo "Set PORT to a free value or stop the existing server."
    exit 1
  fi

  vllm serve "$MODEL_SNAPSHOT" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    > "$SERVER_LOG" 2>&1 &

  SERVER_PID=$!
}

SERVER_READY=0
for ATTEMPT in $(seq 1 "$SERVER_START_ATTEMPTS"); do
  echo "Starting server attempt ${ATTEMPT}/${SERVER_START_ATTEMPTS}..."
  start_server
  echo "Waiting for vLLM server on http://127.0.0.1:${PORT}/health ..."
  SECONDS_WAITED=0
  until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited before becoming ready. Last 120 server log lines:"
      tail -n 120 "$SERVER_LOG" || true
      break
    fi
    if (( SECONDS_WAITED >= SERVER_READY_TIMEOUT )); then
      echo "Timed out waiting for server after ${SERVER_READY_TIMEOUT}s."
      echo "Last 120 server log lines:"
      tail -n 120 "$SERVER_LOG" || true
      break
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
  done
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    SERVER_READY=1
    break
  fi
  cleanup
  SERVER_PID=""
  sleep 10
done

if [[ "$SERVER_READY" != "1" ]]; then
  echo "Server did not become ready after ${SERVER_START_ATTEMPTS} attempt(s)."
  exit 1
fi

echo "Server is ready after ${SECONDS_WAITED}s. Starting benchmark..."

MODEL_NAME="$MODEL_NAME" \
TOKENIZER_NAME="$TOKENIZER_NAME" \
DATASET_PATH="$DATASET_PATH" \
QPS="$QPS" \
END_INDEX="$END_INDEX" \
MAX_TOKENS="$MAX_TOKENS" \
REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
BASE_URL="http://127.0.0.1:${PORT}/v1" \
API_KEY="$API_KEY" \
BLEND_SEPARATOR="$BLEND_SEPARATOR" \
EXPERIMENT_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
OUTPUT_FILE="$OUTPUT_FILE" \
DEBUG_OUTPUT_FILE="$DEBUG_OUTPUT_FILE" \
./launch_vllm.sh 2>&1 | tee "$BENCH_LOG"

echo "Benchmark finished."
echo "Artifacts:"
echo "  metrics: $OUTPUT_FILE"
echo "  debug:   $DEBUG_OUTPUT_FILE"
echo "  server:  $SERVER_LOG"
echo "  bench:   $BENCH_LOG"
