# llm-kv-cache-recompute

This repository is based on [LMCache](https://github.com/LMCache/LMCache) and contains a modified, benchmark-focused version of the project. The original LMCache codebase is licensed under the Apache License 2.0; this repository keeps that license and includes local changes for running and evaluating RAG workloads.

LMCache is a KV cache management layer for LLM serving. It reduces time to first token and improves throughput by storing and reusing KV caches across memory and storage tiers such as GPU, CPU, disk, and remote storage. This is especially useful for long-context and retrieval-augmented generation workloads, where many requests reuse the same documents or context passages.

The main focus of this repository is the RAG benchmark under `benchmarks/rag`. The benchmark compares vLLM and LMCache-based serving paths on JSON datasets, measures request latency and quality metrics, and includes scripts for precomputing reusable document KV cache entries before running the main workload. It also includes experiments with different selection and fusion strategies for selective KV cache recompute, allowing only the most useful parts of cached context to be recomputed or blended during generation.

## Installation

Use Python 3.12. The environment can be created with either `uv` or conda; the example below uses `uv`.

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install -r requirements/build.txt
uv pip install vllm==0.10.0  # pulls in the required torch version
uv pip install -e . --no-build-isolation
```

Important package versions:

- `vllm==0.10.0`
- `transformers<5`

## Tiny CacheBlend Smoke Test

Before running the full RAG benchmark, you can use `tiny_blend_test.py` as a quick build and integration check. This small test verifies that vLLM, LMCache, and CacheBlend can run together, store KV on the first request, and retrieve cached KV on a follow-up request.

The smoke test uses:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- layerwise CacheBlend
- local CPU backend
- small `LMCACHE_CHUNK_SIZE=32`
- disabled prefix caching
- prompts built from `prompt_token_ids`
- prompt chunks separated with `# #`

Successful behavior:

- the first request stores KV in LMCache
- the second request gets LMCache hits and retrieves cached KV

Example successful log lines:

```text
Stored 285 out of total 288 tokens
LMCache hit tokens: 189, need to load: 189
Retrieved 187 out of 189 out of total 189 tokens
```

## Running The RAG Benchmark

The benchmark flow is split into two scripts.

`benchmarks/rag/launch_vllm.sh` runs the baseline experiment. It uses plain vLLM without LMCache or CacheBlend, so the server computes KV for the whole prompt during each measured request. This corresponds to a full prefill / full KV recompute baseline.

`benchmarks/rag/launch_lmcache.sh` runs the LMCache + CacheBlend experiment. It first performs a document precompute phase, where one lightweight request per document populates LMCache-backed storage with reusable document KV. It then runs the measured benchmark, where CacheBlend can load cached KV and selectively recompute or fuse only the needed subset during generation.

To keep the comparison fair, both runs should use the same model and serving limits, including `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization`.

### Example Experiment

The following example uses:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- dataset: `benchmarks/rag/inputs/musique_s.json`
- benchmark subset: first `10` requests

Start the baseline vLLM server in one terminal:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

Run the baseline benchmark in another terminal:

```bash
END_INDEX=10 \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
DATASET_PATH=benchmarks/rag/inputs/musique_s.json \
bash benchmarks/rag/launch_vllm.sh
```

This produces a baseline CSV such as `musique_s_vllm_qps_3.5.csv`.

Stop the baseline server before starting the LMCache-enabled run. Then start the LMCache + CacheBlend server:

```bash
LMCACHE_CONFIG_FILE=benchmarks/rag/lmcache_blend.yaml \
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.55 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

Run the LMCache benchmark:

```bash
END_INDEX=10 \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
DATASET_PATH=benchmarks/rag/inputs/musique_s.json \
bash benchmarks/rag/launch_lmcache.sh
```

This produces a CacheBlend CSV such as `musique_s_lmcache_qps_3.5.csv`.

### CacheBlend Configuration

`LMCACHE_CONFIG_FILE` points vLLM/LMCache to the YAML configuration file used for the LMCache + Blend run. In this benchmark it usually points to `benchmarks/rag/lmcache_blend.yaml`:

```bash
export LMCACHE_CONFIG_FILE=benchmarks/rag/lmcache_blend.yaml
```

The selection and fusion behavior is controlled through `extra_config`:

```yaml
extra_config:
  blend_selection_strategy: "topk_diff_k"
  blend_fusion_strategy: "overwrite_selected"
```

`blend_selection_strategy` decides which token positions should be recomputed. `blend_fusion_strategy` decides how the recomputed K/V tensors are merged with the cached K/V tensors.

Available selection strategies:

- `topk_diff_k` selects tokens with the largest squared difference between recomputed and cached K vectors.
- `topk_diff_v` selects tokens with the largest squared difference between recomputed and cached V vectors.
- `topk_diff_kv_sum` selects tokens by the summed K and V differences.
- `topk_diff_kv_weighted` selects tokens by a weighted combination of K and V differences, controlled by the configured K/V weights.
- `topk_diff_k_neighbors` starts from high K-difference tokens and expands to nearby tokens within a final recompute budget.
- `topk_diff_k_span_closing` starts from high K-difference tokens and fills short gaps between selected spans within a final recompute budget.

Available fusion strategies:

- `overwrite_selected` replaces cached K/V at selected positions with recomputed K/V.
- `weighted_selected` blends cached and recomputed K/V at selected positions using fixed cached K/V weights.
- `adaptive_weighted_selected` blends cached and recomputed K/V with token-wise adaptive weights based on how different the recomputed values are from the cached values.

