# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
import contextlib
import os
import time
from pathlib import Path


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def setup_env() -> None:

    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["LMCACHE_CHUNK_SIZE"] = "32"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "2"

    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = " # # "
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    import torch

    torch_lib = str(Path(torch.__file__).resolve().parent / "lib")
    os.environ["TORCH_LIB"] = torch_lib  # optional, only for debugging/convenience

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if old_ld:
        os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{old_ld}"
    else:
        os.environ["LD_LIBRARY_PATH"] = torch_lib


@contextlib.contextmanager
def build_llm():
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    args = EngineArgs(
        model=MODEL,
        kv_transfer_config=ktc,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_prompt(llm: LLM, prompt_token_ids: list[int], tag: str) -> None:
    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=1)
    start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt_token_ids},
        sampling_params=sampling_params,
    )
    print(f"\n--- {tag} ---")
    print(outputs[0].outputs[0].text)
    print(f"time={time.time() - start:.2f}s")


def main() -> None:
    setup_env()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sep = tokenizer.encode(os.environ["LMCACHE_BLEND_SPECIAL_STR"])[1:]

    sys_prompt = tokenizer.encode("You are a helpful assistant.")
    chunk1 = tokenizer.encode(("Cats are mammals. " * 20).strip())[1:]
    chunk2 = tokenizer.encode(("Paris is in France. " * 20).strip())[1:]
    chunk3 = tokenizer.encode(("Python is a programming language. " * 20).strip())[1:]

    tail1 = tokenizer.encode("Summarize the facts.")[1:]
    tail2 = tokenizer.encode("Which facts mention locations?")[1:]

    first_prompt = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + tail1
    second_prompt = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + tail2

    with build_llm() as llm:
        run_prompt(llm, first_prompt, "first")
        time.sleep(1)
        run_prompt(llm, second_prompt, "second")


if __name__ == "__main__":
    main()

