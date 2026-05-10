# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import os

# Third Party
from openai import OpenAI
from transformers import AutoConfig, AutoTokenizer
import yaml

# Local
from utils import (
    PromptBuildMethodType,
    build_fewshot_prompt,
    build_qa_prompt,
    load_dataset,
    tokenize_segment_for_cache,
)


system_prompt_set = {
    PromptBuildMethodType.QA: "You will be asked a question after reading several passages. "  # noqa: E501
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. "
    "The answer should be within 5 words..\nPassages:\n",
    PromptBuildMethodType.FEW_SHOT: "Summarize the dialogue into a few short sentences. "  # noqa: E501
    "The following are some examples.\n\n",
}


@dataclass
class PrecomputeConfig:
    # Model name.
    model: str
    # Tokenizer name.
    tokenizer: str
    # Model config path.
    model_config: str
    # Dataset.
    dataset: str
    # Start index.
    start_idx: int
    # End index.
    end_idx: int
    # KV storage size.
    kv_storage_size: int
    # KV chunk size.
    kv_chunk_size: int
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # API key
    api_key: str
    # Base url
    base_url: str
    # KV cache precision.
    kv_precision: int
    # Max model length for precompute. If <= 0, no limit.
    max_model_len: int


class KVSizeCalculator:
    def __init__(
        self,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        precision: int,
    ):
        self.ratio = num_key_value_heads * head_dim * num_layers * precision * 2

    def get_kv_size(self, token_cnt: int) -> int:
        return token_cnt * self.ratio


def _load_blend_add_special_in_precomp() -> bool:
    config_file = os.getenv("LMCACHE_CONFIG_FILE")
    if not config_file:
        return False
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return False
    return bool(config.get("blend_add_special_in_precomp", False))


class EndpointPrecomputer:
    """Populate the serving endpoint cache by issuing lightweight requests."""

    def __init__(
        self, api_key: str, base_url: str, model: str, tokenizer
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._tokenizer = tokenizer
        self._blend_add_special_in_precomp = _load_blend_add_special_in_precomp()

    def precompute_kv(
        self,
        text_chunk: str,
        force_special_tokens: bool = False,
    ) -> None:
        input_ids = tokenize_segment_for_cache(
            self._tokenizer,
            text_chunk,
            add_special_tokens=(
                self._blend_add_special_in_precomp or force_special_tokens
            ),
        )
        self._client.completions.create(
            model=self.model,
            prompt=input_ids,
            max_tokens=1,
            temperature=0.0,
            extra_body={"ignore_eos": True},
        )


def _get_num_key_value_heads(model_config) -> int:
    if hasattr(model_config, "num_key_value_heads"):
        return model_config.num_key_value_heads
    if hasattr(model_config, "num_attention_heads"):
        return model_config.num_attention_heads
    raise AttributeError("Model config is missing num_key_value_heads")


def _get_head_dim(model_config) -> int:
    if hasattr(model_config, "head_dim"):
        return model_config.head_dim
    if hasattr(model_config, "hidden_size") and hasattr(
        model_config, "num_attention_heads"
    ):
        return model_config.hidden_size // model_config.num_attention_heads
    raise AttributeError("Model config is missing head_dim")


def _build_kv_size_calculator(
    model_config_name: str, kv_precision: int
) -> KVSizeCalculator:
    try:
        model_config = AutoConfig.from_pretrained(model_config_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the model config for size-limited precompute. "
            "If you are working offline, set --end-index and omit "
            "--kv-storage-size so precompute can run without fetching "
            "tokenizer/config files."
        ) from exc
    return KVSizeCalculator(
        _get_num_key_value_heads(model_config),
        _get_head_dim(model_config),
        model_config.num_hidden_layers,
        kv_precision,
    )


def _estimate_case_size(
    doc_prompts: list[str],
    tokenizer,
    round_up_token_cnt: int,
    kv_size_calculator: KVSizeCalculator,
) -> int:
    token_cnt = 0
    for doc_prompt in doc_prompts:
        assert len(doc_prompt) > 0
        input_comps = tokenizer(doc_prompt, add_special_tokens=False).input_ids
        assert len(input_comps) > 0
        temp_cnt = len(input_comps)
        temp_cnt = ((temp_cnt + round_up_token_cnt - 1) // round_up_token_cnt) * (
            round_up_token_cnt
        )
        token_cnt += temp_cnt
    assert token_cnt > 0, f"token_cnt {token_cnt} <= 0"
    return kv_size_calculator.get_kv_size(token_cnt)


def precompute_all_kv(config: PrecomputeConfig) -> Tuple[int, int, str]:
    tokenizer = None
    kv_size_calculator: Optional[KVSizeCalculator] = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the tokenizer for precompute."
        ) from exc
    kv_size_calculator = _build_kv_size_calculator(
        config.model_config, config.kv_precision
    )
    eval_dataset = load_dataset(config.dataset)
    start_idx = config.start_idx
    end_idx = config.end_idx
    if end_idx >= 0:
        assert end_idx <= len(eval_dataset), (
            f"end_index {end_idx} > length of dataset {len(eval_dataset)}"
        )
    assert start_idx >= 0, f"start_idx {start_idx} < 0"
    assert start_idx < len(eval_dataset), (
        f"start_idx {start_idx} >= length of dataset {len(eval_dataset)}"
    )
    precompute_kv = EndpointPrecomputer(
        config.api_key, config.base_url, config.model, tokenizer
    )
    current_size_taken = 0
    size_upper_bound = config.kv_storage_size
    has_size_limit = size_upper_bound > 0
    current_idx = start_idx
    round_up_token_cnt = config.kv_chunk_size
    assert round_up_token_cnt >= 1

    system_prompt = system_prompt_set[config.prompt_build_method]
    system_prompt_ids = tokenize_segment_for_cache(
        tokenizer,
        system_prompt,
        add_special_tokens=True,
    )
    system_prompt_size = kv_size_calculator.get_kv_size(len(system_prompt_ids))
    can_fit_system_prompt = (
        not has_size_limit or current_size_taken + system_prompt_size <= size_upper_bound
    )
    if can_fit_system_prompt:
        precompute_kv.precompute_kv(system_prompt, force_special_tokens=True)
        if has_size_limit:
            current_size_taken += system_prompt_size

    while True:
        if end_idx >= 0:
            if current_idx >= end_idx:
                break
        else:
            if current_size_taken >= size_upper_bound or current_idx >= len(
                eval_dataset
            ):
                break
        example = eval_dataset[current_idx]
        doc_prompts = None
        this_case_size = 0
        if config.prompt_build_method == PromptBuildMethodType.QA:
            doc_prompts, _ = build_qa_prompt(example, "")
        elif config.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
            doc_prompts, _ = build_fewshot_prompt(example)
        assert doc_prompts is not None
        processed_prompts = []
        if has_size_limit:
            assert tokenizer is not None
            assert kv_size_calculator is not None
            token_cnt = 0
            skip_example = False
            for doc_prompt in doc_prompts:
                input_comps = tokenize_segment_for_cache(
                    tokenizer,
                    doc_prompt,
                    add_special_tokens=precompute_kv._blend_add_special_in_precomp,
                )
                if config.max_model_len > 0 and len(input_comps) > config.max_model_len:
                    skip_example = True
                    break
                temp_cnt = len(input_comps)
                temp_cnt = (
                    (temp_cnt + round_up_token_cnt - 1) // round_up_token_cnt
                ) * round_up_token_cnt
                token_cnt += temp_cnt
                processed_prompts.append(doc_prompt)
            if skip_example:
                current_idx += 1
                continue
            this_case_size = kv_size_calculator.get_kv_size(token_cnt)
            if current_size_taken + this_case_size > size_upper_bound:
                break
        else:
            processed_prompts = doc_prompts
        for prompt in processed_prompts:
            precompute_kv.precompute_kv(prompt)
        current_idx += 1
        if has_size_limit:
            current_size_taken += this_case_size

    return start_idx, current_idx, precompute_kv.model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG precompute configurations.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name")
    parser.add_argument(
        "--model-config", type=str, default="", help="Model config path"
    )
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
    )
    parser.add_argument(
        "--kv-storage-size", type=str, default="", help="KV storage size"
    )
    parser.add_argument(
        "--kv-chunk-size", type=int, default=256, help="KV storage chunk size"
    )
    parser.add_argument(
        "--kv-precision-bit",
        type=int,
        default=16,
        help="KV cache precision bit",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=-1,
        help="Max model length used for precompute. If <=0, no limit.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key of the serving engine endpoint",
    )
    args = parser.parse_args()
    return args


def parse_size(size: str) -> int:
    if len(size) == 0:
        return -1
    else:
        size = size.upper()
        if size.endswith("KB"):
            return int(size[:-2]) * 1024
        elif size.endswith("MB"):
            return int(size[:-2]) * 1024 * 1024
        elif size.endswith("GB"):
            return int(size[:-2]) * 1024 * 1024 * 1024
        elif size.endswith("TB"):
            return int(size[:-2]) * 1024 * 1024 * 1024 * 1024
        elif size.endswith("B"):
            return int(size[:-1])
        else:
            raise ValueError(f"Invalid size unit {size}")


def parse_prompt_build_method(
    prompt_build_method: str,
) -> PromptBuildMethodType:
    prompt_build_method = prompt_build_method.upper()
    if prompt_build_method == "QA":
        return PromptBuildMethodType.QA
    elif prompt_build_method == "FEW_SHOT":
        return PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")


def run_precompute(args):
    kv_storage_size = parse_size(args.kv_storage_size)
    kv_chunk_size = args.kv_chunk_size
    prompt_build_method = parse_prompt_build_method(args.prompt_build_method)
    kv_precision_bit = args.kv_precision_bit
    assert kv_precision_bit % 8 == 0, (
        f"kv_precision_bit {kv_precision_bit} is not a multiple of 8"
    )
    kv_precision = kv_precision_bit // 8
    config = PrecomputeConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        model_config=args.model_config,
        dataset=args.dataset,
        start_idx=args.start_index,
        end_idx=args.end_index,
        kv_storage_size=kv_storage_size,
        kv_chunk_size=kv_chunk_size,
        prompt_build_method=prompt_build_method,
        api_key=args.api_key,
        base_url=args.base_url,
        kv_precision=kv_precision,
        max_model_len=args.max_model_len,
    )
    start_idx, end_idx, model_name = precompute_all_kv(config)
    return start_idx, end_idx, model_name


def main():
    args = parse_arguments()
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model
    if len(args.model_config) == 0:
        args.model_config = args.model
    start_idx, end_idx, model_name = run_precompute(args)
    print(f"Precompute from {start_idx} to {end_idx} for model {model_name}")


if __name__ == "__main__":
    main()
