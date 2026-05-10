# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse

# Third Party
from transformers import AutoTokenizer

# Local
from rag import query_prompt_set, system_prompt_set
from utils import (
    PromptBuildMethodType,
    build_fewshot_prompt,
    build_qa_prompt,
    build_rag_prompt_token_ids,
    load_dataset,
    tokenize_segment_for_cache,
)


def parse_prompt_build_method(prompt_build_method: str) -> PromptBuildMethodType:
    prompt_build_method = prompt_build_method.upper()
    if prompt_build_method == "QA":
        return PromptBuildMethodType.QA
    if prompt_build_method == "FEW_SHOT":
        return PromptBuildMethodType.FEW_SHOT
    raise ValueError(f"Invalid prompt build method {prompt_build_method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare precompute doc segments with final RAG prompt segments."
    )
    parser.add_argument("--model", type=str, required=True, help="Model/tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Dataset example index to inspect",
    )
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method: QA or FEW_SHOT",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" # # ",
        help="Blend separator string",
    )
    parser.add_argument(
        "--blend-add-special-in-precomp",
        action="store_true",
        help="Whether precompute adds special tokens to each stored doc chunk",
    )
    return parser.parse_args()


def build_doc_prompts(
    example: dict,
    prompt_build_method: PromptBuildMethodType,
) -> tuple[list[str], str]:
    if prompt_build_method == PromptBuildMethodType.QA:
        return build_qa_prompt(example, "")
    return build_fewshot_prompt(example)


def first_mismatch_index(left: list[int], right: list[int]) -> int | None:
    for idx, (l_val, r_val) in enumerate(zip(left, right, strict=False)):
        if l_val != r_val:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def main() -> None:
    args = parse_args()
    prompt_build_method = parse_prompt_build_method(args.prompt_build_method)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset(args.dataset)
    example = dataset[args.example_index]

    system_prompt = system_prompt_set[prompt_build_method]
    query_prompt = query_prompt_set[prompt_build_method]

    prompt_token_ids, doc_prompts = build_rag_prompt_token_ids(
        tokenizer,
        system_prompt,
        example,
        query_prompt,
        args.separator,
        prompt_build_method,
    )
    assert doc_prompts is not None

    system_token_ids = tokenize_segment_for_cache(
        tokenizer,
        system_prompt,
        add_special_tokens=True,
    )
    sep_token_ids = tokenize_segment_for_cache(
        tokenizer,
        args.separator,
        add_special_tokens=False,
    )

    cursor = len(system_token_ids)
    print(f"system_prompt_tokens={len(system_token_ids)}")
    print(f"separator_tokens={len(sep_token_ids)}")
    print(f"full_prompt_tokens={len(prompt_token_ids)}")

    for doc_idx, doc_prompt in enumerate(doc_prompts):
        if sep_token_ids:
            actual_sep = prompt_token_ids[cursor : cursor + len(sep_token_ids)]
            if actual_sep != sep_token_ids:
                print(
                    f"doc[{doc_idx}] separator mismatch at prompt offset {cursor}: "
                    f"expected {sep_token_ids}, got {actual_sep}"
                )
                return
            cursor += len(sep_token_ids)

        precompute_doc_ids = tokenize_segment_for_cache(
            tokenizer,
            doc_prompt,
            add_special_tokens=args.blend_add_special_in_precomp,
        )
        prompt_doc_ids = prompt_token_ids[cursor : cursor + len(precompute_doc_ids)]
        mismatch_idx = first_mismatch_index(precompute_doc_ids, prompt_doc_ids)

        print(
            f"doc[{doc_idx}] precompute_len={len(precompute_doc_ids)} "
            f"prompt_slice_len={len(prompt_doc_ids)} "
            f"match={mismatch_idx is None}"
        )

        if mismatch_idx is not None:
            left_context_start = max(0, mismatch_idx - 10)
            left_context_end = min(len(precompute_doc_ids), mismatch_idx + 10)
            print(f"doc[{doc_idx}] first_mismatch_index={mismatch_idx}")
            print(
                "precompute_ids_context="
                f"{precompute_doc_ids[left_context_start:left_context_end]}"
            )
            print(
                "prompt_ids_context="
                f"{prompt_doc_ids[left_context_start:left_context_end]}"
            )
            print(
                "precompute_text_context="
                f"{tokenizer.decode(precompute_doc_ids[left_context_start:left_context_end])!r}"
            )
            print(
                "prompt_text_context="
                f"{tokenizer.decode(prompt_doc_ids[left_context_start:left_context_end])!r}"
            )
            return

        cursor += len(precompute_doc_ids)

    if sep_token_ids:
        final_sep = prompt_token_ids[cursor : cursor + len(sep_token_ids)]
        print(f"final_separator_match={final_sep == sep_token_ids}")
        cursor += len(sep_token_ids)

    print(f"query_tokens_after_docs={len(prompt_token_ids) - cursor}")
    print("All document segments match exactly.")


if __name__ == "__main__":
    main()
