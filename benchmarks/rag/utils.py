# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from logging import Logger
import asyncio
import collections
import json
import logging
import re
import string
import threading

# Third Party
from rouge_score import rouge_scorer


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, log_level=logging.DEBUG) -> Logger:
    logger = logging.getLogger(name)

    logger.handlers.clear()
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    return logger


class AsyncLoopWrapper:
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _logger = init_logger("AsyncLoopWrapper")

    @classmethod
    def WaitLoop(cls):
        assert cls._loop is not None, "Loop is not started"

        async def wait_for_tasks():
            current_task = asyncio.current_task(cls._loop)
            tasks = [
                task
                for task in asyncio.all_tasks(cls._loop)
                if not task.done() and task is not current_task
            ]
            cls._logger.info(f"Waiting for {len(tasks)} tasks to finish")
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                err_cnt = sum(1 for r in results if isinstance(r, Exception))
                if err_cnt > 0:
                    cls._logger.error(
                        f"{err_cnt} task(s) finished with exceptions"
                    )

        # Schedule the wait_for_tasks coroutine to be executed in the loop
        future = asyncio.run_coroutine_threadsafe(wait_for_tasks(), cls._loop)
        try:
            # Wait for wait_for_tasks to complete
            future.result()
        except Exception as e:
            cls._logger.error(f"Error while waiting for tasks: {e}")

    @classmethod
    def StartLoop(cls):
        if cls._loop is not None:
            cls._logger.warning("Loop is already started")
            return

        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(cls._loop)
            cls._logger.debug("Starting the asyncio loop")
            cls._loop.run_forever()

        cls._thread = threading.Thread(target=run_loop)
        cls._thread.start()

    @classmethod
    def StopLoop(cls):
        assert cls._loop is not None, "Loop is not started"
        assert cls._thread is not None, "Thread is not started"

        def stop_loop():
            cls._logger.debug("Stopping the loop!")
            cls._loop.stop()

        cls._logger.info("Waiting for remaining tasks to finish")
        cls.WaitLoop()

        cls._loop.call_soon_threadsafe(stop_loop)
        cls._thread.join()

    @classmethod
    def GetLoop(cls) -> asyncio.AbstractEventLoop:
        assert cls._loop is not None, "Loop is not started"
        return cls._loop

    @classmethod
    def GetOrStartLoop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None:
            cls.StartLoop()
        assert cls._loop is not None, "Loop is not started"
        return cls._loop


class PromptBuildMethodType(Enum):
    QA = auto()
    FEW_SHOT = auto()


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def parse_generation(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split() and (
        (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no")
    ):
        s = "No"
    return s


def strip_reasoning_trace(text: str | None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if "</think>" in text:
        trailing = text.rsplit("</think>", 1)[-1].strip()
        if trailing:
            text = trailing
        else:
            text = text.replace("</think>", "")

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_qa_prompt(example, query_prompt):
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt


def build_fewshot_prompt(example):
    q = "\n\n" + example["question"]
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt


def compute_f1(a_pred, a_gold):
    return compute_qa_metrics(a_pred, a_gold)["f1"]


def compute_tokenizer_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_qa_metrics(a_pred, a_gold):
    a_pred = parse_generation(a_pred)
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        agreed = int(gold_toks == pred_toks)
        return {
            "precision": float(agreed),
            "recall": float(agreed),
            "f1": float(agreed),
        }
    if num_same == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_rl(pred, gold):
    if pred is None:
        pred = ""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(gold, pred)["rougeL"].fmeasure
    return rougeL


def _strip_bos_token(tokenizer, token_ids: list[int]) -> list[int]:
    bos_token_id = tokenizer.bos_token_id
    if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        return token_ids[1:]
    return token_ids


def tokenize_segment_for_cache(
    tokenizer,
    text: str,
    add_special_tokens: bool = False,
) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=add_special_tokens).input_ids
    if add_special_tokens:
        return token_ids
    return _strip_bos_token(tokenizer, token_ids)


def build_rag_prompt(
    system_prompt,
    example,
    query_prompt,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
):
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")
    final_prompt = separator.join([system_prompt] + doc_prompts + [q_prompt])
    return final_prompt, doc_prompts


def build_rag_prompt_token_ids(
    tokenizer,
    system_prompt,
    example,
    query_prompt,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
):
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    system_token_ids = tokenize_segment_for_cache(
        tokenizer,
        system_prompt,
        add_special_tokens=True,
    )
    sep_token_ids = []
    if separator:
        sep_token_ids = tokenize_segment_for_cache(
            tokenizer,
            separator,
            add_special_tokens=False,
        )
    doc_token_ids = [
        tokenize_segment_for_cache(
            tokenizer,
            doc_prompt,
            add_special_tokens=False,
        )
        for doc_prompt in doc_prompts
    ]
    query_token_ids = tokenize_segment_for_cache(
        tokenizer,
        q_prompt,
        add_special_tokens=False,
    )

    prompt_token_ids = list(system_token_ids)
    for doc_token_id in doc_token_ids:
        if sep_token_ids:
            prompt_token_ids.extend(sep_token_ids)
        prompt_token_ids.extend(doc_token_id)
    if sep_token_ids:
        prompt_token_ids.extend(sep_token_ids)
    prompt_token_ids.extend(query_token_ids)
    return prompt_token_ids, doc_prompts
