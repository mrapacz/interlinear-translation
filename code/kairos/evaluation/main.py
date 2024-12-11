from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from loguru import logger
from transformers import EvalPrediction, T5TokenizerFast

from kairos.config import TARGET_BLOCK_SEP_TOKEN, get_config
from kairos.data.sentinel_tokens import get_sentinel_token_id
from kairos.data.utils import decode_batch, decode_safe
from kairos.evaluation.compute_text_metrics import compute_text_based_metrics_in_all_variants
from kairos.evaluation.store_intermediate_results import (
    store_all_outputs,
    store_excessive_inference,
    store_inference_examples,
)
from kairos.evaluation.utils import T_PREDICTIONS, T_REFERENCES
from kairos.utils.common import (
    get_num_tokens_until_value,
    get_number_of_blocks,
    trim_arrays_to_have_the_same_number_of_blocks,
    trim_to_number_of_blocks,
)

eval_idx = 1


def get_eval_identifier() -> str:
    global eval_idx
    identifier = f"eval-{eval_idx}"
    eval_idx += 1
    return identifier


def compute_text_metrics(
    *,
    decoded_preds: T_PREDICTIONS,
    decoded_references: T_REFERENCES,
    trimmed_decoded_preds: T_PREDICTIONS,
    trimmed_decoded_references: T_REFERENCES,
) -> dict[str, float]:
    """Computes all text-based metrics."""
    unlimited_metrics = compute_text_based_metrics_in_all_variants(
        predictions=decoded_preds,
        references=decoded_references,
        block_sep_token=TARGET_BLOCK_SEP_TOKEN,
    )

    trimmed_metrics = compute_text_based_metrics_in_all_variants(
        predictions=trimmed_decoded_preds,
        references=trimmed_decoded_references,
        block_sep_token=TARGET_BLOCK_SEP_TOKEN,
    )

    return {f"unlimited_{metric_name}": val for metric_name, val in unlimited_metrics.items()} | {
        f"trimmed_{metric_name}": val for metric_name, val in trimmed_metrics.items()
    }


def compute_raw_metrics(
    tokenizer,
    preds,
    references,
) -> dict[str, Any]:
    target_block_sep_id = get_sentinel_token_id(tokenizer, TARGET_BLOCK_SEP_TOKEN)

    mean_raw_generation_length = np.mean([np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds])

    # np.argmax allows us to take all the tokens until the first 1
    num_predicted_tokens = get_num_tokens_until_value(preds, value=tokenizer.eos_token_id)
    num_label_tokens = get_num_tokens_until_value(references, value=tokenizer.eos_token_id)

    np.where(num_predicted_tokens == 0, preds.shape[1], num_predicted_tokens)
    mean_token_diff = np.mean(num_predicted_tokens - num_label_tokens)
    mean_token_diff_abs = np.mean(np.abs(num_predicted_tokens - num_label_tokens))

    mean_predicted_block_count = np.mean([get_number_of_blocks(arr, block_separator=target_block_sep_id) for arr in preds])
    # mean_input_block_count = np.mean(input_block_counts)
    # mean_labels_block_count = np.mean(total_block_counts)

    # mean_block_excess = mean_predicted_block_count - mean_input_block_count

    return {
        "mean_raw_generation_length": mean_raw_generation_length,
        "mean_predicted_block_count": mean_predicted_block_count,
        # "mean_labels_block_count": mean_labels_block_count,
        # "mean_input_block_count": mean_input_block_count,
        "mean_token_diff": mean_token_diff,
        "mean_token_diff_abs": mean_token_diff_abs,
        # "mean_block_excess": mean_block_excess,
    }


def decode_preds_and_references(
    *,
    tokenizer: T5TokenizerFast,
    preds: np.ndarray,
    references: np.ndarray,
    safe_decode: bool = False,
) -> tuple[list[str], list[list[str]]]:
    decode_fn = decode_safe if safe_decode else decode_batch
    try:
        decoded_predictions: list[str] = decode_fn(tokenizer=tokenizer, batch=preds)
        decoded_references: list[list[str]] = [[decoded_ref] for decoded_ref in decode_fn(tokenizer=tokenizer, batch=references)]
    except OverflowError:
        logger.error("overflow")
        torch.save(preds, "broken_preds.pt")
        torch.save(references, "broken_labels.pt")
        raise
    else:
        return decoded_predictions, decoded_references


def trim_preds_to_total_num_blocks(
    preds: np.ndarray,
    block_counts: list[int],
    block_separator_token_id: int,
) -> np.ndarray:
    preds_trimmed = np.array(
        [
            trim_to_number_of_blocks(
                arr=pred,
                num_blocks=num,
                block_separator=block_separator_token_id,
            )
            for pred, num in zip(preds, block_counts)
        ]
    )
    return preds_trimmed


def get_trimmed_decoded_preds_and_references(
    *,
    tokenizer: T5TokenizerFast,
    preds: np.ndarray,
    references: np.ndarray,
) -> tuple[list[str], list[list[str]]]:
    trimmed_preds, _ = trim_arrays_to_have_the_same_number_of_blocks(
        arr=preds,
        reference=references,
        block_separator=get_sentinel_token_id(tokenizer, TARGET_BLOCK_SEP_TOKEN),
        trimming_strategy="reference",
    )
    trimmed_decoded_preds, trimmed_decoded_references = decode_preds_and_references(
        tokenizer=tokenizer,
        preds=trimmed_preds,
        references=references,
        safe_decode=False,
    )
    return trimmed_decoded_preds, trimmed_decoded_references


def get_compute_metrics(
    *,
    tokenizer: T5TokenizerFast,
    dset: DatasetDict,
    split="test",
    identifier: str | None = None,
    run_inside_training: bool = True,
) -> Callable[[Any], dict[str, float]]:
    sigla = dset[split]["_SS"]

    def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
        """Computes all metrics.

        We used to trim the preds, so that in case the model starts predicting more blocks than the verse has,
        then perhaps it's already figured out the task at hand, so we should not punish it for this preemptively.
        We stopped doing that, though.

        # preds_trimmed = trim_preds_to_total_num_blocks(
        #     preds=preds,
        #     block_counts=total_block_counts,
        #     block_separator_token_id=get_sentinel_token_id(tokenizer, token=TARGET_BLOCK_SEP_TOKEN),
        # )
        """
        preds, references = eval_preds

        # Both preds and references are of shape (eval_size, max_len)
        assert isinstance(preds, np.ndarray) and isinstance(references, np.ndarray)

        # We're decoding both preds and references.
        # We're also making the references a nested list.
        decoded_preds, decoded_references = decode_preds_and_references(
            tokenizer=tokenizer,
            preds=preds,
            references=references,
            safe_decode=False,
        )

        raw_metrics = compute_raw_metrics(
            tokenizer=tokenizer,
            preds=preds,
            references=references,
        )

        trimmed_decoded_preds, trimmed_decoded_references = get_trimmed_decoded_preds_and_references(
            tokenizer=tokenizer,
            preds=preds,
            references=references,
        )

        text_metrics = compute_text_metrics(
            decoded_preds=decoded_preds,
            decoded_references=decoded_references,
            trimmed_decoded_preds=trimmed_decoded_preds,
            trimmed_decoded_references=trimmed_decoded_references,
        )

        metrics = raw_metrics | text_metrics
        eval_identifier = identifier or get_eval_identifier()

        store_inference_examples(
            decoded_preds=decoded_preds,
            decoded_references=decoded_references,
            dset=dset,
            tokenizer=tokenizer,
            identifier=eval_identifier,
            run_inside_training=run_inside_training,
        )
        store_excessive_inference(
            decoded_preds=decoded_preds,
            decoded_references=decoded_references,
            dset=dset,
            split=split,
            identifier=eval_identifier,
            run_inside_training=run_inside_training,
        )

        if get_config().save_model_outputs:
            store_all_outputs(
                split=split,
                sigla=sigla,
                decoded_preds=decoded_preds,
                decoded_references=decoded_references,
                trimmed_decoded_preds=trimmed_decoded_preds,
                trimmed_decoded_references=trimmed_decoded_references,
                metrics=metrics,
                preds=preds,
                references=references,
                identifier=eval_identifier,
            )
        logger.info(f"{metrics = }")

        return metrics

    return compute_metrics
