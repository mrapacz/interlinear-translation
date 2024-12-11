import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import T5TokenizerFast

from kairos.config import ALL_SENTINEL_TOKENS


def postprocess_text(
    preds: list[str],
    labels: list[str],
) -> tuple[list[str], list[list[str]]]:
    stripped_preds = [pred.strip() for pred in preds]
    nested_labels = [[label.strip()] for label in labels]

    return stripped_preds, nested_labels


# We need to hardcode these, since bowphs/GreTa has an empty all_special_tokens list.
# At the same time bowphs/PhilTa reports all special tokens, including the sentinel ones.
# mt5 on the other hand reports only "<unk>", "</s>", "<pad>" as the special tokens.
def get_special_tokens_to_ignore(tokenizer: T5TokenizerFast) -> list[str]:
    return list(set(tokenizer.all_special_tokens + SPECIAL_TOKENS_TO_IGNORE) - set(ALL_SENTINEL_TOKENS))


SPECIAL_TOKENS_TO_IGNORE = ["<unk>", "</s>", "<pad>"]


def remove_special_tokens(text: str, tokens_to_remove: list[str]) -> str:
    for special_token in tokens_to_remove:
        text = text.replace(special_token, "").strip()
    return text


def batch_remove_special_tokens(tokenizer: T5TokenizerFast, batch: list[str]) -> list[str]:
    tokens_to_remove = get_special_tokens_to_ignore(tokenizer)
    return [remove_special_tokens(text=ex, tokens_to_remove=tokens_to_remove) for ex in batch]


def decode_batch(tokenizer: T5TokenizerFast, batch: torch.Tensor | np.ndarray | list[str]) -> list[str]:
    tensor_padded = np.where(batch != -100, batch, tokenizer.pad_token_id)
    batch_decoded: list[str] = tokenizer.batch_decode(tensor_padded, skip_special_tokens=False)

    return batch_remove_special_tokens(tokenizer=tokenizer, batch=batch_decoded)


def decode_safe(tokenizer: T5TokenizerFast, batch: torch.Tensor | np.ndarray | list[str]) -> list[str]:
    """
    Less performant but safer version.

    This should work in case the elements in batch are not of the same size.
    """
    batch_without_negative_nums = [np.where(x != -100, x, tokenizer.pad_token_id) for x in batch]
    batch_decoded: list[str] = [tokenizer.decode(x, skip_special_tokens=False) for x in batch_without_negative_nums]

    return batch_remove_special_tokens(tokenizer=tokenizer, batch=batch_decoded)


def sort_dset_by_length(dset: DatasetDict, column: str) -> DatasetDict:
    return DatasetDict(
        {split: Dataset.from_list(sorted(values, key=lambda x: len(x[column]), reverse=True)) for split, values in dset.items()}
    )
