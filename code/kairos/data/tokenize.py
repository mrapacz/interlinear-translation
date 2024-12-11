from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from datasets.formatting.formatting import LazyBatch, LazyRow
from transformers import AutoTokenizer, BatchEncoding, T5TokenizerFast

from kairos.config import SOURCE_BLOCK_SEP_TOKEN, Checkpoint, SourceType
from kairos.data.formatters import INPUT_TAG_BLOCKS, RAW_INPUT_TEXT_ONLY, RAW_INPUT_TEXT_WITH_MORPHS, RAW_TARGET
from kairos.data.morph_tokenizer import CustomMorphTokenizer, get_morph_tokenizer
from kairos.data.sentinel_tokens import ensure_sentinel_tokens_are_in_place, get_sentinel_token_id
from kairos.utils.common import split_array_to_blocks, trim_arrays_to_have_the_same_number_of_blocks

DTensorT = dict[str, torch.Tensor]

TOKENIZED_TEXT_ONLY_INPUT_IDS = "text_only_input_ids"
TOKENIZED_TEXT_ONLY_ATTENTION_MASK = "text_only_attention_mask"
TOKENIZED_TEXT_ONLY_LABELS = "text_only_labels"
TOKENIZED_TEXT_POS_INPUT_IDS = "text_pos_input_ids"
TOKENIZED_TEXT_POS_ATTENTION_MASK = "text_pos_attention_mask"
TOKENIZED_TEXT_POS_LABELS = "text_pos_labels"

TRIMMED_TEXT_INPUT_IDS = "trimmed_text_input_ids"
TRIMMED_TEXT_ATTENTION_MASK = "trimmed_text_attention_mask"

FINAL_INPUT_MORPHS = "input_morphs"
FINAL_INPUT_MASK = "attention_mask"
FINAL_INPUT_IDS = "input_ids"
FINAL_LABELS = "labels"


def tokenize_text(
    examples: LazyBatch,
    input_col: str,
    target_col: str,
    tokenizer: T5TokenizerFast,
    tokenizer_max_length: int,
) -> BatchEncoding:
    return tokenizer(examples[input_col], text_target=examples[target_col], max_length=tokenizer_max_length, truncation=True)


def set_expected_number_of_blocks():
    pass


def tokenize_morphs(
    example: LazyRow,
    tokenizer: T5TokenizerFast,
    morph_tokenizer: CustomMorphTokenizer,
) -> LazyRow:
    # Tokenize
    encoded_pos_tags = morph_tokenizer.encode(example[INPUT_TAG_BLOCKS])

    # Obtain text blocks
    block_separator_token_id = get_sentinel_token_id(tokenizer=tokenizer, token=SOURCE_BLOCK_SEP_TOKEN)
    tokenized_text = np.array(example[TRIMMED_TEXT_INPUT_IDS])
    tokenized_text_blocks = split_array_to_blocks(tokenized_text, block_separator=block_separator_token_id)

    tokenized_pos_tags = []
    for pos_token_id, tokenized_text_block in zip(encoded_pos_tags, tokenized_text_blocks):
        tokenized_pos_tags += [pos_token_id] * len(tokenized_text_block)

    arr = np.array(tokenized_pos_tags)
    arr[tokenized_text == block_separator_token_id] = morph_tokenizer.block_separator_token_id
    arr[tokenized_text == tokenizer.eos_token_id] = morph_tokenizer.eos_token_id
    arr[tokenized_text == tokenizer.pad_token_id] = morph_tokenizer.pad_token_id
    arr[tokenized_text == tokenizer.unk_token_id] = morph_tokenizer.unk_token_id

    example[FINAL_INPUT_MORPHS] = arr
    return example


def ensure_ends_with_eos_token(arr: np.ndarray, tokenizer_max_length: int, eos_token_id: int) -> np.ndarray:
    # Let's ensure our sequence ends with eos_token_id and at the same time let's make sure we do not exceed tokenizer_max_length
    if len(arr) == tokenizer_max_length or arr[-1] == eos_token_id:
        arr[-1] = eos_token_id
    else:
        # This should be rare, but just in case let's make sure we do have the eos token at the end.
        arr = np.append(arr, eos_token_id)
    assert len(arr) <= tokenizer_max_length
    return arr


def correct_sequence_length(
    example: LazyRow,
    tokenizer: T5TokenizerFast,
    block_separator_token_id: int,
    to_be_corrected_col: str,
    reference_col: str,
    output_text_col: str,
    output_attention_col: str,
    tokenizer_max_length: int,
) -> dict[str, Any]:
    """Number of token groups in the reference array should be >= number of tokens in the model we're testing."""
    assert isinstance(example, LazyRow), "Are you sure you're not batching?"

    main = np.array(example[to_be_corrected_col])
    reference = np.array(example[reference_col])

    # Trim
    assert np.sum(np.where(main == block_separator_token_id, 1, 0)) >= np.sum(
        np.where(reference == block_separator_token_id, 1, 0)
    ), "reference has more token groups than fixed one"

    main_trimmed, _ = trim_arrays_to_have_the_same_number_of_blocks(
        arr=main,
        reference=reference,
        block_separator=block_separator_token_id,
    )
    assert len(main_trimmed), "empty trimmed array"

    main_trimmed = ensure_ends_with_eos_token(main_trimmed, tokenizer_max_length, eos_token_id=tokenizer.eos_token_id)
    main_attention_mask_trimmed = np.ones(main_trimmed.shape)

    return {**example, output_text_col: main_trimmed, output_attention_col: main_attention_mask_trimmed}


def tokenize(
    *,
    dset: DatasetDict,
    source_type: SourceType,
    tokenizer: T5TokenizerFast,
    tokenizer_max_length: int,
):
    # TODO: which columns do we need to have in the dset?
    dset_full_with_tokenized_text = dset.map(  # noqa: FURB184
        tokenize_text,
        fn_kwargs=dict(
            input_col=RAW_INPUT_TEXT_ONLY,  # <- this is where we want just the text blocks
            target_col=RAW_TARGET,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
        ),
        batched=True,
    ).rename_columns(
        {
            "input_ids": TOKENIZED_TEXT_ONLY_INPUT_IDS,
            "attention_mask": TOKENIZED_TEXT_ONLY_ATTENTION_MASK,
            "labels": TOKENIZED_TEXT_ONLY_LABELS,
        }
    )

    dset_tokenized = dset_full_with_tokenized_text.map(  # noqa: FURB184
        tokenize_text,
        fn_kwargs=dict(
            input_col=RAW_INPUT_TEXT_WITH_MORPHS,  # <- this is where we're tokenizing text with morphs
            target_col=RAW_TARGET,
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
        ),
        batched=True,
    ).rename_columns(
        {
            # tm stands for text-(with)-morph
            "input_ids": TOKENIZED_TEXT_POS_INPUT_IDS,
            "attention_mask": TOKENIZED_TEXT_POS_ATTENTION_MASK,
            "labels": TOKENIZED_TEXT_POS_LABELS,
        }
    )
    # TODO: assert dset_tokenized[TEXT_POS_LABELS] == dset_tokenized[TEXT_ONLY_LABELS]

    dset_corrected = dset_tokenized.map(  # noqa: FURB184
        correct_sequence_length,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            tokenizer_max_length=tokenizer_max_length,
            block_separator_token_id=get_sentinel_token_id(tokenizer, SOURCE_BLOCK_SEP_TOKEN),
            to_be_corrected_col=TOKENIZED_TEXT_ONLY_INPUT_IDS,
            reference_col=TOKENIZED_TEXT_POS_INPUT_IDS,
            output_text_col=TRIMMED_TEXT_INPUT_IDS,
            output_attention_col=TRIMMED_TEXT_ATTENTION_MASK,
        ),
    )

    match source_type:
        case SourceType.TEXT_ONLY:
            # return trimmed columns + target
            dset_final = dset_corrected.rename_columns(
                {
                    TRIMMED_TEXT_INPUT_IDS: FINAL_INPUT_IDS,
                    TRIMMED_TEXT_ATTENTION_MASK: FINAL_INPUT_MASK,
                    TOKENIZED_TEXT_ONLY_LABELS: FINAL_LABELS,
                }
            )
        case SourceType.TEXT_WITH_POS:
            # return text with pos + target
            dset_final = dset_corrected.rename_columns(
                {
                    TOKENIZED_TEXT_POS_INPUT_IDS: FINAL_INPUT_IDS,
                    TOKENIZED_TEXT_POS_ATTENTION_MASK: FINAL_INPUT_MASK,
                    TOKENIZED_TEXT_POS_LABELS: FINAL_LABELS,
                }
            )
        case SourceType.TEXT_WITH_POS_EMBEDDINGS:
            # We need to pass the dataset before correction, otherwise we might not get some tokens into the vocab
            get_morph_tokenizer().initialize(dset=dset, tags_col=INPUT_TAG_BLOCKS)

            dset_final = dset_corrected.map(
                tokenize_morphs,
                fn_kwargs=dict(morph_tokenizer=get_morph_tokenizer(), tokenizer=tokenizer),
            ).rename_columns(
                {
                    TRIMMED_TEXT_INPUT_IDS: FINAL_INPUT_IDS,
                    TRIMMED_TEXT_ATTENTION_MASK: FINAL_INPUT_MASK,
                    TOKENIZED_TEXT_ONLY_LABELS: FINAL_LABELS,
                }
            )

        case _:
            assert False, f"Wrong source type passed in, found {source_type} {type(SourceType.TEXT_ONLY)} {type(source_type)}"

    return dset_final


def get_final_columns(source_type: SourceType) -> list[str]:
    return [FINAL_INPUT_IDS, FINAL_INPUT_MASK, FINAL_LABELS] + (
        [FINAL_INPUT_MORPHS] if source_type == SourceType.TEXT_WITH_POS_EMBEDDINGS else []
    )


def get_tokenizer(checkpoint: Checkpoint) -> T5TokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.value, legacy=False)
    ensure_sentinel_tokens_are_in_place(tokenizer)
    if checkpoint == checkpoint.GRETA:
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token_id = 1
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = 2
    return tokenizer
