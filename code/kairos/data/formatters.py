from datasets import DatasetDict
from datasets.formatting.formatting import LazyBatch

from kairos.config import (
    SOURCE_BLOCK_SEP_TOKEN,
    SOURCE_META_SEP_TOKEN,
    TARGET_BLOCK_SEP_TOKEN,
    Language,
    NormalizationType,
    TagSet,
)

INPUT_TEXT_BLOCKS = "text"
INPUT_TAG_BLOCKS = "pos"
TARGET_BLOCKS = "target"
RAW_INPUT_TEXT_ONLY = "INPUT_TEXT_ONLY"
RAW_INPUT_TEXT_WITH_MORPHS = "INPUT_TEXT_WITH_MORPHS"
RAW_TARGET = "TARGET"


def join_using_separator(sequence: list[str], separator: str) -> str:
    return f" {separator}".join(sequence)


def fmt_verses_without_separators(example: dict, column: str, result_col: str | None = None) -> dict:
    """
    Joins words in a verse without any special separators - use spaces instead§.

    >>> fmt_verses_without_separators(
    ...     example={"text": ["και", "ἐγένετο", "ἡ", "ἡμέρα", "τρίτη"]},
    ...     column="text",
    ... )
    {'text': 'και ἐγένετο ἡ ἡμέρα τρίτη'}
    """
    result_col = result_col or column
    raw_verse = example[column]
    example[result_col] = " ".join(raw_verse)
    return example


def fmt_verses_with_separator(example: dict, column: str, separator: str, result_col: str | None = None) -> dict:
    """
    Joins words in a verse using the separator.

    >>> fmt_verses_with_separator(
    ...     example={"text": ["και", "ἐγένετο", "ἡ", "ἡμέρα", "τρίτη"]},
    ...     column="text",
    ...     separator=" ",
    ... )
    {'text': 'και  ἐγένετο  ἡ  ἡμέρα  τρίτη'}

    >>> fmt_verses_with_separator(
    ...     example={"text": ["και", "ἐγένετο", "ἡ", "ἡμέρα", "τρίτη"]},
    ...     column="text",
    ...     separator="<2>",
    ...     result_col="text_joined",
    ... )
    {'text': ['και', 'ἐγένετο', 'ἡ', 'ἡμέρα', 'τρίτη'], 'text_joined': 'και <2>ἐγένετο <2>ἡ <2>ἡμέρα <2>τρίτη'}
    """
    result_col = result_col or column
    raw_verse = example[column]
    # We need to add the space here, otherwise the special token may not be tokenized properly...
    example[result_col] = join_using_separator(raw_verse, separator)
    return example


def fmt_verse_with_morphs(
    example: dict,
    text_col: str,
    morph_col: str,
    block_sep: str,
    word_morph_sep: str,
    result_col: str,
) -> dict:
    """
    Joins words and morphs in a verse using the separator.

    >>> fmt_verse_with_morphs(
    ...     example={"text": ["και", "ἐγένετο"], "pos": ["CONJ", "VERB"]},
    ...     text_col="text",
    ...     morph_col="pos",
    ...     block_sep="<0>",
    ...     word_morph_sep="<1>",
    ...     result_col="text_with_morphs",
    ... )
    {'text': ['και', 'ἐγένετο'], 'pos': ['CONJ', 'VERB'], 'text_with_morphs': 'και <1>CONJ <0>ἐγένετο <1>VERB'}
    """
    words = example[text_col]
    morphs = example[morph_col]
    assert len(words) == len(morphs), (
        f"Expected to have the exact same number of words and morph tags. "
        f"Something seems to be wrong. In ({example['_SS']}) got:\n\t{words = }\n\t{morphs = }"
    )
    word_morph_blocks = [join_using_separator([word, morph], separator=word_morph_sep) for word, morph in zip(words, morphs)]

    example[result_col] = join_using_separator(word_morph_blocks, separator=block_sep)

    return example


def get_source_col_name(language: Language, normalization: NormalizationType) -> str:
    return f"{language.value}_GREEK_{normalization.value}".upper()


def get_tags_col_name(language: Language, tagset: TagSet) -> str:
    if tagset == TagSet.UNUSED:
        tagset = TagSet.OBLUBIENICA
    return f"{language.value}_TAGS_{tagset.value}".upper()


def get_target_col_name(language: Language) -> str:
    return f"{language.value}_TRANS".upper()


def choose_columns(examples: LazyBatch, language: Language, normalization: NormalizationType, tagset: TagSet) -> LazyBatch:
    source_col = get_source_col_name(language=language, normalization=normalization)
    tag_col = get_tags_col_name(language=language, tagset=tagset)
    target_col = get_target_col_name(language=language)

    examples[INPUT_TEXT_BLOCKS] = examples[source_col]
    examples[INPUT_TAG_BLOCKS] = examples[tag_col]
    examples[TARGET_BLOCKS] = examples[target_col]
    return examples


def format_columns(
    dset_raw: DatasetDict,
    language: Language,
    normalization: NormalizationType,
    tagset: TagSet,
) -> DatasetDict:
    # First let's determine which columns we want to use.
    chosen_columns = [INPUT_TEXT_BLOCKS, INPUT_TAG_BLOCKS, TARGET_BLOCKS]
    dset_with_chosen_columns_and_joined_sequences = (
        dset_raw.map(
            choose_columns,
            fn_kwargs=dict(language=language, normalization=normalization, tagset=tagset),
            batched=True,
            desc=" ".join(chosen_columns),
        )
        .select_columns([*chosen_columns, "_SS"])
        .map(
            fmt_verses_with_separator,
            fn_kwargs=dict(column=INPUT_TEXT_BLOCKS, separator=SOURCE_BLOCK_SEP_TOKEN, result_col=RAW_INPUT_TEXT_ONLY),
            desc=INPUT_TEXT_BLOCKS,
        )
        .map(
            fmt_verse_with_morphs,
            fn_kwargs=dict(
                text_col=INPUT_TEXT_BLOCKS,
                morph_col=INPUT_TAG_BLOCKS,
                block_sep=SOURCE_BLOCK_SEP_TOKEN,
                word_morph_sep=SOURCE_META_SEP_TOKEN,
                result_col=RAW_INPUT_TEXT_WITH_MORPHS,
            ),
            desc=RAW_INPUT_TEXT_WITH_MORPHS,
        )
    )

    dset_full = dset_with_chosen_columns_and_joined_sequences.map(  # noqa: FURB184
        fmt_verses_with_separator,
        fn_kwargs=dict(column=TARGET_BLOCKS, separator=TARGET_BLOCK_SEP_TOKEN, result_col=RAW_TARGET),
        desc=RAW_TARGET,
    )
    return dset_full
