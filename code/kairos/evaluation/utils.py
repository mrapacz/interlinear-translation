import numpy as np

from kairos.config import TARGET_BLOCK_SEP_TOKEN

T_PREDICTIONS = list[str]
T_REFERENCES = list[list[str]]


def ensure_label(label: list | np.ndarray) -> None:
    assert type(label) in (list, np.ndarray), f"labels are not a list 😘 | {label}"
    assert len(label) == 1, f"There were more than 1 references - please update the code 🫡 | {label}"


def get_text_blocks(text: str, sep: str, skip_empty_blocks: bool = True) -> T_PREDICTIONS:
    """
    >>> get_text_blocks(text="this is <sep> an example <sep> sentence", sep="<sep>")
    ['this is', 'an example', 'sentence']

    >>> get_text_blocks(text="this is <sep> an example <sep> <sep> sentence", sep="<sep>")
    ['this is', 'an example', 'sentence']

    >>> get_text_blocks(text="this is <sep> an example <sep> <sep> sentence", sep="<sep>", skip_empty_blocks=False)
    ['this is', 'an example', '', 'sentence']
    """
    blocks = [block.strip() for block in text.split(sep)]
    if skip_empty_blocks:
        return [b for b in blocks if b]
    return blocks


def text_without_separators(text: str, separator: str) -> str:
    """
    >>> text_without_separators("this is <sep> an example <sep> <sep> text <sep>", separator="<sep>")
    'this is an example text'
    """
    return " ".join(get_text_blocks(text=text, sep=separator, skip_empty_blocks=True))


def text_with_unified_blocks(text: str, separator: str) -> str:
    """
    >>> text_with_unified_blocks("this is <sep> an example <sep> <sep> text <sep>", separator="<sep>")
    'thisis anexample text'
    """
    blocks = get_text_blocks(text=text, sep=separator, skip_empty_blocks=True)
    unified_blocks = ["".join(block.split()) for block in blocks]
    return " ".join(unified_blocks)


def get_preds_labels_as_blocks(
    predictions: T_PREDICTIONS, references: T_REFERENCES, separator: str, skip_empty_blocks: bool
) -> tuple[list[list[str]], list[list[list[str]]]]:
    """
    Transforms each pred and ref into a list of blocks.

    >>> sent = "this is <sep> an example <sep> <sep> text <sep>"
    >>> predictions = [sent]
    >>> references=[[sent]]
    >>> get_preds_labels_as_blocks(predictions=predictions, references=references, separator="<sep>", skip_empty_blocks=False)
    ([['this is', 'an example', '', 'text', '']], [[['this is', 'an example', '', 'text', '']]])
    >>> get_preds_labels_as_blocks(predictions=predictions, references=references, separator="<sep>", skip_empty_blocks=True)
    ([['this is', 'an example', 'text']], [[['this is', 'an example', 'text']]])
    """
    predictions_as_blocks = [
        get_text_blocks(text=pred, sep=separator, skip_empty_blocks=skip_empty_blocks) for pred in predictions
    ]
    references_as_blocks = [
        [get_text_blocks(text=ref, sep=separator, skip_empty_blocks=skip_empty_blocks) for ref in ref_group]
        for ref_group in references
    ]
    return predictions_as_blocks, references_as_blocks


def get_preds_labels_without_separators(
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
    block_sep_token: str = TARGET_BLOCK_SEP_TOKEN,
) -> tuple[T_PREDICTIONS, T_REFERENCES]:
    predictions_without_separators = [text_without_separators(text=pred, separator=block_sep_token) for pred in predictions]
    references_without_separators = [
        [text_without_separators(text=ref, separator=block_sep_token) for ref in ref_group] for ref_group in references
    ]
    return (predictions_without_separators, references_without_separators)


def get_preds_labels_in_blocks(
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
    block_sep_token: str = TARGET_BLOCK_SEP_TOKEN,
) -> tuple[T_PREDICTIONS, T_REFERENCES]:
    predictions_in_blocks = [text_with_unified_blocks(text=pred, separator=block_sep_token) for pred in predictions]
    references_in_blocks = [
        [text_with_unified_blocks(text=ref, separator=block_sep_token) for ref in ref_group] for ref_group in references
    ]
    return (predictions_in_blocks, references_in_blocks)
