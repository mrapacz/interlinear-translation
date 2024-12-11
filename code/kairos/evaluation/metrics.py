import numpy as np
from Levenshtein import ratio

from kairos.config import TARGET_BLOCK_SEP_TOKEN
from kairos.evaluation.utils import T_PREDICTIONS, T_REFERENCES, get_preds_labels_as_blocks, get_preds_labels_without_separators


def exact_match(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
) -> dict[str, float]:
    scores = []
    for pred, labels in zip(predictions, references):
        assert len(labels) == 1
        label = labels[0]

        scores.append(1 if pred == label else 0)
    mean_score = float(np.mean(scores))
    return {"score": mean_score}


def _levenshtein(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
) -> dict[str, float]:
    """
    Predictions = [ "- <s> to give <s> wisdom <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness <s> of sins
    <s> of them" ] references = [ ["- <s> to give <s> knowledge <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness
    <s> of [the] sins <s> of them"]  # noqa: E501 ]
    """
    scores = []
    for pred, labels in zip(predictions, references):
        assert len(labels) == 1
        label = labels[0]

        naive_score = ratio(pred, label)
        scores.append(naive_score)
    mean_score = float(np.mean(scores))
    return {"score": mean_score}


def levenshtein_naive(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
):
    return _levenshtein(predictions=predictions, references=references)


def levenshtein_ignore_seps(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
):
    preds_no_seps, refs_no_seps = get_preds_labels_without_separators(predictions=predictions, references=references)
    return _levenshtein(predictions=preds_no_seps, references=refs_no_seps)


def levenshtein_blocks(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
    skip_empty_blocks: bool,
    block_separator: str = TARGET_BLOCK_SEP_TOKEN,
):
    """
    Treats each of the blocks as a different 'token'.

    In case preds and references are the same, empty blocks should not matter:
    >>> predictions = ["this is <s> an example <s>"]
    >>> references = [["this is <s> an example <s>"]]
    >>> levenshtein_blocks(predictions=predictions, references=references, skip_empty_blocks=True, block_separator="<s>")
    {'score': 1.0}
    >>> levenshtein_blocks(predictions=predictions, references=references, skip_empty_blocks=False, block_separator="<s>")
    {'score': 1.0}

    Now the following should work the same as levenshtein distance between "ab" and "ac"
    >>> predictions = ["this is <s> example <s>"]
    >>> levenshtein_blocks(predictions=predictions, references=references, skip_empty_blocks=True, block_separator="<s>")
    {'score': 0.5}

    If we do not ignore empty blocks, they are going to be treated as actual 'words', so the following should be analogous to
    >>> f'{ratio("adec", "abc"):.2f}'
    '0.57'
    >>> predictions = ["this is <s> an <s> example <s>"]
    >>> score = levenshtein_blocks(predictions=predictions, references=references, skip_empty_blocks=False, block_separator="<s>")
    >>> {key: f"{value:.2f}" for key, value in score.items()}
    {'score': '0.57'}
    """
    preds_blocks, refs_blocks = get_preds_labels_as_blocks(
        predictions=predictions,
        references=references,
        separator=block_separator,
        skip_empty_blocks=skip_empty_blocks,
    )
    # Note: it's fine to ignore arg-type below.
    # Levenshtein can handle both ratio("a", "b") as well as ratio(["a", "b"], ["c", "d"])
    # TODO: We could clean this up and fix types.
    return _levenshtein(predictions=preds_blocks, references=refs_blocks)  # type: ignore[arg-type]
