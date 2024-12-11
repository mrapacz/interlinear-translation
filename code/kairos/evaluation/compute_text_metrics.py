from kairos.config import TARGET_BLOCK_SEP_TOKEN
from kairos.evaluation.external_metrics import get_bleu, get_rouge
from kairos.evaluation.metrics import exact_match, levenshtein_blocks, levenshtein_ignore_seps, levenshtein_naive
from kairos.evaluation.utils import T_PREDICTIONS, T_REFERENCES, get_preds_labels_in_blocks, get_preds_labels_without_separators


def compute_format_unaware_metrics(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
) -> dict[str, float]:
    """These metrics are computed the same way, regardless of whether we've removed separators or joined blocks of text."""
    bleu_scores = get_bleu().compute(predictions=predictions, references=references)
    rouge_scores = get_rouge().compute(predictions=predictions, references=references)
    exact_match_score = exact_match(predictions=predictions, references=references)
    return {
        "bleu": bleu_scores["score"],
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "exact_match": exact_match_score["score"],
    }


def compute_format_aware_metrics(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
    block_sep_token: str = TARGET_BLOCK_SEP_TOKEN,
) -> dict[str, float]:
    """
    Computes metrics which could not simply accept texts joined into blocks.

    Currently, this is only Levenshtein family.
    """
    metrics_levenshtein_naive = levenshtein_naive(predictions=predictions, references=references)["score"]
    metrics_levenshtein_ignore_seps = levenshtein_ignore_seps(predictions=predictions, references=references)["score"]
    metrics_levenshtein_blocks_skip_empty = levenshtein_blocks(
        predictions=predictions,
        references=references,
        skip_empty_blocks=True,
        block_separator=block_sep_token,
    )["score"]
    metrics_levenshtein_blocks_keep_empty = levenshtein_blocks(
        predictions=predictions,
        references=references,
        skip_empty_blocks=False,
        block_separator=block_sep_token,
    )["score"]

    return {
        "naive_levenshtein": metrics_levenshtein_naive,
        "no_sep_levenshtein": metrics_levenshtein_ignore_seps,
        "block_levenshtein_skip_empty": metrics_levenshtein_blocks_skip_empty,
        "block_levenshtein_keep_empty": metrics_levenshtein_blocks_keep_empty,
    }


def compute_text_based_metrics_in_all_variants(
    *,
    predictions: T_PREDICTIONS,
    references: T_REFERENCES,
    block_sep_token: str = TARGET_BLOCK_SEP_TOKEN,
) -> dict[str, float]:
    """
    Computes Text metrics in three manners - vanilla (unaware of separators), ignoring separators and block-based.

    Example (Luke 1,77)
    If the translation is 'perfect' i.e. 1:1 against the reference, we should have 100 in in every scenario:
    >>> predictions = [
    ...     "- <s> to give <s> knowledge <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness <s> of [the] sins <s> of them"  # noqa: E501
    ... ]
    >>> references = [
    ...     ["- <s> to give <s> knowledge <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness <s> of [the] sins <s> of them"]  # noqa: E501
    ... ]
    >>> compute_text_based_metrics_in_all_variants(predictions=predictions, references=references, block_sep_token="<s>")
    {'naive_bleu': 100.00000000000004, 'naive_rouge1': 1.0, 'naive_rouge2': 1.0, 'naive_rougeL': 1.0, 'naive_rougeLsum': 1.0, 'naive_exact_match': 1.0, 'nosep_bleu': 100.00000000000004, 'nosep_rouge1': 1.0, 'nosep_rouge2': 1.0, 'nosep_rougeL': 1.0, 'nosep_rougeLsum': 1.0, 'nosep_exact_match': 1.0, 'block_bleu': 100.00000000000004, 'block_rouge1': 1.0, 'block_rouge2': 1.0, 'block_rougeL': 1.0, 'block_rougeLsum': 1.0, 'block_exact_match': 1.0, 'naive_levenshtein': 1.0, 'no_sep_levenshtein': 1.0, 'block_levenshtein_skip_empty': 1.0, 'block_levenshtein_keep_empty': 1.0}

    Now let's asssume our models mistakes changes 'knowledge' -> 'wisdom' and 'of [the] sins' -> 'of sins'
    >>> predictions = [
    ...     "- <s> to give <s> wisdom <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness <s> of sins <s> of them"
    ... ]
    >>> references = [
    ...     ["- <s> to give <s> knowledge <s> of salvation <s> to <s> people <s> of Him <s> in <s> forgiveness <s> of [the] sins <s> of them"]  # noqa: E501
    ... ]
    >>> compute_text_based_metrics_in_all_variants(predictions=predictions, references=references, block_sep_token="<s>")
    {'naive_bleu': 84.99841071762361, 'naive_rouge1': 0.9411764705882353, 'naive_rouge2': 0.8571428571428572, 'naive_rougeL': 0.9411764705882353, 'naive_rougeLsum': 0.9411764705882353, 'naive_exact_match': 0.0, 'nosep_bleu': 56.93976802105479, 'nosep_rouge1': 0.9032258064516129, 'nosep_rouge2': 0.7586206896551724, 'nosep_rougeL': 0.9032258064516129, 'nosep_rougeLsum': 0.9032258064516129, 'nosep_exact_match': 0.0, 'block_bleu': 37.178099888227045, 'block_rouge1': 0.7272727272727272, 'block_rouge2': 0.5, 'block_rougeL': 0.7272727272727272, 'block_rougeLsum': 0.7272727272727272, 'block_exact_match': 0.0, 'naive_levenshtein': 0.9300411522633745, 'no_sep_levenshtein': 0.9300411522633745, 'block_levenshtein_skip_empty': 0.8181818181818181, 'block_levenshtein_keep_empty': 0.8181818181818181}
    """
    # Naive metrics
    metrics_naive = compute_format_unaware_metrics(predictions=predictions, references=references)

    # Metrics ignoring separators
    preds_no_seps, references_no_seps = get_preds_labels_without_separators(
        predictions=predictions,
        references=references,
        block_sep_token=block_sep_token,
    )
    metrics_ignore_seps = compute_format_unaware_metrics(predictions=preds_no_seps, references=references_no_seps)

    # Metrics treating blocks as individual words
    preds_blocks, refs_blocks = get_preds_labels_in_blocks(
        predictions=predictions,
        references=references,
        block_sep_token=block_sep_token,
    )
    metrics_blocks = compute_format_unaware_metrics(predictions=preds_blocks, references=refs_blocks)

    # Additionally, compute levenshtein metrics
    levenshtein_metrics = compute_format_aware_metrics(
        predictions=predictions,
        references=references,
        block_sep_token=block_sep_token,
    )

    return (
        {f"naive_{metric_name}": val for metric_name, val in metrics_naive.items()}
        | {f"nosep_{metric_name}": val for metric_name, val in metrics_ignore_seps.items()}  # noqa: W503
        | {f"block_{metric_name}": val for metric_name, val in metrics_blocks.items()}  # noqa: W503
        | levenshtein_metrics  # noqa: W503
    )
