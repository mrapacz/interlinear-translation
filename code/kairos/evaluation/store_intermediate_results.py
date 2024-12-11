import json

import torch
from datasets import DatasetDict
from loguru import logger
from transformers import T5TokenizerFast

from kairos.config import TARGET_BLOCK_SEP_TOKEN, get_logconf
from kairos.evaluation.utils import T_PREDICTIONS, T_REFERENCES
from kairos.utils.common import simplify_sentinel_token_display


def store_results(
    lines: list[str],
    name: str,
    identifier: str,
    run_inside_training: bool = True,
) -> None:
    logconf = get_logconf()
    logger.info(f"{name} - {identifier}")

    file = logconf.out_dir / identifier / name
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "a", encoding="utf-8") as f:
        f.writelines(lines)

    if run_inside_training:
        logconf.run[f"finetuning/file_{name}"].upload(str(file))


def store_inference_examples(
    decoded_preds: T_PREDICTIONS,
    decoded_references: T_REFERENCES,
    dset: DatasetDict,
    tokenizer: T5TokenizerFast,
    identifier: str | None = None,
    run_inside_training: bool = True,
) -> None:
    reference_examples = []
    for sample_idx in range(min(10, len(decoded_preds))):
        preds_simplified = simplify_sentinel_token_display(decoded_preds[sample_idx])
        refs_simplified = simplify_sentinel_token_display(decoded_references[sample_idx][0])

        reference_examples.append((preds_simplified, refs_simplified))

    lines = []
    lines += ["***" * 20 + " eval " + "***" * 20]
    for sample_idx, (simplified_prediction, simplified_reference) in enumerate(reference_examples):
        decoded_input = tokenizer.decode(dset["test"]["input_ids"][sample_idx])
        lines += ["---" * 5 + f" {str(sample_idx).zfill(3)} " + "---" * 5]
        lines += [
            f"{'Input:':<12} {decoded_input}\n\n",
            f"{'Predictions:':<12} {simplified_prediction}\n\n",
            f"{'References:':<12} {simplified_reference}\n\n",
            "\n\n\n",
        ]

    store_results(
        lines=lines,
        name="inference_examples",
        identifier=identifier,
        run_inside_training=run_inside_training,
    )


def store_excessive_inference(
    decoded_preds: T_PREDICTIONS,
    decoded_references: T_REFERENCES,
    dset: DatasetDict,
    split: str,
    identifier: str | None = None,
    run_inside_training: bool = True,
) -> None:
    lines = []

    flattened_refs = [ref[0] for ref in decoded_references]
    for sample_idx, (prediction, ref) in enumerate(zip(decoded_preds, flattened_refs)):
        if (predicted_blocks := prediction.count(TARGET_BLOCK_SEP_TOKEN)) > ref.count(TARGET_BLOCK_SEP_TOKEN):
            sigla = dset[split]["_SS"][sample_idx]
            lines += [
                f"{sigla} {predicted_blocks = }",
                "\n",
                f"{'Pred:':<5} {simplify_sentinel_token_display(prediction)}",
                "\n",
                f"{'Ref:':<5} {simplify_sentinel_token_display(ref)}",
                "\n\n",
            ]

    lines = lines or ["¯\\_(ツ)_/¯"]
    store_results(
        lines,
        f"{split}-excessive-inference",
        identifier=identifier,
        run_inside_training=run_inside_training,
    )


def store_all_outputs(
    split: str,
    sigla: list,
    decoded_preds: list,
    decoded_references: list,
    trimmed_decoded_preds: list,
    trimmed_decoded_references: list,
    metrics: dict,
    preds: torch.Tensor,
    references: torch.Tensor,
    identifier: str | None = None,
):
    save_dir = get_logconf().out_dir / (identifier if identifier is not None else f"manual-outputs-{split}")
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(f"Saving outputs to {save_dir = }")

    to_save_as_json = (
        (sigla, f"{split}-sigla.json"),
        (decoded_preds, f"{split}-decoded_preds.json"),
        (decoded_references, f"{split}-decoded_references.json"),
        (trimmed_decoded_preds, f"{split}-trimmed_decoded_preds.json"),
        (trimmed_decoded_references, f"{split}-trimmed_decoded_references.json"),
        (metrics, f"{split}-metrics.json"),
    )

    for obj, name in to_save_as_json:
        try:
            (save_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        except Exception:
            (save_dir / name).with_suffix(".txt").write_text(str(obj))

    to_save_as_pt = (
        (preds, f"{split}-preds.pt"),
        (references, f"{split}-references.pt"),
    )
    for obj, name in to_save_as_pt:
        torch.save(obj, str(save_dir / name))
