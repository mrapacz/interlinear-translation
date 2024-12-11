import json
from pathlib import Path

import numpy as np
import torch
import tqdm
from datasets import Dataset, DatasetDict
from loguru import logger
from more_itertools import chunked
from transformers import AutoModelForSeq2SeqLM, EvalPrediction, T5TokenizerFast, Trainer

from kairos.config import get_config
from kairos.data.morph_tokenizer import get_morph_tokenizer
from kairos.evaluation.main import get_compute_metrics
from kairos.neptune_utils import log_metrics, tmp_enable_neptune_logging
from kairos.training.main import get_generation_max_length


def pad_to_same_length(tensors: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """Pad a list of tensors to the same length using the specified pad token."""
    correct_length = max(x.shape[0] for x in tensors)
    tensors_padded: list[torch.Tensor] = []

    for tensor in tensors:
        pads = torch.Tensor([pad_token_id] * (correct_length - len(tensor))).type(torch.int)
        tensor_with_pads = torch.concatenate([tensor, pads]) if len(pads) else tensor
        tensors_padded.append(tensor_with_pads)

    return torch.stack(tensors_padded)


def batch_inference(
    model: AutoModelForSeq2SeqLM,
    tokenizer: T5TokenizerFast,
    dset_split: Dataset,
    batch_size: int | None = None,
) -> EvalPrediction:
    """Run batched inference on a dataset."""
    model = model.to("cuda")
    batch_size = batch_size or len(dset_split)

    pad_value = {
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": tokenizer.pad_token_id,
        "labels": tokenizer.pad_token_id,
        "input_morphs": get_morph_tokenizer().pad_token_id,
    }
    input_columns = ["input_ids", "attention_mask"] + (["input_morphs"] if "input_morphs" in dset_split.column_names else [])

    batches = []
    for batch_data in chunked(dset_split, n=batch_size):
        current_batch = {}
        for column in input_columns:
            tensors = [x[column] for x in batch_data]
            current_batch[column] = pad_to_same_length(tensors, pad_token_id=pad_value[column])
        batches.append(current_batch)

    inferences: list[torch.Tensor] = []
    for batch in tqdm.tqdm(batches, desc="Inference"):
        morph_inputs = batch.get("input_morphs", torch.Tensor([])).to("cuda")
        inputs = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")

        morph_kwargs = {"input_morphs": morph_inputs} if "input_morphs" in input_columns else {}

        inference_result = model.generate(
            inputs=inputs,
            attention_mask=attention_mask,
            max_new_tokens=get_generation_max_length(),
            **morph_kwargs,
        )
        inferences.append(inference_result)

    if batch_size < len(dset_split):
        max_len = max(x.shape[-1] for x in inferences)
        logger.warning(f"Padding all tensors to the same length ({max_len = })")
        inferences = [
            torch.nn.functional.pad(tensor, pad=(tokenizer.pad_token_id, max_len - tensor.shape[-1])) for tensor in inferences
        ]

    inferences_flattened = torch.concatenate(inferences)
    inferences_padded = np.array(pad_to_same_length(inferences_flattened, pad_token_id=pad_value["labels"]).cpu())
    labels_padded = np.array(pad_to_same_length(dset_split["labels"], pad_token_id=pad_value["labels"]).cpu())

    return EvalPrediction(predictions=inferences_padded, label_ids=labels_padded)


def run_benchmarks(
    trainer: Trainer,
    dset: DatasetDict,
    batch_size: int | None = None,
    splits: list[str] = ["bench", "test", "train"],
) -> None:
    """Run benchmarks using the existing trainer and preprocessed dataset."""

    batch_size = batch_size or get_config().train_conf.eval_batch_size

    save_dir = Path(trainer.args.output_dir) / "benchmarks"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Benchmark results will be saved to {save_dir = }")

    all_metrics = {}
    for split in splits:
        save_dir_split = save_dir / split
        save_dir_split.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing benchmark metrics for {split = }...")
        compute_metrics = get_compute_metrics(
            tokenizer=trainer.tokenizer,
            dset=dset,
            split=split,
            identifier=f"benchmark-{split}",
            run_inside_training=False,
        )
        with torch.no_grad():
            eval_pred = batch_inference(
                model=trainer.model,
                tokenizer=trainer.tokenizer,
                dset_split=dset[split],
                batch_size=batch_size,
            )
        metrics = compute_metrics(eval_pred)
        (save_dir / f"{split}-metrics").with_suffix(".json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        all_metrics[split] = metrics
    (save_dir / "all_metrics").with_suffix(".json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))

    with tmp_enable_neptune_logging(run_id=get_config().neptune_run_id) as run:
        for split in splits:
            log_metrics(run=run, metrics=all_metrics[split], split=split)
    logger.success(f"Benchmark results saved to {save_dir = }")
