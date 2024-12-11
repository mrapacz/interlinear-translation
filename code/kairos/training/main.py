from collections.abc import Callable

from datasets import DatasetDict
from torch.utils.tensorboard import SummaryWriter
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration, T5TokenizerFast
from transformers.integrations import NeptuneCallback, TensorBoardCallback

from kairos.batch_size_settings import get_gradient_accumulation_steps
from kairos.config import SINGLETON, SourceType, get_config, get_logconf
from kairos.models.generation import get_generation_max_length
from kairos.training.morph_data_collator import PosEmbeddingAwareDataCollator
from kairos.training.morph_trainer import PosEmbeddingAwareTrainer
from kairos.training.morph_trainer_callback import MorphTrainerCallback


def compute_eval_steps(
    dset: DatasetDict,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    eval_frequency: int,
):
    """
    Compute the number of evaluation steps based on the training batch size and a DatasetDict containing 'train' split.

    Args:
        train_dataset_dict (DatasetDict): A DatasetDict containing a 'train' split.
        train_batch_size (int): Batch size for training.
        gradient_accumulation_steps (int): Gradient accumulation steps.
        eval_frequency (int): Desired frequency of evaluation per epoch.

    Returns:
        int: Number of evaluation steps.

    """
    train_dataset_size = len(dset["train"])

    # Compute the number of evaluation steps
    eval_steps = (train_dataset_size // (train_batch_size * gradient_accumulation_steps)) // eval_frequency

    return eval_steps


def get_training_arguments_from_conf(
    dset: DatasetDict,
) -> Seq2SeqTrainingArguments:
    trainconf = get_config().train_conf
    gradient_accumulation_steps = 1 if get_config().is_dry_run else get_gradient_accumulation_steps()
    # eval_steps = (
    #     5
    #     if get_config().is_dry_run
    #     else compute_eval_steps(
    #         dset,
    #         train_batch_size=trainconf.train_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         eval_frequency=10 if get_config().is_dry_run else trainconf.num_evals_per_epoch,
    #     )
    # )
    num_train_epochs = (
        3 if get_config().is_dry_run else trainconf.num_train_epochs
    )  # Let's have two epochs in dry runs, just in case outputs from one eval were to interfere with the next one.
    return Seq2SeqTrainingArguments(
        evaluation_strategy="epoch",
        # eval_steps=eval_steps,
        save_strategy="epoch",
        # save_steps=eval_steps,
        per_device_train_batch_size=trainconf.train_batch_size,
        per_device_eval_batch_size=trainconf.eval_batch_size,
        save_total_limit=trainconf.save_total_limit,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        generation_max_length=get_generation_max_length(),
        logging_steps=trainconf.logging_steps,
        output_dir=get_logconf().output_dir,
        logging_dir=get_logconf().log_dir,
        logging_first_step=True,
        predict_with_generate=True,
        push_to_hub=False,
        report_to="none",
        # If we don't specify remove_unused_columns, the tokenized morphological forms will be dropped
        # Can we do this cleaner?
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model=trainconf.metric_for_best_model,
        greater_is_better=trainconf.greater_is_better,
    )


def get_trainer(
    training_args: Seq2SeqTrainingArguments,
    compute_metrics: Callable,
    model: T5ForConditionalGeneration,
    dset: DatasetDict,
    tokenizer: T5TokenizerFast,
):
    sourceconf = get_config().source_conf
    morphconf = get_config().morph_conf

    base_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dset["train"],
        eval_dataset=dset["test"],
        tokenizer=tokenizer,
        data_collator=PosEmbeddingAwareDataCollator(tokenizer=tokenizer, model=model, source_type=sourceconf.source_type),
        compute_metrics=compute_metrics,
        callbacks=[
            NeptuneCallback(run=get_logconf().run),
            TensorBoardCallback(tb_writer=SummaryWriter(log_dir=get_logconf().log_dir)),
        ],
    )
    match sourceconf.source_type:
        case SourceType.TEXT_WITH_POS_EMBEDDINGS:
            assert morphconf is not None
            morph_kwargs = base_kwargs | {
                "morph_learning_rate": morphconf.morph_learning_rate,
                "callbacks": base_kwargs["callbacks"] + [MorphTrainerCallback()],
            }
            trainer = PosEmbeddingAwareTrainer(**morph_kwargs)

        case SourceType.TEXT_WITH_POS | SourceType.TEXT_ONLY:
            trainer = Seq2SeqTrainer(**base_kwargs)
        case _:
            assert False

    SINGLETON.trainer = trainer
    return trainer
