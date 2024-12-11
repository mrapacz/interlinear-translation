import argparse
import os
from enum import Enum
from functools import lru_cache
from typing import Any

from loguru import logger

from kairos.config import (
    SINGLETON,
    Checkpoint,
    Config,
    Language,
    LogConf,
    MorphArchitecture,
    MorphSpecificConf,
    NormalizationType,
    SourceConf,
    SourceType,
    TagSet,
    TrainConf,
)


def get_enum_values_for_parser(enum_cls: type[Enum], lowercase: bool = True) -> list[str]:
    """Get lowercase string values of an enum class."""
    return [enum.value.lower() if lowercase else enum.value for enum in enum_cls]


def parse_values_into_enum(enum_cls: type[Enum], value: str) -> Any:
    """Parse a string value into the corresponding enum value, considering casing."""
    try:
        return next(e for e in enum_cls if e.value.lower() == value.lower())
    except StopIteration:
        raise ValueError(f"'{value}' is not a valid enum value")


parser = argparse.ArgumentParser(description="Arguments for Seq2SeqTrainingArguments and Seq2SeqTrainer", exit_on_error=False)

custom = parser.add_argument_group("custom - code specific")

custom.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Name of the run",
)
custom.add_argument(
    "--tags",
    type=str,
    default=[],
    help="neptune tags",
    nargs="+",
)
custom.add_argument(
    "--neptune_run_id",
    type=str,
    default=None,
    required=False,
    help="Neptune run id if available",
)
custom.add_argument(
    "--dry_run",
    action="store_true",
    help="If passed, we're only going to use 1 batch of data and the model will train for 1 epoch",
)
custom.add_argument(
    "--source_type",
    choices=get_enum_values_for_parser(SourceType),
    type=str,
    # default="morph",
    default=SourceConf.source_type.value,
    help="source translation format",
)

custom.add_argument(
    "--language",
    choices=get_enum_values_for_parser(Language),
    type=str,
    default=SourceConf.language.value,
    help="source translation format",
)

custom.add_argument(
    "--normalization",
    choices=get_enum_values_for_parser(NormalizationType),
    type=str,
    default=SourceConf.normalization.value,
)

custom.add_argument(
    "--tagset",
    choices=get_enum_values_for_parser(TagSet),
    type=str,
    default=TagSet.UNUSED.value,
)

custom.add_argument(
    "--skip_truncation",
    action="store_true",
    help="If this flag is set, we will not truncate the inputs to match the most pessimistic block count",
)

parser.add_argument("--morph_learning_rate", type=float, default=MorphSpecificConf.morph_learning_rate)
parser.add_argument(
    "--morph_architecture",
    type=str,
    choices=get_enum_values_for_parser(MorphArchitecture),
    default=MorphSpecificConf.arch.value,
)
parser.add_argument(
    "--compressed_embedding_size",
    type=int,
    default=64,
)
parser.add_argument("--pos_embedding_dim", type=int, default=64)

parser.add_argument(
    "--num_train_samples",
    type=int,
    default=None,
    help="Number of train samples to use",
)

custom.add_argument(
    "--checkpoint",
    type=str,
    choices=get_enum_values_for_parser(Checkpoint, lowercase=False),
    default=SourceConf.checkpoint.value,
)


tokenizer_parser = parser.add_argument_group("Tokenizer")
tokenizer_parser.add_argument(
    "--tokenizer_max_length",
    type=int,
    default=SourceConf.tokenizer_max_length,
    help="Max tokenized length",
)

debug = parser.add_argument_group("Debug")
debug.add_argument(
    "--save_grads",
    action="store_true",
    help="debug: to save gradients on morph layer during training. This applies only to the morph_embeddings strategy.",
)
debug.add_argument(
    "--save_grads_every",
    help="debug: how often to save grads",
    type=int,
    default=None,
)

train_parser = parser.add_argument_group("Training args")

train_parser.add_argument("--eval_steps", type=int, default=TrainConf.eval_steps, help="evaluation steps")
train_parser.add_argument(
    "--train_batch_size",
    type=int,
    default=TrainConf.train_batch_size,
    help="train batch size per device",
)

train_parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=TrainConf.eval_batch_size,
    help="train batch size per device",
)


train_parser.add_argument(
    "--save_total_limit",
    type=int,
    default=TrainConf.save_total_limit,
    help="total number of checkpoints to save",
)
train_parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=TrainConf.num_train_epochs,
    help="number of training epochs",
)
train_parser.add_argument("--logging_steps", type=int, default=TrainConf.logging_steps, help="logging frequency")
train_parser.add_argument("--optim", type=str, default=TrainConf.optimizer, help="optimizer to use")
train_parser.add_argument("--learning_rate", type=float, default=TrainConf.learning_rate, help="learning rate")
train_parser.add_argument(
    "--generation_max_length",
    type=int,
    default=TrainConf.generation_max_length,
    help="The `max_length` to use on each evaluation loop when `predict_with_generate=True`",
)


def parse_args() -> argparse.Namespace:
    args = parser.parse_args()
    SINGLETON.args = args
    return args


@lru_cache(maxsize=1)
def parse_args_into_config() -> Config:
    args = parse_args()

    if args.tagset == TagSet.UNUSED.value and args.source_type != SourceType.TEXT_ONLY.value:
        raise ValueError("Tagset UNUSED can only be used with source type TEXT_ONLY")

    source_config = SourceConf(
        checkpoint=Checkpoint(args.checkpoint),
        language=parse_values_into_enum(Language, args.language),
        normalization=parse_values_into_enum(NormalizationType, args.normalization),
        source_type=parse_values_into_enum(SourceType, args.source_type),
        tag_set=parse_values_into_enum(TagSet, args.tagset),
        truncate_to_most_pessimistic_block_count=not args.skip_truncation,
        tokenizer_max_length=args.tokenizer_max_length,
    )
    logging_config = LogConf()

    training_config = TrainConf(
        eval_steps=args.eval_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        generation_max_length=args.generation_max_length,
        num_train_samples=args.num_train_samples,
    )
    if source_config.source_type == SourceType.TEXT_WITH_POS_EMBEDDINGS:
        morph_config = MorphSpecificConf(
            debug_morph_embeddings_mode=args.save_grads,
            save_grads_every=args.save_grads_every,
            arch=parse_values_into_enum(MorphArchitecture, args.morph_architecture),
            morph_learning_rate=args.morph_learning_rate,
            compressed_embedding_size=args.compressed_embedding_size,
            pos_embedding_dim=args.pos_embedding_dim,
        )
    else:
        morph_config = None

    SINGLETON.args = args
    SINGLETON.is_dry_run = args.dry_run
    if SINGLETON.is_dry_run:
        logger.warning("Running in the DRY RUN mode.")

    neptune_run_id = args.neptune_run_id or os.getenv("KAIROS_NEPTUNE_RUN_ID")
    return Config(
        source_conf=source_config,
        logconf=logging_config,
        train_conf=training_config,
        morph_conf=morph_config,
        tags=args.tags,
        neptune_run_id=neptune_run_id,
        is_dry_run=SINGLETON.is_dry_run,
        run_name=args.run_name,
    )
