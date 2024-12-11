from loguru import logger

from kairos.benchmark import run_benchmarks
from kairos.config import SINGLETON, Config
from kairos.data.main import load_dataset, tokenize_dataset
from kairos.data.tokenize import get_final_columns, get_tokenizer
from kairos.evaluation.main import get_compute_metrics
from kairos.models.main import get_model
from kairos.parser import parse_args_into_config
from kairos.training.main import get_trainer, get_training_arguments_from_conf

logger = logger.opt(colors=True)


def main():
    logger.info("init.")  # noqa: F823

    config: Config = parse_args_into_config()
    SINGLETON.config = config

    source_type = config.source_conf.source_type

    tokenizer = get_tokenizer(config.source_conf.checkpoint)

    dset_raw = load_dataset()
    dset_preprocessed = tokenize_dataset(
        dset_raw,
        tokenizer=tokenizer,
        source_conf=config.source_conf,
        drop_unneeded_cols=False,
    )

    del dset_raw

    dset_preprocessed.set_format("torch")

    model = get_model(tokenizer=tokenizer, source_type=source_type)
    compute_metrics = get_compute_metrics(
        tokenizer=tokenizer,
        dset=dset_preprocessed,
    )

    dset_pure_inputs = dset_preprocessed.select_columns(get_final_columns(source_type=source_type))
    training_args = get_training_arguments_from_conf(dset=dset_pure_inputs)
    trainer = get_trainer(
        training_args=training_args,
        model=model,
        tokenizer=tokenizer,
        dset=dset_pure_inputs,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    logger.info("Training finished, computing benchmarks")
    run_benchmarks(trainer=trainer, dset=dset_preprocessed)


if __name__ == "__main__":
    main()
