import os
from pathlib import Path

from datasets import DatasetDict, disable_caching
from datasets.utils.logging import disable_progress_bar
from loguru import logger
from transformers import T5TokenizerFast

from kairos.config import SINGLETON, SourceConf, get_config
from kairos.data.formatters import format_columns
from kairos.data.tokenize import get_final_columns, tokenize

disable_progress_bar()
disable_caching()


def get_project_root() -> Path:
    project_dir = [p for p in Path(__file__).parents[2:] if p.parts[-1] == "kairos"][0]
    assert project_dir.exists()
    return project_dir


def choose_dataset_from_dir(dset_dir: Path) -> Path:
    tokenizer_max_length = get_config().source_conf.tokenizer_max_length
    if tokenizer_max_length is not None:
        dset_name = f"dset_{tokenizer_max_length}"
        if (dset_path := dset_dir / dset_name).exists():
            get_config().train_conf.found_truncated_dataset = True
            return dset_path

    dset_name = "dset_with_benchmarks"
    if (dset_path := dset_dir / dset_name).exists():
        return dset_path

    raise FileNotFoundError(f"No dataset found in {dset_dir}")


def get_dataset_path() -> Path:
    if (dset_path := os.getenv("META_DATASET_DIR")) is not None:
        assert Path(dset_path).exists()
        return Path(dset_path)

    if (dset_dir := Path(os.getenv("KAIROS_DATASETS_DIR", "--MISSING--"))).exists():
        return choose_dataset_from_dir(dset_dir)

    if (dset_dir := get_project_root() / "data").exists():
        return choose_dataset_from_dir(dset_dir)

    assert False


def adjust_dataset_for_dry_run(dset: DatasetDict) -> DatasetDict:
    logger.info("Sampling dataset for dry run - 10x train batches, 2x validation batches")
    conf = SINGLETON.config
    dset["train"] = dset["train"].select(range(conf.train_conf.train_batch_size * 2))
    dset["test"] = dset["test"].select(range(conf.train_conf.eval_batch_size * 2))
    dset["bench"] = dset["bench"].select(range(conf.train_conf.eval_batch_size * 2))
    return dset


def load_dataset() -> DatasetDict:
    dset_path = get_dataset_path()
    dset = DatasetDict.load_from_disk(dset_path)
    logger.success(f"Found dataset at {dset_path =} {dset =}")

    assert (conf := SINGLETON.config) is not None

    num_train_samples = conf.train_conf.num_train_samples
    if num_train_samples is not None:
        logger.info(
            f"Selecting only the first {num_train_samples} train samples "
            f"({num_train_samples / len(dset['train']):.2%} of the train split"
        )
        dset["train"] = dset["train"].select(range(num_train_samples))

    if conf.is_dry_run:
        dset = adjust_dataset_for_dry_run(dset)

    logger.info(f"Post-adjustment dataset: {dset =}")
    return dset


def tokenize_dataset(
    dset_raw: DatasetDict,
    tokenizer: T5TokenizerFast,
    source_conf: SourceConf,
    drop_unneeded_cols: bool = True,
) -> DatasetDict:
    # choose, rename and format the right columns
    dset = format_columns(
        dset_raw, language=source_conf.language, normalization=source_conf.normalization, tagset=source_conf.tag_set
    )

    # tokenize
    dset_tokenized = tokenize(
        dset=dset,
        source_type=source_conf.source_type,
        tokenizer=tokenizer,
        tokenizer_max_length=source_conf.tokenizer_max_length,
    )
    dset_tokenized.set_format("numpy")

    if drop_unneeded_cols:
        final_cols = get_final_columns(source_type=source_conf.source_type)
        return dset_tokenized.select_columns(final_cols + ["_SS"])

    return dset_tokenized
