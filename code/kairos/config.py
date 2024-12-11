from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
from argparse import Namespace
from enum import Enum
from functools import lru_cache
from pathlib import Path

from neptune import Run
from transformers import Trainer


def get_experiment_name_from_metadata() -> str | None:
    try:
        return json.loads(os.getenv("KAIROS_WORKER_RAW_TASK_META"))["experiment"]
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_output_dir() -> Path:
    task_dir = os.getenv("TASK_DIR")
    if os.getenv("KAIROS_SLURM_SBATCH_RUN") and task_dir is None:
        raise ValueError("TASK_DIR is not set and running in SBATCH mode.")

    if task_dir:
        return Path(task_dir)

    assert os.environ["KAIROS_WORKSPACES_DIR"]
    out_dir = Path(os.environ["KAIROS_WORKSPACES_DIR"])
    if (conf := get_config()).run_name:
        return out_dir / Path(conf.run_name)

    return out_dir / Path(dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


SOURCE_BLOCK_SEP_TOKEN = "<extra_id_0>"
SOURCE_META_SEP_TOKEN = "<extra_id_1>"
TARGET_BLOCK_SEP_TOKEN = "<extra_id_2>"


class Sentinel(Enum):
    WORD_SEP = SOURCE_BLOCK_SEP_TOKEN
    META_SEP = SOURCE_META_SEP_TOKEN
    TARGET_SEP = TARGET_BLOCK_SEP_TOKEN


ALL_SENTINEL_TOKENS = [sent.value for sent in Sentinel]


class Checkpoint(Enum):
    MT5LARGE = "google/mt5-large"
    MT5BASE = "google/mt5-base"
    GRETA = "bowphs/GreTa"
    PHILTA = "bowphs/PhilTa"


class SourceType(Enum):
    TEXT_ONLY = "text-only"
    TEXT_WITH_POS = "text-with-pos"
    TEXT_WITH_POS_EMBEDDINGS = "text-with-pos-embeddings"


class NormalizationType(Enum):
    WITH_DIACRITICS = "diacritics"
    NORMALIZED = "normalized"


class Language(Enum):
    PL = "pl"
    EN = "en"


class TagSet(Enum):
    OBLUBIENICA = "oblubienica"
    BIBLEHUB = "biblehub"
    UNUSED = "unused"


class MorphArchitecture(Enum):
    SIMPLE_SUM = "simple-sum"
    AUTOENCODER = "autoencoder"
    CONCATENATE = "concatenate"


@dataclasses.dataclass
class TrainConf:
    # TODO: Remove, this is no longer used as the generation max length is suited to
    # specific model's efficiency for fair comparison.
    # See compute_generation_max_length below.
    generation_max_length: int = 256

    train_batch_size: int = ...  # required
    eval_batch_size: int = ...  # required
    virtual_batch_size: int = 32  # effective batch size

    optimizer: str = "adafactor"
    learning_rate: float = 1e-3

    eval_steps: int = 40
    num_evals_per_epoch: int = 2

    save_total_limit: int = 1
    metric_for_best_model = "trimmed_nosep_bleu"  # trimmed vs untrimmed made no difference
    greater_is_better = True

    logging_steps: int = 10

    num_train_epochs: int = 10
    compute_generation_max_length: bool = True
    found_truncated_dataset: bool = False
    num_train_samples: int | None = None


@dataclasses.dataclass
class MorphSpecificConf:
    compressed_embedding_size: int
    pos_embedding_dim: int
    morph_learning_rate: float = 1e-3
    arch: MorphArchitecture = MorphArchitecture.SIMPLE_SUM

    save_grads_every: int = 0
    debug_morph_embeddings_mode: bool = False


@dataclasses.dataclass
class LogConf:
    run: Run | None = None

    @property
    def run_id(self) -> str:
        assert self.run is not None
        return self.run._sys_id

    @property
    def out_dir(self) -> Path:
        assert self.run is not None

        return get_output_dir()

    @property
    def output_dir(self) -> str:
        return str(self.out_dir)

    @property
    def log_dir(self) -> str:
        return str(self.out_dir / "logs")

    @property
    def final_model_path(self) -> str:
        return str(self.out_dir / "best_model")

    @property
    def morph_vocabulary(self) -> str:
        return str(self.out_dir / "morph_vocabulary.json")

    @property
    def debug_full_dataset(self) -> str:
        return str(self.out_dir / "debug_full_dataset")

    @property
    def log_command_file(self) -> str:
        return str(self.out_dir / "full_command.txt")

    @property
    def global_log_dir(self) -> Path:
        return Path(os.getenv("KAIROS_TASKS_DIR"))

    def setup(self) -> LogConf:
        self.out_dir.mkdir(parents=True, exist_ok=True)  # this may have already been set up by the worker.
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)  # this may have already been set up by the worker.
        return self


@dataclasses.dataclass
class SourceConf:
    """Contains configs related to how the data will be presented to the model."""

    checkpoint: Checkpoint = Checkpoint.MT5BASE
    language: Language = Language.PL
    normalization: NormalizationType = NormalizationType.NORMALIZED
    source_type: SourceType = SourceType.TEXT_WITH_POS_EMBEDDINGS
    tag_set: TagSet = TagSet.BIBLEHUB
    tokenizer_max_length: int = 512
    truncate_to_most_pessimistic_block_count: bool = True


@dataclasses.dataclass
class Config:
    logconf: LogConf
    source_conf: SourceConf
    train_conf: TrainConf
    morph_conf: MorphSpecificConf | None = None

    tags: list[str] = dataclasses.field(default_factory=list)
    neptune_run_id: str | None = None
    is_dry_run: bool = False
    dry_run_sort_columns_by_longest_first: bool = True
    sort_dataset_by_longest_first: bool = False  # should already come pre-sorted
    run_name: str | None = None
    save_model_outputs: bool = True


@dataclasses.dataclass
class Singleton:
    args: Namespace | None = None
    run: Run | None = None
    config: Config | None = None
    is_dry_run: bool = False
    env: str = "prod"
    trainer: Trainer | None = None


SINGLETON = Singleton()


def get_logconf() -> LogConf:
    assert SINGLETON.config
    assert (logconf := SINGLETON.config.logconf)
    return logconf


def get_config() -> Config:
    assert SINGLETON.config
    return SINGLETON.config
