from contextlib import contextmanager
from typing import Any, Generator

import neptune
from loguru import logger
from neptune import Run

from kairos.config import SINGLETON


def init_neptune_run(run_id: str | None = None) -> Run:
    if run_id is not None or (config := SINGLETON.config) is not None and (neptune_run_id := config.neptune_run_id) is not None:
        run_id = run_id or neptune_run_id
        run = neptune.init_run(
            with_id=run_id,
            capture_hardware_metrics=True,
            capture_stderr=True,
            capture_stdout=True,
        )
        logger.success(f"Successfully initialized Neptune run with passed neptune id: {neptune_run_id}")
    else:
        run = neptune.init_run(
            capture_hardware_metrics=True,
            capture_stderr=True,
            capture_stdout=True,
        )
        logger.success(f"Initialized a brand new Neptune run with id: {run._sys_id}")

    SINGLETON.run = run
    assert SINGLETON.config
    SINGLETON.config.neptune_run_id = run._sys_id
    return run


def stop_ongoing_neptune_runs() -> None:
    if (run := SINGLETON.run) is not None:
        logger.warning("Stopping a previous run... 🥸")
        run.stop()


@contextmanager
def tmp_enable_neptune_logging(run_id: str) -> Generator[Run, None, None]:
    logger.info(f"Temporarily enabling Neptune logging for run with id: {run_id}")
    run = neptune.init_run(with_id=run_id)
    yield run
    logger.info(f"Stopping Neptune logging for run with id: {run_id}")
    run.stop()


def log_metrics(run: Run, metrics: dict[str, Any], split: str) -> None:
    for metric_name, metric_value in metrics.items():
        run[f"benchmark/{split}/{metric_name}"].log(metric_value)
