"""
This module deals with generation-related logic.
"""

from kairos.config import Checkpoint, Language, get_config


def calculate_generation_max_length(checkpoint: Checkpoint, language: Language) -> int:
    """
    Calculate the generation_max_length parameter based on the checkpoint and language.

    We're setting the values to the maximum number of tokens in the target language for the whole dataset.
    The following table was used to calculate the generation_max_length parameter.

    +-----------------+----------+----------------+
    |    Checkpoint   | Language | Max Num Tokens |
    +-----------------+----------+----------------+
    |  bowphs/PhilTa  |    pl    |      331       |
    |  bowphs/PhilTa  |    en    |      212       |
    | google/mt5-base |    pl    |      163       |
    | google/mt5-base |    en    |      149       |
    |   bowphs/GreTa  |    pl    |      309       |
    |   bowphs/GreTa  |    en    |      301       |
    +-----------------+----------+----------------+
    """
    return {
        Checkpoint.MT5BASE: {
            Language.PL: 163,
            Language.EN: 149,
        },
        Checkpoint.MT5LARGE: {
            Language.PL: 163,
            Language.EN: 149,
        },
        Checkpoint.PHILTA: {
            Language.PL: 331,
            Language.EN: 212,
        },
        Checkpoint.GRETA: {
            Language.PL: 309,
            Language.EN: 301,
        },
    }[checkpoint][language]


def get_generation_max_length() -> int:
    if get_config().train_conf.compute_generation_max_length:
        return calculate_generation_max_length(
            get_config().source_conf.checkpoint,
            get_config().source_conf.language,
        )
    return get_config().train_conf.generation_max_length
