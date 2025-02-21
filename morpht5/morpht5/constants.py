from enum import Enum


class SentinelToken(Enum):
    """Sentinel tokens used to separate words, morphological tags, and targets in the source and target corpora."""

    SOURCE = "<extra_id_0>"
    TAG = "<extra_id_1>"
    TARGET = "<extra_id_2>"
