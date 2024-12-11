from functools import lru_cache

import evaluate


@lru_cache
def get_bleu():
    return evaluate.load("sacrebleu")


@lru_cache
def get_rouge():
    return evaluate.load("rouge")
