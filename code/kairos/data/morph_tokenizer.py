import json
from functools import lru_cache
from itertools import chain
from pathlib import Path

from datasets import DatasetDict

from kairos.config import SOURCE_BLOCK_SEP_TOKEN
from kairos.utils.common import dump_as_json


class CustomMorphTokenizer:
    _special_tokens_mapping = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        SOURCE_BLOCK_SEP_TOKEN: 3,
    }

    def __init__(self):
        self.encodings = None
        self.unique_tags = None

    def __str__(self):
        return f"{self.__class__.__name__}({self.special_tokens_ids_mapping}, vocab_size={self.vocabulary_size})"

    @property
    def special_tokens_ids_mapping(self) -> dict[str, int]:
        return self._special_tokens_mapping

    @property
    def pad_token_id(self):
        return self.special_tokens_ids_mapping["<pad>"]

    @property
    def eos_token_id(self):
        return self.special_tokens_ids_mapping["<eos>"]

    @property
    def unk_token_id(self):
        return self.special_tokens_ids_mapping["<unk>"]

    @property
    def block_separator_token_id(self):
        return self.special_tokens_ids_mapping[SOURCE_BLOCK_SEP_TOKEN]

    def initialize(self, dset: DatasetDict, tags_col: str) -> None:
        self.unique_tags = self.get_unique_tags(dset=dset, tags_col=tags_col)
        self.encodings = self.calculate_encodings()

    def get_unique_tags(self, dset: DatasetDict, tags_col: str) -> list[str]:
        """Calculates a set of unique tags present in the dataset."""
        return list(self.special_tokens_ids_mapping.keys()) + list(
            set(chain.from_iterable(dset["train"][tags_col] + dset["test"][tags_col]))
        )

    def calculate_encodings(self) -> dict[str, int]:
        return {morph: i_morph for i_morph, morph in enumerate(self.unique_tags)}

    def save_to_disk(self, path: Path):
        dump_as_json(
            data={
                "special_tokens": self.special_tokens_ids_mapping,
                "encodings": self.encodings,
            },
            path=path,
        )

    @classmethod
    def load_from_disk(cls, path: Path) -> "CustomMorphTokenizer":
        data = json.loads(path.read_text())
        tokenizer = cls()
        tokenizer._special_tokens_mapping = data["special_tokens"]
        tokenizer.encodings = data["encodings"]
        tokenizer.unique_tags = set(tokenizer.encodings)

        return tokenizer

    @property
    def vocabulary_size(self) -> int:
        return len(self.encodings)

    def encode(self, tags: list[str]) -> list[int]:
        return [self.encodings.get(tag, self.unk_token_id) for tag in tags]


# We need to make it a global 'singleton' so that we can pass the information about the number of unique tags to the model.
# This could probably cleaner.
@lru_cache(maxsize=1)
def get_morph_tokenizer():
    return CustomMorphTokenizer()
