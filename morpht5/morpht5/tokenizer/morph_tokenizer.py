import json
from pathlib import Path
from typing import Iterable

from transformers.utils import logging

logger = logging.get_logger(__name__)


class MorphTokenizer:
    """Handles morphological tokenization with special tokens support."""

    def __init__(
        self,
        morph_encodings: dict[str, int] | None = None,
        special_tokens_map: dict[str, str] | None = None,
    ):
        self.morph_encodings = morph_encodings or {}

        self.special_tokens_map: dict[str, str] = {
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "block_separator_token": "<extra_id_0>",
        } | (special_tokens_map or {})
        self._special_token_ids = {token: idx for idx, token in enumerate(self.special_tokens_map.values())}

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids[self.special_tokens_map["pad_token"]]

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids[self.special_tokens_map["eos_token"]]

    @property
    def unk_token_id(self) -> int:
        return self._special_token_ids[self.special_tokens_map["unk_token"]]

    @property
    def block_separator_token(self) -> str:
        return self.special_tokens_map["block_separator_token"]

    @property
    def block_separator_token_id(self) -> int:
        return self._special_token_ids[self.special_tokens_map["block_separator_token"]]

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens to ids."""
        return self.morph_encodings

    @property
    def vocab_size(self) -> int:
        """Return size of the vocabulary."""
        return len(self.morph_encodings)

    def save_pretrained(
        self,
        save_directory: str | Path,
        subfolder: str = "morph_tokenizer",
    ) -> None:
        """Save tokenizer configuration to a directory."""
        config = {
            "morph_encodings": self.morph_encodings,
            "special_tokens_map": self.special_tokens_map,
        }
        save_directory = Path(save_directory)
        if subfolder:
            save_directory = save_directory / subfolder
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / "tokenizer_config.json"
        logger.info(f"morph tokenizer config saved to {save_path}")
        save_path.write_text(json.dumps(config, indent=2))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        subfolder: str = "morph_tokenizer",
        cache_dir: Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | None = None,
        revision: str = "main",
    ) -> "MorphTokenizer":
        """Load a pretrained tokenizer from a local directory or HuggingFace Hub.

        Args:
            pretrained_path: Local directory path or HuggingFace Hub model id
            subfolder: Optional subfolder within the model directory
            cache_dir: Directory to cache downloaded files
            force_download: Force download even if files exist in cache
            local_files_only: Use only local files, don't download
            token: HuggingFace Hub token for private repos
            revision: Git revision to use when downloading from Hub
        """
        from huggingface_hub import hf_hub_download

        if isinstance(pretrained_path, Path) or Path(pretrained_path).exists():
            config_path = Path(pretrained_path)
            if subfolder:
                config_path = config_path / subfolder
            config_path = config_path / "tokenizer_config.json"
            logger.info(f"Loading tokenizer config from local file: {config_path}")
            config = json.loads(config_path.read_text())
        else:
            # Assume it's a HuggingFace Hub model ID
            try:
                config_path = hf_hub_download(
                    repo_id=str(pretrained_path),
                    filename=f"{subfolder}/tokenizer_config.json" if subfolder else "tokenizer_config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    resume_download=None,
                )
                logger.info(f"Loading tokenizer config from HuggingFace Hub: {config_path}")
                config = json.loads(Path(config_path).read_text())
            except Exception as e:
                raise EnvironmentError(
                    f"Can't load tokenizer for '{pretrained_path}'. Make sure it's a valid path or HuggingFace Hub model ID."
                ) from e

        return cls(
            morph_encodings=config["morph_encodings"],
            special_tokens_map=config["special_tokens_map"],
        )

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to their corresponding ids."""
        return [self.morph_encodings.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert a list of ids to their corresponding tokens."""
        id_to_token = {v: k for k, v in self.morph_encodings.items()}
        return [id_to_token[id] for id in ids]

    def initialize_vocab(self, tags: Iterable[str]) -> None:
        """Initialize vocabulary from list of tags."""
        unique_tags = set(tags)
        self.morph_encodings = {token: idx for idx, token in enumerate(list(self._special_token_ids.keys()) + list(unique_tags))}

    def __repr__(self) -> str:
        """Return a detailed string representation of the tokenizer."""
        return f"MorphTokenizer(vocab_size={self.vocab_size}, special_tokens_map={self.special_tokens_map})"

    def __str__(self) -> str:
        """Return a concise string representation of the tokenizer."""
        return f"MorphTokenizer(vocab_size={self.vocab_size})"

    def encode(self, tags: list[str]) -> list[int]:
        """Convert tags to token ids."""
        return self.convert_tokens_to_ids(tags)

    def decode(self, ids: list[int]) -> list[str]:
        """Convert token ids back to tags."""
        return self.convert_ids_to_tokens(ids)
