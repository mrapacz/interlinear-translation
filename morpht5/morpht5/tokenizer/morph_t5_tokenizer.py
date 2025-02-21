from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizer, T5TokenizerFast
from transformers.tokenization_utils import PaddingStrategy

from .morph_tokenizer import MorphTokenizer


class MorphT5Tokenizer(PreTrainedTokenizer):
    """
    MorphT5 Tokenizer for encoding text and morphological tags.

    This tokenizer combines a T5Tokenizer for text tokenization with a MorphTokenizer for morphological tokenization,
    ensuring alignment between text inputs and morphological tags.

    Example:
        ```python
        from transformers import AutoTokenizer

        # Initialize from base T5 tokenizer
        tokenizer = MorphT5Tokenizer.from_base_tokenizer("google/mt5-base")

        # Initialize morphological vocabulary
        tokenizer.initialize_morph_vocab(dataset, tags_column="morph_tags")

        # Or load a complete pretrained MorphT5Tokenizer
        tokenizer = MorphT5Tokenizer.from_pretrained("mrapacz/morph-t5-model")
        ```
    """

    text_tokenizer: T5TokenizerFast
    morph_tokenizer: MorphTokenizer
    model_input_names = ["input_ids", "attention_mask", "input_morphs"]

    def __init__(
        self,
        text_tokenizer: T5TokenizerFast,
        morph_tokenizer: MorphTokenizer | None = None,
        **kwargs,
    ):
        """Initialize tokenizer with text tokenizer and optional morph tokenizer."""
        self.text_tokenizer = text_tokenizer
        self.morph_tokenizer = morph_tokenizer or MorphTokenizer()
        # Copy attributes from text tokenizer
        super().__init__(**kwargs)

        self.pad_token = self.text_tokenizer.pad_token
        self.eos_token = self.text_tokenizer.eos_token

    @classmethod
    def from_base_tokenizer(
        cls,
        base_tokenizer_path: str | Path,
    ) -> "MorphT5Tokenizer":
        """Initialize from a base T5 tokenizer without morphological capabilities."""
        text_tokenizer = T5TokenizerFast.from_pretrained(base_tokenizer_path)
        return cls(text_tokenizer=text_tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "MorphT5Tokenizer":
        """Load both text and morphological tokenizers from a pretrained model."""
        path = Path(pretrained_model_name_or_path)

        text_tokenizer = T5TokenizerFast.from_pretrained(
            path,
            subfolder="text_tokenizer",
        )

        morph_tokenizer = MorphTokenizer.from_pretrained(
            path,
            subfolder="morph_tokenizer",
        )

        return cls(text_tokenizer=text_tokenizer, morph_tokenizer=morph_tokenizer, **kwargs)

    def save_pretrained(
        self,
        save_directory: str | Path,
        **kwargs,
    ) -> None:
        """Save both text and morphological tokenizers."""
        save_directory = Path(save_directory)
        self.text_tokenizer.save_pretrained(save_directory / "text_tokenizer")
        self.morph_tokenizer.save_pretrained(save_directory, subfolder="morph_tokenizer")

    def __repr__(self) -> str:
        """Return a detailed string representation of the tokenizer."""
        text_vocab_size = len(self.text_tokenizer.get_vocab())
        morph_vocab_size = self.morph_tokenizer.vocab_size

        return (
            f"MorphT5Tokenizer(text_vocab_size={text_vocab_size}, "
            f"morph_vocab_size={morph_vocab_size}, "
            f"pad_token='{self.pad_token}', "
            f"eos_token='{self.eos_token}', "
            f"block_separator='{self.morph_tokenizer.block_separator_token}')"
        )

    def __str__(self) -> str:
        """Return a concise string representation of the tokenizer."""
        return (
            f"MorphT5Tokenizer(text_vocab={len(self.text_tokenizer.get_vocab())}, morph_vocab={self.morph_tokenizer.vocab_size})"
        )

    def initialize_morph_vocab(
        self,
        dset: "Dataset",
        tags_col: str,
    ) -> None:
        self.morph_tokenizer.initialize_vocab_from_dataset(dset=dset, tags_col=tags_col)

    def __call__(
        self,
        text: list[str] | list[list[str]],
        text_target: list[str] | None = None,
        morph_tags: list[list[str]] | None = None,
        padding: bool | str | PaddingStrategy = True,
        truncation: bool = True,
        max_length: int | None = 512,
        return_tensors: str | None = None,
        **kwargs,
    ):
        """
        Process text and morphological tags.

        Args:
            text: List of text blocks for each example or list of lists for batched input
            text_target: Optional target text
            morph_tags: List of morphological tags corresponding to text blocks
            padding: Padding strategy
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return format for tensors
            **kwargs: Additional arguments
        """
        # Get block separator token
        block_sep = self.morph_tokenizer.block_separator_token

        # Format text with block separators
        if text and isinstance(text[0], str):
            formatted_text = [f" {block_sep} ".join(text)]
        else:
            formatted_text = [f" {block_sep} ".join(example) for example in text]

        encoding = self.text_tokenizer(
            formatted_text,
            text_target=text_target,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        if morph_tags is not None:
            # Ensure morph_tags is a list of lists for batch processing
            if morph_tags and isinstance(morph_tags[0], str):
                morph_tags = [morph_tags]

            morph_ids = [self.morph_tokenizer.encode(tags) for tags in morph_tags]
            block_sep_id = self.text_tokenizer.convert_tokens_to_ids("<extra_id_0>")

            all_morph_arrays = []
            for batch_idx, (tag_ids, input_ids) in enumerate(zip(morph_ids, encoding["input_ids"])):
                text_ids = np.array(input_ids)
                text_blocks = np.split(text_ids, np.where(text_ids == block_sep_id)[0])

                morph_array = []
                for tag_id, text_block in zip(tag_ids, text_blocks):
                    morph_array.extend([tag_id] * len(text_block))

                morph_array = np.array(morph_array)
                morph_array[text_ids == block_sep_id] = self.morph_tokenizer.block_separator_token_id
                morph_array[text_ids == self.text_tokenizer.eos_token_id] = self.morph_tokenizer.eos_token_id
                morph_array[text_ids == self.text_tokenizer.pad_token_id] = self.morph_tokenizer.pad_token_id
                morph_array[text_ids == self.text_tokenizer.unk_token_id] = self.morph_tokenizer.unk_token_id

                all_morph_arrays.append(morph_array)

            encoding["input_morphs"] = all_morph_arrays

            if return_tensors == "pt":
                import torch

                encoding["input_morphs"] = torch.tensor(encoding["input_morphs"])

        return encoding

    def decode(
        self,
        input_ids: list[int],
        skip_special_tokens: bool = True,
        keep_block_separator: bool = False,
    ) -> str:
        """Decode input IDs back to text."""

        if skip_special_tokens and keep_block_separator:
            decoded = self.text_tokenizer.decode(input_ids, skip_special_tokens=False)
            special_tokens = {
                self.text_tokenizer.eos_token,
                self.text_tokenizer.pad_token,
                self.text_tokenizer.unk_token,
            }
            decoded = self.text_tokenizer.decode(input_ids, skip_special_tokens=False)
            for token in special_tokens:
                decoded = decoded.replace(token, "")
            return decoded.strip()

        decoded = self.text_tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)
        return decoded

    @property
    def target_block_separator_token(self) -> str:
        return "<extra_id_2>"

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary mapping tokens to ids."""
        return self.text_tokenizer.get_vocab()
