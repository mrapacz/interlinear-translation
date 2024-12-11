import json
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from kairos.config import ALL_SENTINEL_TOKENS


def yield_token_groups(text: str, split: str, limit: int | None = None) -> Iterator[str]:
    """
    Yields groups of tokens from a string, split by a given delimiter.

    Args:
        text (str): The string to split into token groups.
        split (str): The delimiter to use for splitting the string.
        limit (int, optional): The maximum number of token groups to yield. If None,
            all token groups are yielded. Defaults to None.

    Yields:
        str: Each token group, as a string stripped of leading and trailing whitespace.

    Examples:
        >>> list(yield_token_groups("foo,bar,baz", ","))
        ['foo', 'bar', 'baz']
        >>> list(yield_token_groups(" foo | bar | baz | qux ", "|", limit=2))
        ['foo', 'bar']

    """
    token_groups = (x.strip() for x in text.split(split))
    return islice(token_groups, limit)


def simplify_sentinel_token_display(x: str) -> str:
    """
    Drops unnecessarily long <special-token> ids for better display.

    >>> simplify_sentinel_token_display(f"Hello {ALL_SENTINEL_TOKENS[0]} World")
    'Hello <0> World'
    """
    for idx, token in enumerate(ALL_SENTINEL_TOKENS):
        x = x.replace(token, f"<{idx}>")
    return x


def simplify_decoded_text(text: str, split: str, limit: int | None) -> str:
    """Drops all special tokens from given text."""
    return " ".join(yield_token_groups(text, split=split, limit=limit)).strip()


def get_num_tokens_until_value(arr: np.ndarray, value: int) -> np.ndarray:
    """
    Returns the number of tokens in a numpy array until the first occurrence of a given value.

    Args:
        arr (np.ndarray): The input numpy array.
        value (int): The value to search for in the array.

    Returns:
        np.ndarray: A numpy array of the same shape as the input array, where each element is
        the number of tokens in the input array until the first occurrence of the given value.
        If the given value is not found in an element of the input array, the corresponding
        element in the output array is set to the length of the array.

    Examples:
        >>> arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 11]])
        >>> get_num_tokens_until_value(arr, 1)
        array([0, 4, 2])

    """
    first_occurrence_indices = np.argmax(arr == value, axis=-1)
    lengths_until_first_occurrence = np.where(
        (first_occurrence_indices == 0) & (arr[..., 0] != value), np.shape(arr)[-1], first_occurrence_indices
    )
    return lengths_until_first_occurrence


def split_array_to_blocks(arr: np.ndarray, block_separator: int) -> list[np.ndarray]:
    """
    Splits a given np.ndarray to a list of arrays.

    Each subarray is one block. Note: in case the sequence ends with a separator, the last block might be empty.

    Examples:
    1. Three blocks + an empty one at the end.
    >>> arr = np.array([ 20741, 250099,  10453, 250099,    850, 121178,   2728, 250099])
    >>> split_array_to_blocks(arr, block_separator=250099)
    [array([ 20741, 250099]), array([ 10453, 250099]), array([   850, 121178,   2728, 250099])]

    2. Four blocks, no separator at the end.
    >>> arr = np.array([ 20741, 250099,  10453, 250099,    850, 121178,   2728, 250099, 259,   2616])
    >>> split_array_to_blocks(arr, block_separator=250099)
    [array([ 20741, 250099]), array([ 10453, 250099]), array([   850, 121178,   2728, 250099]), array([ 259, 2616])]

    """
    # np.where will just give us the indices of where the block separators are.
    # We need to bump the numbers by +1 to make sure we're also catching the separators within each group upon the split
    assert isinstance(arr, np.ndarray), f"Expected {np.ndarray}, got {type(arr)}"
    assert isinstance(block_separator, int)
    block_ends_indices = np.where(arr == block_separator)[0] + 1
    return [block for block in np.split(arr, block_ends_indices) if len(block) != 0]


def get_number_of_blocks(arr: np.ndarray, block_separator: int) -> int:
    """
    Get the number of blocks in array.

    The last, incomplete block should still count as one.
    >>> get_number_of_blocks(arr=np.array([1, 0, 2, 0]), block_separator=0)
    2
    >>> get_number_of_blocks(arr=np.array([1, 0, 2]), block_separator=0)
    2
    >>> get_number_of_blocks(arr=np.array([]), block_separator=0)
    0
    >>> get_number_of_blocks(arr=np.array([0, 0, 0]), block_separator=0)
    3
    >>> get_number_of_blocks(arr=np.array([1, 0, 0]), block_separator=0)
    2
    """
    assert isinstance(arr, np.ndarray), f"Expected {np.ndarray}, got {type(arr)}"

    blocks = split_array_to_blocks(arr, block_separator)
    return len(blocks)


def trim_to_number_of_blocks(arr: np.ndarray, num_blocks: int, block_separator: int) -> np.ndarray:
    """
    Trims a given numpy array to only have the specified number of blocks.

    In the following case we have three complete blocks, we only request 2, though:
    >>> trim_to_number_of_blocks(arr=np.array([1, 0, 2, 0, 3, 0]), num_blocks=2, block_separator=0)
    array([1, 0, 2, 0])

    The result should be the same even if the second block is incomplete (except for the separator):
    >>> trim_to_number_of_blocks(arr=np.array([1, 0, 2]), num_blocks=2, block_separator=0)
    array([1, 0, 2])

    Here we request more blocks than available:
    >>> trim_to_number_of_blocks(arr=np.array([1, 0, 2, 0, 3, 0]), num_blocks=4, block_separator=0)
    array([1, 0, 2, 0, 3, 0])
    """
    assert isinstance(arr, np.ndarray), f"Expected {np.ndarray}, got {type(arr)}"

    blocks = split_array_to_blocks(arr, block_separator=block_separator)
    return np.concatenate(blocks[:num_blocks])


def trim_arrays_to_have_the_same_number_of_blocks(
    arr: np.ndarray,
    reference: np.ndarray,
    block_separator: int,
    trimming_strategy: Literal["reference", "minimum"] = "minimum",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split two arrays into subgroups by a specified value, then trim the number of groups to the smaller one.

    >>> arr = np.array([1, 42,  2, 42, 3, 42, 4, 42, 42])
    >>> ref = np.array([1, 42, 2, 42, 3, 42])
    >>> trim_arrays_to_have_the_same_number_of_blocks(arr, ref, block_separator=42)
    (array([ 1, 42,  2, 42,  3, 42]), array([ 1, 42,  2, 42,  3, 42]))

    >>> arr = np.array([1, 42,  2, 42, 3, 42, 4, 42])
    >>> ref = np.array([1, 42, 2, 42, 3, 42, 4])
    >>> trim_arrays_to_have_the_same_number_of_blocks(arr, ref, block_separator=42)
    (array([ 1, 42,  2, 42,  3, 42,  4, 42]), array([ 1, 42,  2, 42,  3, 42,  4]))

    >>> arr = np.array([11, 12, 13, 42,  21, 22, 42,  31, 32, 42, 4, 4, 42, 5, 42, 6])
    >>> ref = np.array([1, 42,  2, 42, 3, 3, 42, 4, 4, 42, 5])
    >>> trim_arrays_to_have_the_same_number_of_blocks(arr, ref, block_separator=42)
    (array([11, 12, 13, 42, 21, 22, 42, 31, 32, 42,  4,  4, 42,  5, 42]), array([ 1, 42,  2, 42,  3,  3, 42,  4,  4, 42,  5]))
    """
    assert isinstance(arr, np.ndarray), f"Expected {np.ndarray}, got {type(arr)}"
    assert isinstance(reference, np.ndarray)

    match trimming_strategy:
        case "reference":
            desired_num_blocks = get_number_of_blocks(reference, block_separator=block_separator)
        case "minimum":
            desired_num_blocks = min(
                get_number_of_blocks(arr, block_separator=block_separator),
                get_number_of_blocks(reference, block_separator=block_separator),
            )
        case _:
            raise ValueError(f"Unknown trimming strategy: {trimming_strategy}")

    return (
        trim_to_number_of_blocks(arr, num_blocks=desired_num_blocks, block_separator=block_separator),
        trim_to_number_of_blocks(reference, num_blocks=desired_num_blocks, block_separator=block_separator),
    )


def print_aligned_vertically(*lists):
    """
    Prints a variable number of lists of strings aligned vertically.

    .>>> a = [('Tymoteuszowi', 'n_ Dat Sg m', 'τιμοθεω'),
    ...      ('umiłowanemu', 'a_ Dat Sg m', 'αγαπητω'),
    ...      ('dziecku', 'n_ Dat Sg n', 'τεκνω'),
    ...      ('łaska', 'n_ Nom Sg f', 'χαρις'),
    ...      ('miłosierdzie', 'n_ Nom Sg m', 'ελεος')]
    .>>> print_aligned_vertically()
    Tymoteuszowi | n_ Dat Sg m  |   τιμοθεω
    umiłowanemu  | a_ Dat Sg m  |   αγαπητω
      dziecku    | n_ Dat Sg n  |    τεκνω
       łaska     | n_ Nom Sg f  |    χαρις
    miłosierdzie | n_ Nom Sg m  |    ελεος
    """
    # Determine the maximum length of each element across all lists
    lengths = [max(len(str(item)) for item in sublist) for sublist in lists]

    # Join each item in each list with the appropriate spacing
    rows = [" | ".join(str(item).center(length) for item, length in zip(row, lengths)) for row in zip(*lists)]

    # Print the aligned rows
    for row in rows:
        print(row)


def pad_to_desired_length(t: torch.Tensor, desired_length: int, value: int) -> torch.Tensor:
    """
    Pad a 1-dimensional PyTorch tensor to the desired length with a specified value.

    Examples:
    >>> t = torch.tensor([1, 2, 3])
    >>> pad_to_desired_length(t, 5, 0)
    tensor([1, 2, 3, 0, 0])

    >>> t = torch.tensor([1, 2, 3, 4, 5])
    >>> pad_to_desired_length(t, 6, 0)
    tensor([1, 2, 3, 4, 5, 0])

    >>> t = torch.tensor([1, 2, 3])
    >>> pad_to_desired_length(t, 5, 0)
    tensor([1, 2, 3, 0, 0])

    """
    assert len(t.shape) == 1, f"Expected the tensor to be 1-dimensional, got {t.shape} shape instead."
    padding_length = desired_length - t.shape[0]
    padding_values = torch.full((padding_length,), value, dtype=t.dtype, device=t.device)
    return torch.cat([t, padding_values])


def dump_as_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
