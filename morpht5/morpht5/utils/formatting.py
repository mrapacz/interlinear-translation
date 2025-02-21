from typing import Sequence


def format_interlinear(
    *sequences: Sequence[str],
    separator: str = " | ",
    min_column_width: int = 4,
    justify: str = "center",
) -> str:
    """
    Format multiple sequences for interlinear display with aligned columns.

    Args:
        *sequences: Variable number of sequences to align
        separator: String to use as column separator
        min_column_width: Minimum width for each column
        justify: Justification for text in columns

    Example:
        >>> greek = ["Λέγει", "αὐτῷ", "ὁ", "Ἰησοῦς"]
        >>> english = ["Says", "to him", "-", "Jesus"]
        >>> print(format_interlinear(greek, english))
        Λέγει  | αὐτῷ   |  ὁ   | Ἰησοῦς
        Says   | to him | -    | Jesus
    """
    if not sequences:
        return ""

    # Validate all sequences have same length
    seq_len = len(sequences[0])
    if not all(len(seq) == seq_len for seq in sequences):
        raise ValueError("All sequences must have the same length")

    # Get maximum width for each column
    col_widths = []
    for col_idx in range(seq_len):
        width = max(max(len(str(seq[col_idx])) for seq in sequences), min_column_width)
        col_widths.append(width)

    match justify.lower():
        case "left":
            justify_func = str.ljust
        case "right":
            justify_func = str.rjust
        case "center" | _:
            justify_func = str.center
    # Build output rows
    rows = []
    for sequence in sequences:
        row = separator.join(justify_func(str(item), width) for item, width in zip(sequence, col_widths))
        rows.append(row)

    return "\n".join(rows)
