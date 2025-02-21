from pathlib import Path

from transformers.utils import cached_file, download_url, is_remote_url


def get_file_path(file_path: str | Path, filename: str, **kwargs) -> Path:
    """Get file path from local disk, remote URL or cache.

    This behavior mimicks the behavior of the `from_pretrained` method in the `PreTrainedTokenizer` class.
    Args:
        file_path: Local path or remote URL
        filename: Name of the file to retrieve
        **kwargs: Additional arguments passed to cached_file

    Returns:
        Path to the requested file
    """
    if isinstance(file_path, str):
        if is_remote_url(file_path):
            return Path(download_url(f"{file_path}/{filename}"))
        return Path(cached_file(file_path, filename, **kwargs))
    return Path(file_path) / filename
