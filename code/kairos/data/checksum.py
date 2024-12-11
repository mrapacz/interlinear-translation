import hashlib
import json

from datasets import DatasetDict


def compute_dataset_checksum(dataset: DatasetDict) -> tuple[str, str]:
    train_checksum = hashlib.md5(json.dumps(list(dataset["train"]["_SS"])).encode()).hexdigest()  # noqa: S324
    test_checksum = hashlib.md5(json.dumps(list(dataset["test"]["_SS"])).encode()).hexdigest()  # noqa: S324

    return train_checksum, test_checksum
