from loguru import logger
from transformers import T5TokenizerFast

from kairos.config import ALL_SENTINEL_TOKENS


def get_sentinel_token_id(tokenizer: T5TokenizerFast, token: str) -> int:
    match tokenizer.encode(token, add_special_tokens=False):
        case [token_id]:
            return token_id
        case something_else:
            raise ValueError(
                f"Each sentinel token should be encoded as exactly one id. Got {len(something_else)}: {something_else}"
            )


def _add_new_sentinel_token(tokenizer: T5TokenizerFast, token: str) -> None:
    tokenizer.add_tokens([token], special_tokens=True)
    try:
        tokenizer.vocab_size = len(tokenizer)
    except AttributeError:
        logger.warning(f"Could not set the .vocab_size attribute on {tokenizer.name_or_path = } {tokenizer.vocab_size = }")
    tokenizer(" ".join(ALL_SENTINEL_TOKENS), add_special_tokens=False)


def ensure_sentinel_tokens_are_in_place(tokenizer: T5TokenizerFast) -> None:
    special_token_ids = []
    for token in ALL_SENTINEL_TOKENS:
        if token not in tokenizer.vocab:
            logger.warning(f"{token = } not found in vocabulary of, adding it manually...")
            _add_new_sentinel_token(tokenizer, token)
        special_token_ids.append(get_sentinel_token_id(tokenizer, token))
    assert len(set(special_token_ids)) == len(ALL_SENTINEL_TOKENS)
    for token in ALL_SENTINEL_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Token {token} is encoded into {len(token_ids)} ids: {token_ids}. "
                "Each sentinel token must be encoded as exactly one id."
            )
