from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast

from kairos.config import SINGLETON, SourceType, get_config
from kairos.models.modeling_morph_mt5 import MT5ForConditionalGeneration as MT5MorphsForConditionalGeneration


def get_model(
    tokenizer: T5TokenizerFast,
    source_type: SourceType,
) -> T5ForConditionalGeneration:
    checkpoint = get_config().source_conf.checkpoint
    match source_type:
        case SourceType.TEXT_ONLY | SourceType.TEXT_WITH_POS:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint.value)
        case SourceType.TEXT_WITH_POS_EMBEDDINGS:
            assert SINGLETON.config is not None
            assert SINGLETON.config.morph_conf is not None
            model = MT5MorphsForConditionalGeneration.from_pretrained(checkpoint.value)
        case _:
            assert False
    model.resize_token_embeddings(len(tokenizer))
    return model
