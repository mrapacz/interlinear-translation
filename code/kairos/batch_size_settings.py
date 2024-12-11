from loguru import logger

from kairos.config import get_config

logger = logger.opt(colors=True)


def get_gradient_accumulation_steps() -> int:
    train_conf = get_config().train_conf
    gradient_accumulation_steps = train_conf.virtual_batch_size // train_conf.train_batch_size
    logger.info(
        f"Setting gradient accumulation steps to <yellow>{gradient_accumulation_steps}</yellow> so that "
        f"train batch size <yellow>{train_conf.train_batch_size}</yellow> x "
        f"gradient accumulation steps <yellow>{gradient_accumulation_steps}</yellow> = "
        f"virtual batch size <yellow>{train_conf.virtual_batch_size}</yellow>"
    )
    return gradient_accumulation_steps
