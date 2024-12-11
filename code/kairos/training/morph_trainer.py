from collections.abc import Callable

import torch
from datasets import Dataset
from loguru import logger
from torch import nn
from transformers import (
    Adafactor,
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
)


class PosEmbeddingAwareTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        args: TrainingArguments = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        # Custom params
        global_learning_rate: float = 1e-3,
        morph_learning_rate: float = 1e-3,
    ):
        self.global_learning_rate = global_learning_rate
        self.morph_learning_rate = morph_learning_rate
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def create_optimizer(self):
        logger.info(
            f"Instantiating optimizer for {self.__class__.__name__} with learning rates = "
            f"{self.global_learning_rate = } {self.morph_learning_rate = }"
        )
        old_params = [value for name, value in self.model.named_parameters() if "morph" not in name]
        new_params = [value for name, value in self.model.named_parameters() if "morph" in name]

        optimizer = Adafactor(
            [{"params": old_params}, {"params": new_params, "lr": self.morph_learning_rate}],
            lr=self.global_learning_rate,
            relative_step=False,
            scale_parameter=False,
        )
        self.optimizer = optimizer
        return self.optimizer
