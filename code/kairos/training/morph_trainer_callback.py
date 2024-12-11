from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MorphTrainerCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        kwargs["model"]._debug_global_step = state.global_step

    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
