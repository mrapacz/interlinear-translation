import dataclasses

import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq

from kairos.config import SourceType
from kairos.data.morph_tokenizer import get_morph_tokenizer
from kairos.data.tokenize import FINAL_INPUT_MORPHS
from kairos.utils.common import pad_to_desired_length


@dataclasses.dataclass
class PosEmbeddingAwareDataCollator(DataCollatorForSeq2Seq):
    """
    DataCollatorForSeq2Seq but aware of another kind of input - input_morphs.

    This data collator is able to handle a parallel input to `input_ids`, namely `input_morphs`. The collator behaves as a regular
    `DataCollatorForSeq2Seq` would do for the standard features of the BatchEncoding. Once the padded batch is received from the
    call to the superclass, the MorphDataCollator forms an `input_morphs` tensor by padding the inputs received.
    """

    source_type: SourceType | None = None

    def __call__(self, features: BatchEncoding, return_tensors=None) -> BatchEncoding:
        if self.source_type != SourceType.TEXT_WITH_POS_EMBEDDINGS:
            return super().__call__(features=features, return_tensors=return_tensors)

        assert (
            FINAL_INPUT_MORPHS in features[0].keys()
        ), f"Expected {FINAL_INPUT_MORPHS = } to be one of the keys of the batch encoding"

        morphs: list[torch.Tensor] = [ex.pop(FINAL_INPUT_MORPHS) for ex in features]
        batch = super().__call__(features=features, return_tensors=return_tensors)

        # The tensor is of size (<batch_size>, <len inputs>)
        # We want to our input morphs to have the same shape.
        # The regular data collator does that for input_ids, but it won't do it for us in the case of pos tokens.
        input_shape = batch.data["input_ids"].shape
        padded_morphs = [
            pad_to_desired_length(t, desired_length=input_shape[-1], value=get_morph_tokenizer().pad_token_id) for t in morphs
        ]
        morph_tensor = torch.vstack(padded_morphs).int()
        assert morph_tensor.shape == input_shape, f"Shape mismatch {morph_tensor.shape = } {input_shape = }"

        batch.data["input_morphs"] = morph_tensor
        return batch
