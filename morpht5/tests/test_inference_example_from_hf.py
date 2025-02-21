from morpht5 import MorphT5AutoForConditionalGeneration, MorphT5Tokenizer


def test_correctly_tokenizes_interlinear_text():
    text = ["Λέγει", "αὐτῷ", "ὁ", "Ἰησοῦς", "Ἔγειρε", "ἆρον", "τὸν", "κράβαττόν", "σου", "καὶ", "περιπάτει"]
    tags = [
        "V-PIA-3S",
        "PPro-DM3S",
        "Art-NMS",
        "N-NMS",
        "V-PMA-2S",
        "V-AMA-2S",
        "Art-AMS",
        "N-AMS",
        "PPro-G2S",
        "Conj",
        "V-PMA-2S",
    ]

    tokenizer = MorphT5Tokenizer.from_pretrained("mrapacz/interlinear-en-philta-emb-auto-diacritics-bh")
    inputs = tokenizer(text=text, morph_tags=tags, return_tensors="pt")

    model = MorphT5AutoForConditionalGeneration.from_pretrained("mrapacz/interlinear-en-philta-emb-auto-diacritics-bh")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        early_stopping=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True, keep_block_separator=True)
    decoded = decoded.replace(tokenizer.target_block_separator_token, " | ")
    assert decoded == "says  |  to him  |  -  |  jesus  |  arise  |  take up  |  the  |  mat  |  of you  |  and  |  walk"
