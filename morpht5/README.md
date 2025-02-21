# morpht5

This package contains the source code for the MorphT5 models used in the [Low Resource Interlinear Translation](https://github.com/mrapacz/loreslm-interlinear-translation) paper.

## Installation

```bash
pip install morpht5
```

## Components

### Models

The package includes three model variants, incorporating morphological features through dedicated embedding layers:
- `MorphT5SumModel`: Model using positional summation for combining text and morphological tag embeddings
- `MorphT5AutoModel`: Base model with autoencoder-style morphological tag embeddings
- `MorphT5ConcatModel`: Model using concatenation for combining text and morphological tag embeddings


### Tokenizer

The package comes with a tokenizer that allows for encoding text and morphological tags into tokens:

```python
>>> from morpht5 import MorphT5Tokenizer
>>> text = ["Λέγει", "αὐτῷ", "ὁ", "Ἰησοῦς", "Ἔγειρε", "ἆρον", "τὸν", "κράβαττόν", "σου", "καὶ", "περιπάτει"]
>>> tags = [
...     "V-PIA-3S",
...     "PPro-DM3S",
...     "Art-NMS",
...     "N-NMS",
...     "V-PMA-2S",
...     "V-AMA-2S",
...     "Art-AMS",
...     "N-AMS",
...     "PPro-G2S",
...     "Conj",
...     "V-PMA-2S",
... ]
>>> tokenizer = MorphT5Tokenizer.from_pretrained("mrapacz/interlinear-en-philta-emb-auto-diacritics-bh")
>>> inputs = tokenizer(text=text, morph_tags=tags, return_tensors="pt")
>>> inputs.keys()
dict_keys(['input_ids', 'attention_mask', 'input_morphs'])
```

### Tagsets

The package comes with Enum classes for two morphological tagsets - one compiled from BibleHub and one compiled from Oblubienica:

```python
from morpht5 import BibleHubTag, OblubienicaTag
```

Both tagsets cover comprehensive morphological features:
- Parts of Speech (Verb, Noun, Adjective, etc.)
- Person (1st, 2nd, 3rd)
- Tense (Present, Imperfect, Future, Aorist, Perfect, Pluperfect)
- Mood (Indicative, Imperative, Subjunctive, Optative, Infinitive, Participle)
- Voice (Active, Middle, Passive)
- Case (Nominative, Vocative, Accusative, Genitive, Dative)
- Number (Singular, Plural)
- Gender (Masculine, Feminine, Neuter)
- Degree (Positive, Comparative, Superlative)

The tagsets differ in their annotation style:
- BibleHub: Compact format (e.g., `V-PIA-3S` for "Verb - Present Indicative Active - 3rd Person Singular")
- Oblubienica: Verbose format (e.g., `vi Pres Act 3 Sg` for the same morphological information)

```python
>>> from morpht5 import BibleHubTag, OblubienicaTag
>>> len(BibleHubTag)
684
>>> len(OblubienicaTag)
1070
```

### Formatting

There's also a utility function for formatting interlinear translations:

```python
>>> from morpht5.utils.formatting import format_interlinear
>>> text = ['Λέγει', 'αὐτῷ', 'ὁ', 'Ἰησοῦς', 'Ἔγειρε', 'ἆρον', 'τὸν', 'κράβαττόν', 'σου', 'καὶ', 'περιπάτει']
>>> tags = ['V-PIA-3S', 'PPro-DM3S', 'Art-NMS', 'N-NMS', 'V-PMA-2S', 'V-AMA-2S', 'Art-AMS', 'N-AMS', 'PPro-G2S', 'Conj', 'V-PMA-2S']
>>> trans_pl = ['Mówi', 'mu', '-', 'Jezus', 'wstawaj', 'weź', '-', 'matę', 'swoją', 'i', 'chodź']
>>> trans_en = ['says', 'to him', '-', 'jesus', 'arise', 'take up', '-', 'mat', 'of you', 'and', 'walk']
>>> print(format_interlinear(text, tags, trans_en, trans_pl))
 Λέγει   |    αὐτῷ   |    ὁ    | Ἰησοῦς |  Ἔγειρε  |   ἆρον   |   τὸν   | κράβαττόν |   σου    | καὶ  | περιπάτει
V-PIA-3S | PPro-DM3S | Art-NMS | N-NMS  | V-PMA-2S | V-AMA-2S | Art-AMS |   N-AMS   | PPro-G2S | Conj |  V-PMA-2S
  says   |   to him  |    -    | jesus  |  arise   | take up  |   the   |    mat    |  of you  | and  |    walk
  Mówi   |     mu    |    -    | Jezus  | wstawaj  |   weź    |    -    |    matę   |  swoją   |  i   |   chodź
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
