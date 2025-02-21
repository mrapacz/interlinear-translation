# morpht5

This package contains the source code for the MorphT5 models used in the [Low Resource Interlinear Translation](https://github.com/mrapacz/loreslm-interlinear-translation) paper.

## Installation

```bash
pip install morpht5
```

## Components

### Models

The package provides several model variants through a unified interface:

```python
from morpht5 import (
    MorphT5AutoModel,
    MorphT5ConcatModel,
    MorphT5SumModel,
    MorphT5AutoForConditionalGeneration
)
```

Three model variants, incorporating morphological features through dedicated embedding layers:
- `MorphT5SumModel`: Model using positional summation for combining text and morphological tag embeddings
- `MorphT5AutoModel`: Base model with autoencoder-style morphological tag embeddings
- `MorphT5ConcatModel`: Model using concatenation for combining text and morphological tag embeddings


### Tokenizer

The package comes with a tokenizer that allows for encoding text and morphological tags into tokens:

```python
from morpht5 import MorphT5Tokenizer
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


## Model Variants

The package includes several pre-trained models with different configurations:
- Base model size (mT5-base) and Large model size (mT5-large)
- Three morphological encoding strategies (auto, sum, concat)
- Support for both tagsets (BibleHub and Oblubienica)
- Target languages: English and Polish

Best performing models achieve:
- English translation: BLEU 56.24 (mT5-large with sum encoding)
- Polish translation: BLEU 58.46 (mT5-large with sum encoding)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
