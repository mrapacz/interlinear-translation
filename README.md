# loreslm-interlinear-translation

This is a repository holding the resources related to the `Low-Resource Interlinear Translation: Morphology-Enhanced Neural Models for Ancient Greek` paper presented at the [LoResLM@COLING2025 workshop](https://loreslm.github.io/).

The article is available at: https://aclanthology.org/2025.loreslm-1.11/

To view the conference poster, see [./resources/poster.pdf](./resources/poster.pdf).

## Resources

### Code

To see the code of (and play with) the modified T5 models, see [./morpht5](./morpht5).
For the code used for training the models, see [./code](./code).

### Dataset

The dataset used for training the models is available at [🤗 Hugging Face](https://huggingface.co/datasets/mrapacz/greek-interlinear-translations).

### Models

The experiments involved fine-tuning T5-family models in 144 configurations. All models are available at [🤗 Hugging Face](https://huggingface.co/mrapacz). You can refer to the following tables for summary per target language:

- [English Models](./resources/model_table_en.md)
- [Polish Models](./resources/model_table_pl.md)

## License

The resources - unless otherwise specified - are licensed under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).


## Citation

```bibtex
@inproceedings{rapacz-smywinski-pohl-2025-low,
    title = "Low-Resource Interlinear Translation: Morphology-Enhanced Neural Models for {A}ncient {G}reek",
    author = "Rapacz, Maciej  and
      Smywi{\'n}ski-Pohl, Aleksander",
    editor = "Hettiarachchi, Hansi  and
      Ranasinghe, Tharindu  and
      Rayson, Paul  and
      Mitkov, Ruslan  and
      Gaber, Mohamed  and
      Premasiri, Damith  and
      Tan, Fiona Anting  and
      Uyangodage, Lasitha",
    booktitle = "Proceedings of the First Workshop on Language Models for Low-Resource Languages",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.loreslm-1.11/",
    pages = "145--165",
    abstract = "Contemporary machine translation systems prioritize fluent, natural-sounding output with flexible word ordering. In contrast, interlinear translation maintains the source text`s syntactic structure by aligning target language words directly beneath their source counterparts. Despite its importance in classical scholarship, automated approaches to interlinear translation remain understudied. We evaluated neural interlinear translation from Ancient Greek to English and Polish using four transformer-based models: two Ancient Greek-specialized (GreTa and PhilTa) and two general-purpose multilingual models (mT5-base and mT5-large). Our approach introduces novel morphological embedding layers and evaluates text preprocessing and tag set selection across 144 experimental configurations using a word-aligned parallel corpus of the Greek New Testament. Results show that morphological features through dedicated embedding layers significantly enhance translation quality, improving BLEU scores by 35{\%} (44.67 {\textrightarrow} 60.40) for English and 38{\%} (42.92 {\textrightarrow} 59.33) for Polish compared to baseline models. PhilTa achieves state-of-the-art performance for English, while mT5-large does so for Polish. Notably, PhilTa maintains stable performance using only 10{\%} of training data. Our findings challenge the assumption that modern neural architectures cannot benefit from explicit morphological annotations. While preprocessing strategies and tag set selection show minimal impact, the substantial gains from morphological embeddings demonstrate their value in low-resource scenarios."
}
```