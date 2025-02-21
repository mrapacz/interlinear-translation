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

The experiments involved fine-tuning T5-family models in 144 configurations. All models are available at [🤗 Hugging Face](https://huggingface.co/mrapacz). You can refer to the following tables for summary per target language.

#### English Models

| Id | Base Model | Encoding | Tag Set | Preprocessing | BLEU | SemScore | Link |
|-------|------------|----------|----------|--------------|------|-----------|------|
|`interlinear-en-philta-emb-auto-diacritics-bh`|PhilTa|emb-auto|BH (Bible Hub)|Diacritics|60.40|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-diacritics-bh)|
|`interlinear-en-philta-emb-sum-diacritics-bh`|PhilTa|emb-sum|BH (Bible Hub)|Diacritics|60.10|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-diacritics-bh)|
|`interlinear-en-philta-emb-sum-diacritics-ob`|PhilTa|emb-sum|OB (Oblubienica)|Diacritics|59.75|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-diacritics-ob)|
|`interlinear-en-philta-emb-auto-diacritics-ob`|PhilTa|emb-auto|OB (Oblubienica)|Diacritics|59.66|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-diacritics-ob)|
|`interlinear-en-philta-emb-auto-normalized-ob`|PhilTa|emb-auto|OB (Oblubienica)|Normalized|56.51|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-normalized-ob)|
|`interlinear-en-mt5-large-emb-auto-diacritics-bh`|mT5-large|emb-auto|BH (Bible Hub)|Diacritics|56.51|0.88|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-diacritics-bh)|
|`interlinear-en-philta-emb-sum-normalized-bh`|PhilTa|emb-sum|BH (Bible Hub)|Normalized|56.24|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-normalized-bh)|
|`interlinear-en-mt5-large-emb-sum-normalized-ob`|mT5-large|emb-sum|OB (Oblubienica)|Normalized|56.24|0.88|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-normalized-ob)|
|`interlinear-en-philta-emb-auto-normalized-bh`|PhilTa|emb-auto|BH (Bible Hub)|Normalized|56.16|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-normalized-bh)|
|`interlinear-en-mt5-large-emb-sum-diacritics-bh`|mT5-large|emb-sum|BH (Bible Hub)|Diacritics|56.03|0.88|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-diacritics-bh)|
|`interlinear-en-philta-emb-concat-diacritics-bh`|PhilTa|emb-concat|BH (Bible Hub)|Diacritics|55.93|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-large-emb-auto-diacritics-ob`|mT5-large|emb-auto|OB (Oblubienica)|Diacritics|55.81|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-diacritics-ob)|
|`interlinear-en-mt5-large-emb-sum-normalized-bh`|mT5-large|emb-sum|BH (Bible Hub)|Normalized|55.61|0.88|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-normalized-bh)|
|`interlinear-en-philta-emb-sum-normalized-ob`|PhilTa|emb-sum|OB (Oblubienica)|Normalized|55.49|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-normalized-ob)|
|`interlinear-en-mt5-large-emb-auto-normalized-ob`|mT5-large|emb-auto|OB (Oblubienica)|Normalized|55.37|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-normalized-ob)|
|`interlinear-en-greta-emb-sum-diacritics-bh`|GreTa|emb-sum|BH (Bible Hub)|Diacritics|55.22|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-diacritics-bh)|
|`interlinear-en-mt5-large-emb-auto-normalized-bh`|mT5-large|emb-auto|BH (Bible Hub)|Normalized|55.12|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-normalized-bh)|
|`interlinear-en-greta-emb-sum-diacritics-ob`|GreTa|emb-sum|OB (Oblubienica)|Diacritics|54.98|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-diacritics-ob)|
|`interlinear-en-greta-emb-auto-diacritics-ob`|GreTa|emb-auto|OB (Oblubienica)|Diacritics|54.98|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-diacritics-ob)|
|`interlinear-en-greta-emb-auto-diacritics-bh`|GreTa|emb-auto|BH (Bible Hub)|Diacritics|54.18|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-diacritics-bh)|
|`interlinear-en-greta-emb-auto-normalized-bh`|GreTa|emb-auto|BH (Bible Hub)|Normalized|53.17|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-normalized-bh)|
|`interlinear-en-greta-emb-auto-normalized-ob`|GreTa|emb-auto|OB (Oblubienica)|Normalized|53.15|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-normalized-ob`|mT5-base|emb-auto|OB (Oblubienica)|Normalized|52.43|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-normalized-ob)|
|`interlinear-en-greta-emb-sum-normalized-ob`|GreTa|emb-sum|OB (Oblubienica)|Normalized|52.39|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-diacritics-ob`|mT5-base|emb-auto|OB (Oblubienica)|Diacritics|52.37|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-diacritics-ob)|
|`interlinear-en-mt5-base-emb-sum-diacritics-bh`|mT5-base|emb-sum|BH (Bible Hub)|Diacritics|52.34|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-diacritics-bh)|
|`interlinear-en-greta-emb-sum-normalized-bh`|GreTa|emb-sum|BH (Bible Hub)|Normalized|51.93|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-normalized-bh)|
|`interlinear-en-mt5-base-emb-sum-diacritics-ob`|mT5-base|emb-sum|OB (Oblubienica)|Diacritics|51.90|0.86|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-diacritics-ob)|
|`interlinear-en-mt5-large-emb-concat-normalized-ob`|mT5-large|emb-concat|OB (Oblubienica)|Normalized|51.04|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-normalized-ob)|
|`interlinear-en-mt5-large-emb-concat-diacritics-bh`|mT5-large|emb-concat|BH (Bible Hub)|Diacritics|50.47|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-base-emb-sum-normalized-ob`|mT5-base|emb-sum|OB (Oblubienica)|Normalized|47.95|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-normalized-bh`|mT5-base|emb-auto|BH (Bible Hub)|Normalized|47.84|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-normalized-bh)|
|`interlinear-en-philta-emb-concat-normalized-bh`|PhilTa|emb-concat|BH (Bible Hub)|Normalized|46.82|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-normalized-bh)|
|`interlinear-en-mt5-large-t-w-t-diacritics-bh`|mT5-large|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|46.00|0.83|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-diacritics-bh)|
|`interlinear-en-mt5-large-t-w-t-diacritics-ob`|mT5-large|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|45.59|0.83|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-diacritics-ob)|
|`interlinear-en-philta-emb-concat-diacritics-ob`|PhilTa|emb-concat|OB (Oblubienica)|Diacritics|45.43|0.83|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-diacritics-ob)|
|`interlinear-en-mt5-large-baseline-diacritics-unused`|mT5-large|baseline (text only, no morphological tags)|Unused|Diacritics|44.67|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-baseline-diacritics-unused)|
|`interlinear-en-mt5-large-t-w-t-normalized-bh`|mT5-large|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|43.97|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-normalized-bh)|
|`interlinear-en-mt5-large-baseline-normalized-unused`|mT5-large|baseline (text only, no morphological tags)|Unused|Normalized|43.64|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-baseline-normalized-unused)|
|`interlinear-en-mt5-base-emb-concat-diacritics-ob`|mT5-base|emb-concat|OB (Oblubienica)|Diacritics|42.59|0.80|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-diacritics-ob)|
|`interlinear-en-philta-baseline-diacritics-unused`|PhilTa|baseline (text only, no morphological tags)|Unused|Diacritics|41.55|0.83|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-baseline-diacritics-unused)|
|`interlinear-en-mt5-large-emb-concat-diacritics-ob`|mT5-large|emb-concat|OB (Oblubienica)|Diacritics|41.18|0.77|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-diacritics-ob)|
|`interlinear-en-philta-t-w-t-diacritics-bh`|PhilTa|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|40.95|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-diacritics-bh)|
|`interlinear-en-philta-t-w-t-diacritics-ob`|PhilTa|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|40.84|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-diacritics-ob)|
|`interlinear-en-philta-emb-concat-normalized-ob`|PhilTa|emb-concat|OB (Oblubienica)|Normalized|40.76|0.78|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-normalized-ob)|
|`interlinear-en-mt5-large-t-w-t-normalized-ob`|mT5-large|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|35.47|0.78|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-normalized-ob)|
|`interlinear-en-philta-t-w-t-normalized-bh`|PhilTa|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|34.25|0.76|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-normalized-bh)|
|`interlinear-en-philta-t-w-t-normalized-ob`|PhilTa|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|33.44|0.76|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-normalized-ob)|
|`interlinear-en-philta-baseline-normalized-unused`|PhilTa|baseline (text only, no morphological tags)|Unused|Normalized|33.24|0.74|[🤗](@https://huggingface.co/mrapacz/interlinear-en-philta-baseline-normalized-unused)|
|`interlinear-en-mt5-base-baseline-diacritics-unused`|mT5-base|baseline (text only, no morphological tags)|Unused|Diacritics|31.61|0.74|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-baseline-diacritics-unused)|
|`interlinear-en-mt5-base-t-w-t-diacritics-bh`|mT5-base|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|30.11|0.74|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-diacritics-bh)|
|`interlinear-en-mt5-base-baseline-normalized-unused`|mT5-base|baseline (text only, no morphological tags)|Unused|Normalized|29.99|0.74|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-baseline-normalized-unused)|
|`interlinear-en-mt5-base-t-w-t-diacritics-ob`|mT5-base|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|29.62|0.74|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-diacritics-ob)|
|`interlinear-en-mt5-base-emb-auto-diacritics-bh`|mT5-base|emb-auto|BH (Bible Hub)|Diacritics|28.52|0.71|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-diacritics-bh)|
|`interlinear-en-mt5-base-t-w-t-normalized-ob`|mT5-base|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|28.39|0.73|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-normalized-ob)|
|`interlinear-en-mt5-base-t-w-t-normalized-bh`|mT5-base|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|27.59|0.72|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-normalized-bh`|mT5-base|emb-concat|BH (Bible Hub)|Normalized|27.32|0.68|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-normalized-bh)|
|`interlinear-en-greta-baseline-diacritics-unused`|GreTa|baseline (text only, no morphological tags)|Unused|Diacritics|17.69|0.56|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-baseline-diacritics-unused)|
|`interlinear-en-greta-baseline-normalized-unused`|GreTa|baseline (text only, no morphological tags)|Unused|Normalized|16.77|0.56|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-baseline-normalized-unused)|
|`interlinear-en-greta-t-w-t-normalized-bh`|GreTa|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|16.13|0.56|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-normalized-bh)|
|`interlinear-en-greta-t-w-t-diacritics-bh`|GreTa|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|14.70|0.55|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-diacritics-bh)|
|`interlinear-en-greta-t-w-t-diacritics-ob`|GreTa|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|14.51|0.55|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-diacritics-ob)|
|`interlinear-en-greta-t-w-t-normalized-ob`|GreTa|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|12.14|0.53|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-normalized-ob)|
|`interlinear-en-greta-emb-concat-diacritics-ob`|GreTa|emb-concat|OB (Oblubienica)|Diacritics|5.48|0.49|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-diacritics-ob)|
|`interlinear-en-greta-emb-concat-normalized-bh`|GreTa|emb-concat|BH (Bible Hub)|Normalized|4.05|0.42|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-normalized-bh)|
|`interlinear-en-greta-emb-concat-normalized-ob`|GreTa|emb-concat|OB (Oblubienica)|Normalized|3.93|0.42|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-normalized-ob)|
|`interlinear-en-greta-emb-concat-diacritics-bh`|GreTa|emb-concat|BH (Bible Hub)|Diacritics|3.58|0.42|[🤗](@https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-base-emb-sum-normalized-bh`|mT5-base|emb-sum|BH (Bible Hub)|Normalized|1.66|0.38|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-diacritics-bh`|mT5-base|emb-concat|BH (Bible Hub)|Diacritics|1.33|0.34|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-large-emb-sum-diacritics-ob`|mT5-large|emb-sum|OB (Oblubienica)|Diacritics|0.83|0.34|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-diacritics-ob)|
|`interlinear-en-mt5-large-emb-concat-normalized-bh`|mT5-large|emb-concat|BH (Bible Hub)|Normalized|0.70|0.37|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-normalized-ob`|mT5-base|emb-concat|OB (Oblubienica)|Normalized|0.69|0.34|[🤗](@https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-normalized-ob)|

#### Polish Models

| Id | Base Model | Encoding | Tag Set | Preprocessing | BLEU | SemScore | Link |
|-------|------------|----------|----------|--------------|------|-----------|------|
|`interlinear-pl-mt5-large-emb-auto-normalized-ob`|mT5-large|emb-auto|OB (Oblubienica)|Normalized|59.33|0.94|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-normalized-ob)|
|`interlinear-pl-mt5-large-emb-auto-diacritics-bh`|mT5-large|emb-auto|BH (Bible Hub)|Diacritics|59.04|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-sum-normalized-ob`|mT5-large|emb-sum|OB (Oblubienica)|Normalized|58.92|0.94|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-large-emb-sum-diacritics-ob`|mT5-large|emb-sum|OB (Oblubienica)|Diacritics|58.90|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-sum-normalized-bh`|mT5-large|emb-sum|BH (Bible Hub)|Normalized|58.46|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-normalized-bh)|
|`interlinear-pl-mt5-large-emb-auto-diacritics-ob`|mT5-large|emb-auto|OB (Oblubienica)|Diacritics|58.44|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-auto-normalized-bh`|mT5-large|emb-auto|BH (Bible Hub)|Normalized|57.42|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-large-emb-sum-diacritics-bh`|mT5-large|emb-sum|BH (Bible Hub)|Diacritics|56.75|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-diacritics-ob`|mT5-large|emb-concat|OB (Oblubienica)|Diacritics|55.55|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-auto-diacritics-bh`|mT5-base|emb-auto|BH (Bible Hub)|Diacritics|54.63|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-normalized-bh`|mT5-large|emb-concat|BH (Bible Hub)|Normalized|54.54|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-normalized-bh)|
|`interlinear-pl-mt5-base-emb-auto-normalized-bh`|mT5-base|emb-auto|BH (Bible Hub)|Normalized|54.47|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-base-emb-sum-diacritics-ob`|mT5-base|emb-sum|OB (Oblubienica)|Diacritics|54.41|0.93|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-auto-diacritics-ob`|mT5-base|emb-auto|OB (Oblubienica)|Diacritics|53.87|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-sum-diacritics-bh`|mT5-base|emb-sum|BH (Bible Hub)|Diacritics|52.54|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-normalized-ob`|mT5-large|emb-concat|OB (Oblubienica)|Normalized|51.75|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-normalized-ob)|
|`interlinear-pl-greta-emb-auto-diacritics-bh`|GreTa|emb-auto|BH (Bible Hub)|Diacritics|51.30|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-diacritics-bh)|
|`interlinear-pl-greta-emb-sum-diacritics-ob`|GreTa|emb-sum|OB (Oblubienica)|Diacritics|51.21|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-diacritics-ob)|
|`interlinear-pl-greta-emb-auto-diacritics-ob`|GreTa|emb-auto|OB (Oblubienica)|Diacritics|51.06|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-diacritics-ob)|
|`interlinear-pl-greta-emb-sum-diacritics-bh`|GreTa|emb-sum|BH (Bible Hub)|Diacritics|50.89|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-sum-normalized-bh`|mT5-base|emb-sum|BH (Bible Hub)|Normalized|50.43|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-auto-normalized-ob`|GreTa|emb-auto|OB (Oblubienica)|Normalized|49.72|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-normalized-ob)|
|`interlinear-pl-greta-emb-sum-normalized-bh`|GreTa|emb-sum|BH (Bible Hub)|Normalized|48.47|0.92|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-auto-normalized-bh`|GreTa|emb-auto|BH (Bible Hub)|Normalized|46.01|0.91|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-base-emb-auto-normalized-ob`|mT5-base|emb-auto|OB (Oblubienica)|Normalized|44.29|0.90|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-normalized-ob)|
|`interlinear-pl-mt5-large-baseline-diacritics-unused`|mT5-large|baseline (text only, no morphological tags)|Unused|Diacritics|42.92|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-baseline-diacritics-unused)|
|`interlinear-pl-mt5-large-t-w-t-diacritics-bh`|mT5-large|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|41.93|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-diacritics-bh)|
|`interlinear-pl-mt5-large-t-w-t-diacritics-ob`|mT5-large|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|41.62|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-diacritics-ob)|
|`interlinear-pl-mt5-large-t-w-t-normalized-ob`|mT5-large|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|41.58|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-normalized-ob)|
|`interlinear-pl-mt5-large-baseline-normalized-unused`|mT5-large|baseline (text only, no morphological tags)|Unused|Normalized|41.05|0.89|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-baseline-normalized-unused)|
|`interlinear-pl-greta-emb-sum-normalized-ob`|GreTa|emb-sum|OB (Oblubienica)|Normalized|32.92|0.87|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-base-baseline-diacritics-unused`|mT5-base|baseline (text only, no morphological tags)|Unused|Diacritics|28.75|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-baseline-diacritics-unused)|
|`interlinear-pl-mt5-base-t-w-t-diacritics-ob`|mT5-base|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|27.72|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-diacritics-ob)|
|`interlinear-pl-mt5-base-baseline-normalized-unused`|mT5-base|baseline (text only, no morphological tags)|Unused|Normalized|26.21|0.85|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-baseline-normalized-unused)|
|`interlinear-pl-mt5-base-t-w-t-normalized-bh`|mT5-base|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|26.07|0.84|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-normalized-bh)|
|`interlinear-pl-mt5-base-t-w-t-diacritics-bh`|mT5-base|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|21.45|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-diacritics-bh)|
|`interlinear-pl-philta-emb-auto-normalized-bh`|PhilTa|emb-auto|BH (Bible Hub)|Normalized|15.37|0.82|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-normalized-bh)|
|`interlinear-pl-philta-emb-auto-diacritics-bh`|PhilTa|emb-auto|BH (Bible Hub)|Diacritics|11.79|0.80|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-diacritics-bh)|
|`interlinear-pl-philta-emb-auto-diacritics-ob`|PhilTa|emb-auto|OB (Oblubienica)|Diacritics|8.24|0.79|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-diacritics-ob)|
|`interlinear-pl-philta-emb-auto-normalized-ob`|PhilTa|emb-auto|OB (Oblubienica)|Normalized|6.23|0.77|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-normalized-ob)|
|`interlinear-pl-philta-emb-sum-diacritics-bh`|PhilTa|emb-sum|BH (Bible Hub)|Diacritics|6.18|0.77|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-diacritics-bh)|
|`interlinear-pl-philta-emb-sum-normalized-ob`|PhilTa|emb-sum|OB (Oblubienica)|Normalized|5.39|0.76|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-base-emb-concat-normalized-bh`|mT5-base|emb-concat|BH (Bible Hub)|Normalized|1.93|0.68|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-normalized-bh)|
|`interlinear-pl-greta-emb-concat-normalized-bh`|GreTa|emb-concat|BH (Bible Hub)|Normalized|1.86|0.62|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-normalized-bh)|
|`interlinear-pl-philta-emb-sum-normalized-bh`|PhilTa|emb-sum|BH (Bible Hub)|Normalized|1.71|0.69|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-concat-normalized-ob`|GreTa|emb-concat|OB (Oblubienica)|Normalized|1.41|0.60|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-normalized-ob)|
|`interlinear-pl-greta-baseline-diacritics-unused`|GreTa|baseline (text only, no morphological tags)|Unused|Diacritics|0.86|0.53|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-baseline-diacritics-unused)|
|`interlinear-pl-greta-emb-concat-diacritics-ob`|GreTa|emb-concat|OB (Oblubienica)|Diacritics|0.84|0.62|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-concat-diacritics-bh`|mT5-base|emb-concat|BH (Bible Hub)|Diacritics|0.79|0.67|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-diacritics-bh)|
|`interlinear-pl-greta-t-w-t-normalized-ob`|GreTa|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|0.78|0.56|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-normalized-ob)|
|`interlinear-pl-greta-t-w-t-diacritics-ob`|GreTa|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|0.74|0.54|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-diacritics-ob)|
|`interlinear-pl-greta-emb-concat-diacritics-bh`|GreTa|emb-concat|BH (Bible Hub)|Diacritics|0.71|0.59|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-sum-normalized-ob`|mT5-base|emb-sum|OB (Oblubienica)|Normalized|0.66|0.65|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-normalized-ob)|
|`interlinear-pl-greta-baseline-normalized-unused`|GreTa|baseline (text only, no morphological tags)|Unused|Normalized|0.63|0.49|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-baseline-normalized-unused)|
|`interlinear-pl-mt5-base-emb-concat-diacritics-ob`|mT5-base|emb-concat|OB (Oblubienica)|Diacritics|0.63|0.63|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-concat-diacritics-bh`|mT5-large|emb-concat|BH (Bible Hub)|Diacritics|0.57|0.68|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-diacritics-bh)|
|`interlinear-pl-greta-t-w-t-normalized-bh`|GreTa|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|0.56|0.49|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-normalized-bh)|
|`interlinear-pl-greta-t-w-t-diacritics-bh`|GreTa|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|0.49|0.51|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-concat-normalized-ob`|mT5-base|emb-concat|OB (Oblubienica)|Normalized|0.45|0.67|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-normalized-ob)|
|`interlinear-pl-philta-emb-concat-normalized-ob`|PhilTa|emb-concat|OB (Oblubienica)|Normalized|0.26|0.58|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-normalized-ob)|
|`interlinear-pl-philta-emb-concat-normalized-bh`|PhilTa|emb-concat|BH (Bible Hub)|Normalized|0.26|0.58|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-normalized-bh)|
|`interlinear-pl-mt5-base-t-w-t-normalized-ob`|mT5-base|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|0.24|0.66|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-normalized-ob)|
|`interlinear-pl-mt5-large-t-w-t-normalized-bh`|mT5-large|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|0.17|0.45|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-normalized-bh)|
|`interlinear-pl-philta-emb-concat-diacritics-ob`|PhilTa|emb-concat|OB (Oblubienica)|Diacritics|0.13|0.53|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-diacritics-ob)|
|`interlinear-pl-philta-emb-sum-diacritics-ob`|PhilTa|emb-sum|OB (Oblubienica)|Diacritics|0.12|0.55|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-diacritics-ob)|
|`interlinear-pl-philta-emb-concat-diacritics-bh`|PhilTa|emb-concat|BH (Bible Hub)|Diacritics|0.11|0.58|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-diacritics-bh)|
|`interlinear-pl-philta-t-w-t-normalized-bh`|PhilTa|t-w-t (tags-within-text)|BH (Bible Hub)|Normalized|0.08|0.52|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-normalized-bh)|
|`interlinear-pl-philta-t-w-t-diacritics-ob`|PhilTa|t-w-t (tags-within-text)|OB (Oblubienica)|Diacritics|0.08|0.56|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-diacritics-ob)|
|`interlinear-pl-philta-baseline-normalized-unused`|PhilTa|baseline (text only, no morphological tags)|Unused|Normalized|0.07|0.42|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-baseline-normalized-unused)|
|`interlinear-pl-philta-t-w-t-normalized-ob`|PhilTa|t-w-t (tags-within-text)|OB (Oblubienica)|Normalized|0.05|0.50|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-normalized-ob)|
|`interlinear-pl-philta-t-w-t-diacritics-bh`|PhilTa|t-w-t (tags-within-text)|BH (Bible Hub)|Diacritics|0.04|0.54|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-diacritics-bh)|
|`interlinear-pl-philta-baseline-diacritics-unused`|PhilTa|baseline (text only, no morphological tags)|Unused|Diacritics|0.03|0.18|[🤗](@https://huggingface.co/mrapacz/interlinear-pl-philta-baseline-diacritics-unused)|


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