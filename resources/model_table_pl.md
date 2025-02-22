# Model Overview

This table presents the performance metrics of various model configurations for Ancient Greek to Polish interlinear translation.

## Polish Models

| Id | Base Model | Encoding | Tag Set | Preprocessing | BLEU | SemScore | Link |
|-------|------------|----------|----------|--------------|------|-----------|------|
|`interlinear-pl-mt5-large-emb-auto-normalized-ob`|mT5-large|emb-auto|Oblubienica|Normalized|59.33|0.94|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-normalized-ob)|
|`interlinear-pl-mt5-large-emb-auto-diacritics-bh`|mT5-large|emb-auto|Bible Hub|Diacritics|59.04|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-sum-normalized-ob`|mT5-large|emb-sum|Oblubienica|Normalized|58.92|0.94|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-large-emb-sum-diacritics-ob`|mT5-large|emb-sum|Oblubienica|Diacritics|58.90|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-sum-normalized-bh`|mT5-large|emb-sum|Bible Hub|Normalized|58.46|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-normalized-bh)|
|`interlinear-pl-mt5-large-emb-auto-diacritics-ob`|mT5-large|emb-auto|Oblubienica|Diacritics|58.44|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-auto-normalized-bh`|mT5-large|emb-auto|Bible Hub|Normalized|57.42|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-large-emb-sum-diacritics-bh`|mT5-large|emb-sum|Bible Hub|Diacritics|56.75|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-diacritics-ob`|mT5-large|emb-concat|Oblubienica|Diacritics|55.55|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-auto-diacritics-bh`|mT5-base|emb-auto|Bible Hub|Diacritics|54.63|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-normalized-bh`|mT5-large|emb-concat|Bible Hub|Normalized|54.54|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-normalized-bh)|
|`interlinear-pl-mt5-base-emb-auto-normalized-bh`|mT5-base|emb-auto|Bible Hub|Normalized|54.47|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-base-emb-sum-diacritics-ob`|mT5-base|emb-sum|Oblubienica|Diacritics|54.41|0.93|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-auto-diacritics-ob`|mT5-base|emb-auto|Oblubienica|Diacritics|53.87|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-sum-diacritics-bh`|mT5-base|emb-sum|Bible Hub|Diacritics|52.54|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-large-emb-concat-normalized-ob`|mT5-large|emb-concat|Oblubienica|Normalized|51.75|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-normalized-ob)|
|`interlinear-pl-greta-emb-auto-diacritics-bh`|GreTa|emb-auto|Bible Hub|Diacritics|51.30|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-diacritics-bh)|
|`interlinear-pl-greta-emb-sum-diacritics-ob`|GreTa|emb-sum|Oblubienica|Diacritics|51.21|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-diacritics-ob)|
|`interlinear-pl-greta-emb-auto-diacritics-ob`|GreTa|emb-auto|Oblubienica|Diacritics|51.06|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-diacritics-ob)|
|`interlinear-pl-greta-emb-sum-diacritics-bh`|GreTa|emb-sum|Bible Hub|Diacritics|50.89|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-sum-normalized-bh`|mT5-base|emb-sum|Bible Hub|Normalized|50.43|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-auto-normalized-ob`|GreTa|emb-auto|Oblubienica|Normalized|49.72|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-normalized-ob)|
|`interlinear-pl-greta-emb-sum-normalized-bh`|GreTa|emb-sum|Bible Hub|Normalized|48.47|0.92|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-auto-normalized-bh`|GreTa|emb-auto|Bible Hub|Normalized|46.01|0.91|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-auto-normalized-bh)|
|`interlinear-pl-mt5-base-emb-auto-normalized-ob`|mT5-base|emb-auto|Oblubienica|Normalized|44.29|0.90|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-auto-normalized-ob)|
|`interlinear-pl-mt5-large-baseline-diacritics-unused`|mT5-large|baseline (text only)|Unused|Diacritics|42.92|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-baseline-diacritics-unused)|
|`interlinear-pl-mt5-large-t-w-t-diacritics-bh`|mT5-large|t-w-t (tags-within-text)|Bible Hub|Diacritics|41.93|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-diacritics-bh)|
|`interlinear-pl-mt5-large-t-w-t-diacritics-ob`|mT5-large|t-w-t (tags-within-text)|Oblubienica|Diacritics|41.62|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-diacritics-ob)|
|`interlinear-pl-mt5-large-t-w-t-normalized-ob`|mT5-large|t-w-t (tags-within-text)|Oblubienica|Normalized|41.58|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-normalized-ob)|
|`interlinear-pl-mt5-large-baseline-normalized-unused`|mT5-large|baseline (text only)|Unused|Normalized|41.05|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-baseline-normalized-unused)|
|`interlinear-pl-greta-emb-sum-normalized-ob`|GreTa|emb-sum|Oblubienica|Normalized|32.92|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-base-baseline-diacritics-unused`|mT5-base|baseline (text only)|Unused|Diacritics|28.75|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-baseline-diacritics-unused)|
|`interlinear-pl-mt5-base-t-w-t-diacritics-ob`|mT5-base|t-w-t (tags-within-text)|Oblubienica|Diacritics|27.72|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-diacritics-ob)|
|`interlinear-pl-mt5-base-baseline-normalized-unused`|mT5-base|baseline (text only)|Unused|Normalized|26.21|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-baseline-normalized-unused)|
|`interlinear-pl-mt5-base-t-w-t-normalized-bh`|mT5-base|t-w-t (tags-within-text)|Bible Hub|Normalized|26.07|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-normalized-bh)|
|`interlinear-pl-mt5-base-t-w-t-diacritics-bh`|mT5-base|t-w-t (tags-within-text)|Bible Hub|Diacritics|21.45|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-diacritics-bh)|
|`interlinear-pl-philta-emb-auto-normalized-bh`|PhilTa|emb-auto|Bible Hub|Normalized|15.37|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-normalized-bh)|
|`interlinear-pl-philta-emb-auto-diacritics-bh`|PhilTa|emb-auto|Bible Hub|Diacritics|11.79|0.80|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-diacritics-bh)|
|`interlinear-pl-philta-emb-auto-diacritics-ob`|PhilTa|emb-auto|Oblubienica|Diacritics|8.24|0.79|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-diacritics-ob)|
|`interlinear-pl-philta-emb-auto-normalized-ob`|PhilTa|emb-auto|Oblubienica|Normalized|6.23|0.77|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-auto-normalized-ob)|
|`interlinear-pl-philta-emb-sum-diacritics-bh`|PhilTa|emb-sum|Bible Hub|Diacritics|6.18|0.77|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-diacritics-bh)|
|`interlinear-pl-philta-emb-sum-normalized-ob`|PhilTa|emb-sum|Oblubienica|Normalized|5.39|0.76|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-normalized-ob)|
|`interlinear-pl-mt5-base-emb-concat-normalized-bh`|mT5-base|emb-concat|Bible Hub|Normalized|1.93|0.68|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-normalized-bh)|
|`interlinear-pl-greta-emb-concat-normalized-bh`|GreTa|emb-concat|Bible Hub|Normalized|1.86|0.62|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-normalized-bh)|
|`interlinear-pl-philta-emb-sum-normalized-bh`|PhilTa|emb-sum|Bible Hub|Normalized|1.71|0.69|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-normalized-bh)|
|`interlinear-pl-greta-emb-concat-normalized-ob`|GreTa|emb-concat|Oblubienica|Normalized|1.41|0.60|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-normalized-ob)|
|`interlinear-pl-greta-baseline-diacritics-unused`|GreTa|baseline (text only)|Unused|Diacritics|0.86|0.53|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-baseline-diacritics-unused)|
|`interlinear-pl-greta-emb-concat-diacritics-ob`|GreTa|emb-concat|Oblubienica|Diacritics|0.84|0.62|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-base-emb-concat-diacritics-bh`|mT5-base|emb-concat|Bible Hub|Diacritics|0.79|0.67|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-diacritics-bh)|
|`interlinear-pl-greta-t-w-t-normalized-ob`|GreTa|t-w-t (tags-within-text)|Oblubienica|Normalized|0.78|0.56|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-normalized-ob)|
|`interlinear-pl-greta-t-w-t-diacritics-ob`|GreTa|t-w-t (tags-within-text)|Oblubienica|Diacritics|0.74|0.54|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-diacritics-ob)|
|`interlinear-pl-greta-emb-concat-diacritics-bh`|GreTa|emb-concat|Bible Hub|Diacritics|0.71|0.59|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-emb-concat-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-sum-normalized-ob`|mT5-base|emb-sum|Oblubienica|Normalized|0.66|0.65|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-sum-normalized-ob)|
|`interlinear-pl-greta-baseline-normalized-unused`|GreTa|baseline (text only)|Unused|Normalized|0.63|0.49|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-baseline-normalized-unused)|
|`interlinear-pl-mt5-base-emb-concat-diacritics-ob`|mT5-base|emb-concat|Oblubienica|Diacritics|0.63|0.63|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-diacritics-ob)|
|`interlinear-pl-mt5-large-emb-concat-diacritics-bh`|mT5-large|emb-concat|Bible Hub|Diacritics|0.57|0.68|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-emb-concat-diacritics-bh)|
|`interlinear-pl-greta-t-w-t-normalized-bh`|GreTa|t-w-t (tags-within-text)|Bible Hub|Normalized|0.56|0.49|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-normalized-bh)|
|`interlinear-pl-greta-t-w-t-diacritics-bh`|GreTa|t-w-t (tags-within-text)|Bible Hub|Diacritics|0.49|0.51|[🤗](https://huggingface.co/mrapacz/interlinear-pl-greta-t-w-t-diacritics-bh)|
|`interlinear-pl-mt5-base-emb-concat-normalized-ob`|mT5-base|emb-concat|Oblubienica|Normalized|0.45|0.67|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-emb-concat-normalized-ob)|
|`interlinear-pl-philta-emb-concat-normalized-ob`|PhilTa|emb-concat|Oblubienica|Normalized|0.26|0.58|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-normalized-ob)|
|`interlinear-pl-philta-emb-concat-normalized-bh`|PhilTa|emb-concat|Bible Hub|Normalized|0.26|0.58|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-normalized-bh)|
|`interlinear-pl-mt5-base-t-w-t-normalized-ob`|mT5-base|t-w-t (tags-within-text)|Oblubienica|Normalized|0.24|0.66|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-base-t-w-t-normalized-ob)|
|`interlinear-pl-mt5-large-t-w-t-normalized-bh`|mT5-large|t-w-t (tags-within-text)|Bible Hub|Normalized|0.17|0.45|[🤗](https://huggingface.co/mrapacz/interlinear-pl-mt5-large-t-w-t-normalized-bh)|
|`interlinear-pl-philta-emb-concat-diacritics-ob`|PhilTa|emb-concat|Oblubienica|Diacritics|0.13|0.53|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-diacritics-ob)|
|`interlinear-pl-philta-emb-sum-diacritics-ob`|PhilTa|emb-sum|Oblubienica|Diacritics|0.12|0.55|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-sum-diacritics-ob)|
|`interlinear-pl-philta-emb-concat-diacritics-bh`|PhilTa|emb-concat|Bible Hub|Diacritics|0.11|0.58|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-emb-concat-diacritics-bh)|
|`interlinear-pl-philta-t-w-t-normalized-bh`|PhilTa|t-w-t (tags-within-text)|Bible Hub|Normalized|0.08|0.52|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-normalized-bh)|
|`interlinear-pl-philta-t-w-t-diacritics-ob`|PhilTa|t-w-t (tags-within-text)|Oblubienica|Diacritics|0.08|0.56|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-diacritics-ob)|
|`interlinear-pl-philta-baseline-normalized-unused`|PhilTa|baseline (text only)|Unused|Normalized|0.07|0.42|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-baseline-normalized-unused)|
|`interlinear-pl-philta-t-w-t-normalized-ob`|PhilTa|t-w-t (tags-within-text)|Oblubienica|Normalized|0.05|0.50|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-normalized-ob)|
|`interlinear-pl-philta-t-w-t-diacritics-bh`|PhilTa|t-w-t (tags-within-text)|Bible Hub|Diacritics|0.04|0.54|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-t-w-t-diacritics-bh)|
|`interlinear-pl-philta-baseline-diacritics-unused`|PhilTa|baseline (text only)|Unused|Diacritics|0.03|0.18|[🤗](https://huggingface.co/mrapacz/interlinear-pl-philta-baseline-diacritics-unused)|
