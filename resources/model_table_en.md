# Model Overview

This table presents the performance metrics of various model configurations for Ancient Greek to English interlinear translation.

## English Models

| Id | Base Model | Encoding | Tag Set | Preprocessing | BLEU | SemScore | Link |
|-------|------------|----------|----------|--------------|------|-----------|------|
|`interlinear-en-philta-emb-auto-diacritics-bh`|PhilTa|emb-auto|Bible Hub|Diacritics|60.40|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-diacritics-bh)|
|`interlinear-en-philta-emb-sum-diacritics-bh`|PhilTa|emb-sum|Bible Hub|Diacritics|60.10|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-diacritics-bh)|
|`interlinear-en-philta-emb-sum-diacritics-ob`|PhilTa|emb-sum|Oblubienica|Diacritics|59.75|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-diacritics-ob)|
|`interlinear-en-philta-emb-auto-diacritics-ob`|PhilTa|emb-auto|Oblubienica|Diacritics|59.66|0.89|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-diacritics-ob)|
|`interlinear-en-philta-emb-auto-normalized-ob`|PhilTa|emb-auto|Oblubienica|Normalized|56.51|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-normalized-ob)|
|`interlinear-en-mt5-large-emb-auto-diacritics-bh`|mT5-large|emb-auto|Bible Hub|Diacritics|56.51|0.88|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-diacritics-bh)|
|`interlinear-en-philta-emb-sum-normalized-bh`|PhilTa|emb-sum|Bible Hub|Normalized|56.24|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-normalized-bh)|
|`interlinear-en-mt5-large-emb-sum-normalized-ob`|mT5-large|emb-sum|Oblubienica|Normalized|56.24|0.88|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-normalized-ob)|
|`interlinear-en-philta-emb-auto-normalized-bh`|PhilTa|emb-auto|Bible Hub|Normalized|56.16|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-auto-normalized-bh)|
|`interlinear-en-mt5-large-emb-sum-diacritics-bh`|mT5-large|emb-sum|Bible Hub|Diacritics|56.03|0.88|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-diacritics-bh)|
|`interlinear-en-philta-emb-concat-diacritics-bh`|PhilTa|emb-concat|Bible Hub|Diacritics|55.93|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-large-emb-auto-diacritics-ob`|mT5-large|emb-auto|Oblubienica|Diacritics|55.81|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-diacritics-ob)|
|`interlinear-en-mt5-large-emb-sum-normalized-bh`|mT5-large|emb-sum|Bible Hub|Normalized|55.61|0.88|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-normalized-bh)|
|`interlinear-en-philta-emb-sum-normalized-ob`|PhilTa|emb-sum|Oblubienica|Normalized|55.49|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-sum-normalized-ob)|
|`interlinear-en-mt5-large-emb-auto-normalized-ob`|mT5-large|emb-auto|Oblubienica|Normalized|55.37|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-normalized-ob)|
|`interlinear-en-greta-emb-sum-diacritics-bh`|GreTa|emb-sum|Bible Hub|Diacritics|55.22|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-diacritics-bh)|
|`interlinear-en-mt5-large-emb-auto-normalized-bh`|mT5-large|emb-auto|Bible Hub|Normalized|55.12|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-auto-normalized-bh)|
|`interlinear-en-greta-emb-sum-diacritics-ob`|GreTa|emb-sum|Oblubienica|Diacritics|54.98|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-diacritics-ob)|
|`interlinear-en-greta-emb-auto-diacritics-ob`|GreTa|emb-auto|Oblubienica|Diacritics|54.98|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-diacritics-ob)|
|`interlinear-en-greta-emb-auto-diacritics-bh`|GreTa|emb-auto|Bible Hub|Diacritics|54.18|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-diacritics-bh)|
|`interlinear-en-greta-emb-auto-normalized-bh`|GreTa|emb-auto|Bible Hub|Normalized|53.17|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-normalized-bh)|
|`interlinear-en-greta-emb-auto-normalized-ob`|GreTa|emb-auto|Oblubienica|Normalized|53.15|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-auto-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-normalized-ob`|mT5-base|emb-auto|Oblubienica|Normalized|52.43|0.87|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-normalized-ob)|
|`interlinear-en-greta-emb-sum-normalized-ob`|GreTa|emb-sum|Oblubienica|Normalized|52.39|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-diacritics-ob`|mT5-base|emb-auto|Oblubienica|Diacritics|52.37|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-diacritics-ob)|
|`interlinear-en-mt5-base-emb-sum-diacritics-bh`|mT5-base|emb-sum|Bible Hub|Diacritics|52.34|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-diacritics-bh)|
|`interlinear-en-greta-emb-sum-normalized-bh`|GreTa|emb-sum|Bible Hub|Normalized|51.93|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-sum-normalized-bh)|
|`interlinear-en-mt5-base-emb-sum-diacritics-ob`|mT5-base|emb-sum|Oblubienica|Diacritics|51.90|0.86|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-diacritics-ob)|
|`interlinear-en-mt5-large-emb-concat-normalized-ob`|mT5-large|emb-concat|Oblubienica|Normalized|51.04|0.85|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-normalized-ob)|
|`interlinear-en-mt5-large-emb-concat-diacritics-bh`|mT5-large|emb-concat|Bible Hub|Diacritics|50.47|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-base-emb-sum-normalized-ob`|mT5-base|emb-sum|Oblubienica|Normalized|47.95|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-normalized-ob)|
|`interlinear-en-mt5-base-emb-auto-normalized-bh`|mT5-base|emb-auto|Bible Hub|Normalized|47.84|0.84|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-normalized-bh)|
|`interlinear-en-philta-emb-concat-normalized-bh`|PhilTa|emb-concat|Bible Hub|Normalized|46.82|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-normalized-bh)|
|`interlinear-en-mt5-large-t-w-t-diacritics-bh`|mT5-large|t-w-t (tags-within-text)|Bible Hub|Diacritics|46.00|0.83|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-diacritics-bh)|
|`interlinear-en-mt5-large-t-w-t-diacritics-ob`|mT5-large|t-w-t (tags-within-text)|Oblubienica|Diacritics|45.59|0.83|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-diacritics-ob)|
|`interlinear-en-philta-emb-concat-diacritics-ob`|PhilTa|emb-concat|Oblubienica|Diacritics|45.43|0.83|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-diacritics-ob)|
|`interlinear-en-mt5-large-baseline-diacritics-unused`|mT5-large|baseline (text only)|Unused|Diacritics|44.67|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-baseline-diacritics-unused)|
|`interlinear-en-mt5-large-t-w-t-normalized-bh`|mT5-large|t-w-t (tags-within-text)|Bible Hub|Normalized|43.97|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-normalized-bh)|
|`interlinear-en-mt5-large-baseline-normalized-unused`|mT5-large|baseline (text only)|Unused|Normalized|43.64|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-baseline-normalized-unused)|
|`interlinear-en-mt5-base-emb-concat-diacritics-ob`|mT5-base|emb-concat|Oblubienica|Diacritics|42.59|0.80|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-diacritics-ob)|
|`interlinear-en-philta-baseline-diacritics-unused`|PhilTa|baseline (text only)|Unused|Diacritics|41.55|0.83|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-baseline-diacritics-unused)|
|`interlinear-en-mt5-large-emb-concat-diacritics-ob`|mT5-large|emb-concat|Oblubienica|Diacritics|41.18|0.77|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-diacritics-ob)|
|`interlinear-en-philta-t-w-t-diacritics-bh`|PhilTa|t-w-t (tags-within-text)|Bible Hub|Diacritics|40.95|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-diacritics-bh)|
|`interlinear-en-philta-t-w-t-diacritics-ob`|PhilTa|t-w-t (tags-within-text)|Oblubienica|Diacritics|40.84|0.82|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-diacritics-ob)|
|`interlinear-en-philta-emb-concat-normalized-ob`|PhilTa|emb-concat|Oblubienica|Normalized|40.76|0.78|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-emb-concat-normalized-ob)|
|`interlinear-en-mt5-large-t-w-t-normalized-ob`|mT5-large|t-w-t (tags-within-text)|Oblubienica|Normalized|35.47|0.78|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-t-w-t-normalized-ob)|
|`interlinear-en-philta-t-w-t-normalized-bh`|PhilTa|t-w-t (tags-within-text)|Bible Hub|Normalized|34.25|0.76|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-normalized-bh)|
|`interlinear-en-philta-t-w-t-normalized-ob`|PhilTa|t-w-t (tags-within-text)|Oblubienica|Normalized|33.44|0.76|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-t-w-t-normalized-ob)|
|`interlinear-en-philta-baseline-normalized-unused`|PhilTa|baseline (text only)|Unused|Normalized|33.24|0.74|[🤗](https://huggingface.co/mrapacz/interlinear-en-philta-baseline-normalized-unused)|
|`interlinear-en-mt5-base-baseline-diacritics-unused`|mT5-base|baseline (text only)|Unused|Diacritics|31.61|0.74|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-baseline-diacritics-unused)|
|`interlinear-en-mt5-base-t-w-t-diacritics-bh`|mT5-base|t-w-t (tags-within-text)|Bible Hub|Diacritics|30.11|0.74|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-diacritics-bh)|
|`interlinear-en-mt5-base-baseline-normalized-unused`|mT5-base|baseline (text only)|Unused|Normalized|29.99|0.74|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-baseline-normalized-unused)|
|`interlinear-en-mt5-base-t-w-t-diacritics-ob`|mT5-base|t-w-t (tags-within-text)|Oblubienica|Diacritics|29.62|0.74|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-diacritics-ob)|
|`interlinear-en-mt5-base-emb-auto-diacritics-bh`|mT5-base|emb-auto|Bible Hub|Diacritics|28.52|0.71|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-auto-diacritics-bh)|
|`interlinear-en-mt5-base-t-w-t-normalized-ob`|mT5-base|t-w-t (tags-within-text)|Oblubienica|Normalized|28.39|0.73|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-normalized-ob)|
|`interlinear-en-mt5-base-t-w-t-normalized-bh`|mT5-base|t-w-t (tags-within-text)|Bible Hub|Normalized|27.59|0.72|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-t-w-t-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-normalized-bh`|mT5-base|emb-concat|Bible Hub|Normalized|27.32|0.68|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-normalized-bh)|
|`interlinear-en-greta-baseline-diacritics-unused`|GreTa|baseline (text only)|Unused|Diacritics|17.69|0.56|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-baseline-diacritics-unused)|
|`interlinear-en-greta-baseline-normalized-unused`|GreTa|baseline (text only)|Unused|Normalized|16.77|0.56|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-baseline-normalized-unused)|
|`interlinear-en-greta-t-w-t-normalized-bh`|GreTa|t-w-t (tags-within-text)|Bible Hub|Normalized|16.13|0.56|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-normalized-bh)|
|`interlinear-en-greta-t-w-t-diacritics-bh`|GreTa|t-w-t (tags-within-text)|Bible Hub|Diacritics|14.70|0.55|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-diacritics-bh)|
|`interlinear-en-greta-t-w-t-diacritics-ob`|GreTa|t-w-t (tags-within-text)|Oblubienica|Diacritics|14.51|0.55|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-diacritics-ob)|
|`interlinear-en-greta-t-w-t-normalized-ob`|GreTa|t-w-t (tags-within-text)|Oblubienica|Normalized|12.14|0.53|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-t-w-t-normalized-ob)|
|`interlinear-en-greta-emb-concat-diacritics-ob`|GreTa|emb-concat|Oblubienica|Diacritics|5.48|0.49|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-diacritics-ob)|
|`interlinear-en-greta-emb-concat-normalized-bh`|GreTa|emb-concat|Bible Hub|Normalized|4.05|0.42|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-normalized-bh)|
|`interlinear-en-greta-emb-concat-normalized-ob`|GreTa|emb-concat|Oblubienica|Normalized|3.93|0.42|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-normalized-ob)|
|`interlinear-en-greta-emb-concat-diacritics-bh`|GreTa|emb-concat|Bible Hub|Diacritics|3.58|0.42|[🤗](https://huggingface.co/mrapacz/interlinear-en-greta-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-base-emb-sum-normalized-bh`|mT5-base|emb-sum|Bible Hub|Normalized|1.66|0.38|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-sum-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-diacritics-bh`|mT5-base|emb-concat|Bible Hub|Diacritics|1.33|0.34|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-diacritics-bh)|
|`interlinear-en-mt5-large-emb-sum-diacritics-ob`|mT5-large|emb-sum|Oblubienica|Diacritics|0.83|0.34|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-sum-diacritics-ob)|
|`interlinear-en-mt5-large-emb-concat-normalized-bh`|mT5-large|emb-concat|Bible Hub|Normalized|0.70|0.37|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-large-emb-concat-normalized-bh)|
|`interlinear-en-mt5-base-emb-concat-normalized-ob`|mT5-base|emb-concat|Oblubienica|Normalized|0.69|0.34|[🤗](https://huggingface.co/mrapacz/interlinear-en-mt5-base-emb-concat-normalized-ob)|
