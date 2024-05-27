# multilingual-resource-models

Research code for: **A Unified Framework for Low-Resource African Natural Language Processing: Cross-Lingual Transfer, Embedding Augmentation, and Knowledge Distillation**

The full paper is available at [paper/ieee_conference_template.pdf](paper/ieee_conference_template.pdf).

---

## Overview

This repository implements three complementary techniques for improving NLP on low-resource African languages:

1. **Cross-Lingual Transfer Benchmark** — zero-shot transfer from Kinyarwanda to Kirundi (KINNEWS corpus, 14-class classification) across six architectures: mBERT, AfriBERTa, BantuBERTa, BiGRU, TextCNN, and CharCNN. Introduces a catastrophic forgetting percentage (CF%) metric.

2. **MAGE: Multi-Head Attention Guided Embeddings** — language-independent augmentation pipeline for low-resource sentiment analysis on AfriSenti tweets, combining LiDA noise augmentation with DAE/VAE refinement and a 4-head attention classifier.

3. **AfroXLMR-Comet: Knowledge Distillation** — distils AfroXLMR-Large (559.9M params) into AfroXLMR-Comet (68.9M params) using KL divergence on soft outputs and MSE on attention maps, achieving 87.7% size reduction and 20x inference speedup.

---

## Model Comparison

| Model | Task | F1 (%) | Params | Inference |
|---|---|---|---|---|
| AfriBERTa (fine-tuned) | News Classification (Kirundi) | 87.87 | 126M | ~150 ms |
| BantuBERTa (fine-tuned) | News Classification (Kirundi) | 86.06 | 126M | ~160 ms |
| mBERT (fine-tuned) | News Classification (Kirundi) | 84.22 | 178M | ~200 ms |
| BiGRU | News Classification (Kirundi) | 87.90 | ~2M | ~15 ms |
| TextCNN | News Classification (Kirundi) | 57.32 | ~1M | ~8 ms |
| CharCNN | News Classification (Kirundi) | 47.64 | ~3M | ~10 ms |
| MAGE+DAE (avg) | Sentiment (AfriSenti) | 56.86 | 126M+ | ~160 ms |
| MAGE+VAE (avg) | Sentiment (AfriSenti) | 55.81 | 126M+ | ~165 ms |
| AfroXLMR-Large (teacher) | Sentiment (AfriSenti) | 73.22 | 559.9M | 293.9 ms |
| AfroXLMR-Comet (student) | Sentiment (AfriSenti) | 65.28 | 68.9M | **14.0 ms** |

---

## Quick Start

```bash
git clone https://github.com/omprxkash/multilingual-resource-models
cd multilingual-resource-models
pip install -e ".[dev]"

# Download datasets (KINNEWS, AfriSenti)
python scripts/download_data.py

# Cross-lingual transfer
python scripts/train_crosslingual.py --config configs/crosslingual.yaml --model afriberta --phase all

# MAGE augmentation + sentiment
python scripts/train_mage.py --config configs/mage.yaml --language kin --model mage_dae

# Knowledge distillation
python scripts/train_distill.py --config configs/distillation.yaml --language kin --phase all

# Run tests
python -m pytest tests/ -v
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Dataset statistics, class distributions, vocabulary overlap |
| [02_crosslingual_transfer.ipynb](notebooks/02_crosslingual_transfer.ipynb) | Full benchmark with forgetting analysis |
| [03_mage_augmentation.ipynb](notebooks/03_mage_augmentation.ipynb) | LiDA, DAE/VAE, MAGE training and ablation |
| [04_knowledge_distillation.ipynb](notebooks/04_knowledge_distillation.ipynb) | Distillation training and efficiency benchmarking |
| [05_results_and_figures.ipynb](notebooks/05_results_and_figures.ipynb) | All tables and figures |

---

## Project Structure

```
multilingual-resource-models/
├── src/mrm/
│   ├── data/           download, preprocessing, dataset wrappers
│   ├── models/         all 9 model architectures
│   ├── augmentation/   LiDA
│   ├── training/       Trainer, distillation loop, callbacks
│   └── evaluation/     metrics, evaluation pipelines
├── scripts/            training entry points
├── configs/            YAML hyperparameter files
├── notebooks/          5 Jupyter notebooks
├── tests/              unit tests
└── paper/              IEEE paper (PDF + LaTeX source)
```
