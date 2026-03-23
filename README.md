# multilingual-resource-models

Research code for: **A Unified Framework for Low-Resource African Natural Language Processing:
Cross-Lingual Transfer, Embedding Augmentation, and Knowledge Distillation**

Developed as part of a collaborative research initiative with the
[Data Science for Social Impact (DSFSI) Laboratory](https://dsfsi.github.io/),
University of Pretoria, under the supervision of Prof. Vukosi Marivate.
This project is led by a team of eight researchers working on open-source
multilingual NLP models for underserved African language communities.

---

## Why This Work Matters

Africa is home to more than 2,000 living languages, yet the overwhelming majority
are absent from modern NLP benchmarks. Languages like Kinyarwanda, Kirundi, Swahili,
Hausa, Igbo, and Yoruba are spoken by hundreds of millions of people, but their speakers
are excluded from AI-powered services that depend on language understanding—search,
content moderation, health information, and educational tools among them. I started this
project to directly address that gap.

The first problem is **data scarcity**. Annotated corpora for Bantu languages are
orders of magnitude smaller than those available for English or Mandarin. Training a
model from scratch is impractical; instead, we must transfer knowledge from related
high-resource settings and augment the limited data we do have.

The second problem is **cross-lingual transfer reliability**. Even closely related
languages like Kinyarwanda and Kirundi are not interchangeable: their shared vocabulary
is partial, and models trained on one language degrade unpredictably on the other. I
wanted to quantify exactly how much knowledge is transferred and how much is lost—what
I call the catastrophic forgetting metric.

The third problem is **deployment at scale**. State-of-the-art multilingual models
contain hundreds of millions of parameters and require GPUs to run at usable speed.
In Sub-Saharan Africa, where compute infrastructure is limited and connectivity is
intermittent, a 2-gigabyte model is not a viable tool. This project distils those large
models into compact equivalents that can run on modest hardware without sacrificing
most of the accuracy.

---

## Three Research Contributions

### 1. Cross-Lingual Transfer Benchmark

I benchmark zero-shot cross-lingual transfer from Kinyarwanda to Kirundi on the KINNEWS
corpus (14-class news classification). Six architectures are evaluated: mBERT, AfriBERTa,
BantuBERTa, BiGRU, TextCNN, and CharCNN. A novel **catastrophic forgetting percentage
(CF%)** tracks how much source-language accuracy is lost after target-language fine-tuning.

AfriBERTa achieves the best accuracy at **88.3%** with only **5.1% catastrophic forgetting**,
confirming that Africa-specific pretraining strongly outperforms generic multilingual models
for Bantu language transfer.

```bash
python scripts/train_crosslingual.py --model afriberta --phase all
```

### 2. MAGE: Language-Independent Augmentation + Attention Classifier

I implement the MAGE (Multi-Head Attention Guided Embeddings) pipeline for low-resource
sentiment analysis on AfriSenti tweets. The pipeline extracts AfriBERTa sentence embeddings,
applies LiDA (embedding-space noise augmentation), refines embeddings through a Denoising
Autoencoder or Variational Autoencoder, and classifies with a 4-head attention ensemble.

MAGE+DAE achieves a **+3.64 percentage-point** improvement in weighted F1 over the logistic
regression baseline—without any language-specific resources.

```bash
python scripts/train_mage.py --language kin --model mage_dae
```

### 3. AfroXLMR-Comet: Knowledge Distillation

I distil AfroXLMR-Large (559.9M parameters, 2.1 GB) into AfroXLMR-Comet (68.9M parameters,
263 MB) using a hybrid loss: KL divergence on soft output probabilities (temperature T=2)
plus MSE on mean-pooled attention maps (alpha=0.5). AfroXLMR-Comet is **87.7% smaller**
and **20x faster** while retaining **89.2% of teacher F1** across five African languages.

```bash
python scripts/train_distill.py --phase all --language kin
```

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

# Download all datasets (KINNEWS, AfriSenti)
python scripts/download_data.py

# Paper 1: cross-lingual transfer (all 6 models)
python scripts/train_crosslingual.py --config configs/crosslingual.yaml --model afriberta --phase all

# Paper 2: MAGE augmentation + sentiment
python scripts/train_mage.py --config configs/mage.yaml --language kin --model mage_dae

# Paper 3: knowledge distillation
python scripts/train_distill.py --config configs/distillation.yaml --language kin --phase all

# Run unit tests
python -m pytest tests/ -v
```

---

## Notebooks

| Notebook | Description |
|---|---|
| [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb) | Dataset statistics, class distributions, vocabulary overlap |
| [02_crosslingual_transfer.ipynb](notebooks/02_crosslingual_transfer.ipynb) | Paper 1: full benchmark with forgetting analysis |
| [03_mage_augmentation.ipynb](notebooks/03_mage_augmentation.ipynb) | Paper 2: LiDA, DAE/VAE, MAGE training and ablation |
| [04_knowledge_distillation.ipynb](notebooks/04_knowledge_distillation.ipynb) | Paper 3: distillation training and efficiency benchmarking |
| [05_results_and_figures.ipynb](notebooks/05_results_and_figures.ipynb) | All tables and figures for the IEEE paper |

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
└── paper/              IEEE LaTeX paper + BibTeX
```

---

## Citation

If you use this work, please cite the three underlying papers:

```bibtex
@article{thangaraj2024crosslingual,
  title   = {Cross-lingual transfer of multilingual models on low resource African languages},
  author  = {Thangaraj, Harish and Chenat, Ananya and Walia, Jaskaran Singh and Marivate, Vukosi},
  journal = {arXiv:2409.10965},
  year    = {2024}
}

@article{vashisht2025mage,
  title   = {MAGE: Multi-head attention guided embeddings for low resource sentiment classification},
  author  = {Vashisht, Varun and Singh, Samar and Konduskar, Mihir and Walia, Jaskaran Singh and Marivate, Vukosi},
  journal = {arXiv:2502.17987},
  year    = {2025}
}

@article{raju2025comet,
  title   = {AfroXLMR-Comet: Multilingual knowledge distillation with attention matching for low-resource languages},
  author  = {Raju, Joshua Sakthivel and S, Sanjay and Walia, Jaskaran Singh and Raghav, Srinivas and Marivate, Vukosi},
  journal = {arXiv:2502.18020},
  year    = {2025}
}
```

---

## Acknowledgements

DSFSI Research Laboratory, University of Pretoria.
Prof. Vukosi Marivate for research direction and mentorship.
The Masakhane NLP community for open-sourcing African language datasets and models.
