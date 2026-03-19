#!/usr/bin/env python
"""End-to-end Paper 2 MAGE augmentation experiments.

Usage:
    python scripts/train_mage.py --language kin --model mage_dae
    python scripts/train_mage.py --language swa --model lstm_baseline
    python scripts/train_mage.py --language kin --model lr_baseline
    python scripts/train_mage.py --language kin --model all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrm.data.datasets import HFTextDataset, EmbeddingDataset
from mrm.augmentation.lida import LiDA
from mrm.models.autoencoder import DenoisingAutoencoder, VariationalAutoencoder, train_autoencoder
from mrm.models.mage import MAGEClassifier, LSTMClassifier, LogisticRegressionWrapper
from mrm.models.transformer_clf import load_transformer_classifier
from mrm.training.callbacks import EarlyStopping
from mrm.evaluation.metrics import weighted_f1, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mage.yaml")
    p.add_argument("--language", default="kin", choices=["kin", "swa", "tso", "hau", "ibo", "yor"])
    p.add_argument(
        "--model",
        choices=["mage_dae", "mage_vae", "lstm_baseline", "lr_baseline", "all"],
        default="mage_dae",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_embeddings(cfg: dict, lang: str, device: str, dry_run: bool) -> tuple:
    """Load AfriSenti data, extract AfriBERTa embeddings, apply LiDA augmentation."""
    import pandas as pd

    emb_model_name = cfg["embedding_model"]
    emb_model, tokenizer = load_transformer_classifier(emb_model_name, num_labels=3)
    emb_model = emb_model.base_model  # use encoder only, no classification head
    emb_model.to(device).eval()

    train_path = f"data/processed/afrisenti/{lang}_train.csv"
    test_path = f"data/processed/afrisenti/{lang}_test.csv"

    if not Path(train_path).exists():
        logger.warning("data not found — run scripts/download_data.py first")
        # Return tiny synthetic data for dry-run/testing
        n = 8 if dry_run else 16
        return (
            torch.randn(n * 2, 768), torch.randint(0, 3, (n * 2,)),
            torch.randn(n, 768), torch.randint(0, 3, (n,)),
        )

    train_ds = HFTextDataset.from_csv(train_path, tokenizer, cfg["max_length"])
    test_ds = HFTextDataset.from_csv(test_path, tokenizer, cfg["max_length"])
    if dry_run:
        train_ds = torch.utils.data.Subset(train_ds, range(8))
        test_ds = torch.utils.data.Subset(test_ds, range(8))

    lida = LiDA(r_min=cfg["lida"]["r_min"], r_max=cfg["lida"]["r_max"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    train_emb, train_labels = lida.augment_dataset(
        emb_model, train_loader, device=device, n_augmented=cfg["lida"]["n_augmented"]
    )
    test_emb, test_labels = lida.augment_dataset(
        emb_model, test_loader, device=device, n_augmented=0
    )

    return train_emb, train_labels, test_emb, test_labels


def run_mage(cfg: dict, variant: str, train_emb, train_labels, test_emb, test_labels, device: str, dry_run: bool):
    mcfg = cfg["mage"]
    ae_cfg = cfg["dae"] if "dae" in variant else cfg["vae"]
    is_vae = "vae" in variant

    logger.info("training %s autoencoder …", "VAE" if is_vae else "DAE")
    ae_cls = VariationalAutoencoder if is_vae else DenoisingAutoencoder
    ae_kwargs = {k: v for k, v in ae_cfg.items() if k in ("input_dim", "latent_dim", "hidden_dim", "dropout") and v is not None}
    ae = ae_cls(**ae_kwargs)
    n_epochs = 2 if dry_run else ae_cfg["epochs"]
    train_autoencoder(ae, train_emb, num_epochs=n_epochs, lr=ae_cfg["lr"],
                      batch_size=ae_cfg["batch_size"], device=device, is_vae=is_vae)

    # Generate refined embeddings via AE bottleneck
    ae.to(device).eval()
    with torch.no_grad():
        if is_vae:
            refined_train, _, _ = ae(train_emb.to(device))
            refined_test, _, _ = ae(test_emb.to(device))
        else:
            refined_train = ae(train_emb.to(device))
            refined_test = ae(test_emb.to(device))
    refined_train = refined_train.cpu()
    refined_test = refined_test.cpu()

    logger.info("training MAGE classifier …")
    model = MAGEClassifier(embed_dim=mcfg["embed_dim"], num_heads=mcfg["num_heads"],
                           num_classes=mcfg["num_classes"], dropout=mcfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=3, metric="f1")

    train_loader = torch.utils.data.DataLoader(
        EmbeddingDataset(refined_train, train_labels),
        batch_size=mcfg["batch_size"], shuffle=True,
    )
    n_epochs = 2 if dry_run else mcfg["epochs"]
    for epoch in range(n_epochs):
        model.train()
        for embs, lbls in train_loader:
            embs, lbls = embs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(embs), lbls)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(refined_test.to(device)).argmax(dim=1).cpu().tolist()
    labels = test_labels.tolist()
    f1 = weighted_f1(labels, preds)
    acc = accuracy_score(labels, preds)
    logger.info("MAGE+%s — F1: %.4f  Acc: %.4f", "VAE" if is_vae else "DAE", f1, acc)
    return {"f1": f1, "accuracy": acc}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))

    logger.info("extracting AfriBERTa embeddings for language: %s", args.language)
    train_emb, train_labels, test_emb, test_labels = extract_embeddings(
        cfg, args.language, args.device, args.dry_run
    )

    variants = ["mage_dae", "mage_vae", "lstm_baseline", "lr_baseline"] if args.model == "all" else [args.model]
    results = {}

    for variant in variants:
        logger.info("=== %s ===", variant)
        if variant in ("mage_dae", "mage_vae"):
            results[variant] = run_mage(cfg, variant, train_emb, train_labels, test_emb, test_labels, args.device, args.dry_run)
        elif variant == "lstm_baseline":
            lcfg = cfg["lstm_baseline"]
            model = LSTMClassifier(num_classes=lcfg["num_classes"]).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lcfg["lr"])
            criterion = torch.nn.CrossEntropyLoss()
            loader = torch.utils.data.DataLoader(
                EmbeddingDataset(train_emb, train_labels), batch_size=lcfg["batch_size"], shuffle=True
            )
            n_ep = 2 if args.dry_run else lcfg["epochs"]
            for _ in range(n_ep):
                for embs, lbls in loader:
                    optimizer.zero_grad()
                    criterion(model(embs.to(args.device)), lbls.to(args.device)).backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                preds = model(test_emb.to(args.device)).argmax(1).cpu().tolist()
            f1 = weighted_f1(test_labels.tolist(), preds)
            results[variant] = {"f1": f1}
            logger.info("LSTM baseline — F1: %.4f", f1)
        elif variant == "lr_baseline":
            lr_model = LogisticRegressionWrapper()
            lr_model.fit(train_emb, train_labels)
            preds = lr_model.predict(test_emb)
            f1 = weighted_f1(test_labels.tolist(), preds.tolist())
            results[variant] = {"f1": f1}
            logger.info("LR baseline — F1: %.4f", f1)

    logger.info("=== summary for language %s ===", args.language)
    for name, res in results.items():
        logger.info("  %s: %s", name, res)


if __name__ == "__main__":
    main()
