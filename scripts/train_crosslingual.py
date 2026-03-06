#!/usr/bin/env python
"""End-to-end Paper 1 cross-lingual transfer experiments.

Usage:
    # Train AfriBERTa on Kinyarwanda, zero-shot eval on Kirundi, then fine-tune:
    python scripts/train_crosslingual.py --model afriberta --phase all

    # Single transformer, only source training:
    python scripts/train_crosslingual.py --model mbert --phase train_source

    # Traditional BiGRU baseline:
    python scripts/train_crosslingual.py --model bigru --phase all

    # Dry run (1 batch, no save):
    python scripts/train_crosslingual.py --model afriberta --phase all --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrm.data.datasets import HFTextDataset, SimpleVocabDataset
from mrm.data.preprocessing import build_word2vec_embeddings
from mrm.models.transformer_clf import load_transformer_classifier, make_training_args, SUPPORTED_MODELS
from mrm.models.bigru import BiGRUClassifier, train_epoch, evaluate_epoch
from mrm.models.cnn_text import TextCNNClassifier
from mrm.models.cnn_text import train_epoch as cnn_train, evaluate_epoch as cnn_eval
from mrm.models.char_cnn import CharCNNClassifier, CharDataset
from mrm.training.trainer import build_hf_trainer, compute_metrics
from mrm.training.callbacks import ForgettingMonitor
from mrm.evaluation.metrics import weighted_f1, accuracy_score, compute_forgetting_percentage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/crosslingual.yaml")
    p.add_argument(
        "--model",
        choices=list(SUPPORTED_MODELS.keys()) + ["bigru", "cnn", "charcnn", "all"],
        default="afriberta",
    )
    p.add_argument(
        "--phase",
        choices=["train_source", "zero_shot", "finetune_target", "all"],
        default="all",
    )
    p.add_argument("--dry-run", action="store_true", help="run 1 step to verify setup")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_transformer(cfg: dict, model_name: str, phase: str, device: str, dry_run: bool):
    tcfg = cfg["transformers"]
    model, tokenizer = load_transformer_classifier(model_name, num_labels=tcfg["num_labels"])
    model.to(device)

    kin_train = HFTextDataset.from_csv(cfg["data"]["kinnews_train"], tokenizer, tcfg["max_length"])
    kin_test = HFTextDataset.from_csv(cfg["data"]["kinnews_test"], tokenizer, tcfg["max_length"])
    kir_train = HFTextDataset.from_csv(cfg["data"]["kirnews_train"], tokenizer, tcfg["max_length"])
    kir_test = HFTextDataset.from_csv(cfg["data"]["kirnews_test"], tokenizer, tcfg["max_length"])

    out_dir = f"{tcfg['output_dir']}/{model_name}"

    if dry_run:
        kin_train = torch.utils.data.Subset(kin_train, range(min(8, len(kin_train))))
        kin_test = torch.utils.data.Subset(kin_test, range(min(8, len(kin_test))))
        kir_test = torch.utils.data.Subset(kir_test, range(min(8, len(kir_test))))
        kir_train = torch.utils.data.Subset(kir_train, range(min(8, len(kir_train))))
        tcfg = {**tcfg, "num_epochs": 1}

    forgetting_monitor = ForgettingMonitor(kin_test, device=device)

    if phase in ("train_source", "all"):
        logger.info("=== phase: train on Kinyarwanda ===")
        args = make_training_args(
            f"{out_dir}/source",
            num_epochs=tcfg["num_epochs"],
            batch_size=tcfg["batch_size"],
            lr=tcfg["learning_rate"],
            warmup_steps=tcfg["warmup_steps"],
            weight_decay=tcfg["weight_decay"],
            fp16=tcfg.get("fp16", False),
        )
        trainer = build_hf_trainer(model, kin_train, kin_test, args, compute_metrics)
        trainer.train()
        logger.info("source training complete")

    if phase in ("zero_shot", "all"):
        logger.info("=== phase: zero-shot eval on Kirundi ===")
        forgetting_monitor.snapshot_source_accuracy(model)
        # zero-shot: evaluate on Kirundi without any Kirundi training
        trainer_eval = build_hf_trainer(
            model, kir_test, kir_test,
            make_training_args(f"{out_dir}/tmp"), compute_metrics,
        )
        result = trainer_eval.evaluate(kir_test)
        logger.info("zero-shot Kirundi: %s", result)

    if phase in ("finetune_target", "all"):
        logger.info("=== phase: fine-tune on Kirundi ===")
        forgetting_monitor.snapshot_source_accuracy(model)
        args_ft = make_training_args(
            f"{out_dir}/finetuned",
            num_epochs=max(1, tcfg["num_epochs"] // 2),
            batch_size=tcfg["batch_size"],
            lr=tcfg["learning_rate"] * 0.5,
        )
        trainer_ft = build_hf_trainer(model, kir_train, kir_test, args_ft, compute_metrics)
        trainer_ft.train()
        ft_result = trainer_ft.evaluate(kir_test)
        logger.info("fine-tuned Kirundi: %s", ft_result)
        forgetting = forgetting_monitor.compute_forgetting(model)
        logger.info("catastrophic forgetting: %.2f%%", forgetting["forgetting_pct"])


def run_bigru(cfg: dict, device: str, dry_run: bool):
    bcfg = cfg["traditional"]["bigru"]
    train_ds, test_kin = SimpleVocabDataset.build_pair(
        cfg["data"]["kinnews_train"], cfg["data"]["kinnews_test"],
    )
    _, test_kir = SimpleVocabDataset.build_pair(
        cfg["data"]["kirnews_train"], cfg["data"]["kirnews_test"],
        vocab=train_ds.vocab if hasattr(train_ds, "vocab") else None,
    )

    emb_path = cfg["data"]["embeddings_kin"]
    pretrained_vecs = None
    if Path(emb_path).exists():
        pretrained_vecs = train_ds.load_pretrained_embeddings(emb_path, bcfg["embedding_dim"])

    model = BiGRUClassifier(
        vocab_size=len(train_ds.vocab),
        embedding_dim=bcfg["embedding_dim"],
        hidden_dim=bcfg["hidden_dim"],
        output_dim=cfg["transformers"]["num_labels"],
        n_layers=bcfg["n_layers"],
        dropout=bcfg["dropout"],
    ).to(device)

    if pretrained_vecs is not None:
        model.load_pretrained_embeddings(pretrained_vecs.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=bcfg["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    n_epochs = 1 if dry_run else bcfg["epochs"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bcfg["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_kin, batch_size=bcfg["batch_size"])

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate_epoch(model, test_loader, criterion, device)
        logger.info("BiGRU epoch %d — train_loss %.4f train_acc %.4f val_acc %.4f",
                    epoch + 1, tr_loss, tr_acc, vl_acc)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    models_to_run = list(SUPPORTED_MODELS.keys()) + ["bigru", "cnn", "charcnn"] if args.model == "all" else [args.model]

    for model_name in models_to_run:
        logger.info("=== model: %s ===", model_name)
        if model_name in SUPPORTED_MODELS:
            run_transformer(cfg, model_name, args.phase, args.device, args.dry_run)
        elif model_name == "bigru":
            run_bigru(cfg, args.device, args.dry_run)
        else:
            logger.warning("skipping %s — use notebook for full traditional model run", model_name)


if __name__ == "__main__":
    main()
