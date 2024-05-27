#!/usr/bin/env python
"""End-to-end Paper 3 knowledge distillation experiments.

Usage:
    # Stage 1: distill AfroXLMR-Large → AfroXLMR-Comet student
    python scripts/train_distill.py --phase distill --language kin

    # Stage 2: fine-tune distilled student on a specific language
    python scripts/train_distill.py --phase finetune --language swa \\
        --student-checkpoint outputs/distillation/student

    # Both stages sequentially:
    python scripts/train_distill.py --phase all --language kin

    # Dry run (verify setup without real training):
    python scripts/train_distill.py --phase all --language kin --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mrm.data.datasets import HFTextDataset
from mrm.models.distillation import AfroXLMRComet, TEACHER_MODEL
from mrm.models.transformer_clf import load_transformer_classifier, make_training_args
from mrm.training.distill_trainer import DistillationTrainer
from mrm.training.trainer import build_hf_trainer, compute_metrics
from mrm.evaluation.metrics import weighted_f1, parameter_count, size_reduction_pct, speedup_ratio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/distillation.yaml")
    p.add_argument("--phase", choices=["distill", "finetune", "all"], default="all")
    p.add_argument("--language", default="kin")
    p.add_argument("--student-checkpoint", default=None, help="path to saved student for finetune phase")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_tokenizer(teacher_name: str = TEACHER_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(teacher_name)


def load_data(cfg: dict, lang: str, tokenizer, dry_run: bool, split="train"):
    data_dir = cfg["data"]["afrisenti_dir"]
    path = f"{data_dir}/{lang}_{split}.csv"
    if not Path(path).exists():
        logger.warning("data not found at %s — using synthetic data for dry-run", path)
        from mrm.data.datasets import HFTextDataset
        texts = ["dummy text for testing"] * (8 if dry_run else 32)
        labels = [0] * len(texts)
        return HFTextDataset(texts, labels, tokenizer, cfg["data"]["max_length"])
    ds = HFTextDataset.from_csv(path, tokenizer, cfg["data"]["max_length"])
    if dry_run:
        ds = torch.utils.data.Subset(ds, range(min(8, len(ds))))
    return ds


def run_distillation(cfg: dict, lang: str, device: str, dry_run: bool) -> str:
    """Stage 1: distill teacher into student."""
    dcfg = cfg["distillation"]
    tokenizer = get_tokenizer(cfg["teacher_model"])

    logger.info("loading teacher model: %s", cfg["teacher_model"])
    teacher_model, _ = load_transformer_classifier(
        cfg["teacher_model"], num_labels=3
    )
    teacher_model.eval()

    logger.info("building student model (AfroXLMR-Comet)")
    student = AfroXLMRComet.from_scratch(teacher_name=cfg["teacher_model"], num_labels=3)

    n_teacher = parameter_count(teacher_model)
    n_student = student.num_parameters
    logger.info("teacher: %dM params | student: %dM params (%.1f%% reduction)",
                n_teacher // 1_000_000, n_student // 1_000_000,
                size_reduction_pct(n_teacher, n_student))

    train_ds = load_data(cfg, lang, tokenizer, dry_run, "train")
    test_ds = load_data(cfg, lang, tokenizer, dry_run, "test")

    batch_size = dcfg["batch_size"]
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size * 2)

    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        seq_len=cfg["data"]["max_length"],
        temperature=dcfg["temperature"],
        alpha=dcfg["alpha"],
        lr=dcfg["learning_rate"],
        num_epochs=2 if dry_run else dcfg["num_epochs"],
        grad_accum_steps=dcfg["grad_accum_steps"],
        patience=dcfg["patience"],
        fp16=dcfg["fp16"] and device.startswith("cuda"),
        device=device,
        output_dir=dcfg["output_dir"],
    )

    trainer.train()
    student_path = f"{dcfg['output_dir']}/student_{lang}"
    trainer.save_student(student_path)
    return student_path


def run_finetuning(cfg: dict, lang: str, device: str, dry_run: bool, student_checkpoint: str = None):
    """Stage 2: fine-tune distilled student on per-language AfriSenti data."""
    fcfg = cfg["finetuning"]
    tokenizer = get_tokenizer(cfg["teacher_model"])

    if student_checkpoint:
        from transformers import AutoModelForSequenceClassification
        import torch.nn as nn
        model = AutoModelForSequenceClassification.from_pretrained(student_checkpoint, num_labels=3)
    else:
        student = AfroXLMRComet.from_scratch(teacher_name=cfg["teacher_model"], num_labels=3)
        model = student.model

    train_ds = load_data(cfg, lang, tokenizer, dry_run, "train")
    test_ds = load_data(cfg, lang, tokenizer, dry_run, "test")

    args = make_training_args(
        f"{fcfg['output_dir']}/{lang}",
        num_epochs=2 if dry_run else fcfg["num_epochs"],
        batch_size=fcfg["batch_size"],
        lr=fcfg["learning_rate"],
        fp16=device.startswith("cuda"),
    )
    trainer = build_hf_trainer(model, train_ds, test_ds, args, compute_metrics)
    trainer.train()
    results = trainer.evaluate(test_ds)
    logger.info("fine-tuned student [%s] — F1: %.4f  Acc: %.4f",
                lang, results.get("eval_f1", 0), results.get("eval_accuracy", 0))
    return results


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))

    student_ckpt = args.student_checkpoint
    if args.phase in ("distill", "all"):
        student_ckpt = run_distillation(cfg, args.language, args.device, args.dry_run)

    if args.phase in ("finetune", "all"):
        run_finetuning(cfg, args.language, args.device, args.dry_run, student_ckpt)


if __name__ == "__main__":
    main()
