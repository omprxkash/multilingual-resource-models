"""Evaluation entry points for all three papers."""

import logging
import torch
from torch.utils.data import DataLoader

from mrm.evaluation.metrics import weighted_f1, accuracy_score, compute_forgetting_percentage

logger = logging.getLogger(__name__)


def evaluate_crosslingual(
    model,
    source_dataset,
    target_dataset,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Paper 1 evaluation: source + target accuracy/F1, plus forgetting %.

    Run this twice — before and after fine-tuning — and compute forgetting.
    """
    def _run(dataset):
        model.eval()
        all_preds, all_labels = [], []
        loader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    labels = batch.pop("labels")
                    inp = {k: v.to(device) for k, v in batch.items()}
                    preds = model(**inp).logits.argmax(-1).cpu()
                else:
                    x, labels = batch
                    preds = model(x.to(device)).argmax(-1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        return all_labels, all_preds

    src_labels, src_preds = _run(source_dataset)
    tgt_labels, tgt_preds = _run(target_dataset)

    return {
        "source_accuracy": accuracy_score(src_labels, src_preds),
        "source_f1": weighted_f1(src_labels, src_preds),
        "target_accuracy": accuracy_score(tgt_labels, tgt_preds),
        "target_f1": weighted_f1(tgt_labels, tgt_preds),
    }


def evaluate_sentiment(
    model,
    test_dataset,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Paper 2 / Paper 3 evaluation: sentiment F1 on AfriSenti test split."""
    model.eval()
    all_preds, all_labels = [], []
    loader = DataLoader(test_dataset, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                embs, labels = batch
                preds = model(embs.to(device)).argmax(-1).cpu()
            else:
                labels = batch.pop("labels")
                inp = {k: v.to(device) for k, v in batch.items()}
                preds = model(**inp).logits.argmax(-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": weighted_f1(all_labels, all_preds),
    }
