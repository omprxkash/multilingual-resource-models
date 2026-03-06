"""HuggingFace Trainer wrapper with standard compute_metrics for all papers."""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import Trainer


def compute_metrics(eval_pred) -> dict:
    """Standard compute_metrics for HuggingFace Trainer.

    Returns accuracy and weighted F1 — the two metrics reported across all
    three papers in the benchmark tables.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def build_hf_trainer(
    model,
    train_dataset,
    eval_dataset,
    training_args,
    compute_metrics_fn=None,
) -> Trainer:
    """Thin wrapper around HuggingFace Trainer with sensible defaults."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn or compute_metrics,
    )
