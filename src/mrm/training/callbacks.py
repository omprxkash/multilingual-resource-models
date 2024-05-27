"""Training callbacks: ForgettingMonitor and EarlyStopping.

ForgettingMonitor implements the novel catastrophic-forgetting metric from
Paper 1: it snapshots source-language (Kinyarwanda) accuracy before and
after target-language (Kirundi) fine-tuning, then reports the percentage drop.
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ForgettingMonitor:
    """Track source-language accuracy before and after target fine-tuning.

    Catastrophic forgetting % = (acc_before - acc_after) / acc_before × 100

    Transformers typically show 3-5% forgetting; traditional models 70-75%.
    """

    def __init__(
        self,
        source_dataset,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
        self.device = device
        self._acc_before: float = None
        self._acc_after: float = None

    def _eval_accuracy(self, model) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.source_loader:
                if isinstance(batch, dict):
                    labels = batch.pop("labels").to(self.device)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = outputs.logits.argmax(dim=-1)
                else:
                    x, labels = batch
                    x, labels = x.to(self.device), labels.to(self.device)
                    preds = model(x).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def snapshot_source_accuracy(self, model) -> float:
        """Evaluate and store pre-fine-tuning source accuracy."""
        self._acc_before = self._eval_accuracy(model)
        logger.info("source accuracy before fine-tuning: %.4f", self._acc_before)
        return self._acc_before

    def compute_forgetting(self, model) -> dict:
        """Evaluate post-fine-tuning source accuracy and compute forgetting %."""
        if self._acc_before is None:
            raise RuntimeError("call snapshot_source_accuracy() first")
        self._acc_after = self._eval_accuracy(model)
        forgetting_pct = (self._acc_before - self._acc_after) / self._acc_before * 100
        result = {
            "acc_before": self._acc_before,
            "acc_after": self._acc_after,
            "forgetting_pct": forgetting_pct,
        }
        logger.info(
            "catastrophic forgetting: %.2f%% (%.4f → %.4f)",
            forgetting_pct, self._acc_before, self._acc_after,
        )
        return result


class EarlyStopping:
    """Patience-based early stopping, used in Paper 2 and Paper 3 training."""

    def __init__(self, patience: int = 3, metric: str = "eval_f1", greater_is_better: bool = True):
        self.patience = patience
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best: float = None
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metrics: dict) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.greater_is_better else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
