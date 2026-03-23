"""Evaluation metrics shared across all three papers."""

import time
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score as _accuracy_score


def weighted_f1(y_true: list, y_pred: list) -> float:
    """Weighted F1 score — primary metric for all three papers."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def macro_f1(y_true: list, y_pred: list) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy_score(y_true: list, y_pred: list) -> float:
    return float(_accuracy_score(y_true, y_pred))


def compute_forgetting_percentage(acc_before: float, acc_after: float) -> float:
    """Novel metric from Paper 1: percentage accuracy drop on source language.

    Measures how much a model forgets Kinyarwanda after fine-tuning on Kirundi.
    Transformers: ~3-5% forgetting.  Traditional models: ~70-75% forgetting.
    """
    if acc_before == 0:
        return 0.0
    return (acc_before - acc_after) / acc_before * 100.0


def parameter_count(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def size_reduction_pct(teacher_params: int, student_params: int) -> float:
    """Percentage reduction in parameter count from teacher to student."""
    return (teacher_params - student_params) / teacher_params * 100.0


def speedup_ratio(teacher_ms: float, student_ms: float) -> float:
    """Inference speedup factor: teacher_time / student_time."""
    return teacher_ms / student_ms if student_ms > 0 else float("inf")


def benchmark_inference_speed(
    model: torch.nn.Module,
    inputs: dict,
    n_runs: int = 100,
    device: str = "cpu",
) -> float:
    """Measure mean inference latency in milliseconds.

    Runs n_runs forward passes and returns the mean wall-clock time.
    Warms up for 10 runs first to avoid JIT compilation effects.
    """
    model.eval().to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(**inputs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(**inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))
