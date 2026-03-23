"""Unit tests for the evaluation metrics module."""

import pytest
from mrm.evaluation.metrics import (
    weighted_f1,
    macro_f1,
    accuracy_score,
    compute_forgetting_percentage,
    size_reduction_pct,
    speedup_ratio,
    parameter_count,
)
import torch
import torch.nn as nn


def test_accuracy_perfect():
    assert accuracy_score([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_accuracy_zero():
    assert accuracy_score([0, 1, 2], [1, 2, 0]) == pytest.approx(0.0)


def test_weighted_f1_perfect():
    assert weighted_f1([0, 1, 2, 2], [0, 1, 2, 2]) == pytest.approx(1.0)


def test_weighted_f1_binary():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    score = weighted_f1(y_true, y_pred)
    assert 0.0 <= score <= 1.0


def test_forgetting_percentage_correct():
    acc_before = 0.85
    acc_after = 0.60
    expected = (0.85 - 0.60) / 0.85 * 100
    result = compute_forgetting_percentage(acc_before, acc_after)
    assert result == pytest.approx(expected)


def test_forgetting_percentage_no_forgetting():
    assert compute_forgetting_percentage(0.80, 0.80) == pytest.approx(0.0)


def test_forgetting_percentage_zero_before():
    assert compute_forgetting_percentage(0.0, 0.5) == pytest.approx(0.0)


def test_size_reduction_pct():
    teacher = 559_890_432
    student = 68_937_216
    pct = size_reduction_pct(teacher, student)
    assert 87.0 <= pct <= 88.0  # paper reports 87.69%


def test_speedup_ratio():
    ratio = speedup_ratio(293.9, 14.0)
    assert 20.0 <= ratio <= 21.5  # paper reports ~20x speedup


def test_speedup_ratio_zero_student():
    assert speedup_ratio(100.0, 0.0) == float("inf")


def test_parameter_count():
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
    expected = 10 * 20 + 20 + 20 * 5 + 5
    assert parameter_count(model) == expected
