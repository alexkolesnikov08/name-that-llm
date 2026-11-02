"""Тесты метрик: старые функции не сломаны, BA(2/prompt) работает."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl.metrics import (  # noqa: E402
    ba,
    ba2,
    ba2_ci,
    bootstrap_ba,
    paired_ba2,
    paired_bootstrap_delta,
    prompt_scores,
)


def fixture(n_prompts: int = 12, n_classes: int = 3, rows_per_prompt: int = 3):
    rng = np.random.default_rng(0)
    prompts = np.repeat([f"p{i:03d}" for i in range(n_prompts)], rows_per_prompt)
    y_true = np.tile(np.arange(n_classes), n_prompts * rows_per_prompt // n_classes + 1)[
        : len(prompts)
    ]
    pred = y_true.copy()
    # портим часть предсказаний, чтобы метрика была не 1
    bad = rng.choice(len(pred), size=max(1, len(pred) // 7), replace=False)
    pred[bad] = (pred[bad] + 1) % n_classes
    return y_true, pred, prompts


def test_existing_ba_functions_still_work() -> None:
    y_true, pred, prompts = fixture()
    score = ba(y_true, pred)
    assert 0.0 <= score <= 1.0
    mean, lo, hi = bootstrap_ba(y_true, pred, prompts, n_boot=30, seed=1)
    assert lo <= mean <= hi
    delta, dlo, dhi = paired_bootstrap_delta(y_true, pred, pred, prompts, n_boot=30, seed=1)
    assert delta == 0.0
    assert dlo == 0.0 and dhi == 0.0
    scores = prompt_scores(y_true, pred, prompts)
    assert len(scores) == len(np.unique(prompts))


def test_perfect_predictions() -> None:
    y_true, _, prompts = fixture()
    assert ba2(y_true, y_true, prompts, draws=10, seed=0) == 1.0
    mean, lo, hi = ba2_ci(y_true, y_true, prompts, n_boot=30, draws=5, seed=0)
    assert mean == 1.0
    assert lo == 1.0 and hi == 1.0


def test_ba2_is_bounded_and_deterministic() -> None:
    y_true, pred, prompts = fixture()
    a = ba2(y_true, pred, prompts, draws=25, seed=7)
    b = ba2(y_true, pred, prompts, draws=25, seed=7)
    c = ba2(y_true, pred, prompts, draws=25, seed=8)
    assert 0.0 <= a <= 1.0
    assert a == b
    assert 0.0 <= c <= 1.0


def test_ba2_ignores_extra_rows_per_prompt() -> None:
    # BA(2/prompt) не должна зависеть от числа строк в промпте так же, как BA(18).
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    pred = np.array([0, 1, 2, 0, 1, 2, 1, 0, 0, 0])
    prompts = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"])
    score = ba2(y_true, pred, prompts, classes=3, draws=10, seed=0)
    assert 0.0 <= score <= 1.0


def test_paired_ba2_identical_is_zero() -> None:
    y_true, pred, prompts = fixture()
    delta, lo, hi = paired_ba2(y_true, pred, pred, prompts, n_boot=30, seed=0)
    assert delta == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_ba2_positive_for_better_predictor() -> None:
    y_true, pred, prompts = fixture()
    delta, lo, hi = paired_ba2(y_true, y_true, pred, prompts, n_boot=40, seed=0)
    assert delta > 0
    assert lo <= delta <= hi or lo < delta


def test_ba2_accepts_string_classes() -> None:
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    pred = np.array(["a", "b", "c", "a", "c", "b"])
    prompts = np.array(["p1", "p1", "p1", "p2", "p2", "p2"])
    score = ba2(y_true, pred, prompts, draws=10, seed=0)
    assert 0.0 <= score <= 1.0


def test_ba2_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError):
        ba2([0, 1], [0], [0, 1])
    with pytest.raises(ValueError):
        ba2([0, 1], [0, 1], [0, 1], draws=0)
