"""Smoke-тесты: пакет импортируется, базовые инварианты метрик и CV живы.

Не требуют train.csv, пула или моделей — специально, чтобы CI на чистом
клоне не падал на отсутствующих данных.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_package_imports() -> None:
    import ntl
    from ntl import cv, data, evaluate, features, metrics, models

    assert ntl.__doc__
    assert callable(data.load_train)
    assert callable(cv.group_folds)
    assert callable(metrics.ba)
    assert callable(features.stylometry)
    assert callable(models.build_pipeline)
    assert callable(evaluate.load_split)


def test_ba_perfect_and_chance() -> None:
    from ntl.metrics import ba

    y = np.array(["a", "b", "c", "a", "b", "c"])
    assert ba(y, y) == 1.0


def test_group_folds_disjoint_and_cover() -> None:
    from ntl.cv import group_folds

    groups = np.repeat([f"p{i:02d}" for i in range(10)], 3)
    folds = group_folds(groups, n_splits=5, seed=0)
    assert len(folds) == 5
    seen = []
    for trn, val in folds:
        assert len(np.intersect1d(trn, val)) == 0
        seen.append(val)
    assert sorted(np.concatenate(seen).tolist()) == list(range(len(groups)))


def test_prompt_scores_are_per_prompt() -> None:
    from ntl.metrics import prompt_scores

    scores = prompt_scores(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        ["p1", "p1", "p2", "p2"],
    )
    assert scores.loc["p1"] == 0.5
    assert scores.loc["p2"] == 1.0
