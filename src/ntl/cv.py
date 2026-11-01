"""Кросс-валидация по промптам.

Тестовые промпты не пересекаются с train, поэтому фолды режем по prompt_id.
Внутри промпта всегда все 18 классов — сплит автоматически сбалансирован.
"""
from __future__ import annotations

import numpy as np


def group_folds(groups, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Детерминированные фолды по группам (промптам)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < n_splits:
        raise ValueError("групп меньше, чем фолдов")
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    fold_idx = np.array([fold_of[g] for g in groups])
    return [
        (np.flatnonzero(fold_idx != k), np.flatnonzero(fold_idx == k))
        for k in range(n_splits)
    ]


def oof_predict(model_factory, texts, labels, groups, n_splits: int = 5, seed: int = 0):
    """OOF-предсказания для одного повтора CV."""
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    pred = np.empty(len(texts), dtype=object)
    for trn, val in group_folds(groups, n_splits=n_splits, seed=seed):
        model = model_factory()
        model.fit(texts[trn], labels[trn])
        pred[val] = model.predict(texts[val])
    return pred


def oof_proba(model_factory, texts, labels, groups, n_splits: int = 5, seed: int = 0):
    """OOF-вероятности и порядок классов (для стекинга)."""
    texts = np.asarray(texts)
    labels = np.asarray(labels)
    classes = None
    proba = None
    for trn, val in group_folds(groups, n_splits=n_splits, seed=seed):
        model = model_factory()
        model.fit(texts[trn], labels[trn])
        p = model.predict_proba(texts[val])
        if classes is None:
            classes = model.classes_
            proba = np.zeros((len(texts), len(classes)))
        proba[val] = p
    return proba, classes
