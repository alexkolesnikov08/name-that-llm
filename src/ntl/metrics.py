"""Метрика: balanced accuracy и доверительные интервалы по промптам.

Единица анализа — промпт: в train каждый промпт даёт по одному ответу каждой
из 18 моделей, поэтому BA = среднее по промптам от доли верных внутри промпта.
Ресемплим именно промпты, а не строки: строки внутри промпта зависимы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

MetricArrays = tuple[np.ndarray, np.ndarray]


def ba(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def prompt_scores(y_true, y_pred, prompt_ids) -> pd.Series:
    """Доля верных внутри каждого промпта."""
    correct = np.asarray(y_true) == np.asarray(y_pred)
    return pd.Series(correct, index=pd.Index(prompt_ids, name="prompt")).groupby("prompt").mean()


def bootstrap_ba(
    y_true,
    y_pred,
    prompt_ids,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """BA и 95% CI, ресемплируя промпты с возвращением."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prompt_ids = np.asarray(prompt_ids)
    prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([np.flatnonzero(prompt_ids == p) for p in draw])
        stats[i] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def paired_bootstrap_delta(
    y_true,
    pred_a,
    pred_b,
    prompt_ids,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Δ = BA(a) − BA(b) и 95% CI, парно по промптам.

    Возвращает (delta, lo, hi). Если CI не содержит 0 — разница значима.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    prompt_ids = np.asarray(prompt_ids)
    prompts = np.unique(prompt_ids)
    rng = np.random.default_rng(seed)
    idx_by_prompt = {p: np.flatnonzero(prompt_ids == p) for p in prompts}
    stats = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.choice(prompts, size=len(prompts), replace=True)
        idx = np.concatenate([idx_by_prompt[p] for p in draw])
        stats[i] = balanced_accuracy_score(y_true[idx], pred_a[idx]) - balanced_accuracy_score(
            y_true[idx], pred_b[idx]
        )
    lo, hi = np.percentile(stats, [2.5, 97.5])
    delta = balanced_accuracy_score(y_true, pred_a) - balanced_accuracy_score(y_true, pred_b)
    return float(delta), float(lo), float(hi)
