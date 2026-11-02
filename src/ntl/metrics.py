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


# ---------------------------------------------------------------------------
# BA(2/prompt): метрика, приближенная к скрытому тесту.
#
# В train на промпт приходится 18 ответов, а в тесте — примерно два. BA по 18
# ответам даёт модели больше контекста для выравнивания и оптимистична. Здесь
# на каждом розыгрыше берём не больше двух строк на промпт, считаем обычную
# balanced accuracy и усредняем по розыгрышам. Это тот же протокол, что
# `two_per_prompt_ba` в experiments/rnn_style_v2.py.
# ---------------------------------------------------------------------------

ClassesLike = int | np.ndarray | list | tuple | None


def _classes_for(y_true: np.ndarray, y_pred: np.ndarray, classes: ClassesLike = None) -> np.ndarray:
    if classes is None:
        return np.unique(np.concatenate([y_true, y_pred]))
    if isinstance(classes, (int, np.integer)):
        return np.arange(int(classes))
    return np.asarray(classes)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> float:
    """BA с явным списком классов; отсутствующие в y_true классы пропускаются."""
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == c)))
    return float(np.mean(recalls)) if recalls else 0.0


def _prompt_index(prompt_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    prompts = np.unique(prompt_ids)
    return prompts, {p: np.flatnonzero(prompt_ids == p) for p in prompts}


def _pick_two(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= 2:
        return idx
    return rng.choice(idx, size=2, replace=False)


def _draw_two_per_prompt(
    prompts: np.ndarray,
    idx_by_prompt: dict,
    rng: np.random.Generator,
    *,
    resample_prompts: bool = False,
) -> np.ndarray:
    """Индексы строк: по <=2 на промпт; при resample_prompts — bootstrap промптов."""
    draw = rng.choice(prompts, size=len(prompts), replace=True) if resample_prompts else prompts
    parts = [_pick_two(idx_by_prompt[p], rng) for p in draw]
    return np.concatenate(parts) if parts else np.empty(0, dtype=int)


def ba2(
    y_true,
    y_pred,
    prompt_ids,
    classes: ClassesLike = None,
    draws: int = 25,
    seed: int = 0,
) -> float:
    """BA(2/prompt): среднее BA по `draws` розыгрышам не более 2 строк на промпт.

    `classes` можно передать числом (тогда классы 0..n-1) или явным списком.
    По умолчанию берётся объединение меток из y_true и y_pred.
    """
    if draws < 1:
        raise ValueError("draws должен быть >= 1")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prompt_ids = np.asarray(prompt_ids)
    if not (len(y_true) == len(y_pred) == len(prompt_ids)):
        raise ValueError("y_true, y_pred и prompt_ids должны быть одной длины")
    cls = _classes_for(y_true, y_pred, classes)
    prompts, idx_by_prompt = _prompt_index(prompt_ids)
    rng = np.random.default_rng(seed)
    scores = np.empty(draws)
    for i in range(draws):
        idx = _draw_two_per_prompt(prompts, idx_by_prompt, rng)
        scores[i] = _balanced_accuracy(y_true[idx], y_pred[idx], cls)
    return float(np.mean(scores))


def ba2_ci(
    y_true,
    y_pred,
    prompt_ids,
    classes: ClassesLike = None,
    n_boot: int = 1000,
    draws: int = 25,
    seed: int = 0,
) -> tuple[float, float, float]:
    """BA(2/prompt) и 95% bootstrap-CI, ресемплируя промпты с возвращением.

    В каждом bootstrap-повторе промпты выбираются с возвращением, затем из
    каждого выбранного промпта берётся не больше двух строк. Возвращает
    (mean, lo, hi), как `bootstrap_ba`.
    """
    if n_boot < 1:
        raise ValueError("n_boot должен быть >= 1")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prompt_ids = np.asarray(prompt_ids)
    if not (len(y_true) == len(y_pred) == len(prompt_ids)):
        raise ValueError("y_true, y_pred и prompt_ids должны быть одной длины")
    cls = _classes_for(y_true, y_pred, classes)
    prompts, idx_by_prompt = _prompt_index(prompt_ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = _draw_two_per_prompt(prompts, idx_by_prompt, rng, resample_prompts=True)
        stats[i] = _balanced_accuracy(y_true[idx], y_pred[idx], cls)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def paired_ba2(
    y_true,
    pred_a,
    pred_b,
    prompt_ids,
    classes: ClassesLike = None,
    n_boot: int = 1000,
    draws: int = 25,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Δ = BA(2/a) − BA(2/b) и 95% CI, парно по промптам.

    Оба предиктора оцениваются на одних и тех же строках в каждом розыгрыше.
    Возвращает (delta, lo, hi).
    """
    if n_boot < 1:
        raise ValueError("n_boot должен быть >= 1")
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    prompt_ids = np.asarray(prompt_ids)
    if not (len(y_true) == len(pred_a) == len(pred_b) == len(prompt_ids)):
        raise ValueError("y_true, pred_a, pred_b и prompt_ids должны быть одной длины")
    cls = _classes_for(y_true, np.concatenate([pred_a, pred_b]), classes)
    prompts, idx_by_prompt = _prompt_index(prompt_ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = _draw_two_per_prompt(prompts, idx_by_prompt, rng, resample_prompts=True)
        stats[i] = _balanced_accuracy(y_true[idx], pred_a[idx], cls) - _balanced_accuracy(
            y_true[idx], pred_b[idx], cls
        )
    lo, hi = np.percentile(stats, [2.5, 97.5])
    delta = ba2(y_true, pred_a, prompt_ids, classes=cls, draws=draws, seed=seed) - ba2(
        y_true, pred_b, prompt_ids, classes=cls, draws=draws, seed=seed
    )
    return float(delta), float(lo), float(hi)
