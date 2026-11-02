"""Контракт OOF-предсказаний на пуле.

OOF-файл содержит по одной строке на пассаж чистого пула::

    id, prompt_id, model, fold, p_<class 1>, ..., p_<class 18>

Требования:

* ``id`` уникален и покрывает ровно ожидаемый набор строк;
* ``prompt_id`` — это ``prompt_canon`` из ``splits/pool_v1.parquet``;
* ``fold`` совпадает с фолдом промпта в сплите;
* вероятности неотрицательны, конечны и в сумме дают 1;
* набор ``p_*`` совпадает с классами из ``labels.json``.

Модуль намеренно не знает, как обучать модель: только читает, пишет и
проверяет формат, чтобы разные эксперименты были сравнимы.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["id", "prompt_id", "model", "fold"]
PROB_PREFIX = "p_"
SUM_ATOL = 1e-5


class OOFValidationError(ValueError):
    """Ошибка контракта OOF; содержит список всех найденных проблем."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("OOF невалиден: " + "; ".join(self.errors))


def prob_columns(labels) -> list[str]:
    """Имена вероятностных колонок в порядке классов."""
    return [f"{PROB_PREFIX}{label}" for label in labels]


def read_oof(path: str | Path) -> pd.DataFrame:
    """Прочитать OOF из CSV или parquet (по расширению)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"неизвестное расширение OOF: {path.suffix!r}")


def write_oof(frame: pd.DataFrame, path: str | Path) -> Path:
    """Записать OOF, создав родительские каталоги."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"неизвестное расширение OOF: {path.suffix!r}")
    return path


def validate_oof(
    frame: pd.DataFrame,
    labels,
    pool_split: pd.DataFrame | None = None,
    expected_ids=None,
    atol: float = SUM_ATOL,
) -> dict:
    """Проверить OOF-фрейм и вернуть отчёт.

    ``labels`` — список классов в порядке колонок (обычно из labels.json).
    ``pool_split`` — таблица с колонками prompt_canon/fold; если передана,
    проверяется соответствие prompt_id → fold. ``expected_ids`` — полный
    набор id, который должен быть в OOF (например, id чистого пула).
    """
    errors: list[str] = []
    warnings: list[str] = []
    labels = list(labels)
    p_cols = prob_columns(labels)

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    missing += [c for c in p_cols if c not in frame.columns]
    extra_p = [c for c in frame.columns if c.startswith(PROB_PREFIX) and c not in p_cols]
    if missing:
        errors.append(f"нет колонок: {missing[:10]}")
    if extra_p:
        errors.append(f"лишние p_* колонки: {extra_p[:10]}")
    if missing:
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "n_rows": int(len(frame)),
            "n_prompts": int(frame["prompt_id"].nunique()) if "prompt_id" in frame else 0,
            "n_models": int(frame["model"].nunique()) if "model" in frame else 0,
        }

    if frame["id"].isna().any():
        errors.append("id: есть пропуски")
    if frame["id"].duplicated().any():
        errors.append(f"id: {int(frame['id'].duplicated().sum())} дублей")
    if frame["prompt_id"].isna().any():
        errors.append("prompt_id: есть пропуски")
    if frame["model"].isna().any():
        errors.append("model: есть пропуски")

    unknown_models = sorted(set(frame["model"].dropna()) - set(labels))
    if unknown_models:
        errors.append(f"model: неизвестные классы {unknown_models[:10]}")
    if not pd.api.types.is_integer_dtype(frame["fold"]):
        errors.append(f"fold: тип {frame['fold'].dtype}, ожидается целый")
    if frame["fold"].isna().any():
        errors.append("fold: есть пропуски")
    bad_folds = sorted(set(frame["fold"].dropna().astype(int)) - set(range(5)))
    if bad_folds:
        errors.append(f"fold: недопустимые значения {bad_folds}")

    probs = frame[p_cols].apply(pd.to_numeric, errors="coerce")
    if probs.isna().any().any():
        errors.append("p_*: есть нечисловые или пропущенные значения")
    else:
        values = probs.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append("p_*: есть inf/nan")
        if (values < -atol).any() or (values > 1 + atol).any():
            errors.append("p_*: значения вне [0, 1]")
        sums = values.sum(axis=1)
        bad_sum = np.abs(sums - 1.0) > atol
        if bad_sum.any():
            errors.append(f"p_*: {int(bad_sum.sum())} строк с суммой != 1 (atol={atol})")

    if expected_ids is not None:
        expected = set(map(str, expected_ids))
        actual = set(map(str, frame["id"]))
        missing_ids = expected - actual
        extra_ids = actual - expected
        if missing_ids:
            errors.append(f"id: нет {len(missing_ids)} ожидаемых строк")
        if extra_ids:
            errors.append(f"id: {len(extra_ids)} лишних строк")

    if pool_split is not None:
        required = {"prompt_canon", "fold"}
        missing_split = required - set(pool_split.columns)
        if missing_split:
            errors.append(f"pool_split: нет колонок {sorted(missing_split)}")
        else:
            fold_map = dict(zip(pool_split["prompt_canon"].astype(str), pool_split["fold"].astype(int)))
            actual_folds = frame[["prompt_id", "fold"]].copy()
            actual_folds["prompt_id"] = actual_folds["prompt_id"].astype(str)
            actual_folds["expected"] = actual_folds["prompt_id"].map(fold_map)
            unknown = actual_folds["expected"].isna()
            if unknown.any():
                errors.append(f"prompt_id: {int(unknown.sum())} строк нет в pool_split")
            known = actual_folds[~unknown]
            mismatch = known["fold"].astype(int) != known["expected"].astype(int)
            if mismatch.any():
                errors.append(f"fold: {int(mismatch.sum())} строк не совпадают со сплитом")

    n_prompts = int(frame["prompt_id"].nunique())
    n_models = int(frame["model"].nunique())
    if n_models != len(labels):
        warnings.append(f"в OOF {n_models} моделей из {len(labels)} классов")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": int(len(frame)),
        "n_prompts": n_prompts,
        "n_models": n_models,
        "classes": labels,
        "prob_columns": p_cols,
    }


def assert_valid_oof(
    frame: pd.DataFrame,
    labels,
    pool_split: pd.DataFrame | None = None,
    expected_ids=None,
    atol: float = SUM_ATOL,
) -> dict:
    """Как validate_oof, но бросает OOFValidationError при ошибках."""
    report = validate_oof(frame, labels, pool_split=pool_split, expected_ids=expected_ids, atol=atol)
    if not report["ok"]:
        raise OOFValidationError(report["errors"])
    return report
