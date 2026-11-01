"""Оценка конфига на фиксированном сплите."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ntl.cv import oof_predict
from ntl.metrics import ba, bootstrap_ba

ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = ROOT / "splits" / "v2.json"


def load_split(path: Path = SPLIT_PATH) -> dict:
    return json.loads(path.read_text())


def folds_for_seed(split: dict, seed: int) -> list[list[str]]:
    for repeat in split["repeats"]:
        if repeat["seed"] == seed:
            return repeat["folds"]
    raise KeyError(f"seed {seed} нет в сплите")


def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return df["text"].values, df["model"].values, df["prompt_id"].values


def evaluate_config(
    cfg: dict,
    df: pd.DataFrame,
    split: dict,
    seeds: list[int],
    model_factory=None,
    n_boot: int = 2000,
) -> dict:
    """OOF BA на dev-пуле для набора повторных разбиений.

    Возвращает mean/std по повторам, pooled BA и bootstrap CI по промптам.
    n_boot=0 отключает bootstrap (для HPO).
    """
    from ntl.models import build_pipeline

    factory = model_factory or (lambda: build_pipeline(cfg))
    dev = df[df["prompt_id"].isin(split["dev_prompts"])]
    texts, labels, groups = arrays(dev)

    oof_true, oof_pred, oof_groups = [], [], []
    per_repeat = []
    for seed in seeds:
        folds = folds_for_seed(split, seed)
        pred = np.empty(len(dev), dtype=object)
        fold_id = np.empty(len(dev), dtype=int)
        for k, val_prompts in enumerate(folds):
            val_mask = dev["prompt_id"].isin(val_prompts).values
            trn_mask = ~val_mask
            model = factory()
            model.fit(texts[trn_mask], labels[trn_mask])
            pred[val_mask] = model.predict(texts[val_mask])
            fold_id[val_mask] = k
        score = ba(labels, pred)
        per_repeat.append(score)
        oof_true.append(labels)
        oof_pred.append(pred)
        oof_groups.append(groups)

    y = np.concatenate(oof_true)
    p = np.concatenate(oof_pred)
    g = np.concatenate(oof_groups)
    if n_boot > 0:
        mean, lo, hi = bootstrap_ba(y, p, g, n_boot=n_boot, seed=0)
    else:
        mean, lo, hi = float("nan"), float("nan"), float("nan")
    return {
        "mean": float(np.mean(per_repeat)),
        "std": float(np.std(per_repeat)),
        "per_repeat": [float(s) for s in per_repeat],
        "pooled_ba": ba(y, p),
        "ci_low": lo,
        "ci_high": hi,
        "n_dev_prompts": len(split["dev_prompts"]),
    }


def evaluate_holdout(cfg: dict, df: pd.DataFrame, split: dict) -> dict:
    """Один замер на locked holdout: обучаем на всём dev, предсказываем holdout.

    Вызывать только для финалистов, не для выбора.
    """
    from ntl.models import build_pipeline

    dev = df[df["prompt_id"].isin(split["dev_prompts"])]
    hold = df[df["prompt_id"].isin(split["holdout_prompts"])]
    model = build_pipeline(cfg)
    model.fit(dev["text"].values, dev["model"].values)
    pred = model.predict(hold["text"].values)
    score = ba(hold["model"].values, pred)
    _, lo, hi = bootstrap_ba(
        hold["model"].values, pred, hold["prompt_id"].values, n_boot=2000, seed=0
    )
    return {"holdout_ba": score, "ci_low": lo, "ci_high": hi, "n": len(hold)}
