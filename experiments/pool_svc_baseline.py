"""Опорная точка: char n-gram SVC на пуле, оценка на официальном train.

Нужна, чтобы понять, наш RNN недоучился или перенос с синтетического пула
на реальные тексты — общая проблема. Протокол тот же: holdout по промптам,
BA по всем 18 ответам и по 2 (как в тесте).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
POOL = Path(sorted(glob.glob(str(ROOT / "research/pool/*.parquet")))[0])
DATA = ROOT / "train.csv"


def ba(y_true, y_pred, classes):
    rec = [float((y_pred[y_true == c] == c).mean()) for c in classes if (y_true == c).any()]
    return float(np.mean(rec))


def two_per_prompt(y_true, y_pred, groups, classes, draws=25, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    vals = []
    for _ in range(draws):
        pick = []
        for idx in by.values():
            pick.extend(idx if len(idx) <= 2 else rng.choice(idx, 2, replace=False).tolist())
        pick = np.asarray(pick)
        vals.append(ba(y_true[pick], y_pred[pick], classes))
    return float(np.mean(vals)), float(np.std(vals))


def main(n_pool: int = 40000, hold_prompts: int = 30, seed: int = 0) -> None:
    pool = pd.read_parquet(POOL, columns=["text", "model", "flag_kb_proxy"])
    pool = pool[~pool["flag_kb_proxy"].astype(bool)]
    pool = pool[pool["text"].notna()].sample(min(n_pool, len(pool)), random_state=seed)
    off = pd.read_csv(DATA)[["text", "model", "prompt_id"]]
    holds = sorted(off["prompt_id"].unique())[:hold_prompts]
    val = off[off["prompt_id"].isin(holds)].reset_index(drop=True)
    tr = off[~off["prompt_id"].isin(holds)].reset_index(drop=True)
    print(f"пул {len(pool)} | train {len(tr)} | val {len(val)}", flush=True)

    Xtr_text = pool["text"].astype(str).tolist() + tr["text"].astype(str).tolist()
    ytr = pool["model"].astype(str).tolist() + tr["model"].astype(str).tolist()

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(Xtr_text)
    Xv = vec.transform(val["text"].astype(str).tolist())
    print(f"признаков {X.shape[1]}", flush=True)
    clf = LinearSVC(C=0.5, dual=True, max_iter=3000).fit(X, ytr)
    pred = clf.predict(Xv)
    classes = sorted(off["model"].unique())
    full = ba(val["model"].to_numpy(), pred, classes)
    two, sd = two_per_prompt(val["model"].to_numpy(), pred, val["prompt_id"].to_numpy(), classes)
    print(f"SVC pool->official: BA(18) {full:.4f} | BA(2) {two:.4f} ± {sd:.4f}", flush=True)


if __name__ == "__main__":
    main()
