"""Стекинг линейных вью: OOF-вероятности → мета-модель.

Оценка честная: базовые модели дают OOF-предсказания внутри фолдов, мета
обучается на OOF-пробах одних промптов и проверяется на других. Повторяем
на 5 разбиениях (сиды 42–46), чтобы не подгоняться под один сплит.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.features import format_features, function_words, skeleton, stylometry
from ntl.metrics import ba

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "stack"
CLASSES = None


def oof_probs(make_model, X, y, folds, groups, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(y), len(classes)))
    for val_prompts in folds:
        val = np.isin(groups, val_prompts)
        model = make_model()
        model.fit(X[~val], y[~val])
        p = model.predict_proba(X[val])
        for j, c in enumerate(model.classes_):
            proba[val, idx[c]] = p[:, j]
    return proba


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    texts = np.asarray(dev["text"].tolist(), dtype=object)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    classes = sorted(df["model"].unique())
    idx = {c: i for i, c in enumerate(classes)}

    emb = np.load(ROOT / "runs" / "emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(emb["ids"])), index=emb["ids"])
    emb = emb["emb"][order.loc[dev["id"]].values]

    stylo, _ = stylometry(texts)
    fmt, _ = format_features(texts)
    func = function_words(texts)
    skel = skeleton(texts)

    views = {
        "char": (texts, lambda: make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "word": (texts, lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "stylo": (stylo, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        "format": (fmt, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        "func": (func, lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "skeleton": (skel, lambda: make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "emb": (emb, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
    }

    seeds = [42, 43, 44, 45, 46]
    probs = {seed: {} for seed in seeds}
    for seed in seeds:
        folds = folds_for_seed(split, seed)
        for name, (X, factory) in views.items():
            path = CACHE / f"{seed}_{name}.npy"
            if path.exists():
                probs[seed][name] = np.load(path)
                continue
            p = oof_probs(factory, X, y, folds, groups, classes)
            np.save(path, p)
            probs[seed][name] = p
            print(f"seed {seed} {name}: {ba(y, [classes[i] for i in p.argmax(1)]):.4f}", flush=True)

    # одиночные вью: средний OOF BA по сидам
    print("\nview      mean BA")
    for name in views:
        scores = [ba(y, [classes[i] for i in probs[s][name].argmax(1)]) for s in seeds]
        print(f"{name:9s} {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    names = list(views)
    Z = {s: np.concatenate([probs[s][n] for n in names], axis=1) for s in seeds}

    # среднее вероятностей
    avg_scores = []
    for s in seeds:
        p = np.mean([probs[s][n] for n in names], axis=0)
        avg_scores.append(ba(y, [classes[i] for i in p.argmax(1)]))
    print(f"\nsimple average: {np.mean(avg_scores):.4f} ± {np.std(avg_scores):.4f}")

    # мета на OOF-пробах, CV по промптам
    meta_scores = []
    for s in seeds:
        folds = folds_for_seed(split, s)
        pred = np.empty(len(y), dtype=object)
        for val_prompts in folds:
            val = np.isin(groups, val_prompts)
            meta = LogisticRegression(C=1, max_iter=3000)
            meta.fit(Z[s][~val], y[~val])
            pred[val] = meta.predict(Z[s][val])
        meta_scores.append(ba(y, pred))
        print(f"meta seed {s}: {meta_scores[-1]:.4f}", flush=True)
    print(f"stack (logreg meta): {np.mean(meta_scores):.4f} ± {np.std(meta_scores):.4f}")


if __name__ == "__main__":
    main()
