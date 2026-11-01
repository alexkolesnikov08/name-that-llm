"""Независимый замер финального стека на locked holdout (30 промптов).

Train — весь dev (120), holdout не участвовал ни в одном выборе.
Схема как на тесте: train и holdout кластеризуются/центрируются раздельно.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.features import format_features, function_words, skeleton, stylometry
from ntl.metrics import ba, bootstrap_ba

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "stack"
SEED = 42


def knn_center(X_emb, texts, k=17):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    Z = normalize(TruncatedSVD(n_components=min(100, M.shape[0] - 1, M.shape[1] - 1)).fit_transform(M))
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, len(Z) - 1)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return X_emb - X_emb[idx].mean(axis=1)


def text_views():
    return {
        "char": (lambda: make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "word": (lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "stylo": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        "format": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        "func": (lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
        "skeleton": (lambda: make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=10, max_iter=3000))),
    }


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    hold = df[df["prompt_id"].isin(split["holdout_prompts"])].reset_index(drop=True)
    classes = sorted(df["model"].unique())
    cidx = {c: i for i, c in enumerate(classes)}

    data = np.load(ROOT / "runs" / "emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    emb_dev = data["emb"][order.loc[dev["id"]].values]
    emb_hold = data["emb"][order.loc[hold["id"]].values]

    emb_dev = knn_center(emb_dev, dev["text"].tolist())
    emb_hold = knn_center(emb_hold, hold["text"].tolist())

    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)

    # OOF-признаки на dev для мета
    emb_oof = np.zeros((len(dev), len(classes)))
    for val_prompts in folds_for_seed(split, SEED):
        val = np.isin(groups, val_prompts)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
        model.fit(emb_dev[~val], y[~val])
        p = model.predict_proba(emb_dev[val])
        for j, c in enumerate(model.classes_):
            emb_oof[val, cidx[c]] = p[:, j]

    other_oof = [np.load(CACHE / f"{SEED}_{n}.npy") for n in text_views()]
    Z_dev = np.concatenate([emb_oof] + other_oof, axis=1)

    # признаки holdout: базовые модели учатся на всём dev
    feats = {
        "char": (dev["text"], hold["text"], "text"),
        "word": (dev["text"], hold["text"], "text"),
        "stylo": (stylometry(dev["text"].tolist())[0], stylometry(hold["text"].tolist())[0], "num"),
        "format": (format_features(dev["text"].tolist())[0], format_features(hold["text"].tolist())[0], "num"),
        "func": (function_words(dev["text"].tolist()), function_words(hold["text"].tolist()), "text"),
        "skeleton": (skeleton(dev["text"].tolist()), skeleton(hold["text"].tolist()), "text"),
    }
    factories = text_views()
    hold_parts = []
    for name, (Xtr, Xte, _) in feats.items():
        model = factories[name]()
        model.fit(Xtr, y)
        p = model.predict_proba(Xte)
        aligned = np.zeros((len(hold), len(classes)))
        for j, c in enumerate(model.classes_):
            aligned[:, cidx[c]] = p[:, j]
        hold_parts.append(aligned)

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
    model.fit(emb_dev, y)
    p = model.predict_proba(emb_hold)
    emb_part = np.zeros((len(hold), len(classes)))
    for j, c in enumerate(model.classes_):
        emb_part[:, cidx[c]] = p[:, j]

    Z_hold = np.concatenate([emb_part] + hold_parts, axis=1)
    meta = LogisticRegression(C=1, max_iter=3000)
    meta.fit(Z_dev, y)
    pred = meta.predict(Z_hold)

    score = ba(hold["model"].values, pred)
    _, lo, hi = bootstrap_ba(
        hold["model"].values, pred, hold["prompt_id"].values, n_boot=2000, seed=0
    )
    print(f"holdout BA={score:.4f} CI=[{lo:.3f},{hi:.3f}] n_prompts={hold['prompt_id'].nunique()}")


if __name__ == "__main__":
    main()
