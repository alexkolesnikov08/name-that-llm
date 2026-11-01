"""Центрирование эмбеддингов по kNN в TF-IDF-пространстве.

Не требует знать число промптов на тесте (в отличие от KMeans с k=n/18).
Сравниваем с KMeans-центрированием на dev v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba

SEEDS = [42, 43, 44, 45, 46]


def knn_center(X_emb, Z, k):
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return X_emb - X_emb[idx].mean(axis=1)


def evaluate(X, y, groups, split):
    scores = []
    for seed in SEEDS:
        pred = np.empty(len(y), dtype=object)
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
            model.fit(X[~val], y[~val])
            pred[val] = model.predict(X[val])
        scores.append(ba(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)

    data = np.load("runs/emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    emb = data["emb"][order.loc[dev["id"]].values]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(dev["text"].tolist())
    Z = normalize(TruncatedSVD(n_components=100, random_state=0).fit_transform(M))

    for k in [9, 17, 25, 35]:
        Xc = knn_center(emb, Z, k)
        score, std = evaluate(Xc, y, groups, split)
        print(f"knn tfidf k={k}: {score:.4f} ± {std:.4f}", flush=True)


if __name__ == "__main__":
    main()
