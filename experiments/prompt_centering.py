"""Можно ли восстановить группы промптов кластеризацией и центрировать по ним.

Оценка на dev: oracle-центрирование даёт 0.4385. Здесь — KMeans и kNN-центрирование,
то есть версии, применимые к тесту без prompt_id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba

SEEDS = [42, 43, 44, 45, 46]


def center_by_labels(X, labels):
    out = X.copy()
    for g in np.unique(labels):
        m = labels == g
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def knn_center(X, k=17):
    Z = normalize(X)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return X - X[idx].mean(axis=1)


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


def purity(labels, groups) -> float:
    frame = pd.DataFrame({"c": labels, "p": groups})
    sizes = frame.groupby("c").size()
    major = frame.groupby("c")["p"].agg(lambda s: s.value_counts().iloc[0])
    return float((major / sizes * sizes).sum() / len(frame))


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)

    data = np.load("runs/emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    X = data["emb"][order.loc[dev["id"]].values]
    n_prompts = dev["prompt_id"].nunique()
    print(f"dev texts={len(X)} prompts={n_prompts}", flush=True)

    for k in [n_prompts, int(n_prompts * 1.1), int(n_prompts * 0.9)]:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(normalize(X))
        sizes = pd.Series(labels).value_counts()
        p = purity(labels, groups)
        score, std = evaluate(center_by_labels(X, labels), y, groups, split)
        print(
            f"kmeans k={k}: purity={p:.3f} size={sizes.mean():.0f}±{sizes.std():.0f} "
            f"BA={score:.4f} ± {std:.4f}",
            flush=True,
        )

    for k in [9, 17, 25]:
        score, std = evaluate(knn_center(X, k), y, groups, split)
        print(f"knn k={k}: BA={score:.4f} ± {std:.4f}", flush=True)

    oracle = center_by_labels(X, groups)
    score, std = evaluate(oracle, y, groups, split)
    print(f"oracle: BA={score:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
