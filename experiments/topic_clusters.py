"""Восстановление групп промптов по содержанию: TF-IDF → SVD → KMeans."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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
    return float(frame.groupby("c")["p"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(frame))


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    texts = dev["text"].tolist()
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    n_prompts = dev["prompt_id"].nunique()

    data = np.load("runs/emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    emb = data["emb"][order.loc[dev["id"]].values]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    print(f"tfidf {M.shape}, prompts={n_prompts}", flush=True)

    for n_comp in [50, 100, 200]:
        Z = normalize(TruncatedSVD(n_components=n_comp, random_state=0).fit_transform(M))
        for k in [n_prompts, int(n_prompts * 1.1)]:
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
            p = purity(labels, groups)
            score, std = evaluate(center_by_labels(emb, labels), y, groups, split)
            print(
                f"svd={n_comp} k={k}: purity={p:.3f} BA={score:.4f} ± {std:.4f}",
                flush=True,
            )

    # aggломеративка на косинусе, если KMeans не тянет
    Z = normalize(TruncatedSVD(n_components=100, random_state=0).fit_transform(M))
    for thresh in [0.3, 0.4, 0.5]:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=thresh, metric="cosine", linkage="average"
        ).fit_predict(Z)
        p = purity(labels, groups)
        score, std = evaluate(center_by_labels(emb, labels), y, groups, split)
        print(
            f"agglo thr={thresh}: clusters={len(set(labels))} purity={p:.3f} "
            f"BA={score:.4f} ± {std:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
