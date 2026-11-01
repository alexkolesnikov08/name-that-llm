"""Стек с центрированием эмбеддингов по TF-IDF-кластерам.

Схема как на тесте: train и val кластеризуются раздельно, k = числу промптов
в подвыборке. Другие вью берутся из закэшированных OOF-вероятностей.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "stack"
SEEDS = [42, 43, 44, 45, 46]
CLASSES: list[str] = []


def cluster_labels(texts, k, seed=0):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    Z = normalize(TruncatedSVD(n_components=100, random_state=seed).fit_transform(M))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def center(X, labels):
    out = X.copy()
    for g in np.unique(labels):
        m = labels == g
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def main() -> None:
    global CLASSES
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    texts = np.asarray(dev["text"].tolist(), dtype=object)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    CLASSES = sorted(df["model"].unique())

    data = np.load(ROOT / "runs" / "emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    emb = data["emb"][order.loc[dev["id"]].values]

    emb_probs = {}
    for seed in SEEDS:
        proba = np.zeros((len(dev), len(CLASSES)))
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            k_trn = dev.loc[~val, "prompt_id"].nunique()
            k_val = dev.loc[val, "prompt_id"].nunique()
            Xtr = center(emb[~val], cluster_labels(texts[~val], k_trn, seed))
            Xval = center(emb[val], cluster_labels(texts[val], k_val, seed))
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
            model.fit(Xtr, y[~val])
            p = model.predict_proba(Xval)
            for j, c in enumerate(model.classes_):
                proba[val, CLASSES.index(c)] = p[:, j]
        emb_probs[seed] = proba
        pred = [CLASSES[i] for i in proba.argmax(1)]
        print(f"seed {seed} centered emb: {ba(y, pred):.4f}", flush=True)

    scores = [ba(y, [CLASSES[i] for i in emb_probs[s].argmax(1)]) for s in SEEDS]
    print(f"centered emb: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    others = ["char", "word", "stylo", "format", "func", "skeleton"]
    for subset_name, names in [
        ("all 7", others + ["emb"]),
        ("emb+stylo+char", ["emb", "stylo", "char"]),
    ]:
        meta_scores = []
        for seed in SEEDS:
            parts = []
            for n in names:
                parts.append(emb_probs[seed] if n == "emb" else np.load(CACHE / f"{seed}_{n}.npy"))
            Z = np.concatenate(parts, axis=1)
            pred = np.empty(len(y), dtype=object)
            for val_prompts in folds_for_seed(split, seed):
                val = np.isin(groups, val_prompts)
                meta = LogisticRegression(C=1, max_iter=3000)
                meta.fit(Z[~val], y[~val])
                pred[val] = meta.predict(Z[val])
            meta_scores.append(ba(y, pred))
        print(f"stack {subset_name}: {np.mean(meta_scores):.4f} ± {np.std(meta_scores):.4f}")


if __name__ == "__main__":
    main()
