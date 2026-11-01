"""Стек с kNN-центрированием эмбеддингов (TF-IDF-пространство, k=17)."""
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
from ntl.metrics import ba, bootstrap_ba

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "stack"
SEEDS = [42, 43, 44, 45, 46]
CLASSES: list[str] = []


def knn_center(X_emb, Z, k=17):
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, len(Z) - 1)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return X_emb - X_emb[idx].mean(axis=1)


def main() -> None:
    global CLASSES
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    texts = dev["text"].tolist()
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    CLASSES = sorted(df["model"].unique())

    data = np.load(ROOT / "runs" / "emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    emb = data["emb"][order.loc[dev["id"]].values]

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    Z = normalize(TruncatedSVD(n_components=100, random_state=0).fit_transform(M))
    emb = knn_center(emb, Z)

    emb_probs = {}
    for seed in SEEDS:
        proba = np.zeros((len(dev), len(CLASSES)))
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
            model.fit(emb[~val], y[~val])
            p = model.predict_proba(emb[val])
            for j, c in enumerate(model.classes_):
                proba[val, CLASSES.index(c)] = p[:, j]
        emb_probs[seed] = proba

    scores = [ba(y, [CLASSES[i] for i in emb_probs[s].argmax(1)]) for s in SEEDS]
    print(f"emb knn-centered: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    others = ["char", "word", "stylo", "format", "func", "skeleton"]
    meta_scores = []
    oof_y, oof_p, oof_g = [], [], []
    for seed in SEEDS:
        Z_all = np.concatenate(
            [emb_probs[seed]] + [np.load(CACHE / f"{seed}_{n}.npy") for n in others], axis=1
        )
        pred = np.empty(len(y), dtype=object)
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            meta = LogisticRegression(C=1, max_iter=3000)
            meta.fit(Z_all[~val], y[~val])
            pred[val] = meta.predict(Z_all[val])
        meta_scores.append(ba(y, pred))
        oof_y.append(y)
        oof_p.append(pred)
        oof_g.append(groups)

    _, lo, hi = bootstrap_ba(
        np.concatenate(oof_y), np.concatenate(oof_p), np.concatenate(oof_g), n_boot=2000, seed=0
    )
    print(
        f"stack knn-centered: {np.mean(meta_scores):.4f} ± {np.std(meta_scores):.4f} "
        f"CI=[{lo:.3f},{hi:.3f}]"
    )


if __name__ == "__main__":
    main()
