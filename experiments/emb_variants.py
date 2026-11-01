"""Варианты эмбеддингов: raw, prompt-centered, + stylometry."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.features import stylometry
from ntl.metrics import ba

SEEDS = [42, 43, 44, 45, 46]


def center_by_group(X, groups):
    out = X.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def run(name, X, y, groups, split, use_stylo=False):
    scores = []
    for seed in SEEDS:
        pred = np.empty(len(y), dtype=object)
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            Xtr, Xval = X[~val], X[val]
            if use_stylo:
                str_, sv = stylo_feats[~val], stylo_feats[val]
                Xtr = np.hstack([Xtr, str_])
                Xval = np.hstack([Xval, sv])
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))
            model.fit(Xtr, y[~val])
            pred[val] = model.predict(Xval)
        scores.append(ba(y, pred))
    print(f"{name:22s} {np.mean(scores):.4f} ± {np.std(scores):.4f}", flush=True)


def main() -> None:
    global stylo_feats
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)

    data = np.load("runs/emb_deberta_small.npz", allow_pickle=True)
    order = pd.Series(range(len(data["ids"])), index=data["ids"])
    X = data["emb"][order.loc[dev["id"]].values]
    stylo_feats, _ = stylometry(dev["text"].tolist())

    run("raw", X, y, groups, split)
    run("prompt-centered", center_by_group(X, groups), y, groups, split)
    run("raw+stylo", X, y, groups, split, use_stylo=True)
    run("centered+stylo", center_by_group(X, groups), y, groups, split, use_stylo=True)


if __name__ == "__main__":
    main()
