"""Абляции стека на закэшированных OOF-вероятностях."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "stack"
SEEDS = [42, 43, 44, 45, 46]


def stack_score(names, y, groups, split, probs):
    scores = []
    for seed in SEEDS:
        Z = np.concatenate([probs[seed][n] for n in names], axis=1)
        pred = np.empty(len(y), dtype=object)
        for val_prompts in folds_for_seed(split, seed):
            val = np.isin(groups, val_prompts)
            from sklearn.linear_model import LogisticRegression

            meta = LogisticRegression(C=1, max_iter=3000)
            meta.fit(Z[~val], y[~val])
            pred[val] = meta.predict(Z[val])
        scores.append(ba(y, pred))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    names = ["char", "word", "stylo", "format", "func", "skeleton", "emb"]
    probs = {s: {n: np.load(CACHE / f"{s}_{n}.npy") for n in names} for s in SEEDS}

    print(f"{'subset':28s} {'BA':>7} {'std':>6}")
    full = ["char", "word", "stylo", "format", "func", "skeleton", "emb"]
    subsets = [("all", full), ("without emb", [n for n in full if n != "emb"])]
    for n in full:
        subsets.append((f"all - {n}", [x for x in full if x != n]))
    subsets += [
        ("emb+stylo+char", ["emb", "stylo", "char"]),
        ("emb+stylo+char+word", ["emb", "stylo", "char", "word"]),
        ("emb+stylo", ["emb", "stylo"]),
        ("emb+char", ["emb", "char"]),
        ("emb+stylo+char+skel", ["emb", "stylo", "char", "skeleton"]),
    ]
    for name, subset in subsets:
        mean, std = stack_score(subset, y, groups, split, probs)
        print(f"{name:28s} {mean:.4f} {std:.4f}")


if __name__ == "__main__":
    main()
