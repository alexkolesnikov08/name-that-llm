"""Линейный probe поверх готовых эмбеддингов, оценка на сплите v2.

Запуск:
    PYTHONPATH=src python3 src/eval_embeddings.py --emb runs/emb_deberta_small.npz --C 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.metrics import ba, bootstrap_ba

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    df = load_train()
    data = np.load(ROOT / args.emb, allow_pickle=True)
    emb = data["emb"]
    ids = data["ids"]
    order = pd.Series(range(len(ids)), index=ids)
    emb = emb[order.loc[df["id"]].values]

    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    dev_pos = df.index[df["prompt_id"].isin(split["dev_prompts"])].values
    X = emb[dev_pos]
    y = dev["model"].values
    groups = dev["prompt_id"].values

    oof_y, oof_p, oof_g, per_repeat = [], [], [], []
    for seed in seeds:
        pred = np.empty(len(dev), dtype=object)
        for val_prompts in folds_for_seed(split, seed):
            val_mask = dev["prompt_id"].isin(val_prompts).values
            model = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000))
            model.fit(X[~val_mask], y[~val_mask])
            pred[val_mask] = model.predict(X[val_mask])
        score = ba(y, pred)
        per_repeat.append(score)
        oof_y.append(y)
        oof_p.append(pred)
        oof_g.append(groups)

    yy, pp, gg = np.concatenate(oof_y), np.concatenate(oof_p), np.concatenate(oof_g)
    _, lo, hi = bootstrap_ba(yy, pp, gg, n_boot=2000, seed=0)
    print(
        f"{args.emb} C={args.C}: {np.mean(per_repeat):.4f} ± {np.std(per_repeat):.4f} "
        f"CI=[{lo:.3f},{hi:.3f}] per_repeat={[round(s, 3) for s in per_repeat]}"
    )


if __name__ == "__main__":
    main()
