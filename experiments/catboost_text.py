"""Честный тест CatBoost с нативными text_features на сыром тексте."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.features import stylometry
from ntl.metrics import ba

SEED = 42


def run(name, data, y, folds, text_features):
    pred = np.empty(len(y), dtype=object)
    t0 = time.time()
    for k, val_prompts in enumerate(folds):
        val = np.isin(groups, val_prompts)
        model = CatBoostClassifier(
            iterations=600,
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=10,
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        )
        train_pool = Pool(data[~val], y[~val], text_features=text_features)
        model.fit(train_pool)
        pred[val] = np.asarray(model.predict(data[val])).ravel()
        print(f"  {name} fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"{name}: BA={ba(y, pred):.4f} ({time.time() - t0:.0f}s)", flush=True)


df = load_train()
split = load_split()
dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
y = dev["model"].to_numpy(dtype=object)
groups = dev["prompt_id"].to_numpy(dtype=object)
folds = folds_for_seed(split, SEED)

raw = pd.DataFrame({"text": dev["text"].astype(str).tolist()})
run("catboost text", raw, y, folds, ["text"])

stylo, names = stylometry(dev["text"].tolist())
with_stylo = pd.DataFrame(stylo, columns=names)
with_stylo["text"] = dev["text"].astype(str).tolist()
run("catboost text+stylo", with_stylo, y, folds, ["text"])
