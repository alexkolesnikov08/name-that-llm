"""Сравнение вью фич и семейств моделей на одних фолдах (dev v2, seed 42).

Линейное: LogReg. Нелинейное: CatBoost и RandomForest.
Текстовые вью для деревьев сжимаются SVD(128) внутри фолда.
"""
from __future__ import annotations

import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntl.data import load_train
from ntl.evaluate import folds_for_seed, load_split
from ntl.features import format_features, function_words, skeleton, stylometry
from ntl.metrics import ba


def catboost() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=10,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )


def rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)


def eval_numeric(X, y, folds, groups):
    scores = {}
    for name, factory in [
        ("linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000))),
        ("catboost", catboost),
        ("rf", rf),
    ]:
        pred = np.empty(len(y), dtype=object)
        t0 = time.time()
        for val_prompts in folds:
            val = np.isin(groups, val_prompts)
            model = factory()
            model.fit(X[~val], y[~val])
            pred[val] = np.asarray(model.predict(X[val])).ravel()
        scores[name] = ba(y, pred)
        print(f"  {name:10s} {scores[name]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return scores


def eval_text(texts, y, folds, groups, vec):
    scores = {}
    for name in ["linear", "catboost", "rf"]:
        pred = np.empty(len(y), dtype=object)
        t0 = time.time()
        for val_prompts in folds:
            val = np.isin(groups, val_prompts)
            v = vec()
            Xtr = v.fit_transform(texts[~val])
            Xval = v.transform(texts[val])
            if name == "linear":
                model = LogisticRegression(C=10, max_iter=3000)
                model.fit(Xtr, y[~val])
                pred[val] = np.asarray(model.predict(Xval)).ravel()
            else:
                svd = TruncatedSVD(n_components=128, random_state=42)
                Ztr = svd.fit_transform(Xtr)
                Zval = svd.transform(Xval)
                model = catboost() if name == "catboost" else rf()
                model.fit(Ztr, y[~val])
                pred[val] = np.asarray(model.predict(Zval)).ravel()
        scores[name] = ba(y, pred)
        print(f"  {name:10s} {scores[name]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return scores


def main() -> None:
    df = load_train()
    split = load_split()
    dev = df[df["prompt_id"].isin(split["dev_prompts"])].reset_index(drop=True)
    texts = np.asarray(dev["text"].tolist(), dtype=object)
    y = dev["model"].to_numpy(dtype=object)
    groups = dev["prompt_id"].to_numpy(dtype=object)
    folds = folds_for_seed(split, 42)

    results = {}

    print("stylo (numeric)", flush=True)
    X, _ = stylometry(texts)
    results["stylo"] = eval_numeric(X, y, folds, groups)

    print("format (numeric)", flush=True)
    X, _ = format_features(texts)
    results["format"] = eval_numeric(X, y, folds, groups)

    print("func_words (text)", flush=True)
    results["func_words"] = eval_text(
        function_words(texts), y, folds, groups,
        lambda: TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
    )

    print("skeleton (text)", flush=True)
    results["skeleton"] = eval_text(
        skeleton(texts), y, folds, groups,
        lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )

    print("char (text)", flush=True)
    results["char"] = eval_text(
        texts, y, folds, groups,
        lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )

    print("word (text)", flush=True)
    results["word"] = eval_text(
        texts, y, folds, groups,
        lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    )

    print("\nview          linear  catboost      rf")
    for view, s in results.items():
        print(f"{view:12s} {s['linear']:.4f}   {s['catboost']:.4f}  {s['rf']:.4f}")


if __name__ == "__main__":
    main()
