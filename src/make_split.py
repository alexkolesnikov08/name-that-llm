"""Сборка фиксированного сплита: 30 промптов holdout + 120 development.

Сплит режется по prompt_id, потому что тестовые промпты не пересекаются
с train. Внутри промпта всегда все 18 моделей, поэтому по классам сплит
сбалансирован автоматически. Дополнительно балансируем длину, форматирование
и главные компоненты тематического TF-IDF, чтобы holdout не оказался
случайно «другим» по стилю.

Результат: splits/v1.json
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ntl.data import load_train

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "splits" / "v2.json"

HOLDOUT_SIZE = 30
DEV_FOLDS = 5
REPEAT_SEEDS = [42, 43, 44, 45, 46]

PUNCT = set(".,;:!?—-–()[]{}\"'“”‘’…")
MARKDOWN = set("*_#`>|[]")


def prompt_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Признаки промпта, посчитанные по его 18 текстам."""
    prompts = sorted(df["prompt_id"].unique())
    rows = []
    joined = []
    for p in prompts:
        texts = df.loc[df["prompt_id"] == p, "text"].tolist()
        joined.append("\n".join(texts))
        wc = np.array([len(t.split()) for t in texts], dtype=float)
        chars = sum(len(t) for t in texts)
        rows.append(
            [
                np.log1p(wc.mean()),
                np.log1p(wc.std()),
                sum(c in PUNCT for t in texts for c in t) / chars,
                sum(c in MARKDOWN for t in texts for c in t) / chars,
                sum(c.isdigit() for t in texts for c in t) / chars,
                sum(c == "\n" for t in texts for c in t) / chars,
            ]
        )

    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    svd = TruncatedSVD(n_components=4, random_state=0)
    pcs = svd.fit_transform(tfidf.fit_transform(joined))

    names = ["log_wc_mean", "log_wc_std", "punct_rate", "md_rate", "digit_rate", "nl_rate"]
    names += [f"pc{i}" for i in range(pcs.shape[1])]
    X = np.column_stack([np.array(rows), pcs])
    return X, names, joined


def greedy_balance(X: np.ndarray, n_buckets: int, seed: int) -> list[list[int]]:
    """Жадно раскладывает объекты по корзинам, выравнивая средние по фичам."""
    rng = np.random.default_rng(seed)
    n = len(X)
    target = X.mean(axis=0)
    sizes = [n // n_buckets + (1 if i < n % n_buckets else 0) for i in range(n_buckets)]
    sums = np.zeros((n_buckets, X.shape[1]))
    counts = np.zeros(n_buckets, dtype=int)
    buckets: list[list[int]] = [[] for _ in range(n_buckets)]

    # сначала раскладываем самые «крайние» объекты, чтобы компенсировать хвосты
    order = rng.permutation(n)
    norm = np.linalg.norm(X - target, axis=1)[order]
    order = order[np.argsort(-norm, kind="stable")]

    for i in order:
        best_b, best_cost = None, np.inf
        for b in range(n_buckets):
            if counts[b] >= sizes[b]:
                continue
            new_mean = (sums[b] + X[i]) / (counts[b] + 1)
            cost = np.sum((new_mean - target) ** 2)
            if cost < best_cost:
                best_b, best_cost = b, cost
        buckets[best_b].append(int(i))
        sums[best_b] += X[i]
        counts[best_b] += 1
    return [sorted(b) for b in buckets]


def smd(X: np.ndarray, idx_a, idx_b) -> np.ndarray:
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    idx_a = np.asarray(idx_a)
    idx_b = np.asarray(idx_b)
    return (X[idx_a].mean(0) - X[idx_b].mean(0)) / sd


def refine_holdout(
    X: np.ndarray, holdout: list[int], dev: list[int], iters: int = 4000, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Локальные свопы holdout↔dev, минимизируем max |SMD|."""
    rng = np.random.default_rng(seed)
    H, D = set(holdout), set(dev)
    best = np.abs(smd(X, sorted(H), sorted(D))).max()
    for _ in range(iters):
        h = int(rng.choice(list(H)))
        d = int(rng.choice(list(D)))
        H2, D2 = (H - {h}) | {d}, (D - {d}) | {h}
        score = np.abs(smd(X, sorted(H2), sorted(D2))).max()
        if score < best:
            H, D, best = H2, D2, score
    return sorted(H), sorted(D)


def random_folds(prompts: list[str], n_splits: int, seed: int) -> list[list[str]]:
    """Случайные фолды равного размера.

    Нужны именно случайные, а не сбалансированные: иначе разные сиды дают
    один и тот же сплит и «повторы» перестают быть повторами.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(prompts))
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, idx in enumerate(order):
        folds[i % n_splits].append(prompts[idx])
    return [sorted(f) for f in folds]


def main() -> None:
    df = load_train()
    X_raw, names, joined = prompt_features(df)
    X = StandardScaler().fit_transform(X_raw)
    prompts = np.array(sorted(df["prompt_id"].unique()))

    # holdout берём из v1, чтобы locked-множество не менялось
    prev = json.loads((ROOT / "splits" / "v1.json").read_text())
    holdout_prompts = prev["holdout_prompts"]
    dev_prompts = prev["dev_prompts"]
    holdout_idx = np.array([int(np.flatnonzero(prompts == p)[0]) for p in holdout_prompts])
    dev_idx = np.array([int(np.flatnonzero(prompts == p)[0]) for p in dev_prompts])

    repeats = []
    for seed in REPEAT_SEEDS:
        folds = random_folds(dev_prompts, DEV_FOLDS, seed=seed)
        repeats.append({"seed": seed, "folds": folds})

    data_bytes = (ROOT / "train.csv").read_bytes()
    payload = {
        "version": 2,
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "n_prompts": len(prompts),
        "holdout_prompts": holdout_prompts,
        "dev_prompts": dev_prompts,
        "repeat_seeds": REPEAT_SEEDS,
        "repeats": repeats,
        "note": "v1 had identical pseudo-repeats; v2 uses random folds",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # диагностика
    print(f"prompts={len(prompts)} holdout={len(holdout_prompts)} dev={len(dev_prompts)}")
    s = smd(X, holdout_idx, dev_idx)
    print(f"max |SMD| holdout vs dev = {np.abs(s).max():.3f}")
    for name, value in sorted(zip(names, s), key=lambda t: -abs(t[1]))[:4]:
        print(f"  {name}: {value:+.3f}")
    for r in repeats:
        sizes = [len(f) for f in r["folds"]]
        print(f"seed={r['seed']} fold sizes={sizes}")
    base = [set(f) for f in repeats[0]["folds"]]
    for r in repeats[1:]:
        same = all(a == b for a, b in zip(base, [set(f) for f in r["folds"]]))
        print(f"seed={r['seed']} identical to seed {repeats[0]['seed']}: {same}")

    # near-duplicate промптов: максимум косинуса между prompt-level документами
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(joined)
    S = cosine_similarity(M)
    np.fill_diagonal(S, 0)
    print(f"max prompt cosine={S.max():.3f}")
    print(f"pairs > 0.90: {(S > 0.90).sum() // 2}")

    # стабильность тематических кластеров
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    labels = [
        KMeans(n_clusters=6, n_init=10, random_state=s).fit_predict(X[:, 6:10])
        for s in range(10)
    ]
    aris = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(len(labels)) for j in range(i + 1, len(labels))
    ]
    print(f"cluster ARI (k=6, 10 seeds) = {np.mean(aris):.2f} +- {np.std(aris):.2f}")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
