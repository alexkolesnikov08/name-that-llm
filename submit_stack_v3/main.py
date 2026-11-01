"""Name That LLM — стек из 7 вью.

7 линейных баз (char/word TF-IDF, стилометрия, формат, function words,
скелет, frozen DeBERTa-small) → OOF-вероятности → LogReg-мета.

Kaggle: данные из /kaggle/input/competitions/name-that-llm,
веса DeBERTa из датасета microsoft-deberta-v3-small-local.
Локально: NTL_DATA_DIR=. и обычный HF-кэш.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoModel, AutoTokenizer

DATA_CANDIDATES = [
    "/kaggle/input/competitions/name-that-llm",
    "/kaggle/input/name-that-llm",
]
MODEL_CANDIDATES = [
    "/kaggle/input/datasets/eric0045/microsoft-deberta-v3-small-local/microsoft-deberta-v3-small",
    "/kaggle/input/datasets/morginalium/deberta-v3-small-ntl",
    os.environ.get("NTL_MODEL_DIR", ""),
    "microsoft/deberta-v3-small",
]

PUNCT = set(".,;:!?—-–()[]{}\"'“”‘’…")
MARKDOWN = set("*_#`>|")
DISCOURSE = [
    "however", "moreover", "therefore", "thus", "ultimately", "consequently",
    "furthermore", "nevertheless", "in conclusion", "in summary", "overall",
    "importantly", "notably", "crucially", "essentially", "fundamentally",
    "it's important to note", "it is important to note", "delve", "tapestry",
    "realm", "foster", "intricate", "nuanced", "crucial", "pivotal", "profound",
]


def data_dir() -> str:
    for path in DATA_CANDIDATES:
        if os.path.exists(os.path.join(path, "train.csv")):
            return path
    return os.environ.get("NTL_DATA_DIR", ".")


def model_dir() -> str:
    for path in MODEL_CANDIDATES:
        if path and os.path.isdir(path):
            return path
    return "microsoft/deberta-v3-small"


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def stylometry(texts) -> np.ndarray:
    rows = []
    for text in texts:
        words = re.findall(r"\S+", text)
        clean = [re.sub(r"[^\w'-]", "", w) for w in words]
        clean = [w for w in clean if w]
        wl = np.array([len(w) for w in clean], dtype=float) if clean else np.array([0.0])
        sents = sentences(text)
        sl = np.array([len(s.split()) for s in sents], dtype=float) if sents else np.array([0.0])
        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)
        lower = [w.lower() for w in clean]
        ttr = len(set(lower)) / max(len(lower), 1)
        hapax = sum(1 for w in set(lower) if lower.count(w) == 1) / max(len(set(lower)), 1)
        stop_rate = sum(w in ENGLISH_STOP_WORDS for w in lower) / max(len(lower), 1)
        punct = [c for c in text if c in PUNCT]
        md = sum(c in MARKDOWN for c in text)
        rows.append([
            len(text), n_words, len(sents), len(re.findall(r"\n\n", text)),
            wl.mean(), wl.std(), np.percentile(wl, 10), np.percentile(wl, 50),
            np.percentile(wl, 90), wl.max(),
            sl.mean(), sl.std(), sl.min(), sl.max(),
            text.count(",") / n_words, text.count(".") / n_words,
            text.count(";") / n_words, text.count(":") / n_words,
            text.count("—") / n_words, text.count("-") / n_words,
            text.count("(") / n_words, text.count(")") / n_words,
            text.count('"') / n_words, text.count("'") / n_words,
            text.count("“") / n_words, text.count("”") / n_words,
            text.count("’") / n_words, text.count("!") / n_words,
            text.count("?") / n_words, text.count("…") / n_words,
            sum(c.isupper() for c in text) / n_chars,
            sum(c.isdigit() for c in text) / n_chars,
            text.count("\n") / n_chars, md / n_chars,
            ttr, hapax, stop_rate,
            len(punct) / n_words,
            sum(w.isupper() and len(w) > 1 for w in clean) / max(len(clean), 1),
            sum(w[0].isupper() for w in clean if w) / max(len(clean), 1),
            float(np.mean([text.lower().count(d) for d in DISCOURSE])),
        ])
    return np.array(rows, dtype=np.float64)


def format_features(texts) -> np.ndarray:
    rows = []
    for text in texts:
        n = max(len(text), 1)
        rows.append([
            text.count("\n"), text.count("\n\n"), text.count("*"), text.count("_"),
            text.count("#"), text.count("`"), text.count(">"), text.count("|"),
            text.count("—"), text.count("–"), text.count("-"), text.count("“"),
            text.count("”"), text.count("’"), text.count("…"), text.count("$"),
            text.count("("), text.count("["), text.count("•"), text.count("1."),
            sum(c.isupper() for c in text) / n,
            sum(c.isdigit() for c in text) / n,
            len(text) / max(len(text.split()), 1),
        ])
    return np.array(rows, dtype=np.float64)


def function_words(texts) -> np.ndarray:
    out = []
    for text in texts:
        toks = re.findall(r"[a-z']+", text.lower())
        out.append(" ".join(t for t in toks if t in ENGLISH_STOP_WORDS))
    return np.array(out, dtype=object)


def skeleton(texts) -> np.ndarray:
    out = []
    for text in texts:
        parts = []
        for tok in re.findall(r"[A-Za-z']+|[^\sA-Za-z0-9]", text):
            if tok[0].isalpha():
                parts.append(tok[0].lower() if tok[0].islower() else tok[0])
            else:
                parts.append(tok)
        out.append(" ".join(parts))
    return np.array(out, dtype=object)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def embed(texts, tokenizer, model, device, batch_size: int = 32, max_length: int = 384) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def cluster_labels(texts, k: int, seed: int = 0) -> np.ndarray:
    """Группы промптов по содержанию: TF-IDF → SVD → KMeans."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = tfidf.fit_transform(texts)
    Z = TruncatedSVD(n_components=min(100, M.shape[1] - 1), random_state=seed).fit_transform(M)
    Z = normalize(Z)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def center_by_labels(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = X.copy()
    for g in np.unique(labels):
        m = labels == g
        out[m] -= out[m].mean(axis=0, keepdims=True)
    return out


def group_folds(groups, n_splits: int = 5, seed: int = 0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    fold_idx = np.array([fold_of[g] for g in groups])
    return [
        (np.flatnonzero(fold_idx != k), np.flatnonzero(fold_idx == k))
        for k in range(n_splits)
    ]


def oof_probs(factory, X, y, groups, classes, n_splits=5, seed=0):
    idx = {c: i for i, c in enumerate(classes)}
    proba = np.zeros((len(y), len(classes)))
    for trn, val in group_folds(groups, n_splits, seed):
        model = factory()
        model.fit(X[trn], y[trn])
        p = model.predict_proba(X[val])
        for j, c in enumerate(model.classes_):
            proba[val, idx[c]] = p[:, j]
    return proba


def main() -> None:
    root = data_dir()
    train = pd.read_csv(os.path.join(root, "train.csv"))
    test = pd.read_csv(os.path.join(root, "test.csv"))
    y = train["model"].to_numpy(dtype=object)
    groups = train["prompt_id"].to_numpy(dtype=object)
    classes = sorted(train["model"].unique())

    tr_texts = np.asarray(train["text"].tolist(), dtype=object)
    te_texts = np.asarray(test["text"].tolist(), dtype=object)
    print(f"train={len(tr_texts)} test={len(te_texts)} emb_device={pick_device()}", flush=True)

    device = pick_device()
    print("model candidates:", [(x, os.path.isdir(x)) for x in MODEL_CANDIDATES], flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir())
    encoder = AutoModel.from_pretrained(model_dir()).to(device).eval()
    tr_emb = embed(tr_texts.tolist(), tokenizer, encoder, device)
    te_emb = embed(te_texts.tolist(), tokenizer, encoder, device)
    print("embeddings ready", flush=True)

    k_tr = int(train["prompt_id"].nunique())
    k_te = max(1, round(len(te_texts) / 18))
    tr_labels = cluster_labels(tr_texts.tolist(), k_tr)
    te_labels = cluster_labels(te_texts.tolist(), k_te)
    tr_emb = center_by_labels(tr_emb, tr_labels)
    te_emb = center_by_labels(te_emb, te_labels)
    print(
        f"clusters train={k_tr} mean_size={len(tr_texts) / k_tr:.1f}; "
        f"test={k_te} mean_size={len(te_texts) / k_te:.1f}",
        flush=True,
    )

    views = {
        "char": (
            tr_texts, te_texts,
            lambda: make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                LogisticRegression(C=10, max_iter=3000)),
        ),
        "word": (
            tr_texts, te_texts,
            lambda: make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                LogisticRegression(C=10, max_iter=3000)),
        ),
        "stylo": (
            stylometry(tr_texts), stylometry(te_texts),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
        ),
        "format": (
            format_features(tr_texts), format_features(te_texts),
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
        ),
        "func": (
            function_words(tr_texts), function_words(te_texts),
            lambda: make_pipeline(
                TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True),
                LogisticRegression(C=10, max_iter=3000)),
        ),
        "skeleton": (
            skeleton(tr_texts), skeleton(te_texts),
            lambda: make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                LogisticRegression(C=10, max_iter=3000)),
        ),
        "emb": (
            tr_emb, te_emb,
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
        ),
    }

    oof, test_probs = [], []
    for name, (Xtr, Xte, factory) in views.items():
        p = oof_probs(factory, Xtr, y, groups, classes)
        oof.append(p)
        model = factory()
        model.fit(Xtr, y)
        q = model.predict_proba(Xte)
        aligned = np.zeros((len(Xte), len(classes)))
        for j, c in enumerate(model.classes_):
            aligned[:, classes.index(c)] = q[:, j]
        test_probs.append(aligned)
        print(f"{name}: oof ok", flush=True)

    Z_train = np.concatenate(oof, axis=1)
    Z_test = np.concatenate(test_probs, axis=1)
    meta = LogisticRegression(C=1, max_iter=3000)
    meta.fit(Z_train, y)
    pred = meta.predict(Z_test)

    submission = pd.DataFrame({"id": test["id"], "model": pred})
    submission.to_parquet("submission.parquet", index=False)
    print(f"rows={len(submission)}")
    print(submission["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
