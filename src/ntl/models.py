"""Фабрика моделей для HPO и экспериментов."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def _char(analyzer: str, ngram: tuple[int, int], min_df: int, sublinear: bool, max_features):
    return TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram,
        min_df=min_df,
        sublinear_tf=sublinear,
        max_features=max_features,
    )


def _word(min_df: int, sublinear: bool, max_features):
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=sublinear,
        max_features=max_features,
    )


def build_pipeline(cfg: dict) -> Pipeline:
    """Собирает TF-IDF + линейный классификатор по словарю гиперпараметров."""
    view = cfg.get("view", "char_wb")
    min_df = cfg.get("min_df", 2)
    sublinear = cfg.get("sublinear_tf", True)
    max_features = cfg.get("max_features")
    ngram = tuple(cfg.get("ngram_range", (2, 5)))

    if view in ("char_wb", "char"):
        feats = _char(view, ngram, min_df, sublinear, max_features)
    elif view == "word":
        feats = _word(min_df, sublinear, max_features)
    elif view == "char+word":
        feats = FeatureUnion(
            [
                ("char", _char("char_wb", ngram, min_df, sublinear, max_features)),
                ("word", _word(min_df, sublinear, max_features)),
            ]
        )
    else:
        raise ValueError(f"unknown view: {view}")

    clf_name = cfg.get("clf", "logreg")
    if clf_name == "logreg":
        clf = LogisticRegression(C=cfg.get("C", 10.0), max_iter=2000)
    elif clf_name == "svc":
        clf = LinearSVC(C=cfg.get("C", 1.0))
    else:
        raise ValueError(f"unknown classifier: {clf_name}")

    return Pipeline([("features", feats), ("clf", clf)])
