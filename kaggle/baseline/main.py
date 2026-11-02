"""Первый сабмит: TF-IDF char_wb + LogReg.

Самодостаточный скрипт для Kaggle Script Kernel. Локально запускается так:
    NTL_DATA_DIR=. python3 submit/main.py
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

CANDIDATE_DIRS = [
    "/kaggle/input/competitions/name-that-llm",
    "/kaggle/input/name-that-llm",
]


def find_data_dir() -> str:
    for path in CANDIDATE_DIRS:
        if os.path.exists(os.path.join(path, "train.csv")):
            return path
    return os.environ.get("NTL_DATA_DIR", ".")


def main() -> None:
    data_dir = find_data_dir()
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    model = make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=300_000,
        ),
        LogisticRegression(C=10, max_iter=2000),
    )
    model.fit(train["text"], train["model"])
    pred = model.predict(test["text"])

    submission = pd.DataFrame({"id": test["id"], "model": pred})
    submission.to_parquet("submission.parquet", index=False)
    print(f"rows={len(submission)} data_dir={data_dir}")
    print(submission["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
