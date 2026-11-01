"""Загрузка данных соревнования.

Путь по умолчанию — корень репозитория; переопределяется через NTL_DATA_DIR.
На Kaggle данные лежат в /kaggle/input/competitions/name-that-llm.
"""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIRS = (
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
)


def data_dir() -> Path:
    """Каталог с train.csv/test.csv."""
    env = os.environ.get("NTL_DATA_DIR")
    if env:
        return Path(env)
    for path in KAGGLE_DIRS:
        if (path / "train.csv").exists():
            return path
    return REPO_ROOT


def load_train() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "test.csv")
