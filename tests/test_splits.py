"""Тесты сплитов на синтетических фикстурах.

Реальные train.csv / pool читаются только если лежат в репозитории; иначе
соответствующие тесты скипаются.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl.splits import (  # noqa: E402
    load_official_split,
    load_pool_split,
    validate_official_split,
    validate_pool_split,
)

MODELS = [f"model-{i:02d}" for i in range(18)]


def make_train(n_prompts: int = 150) -> pd.DataFrame:
    rows = []
    for i in range(n_prompts):
        prompt_id = f"p{i:03d}"
        n_models = 17 if i == 0 else 18
        for j in range(n_models):
            rows.append(
                {
                    "id": f"{prompt_id}-{j}",
                    "prompt_id": prompt_id,
                    "model": MODELS[j],
                    "text": "x" * (100 + i),
                }
            )
    return pd.DataFrame(rows)


def make_official(train: pd.DataFrame) -> dict:
    prompts = sorted(train["prompt_id"].unique())
    blocks = {prompt_id: i % 5 for i, prompt_id in enumerate(prompts)}
    return {
        "version": "test",
        "train_sha256": "0" * 64,
        "n_prompts": len(prompts),
        "n_rows": len(train),
        "method": "synthetic",
        "seed": 42,
        "blocks": blocks,
        "roles": {"dev": [0, 1, 2, 3], "sealed": [4]},
    }


def make_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    split_rows = []
    for c in range(10):
        canon = f"prompt-{c}"
        fold = c % 5
        for m, model in enumerate(MODELS):
            rows.append(
                {
                    "id": f"id-{c}-{m}",
                    "prompt_canon": canon,
                    "model": model,
                    "source_dataset": "owner/source",
                    "flag_kb_proxy": False,
                }
            )
        split_rows.append(
            {
                "prompt_canon": canon,
                "fold": fold,
                "n_rows": 18,
                "n_clean": 18,
                "n_models": 18,
                "n_sources": 1,
                "source_top": "owner/source",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def test_valid_official_split() -> None:
    train = make_train()
    split = make_official(train)
    report = validate_official_split(split, train=train)
    assert report["ok"], report["errors"]
    assert report["block_sizes"] == {0: 30, 1: 30, 2: 30, 3: 30, 4: 30}
    assert report["n_rows"] == 2699
    assert report["block_rows"] == {0: 539, 1: 540, 2: 540, 3: 540, 4: 540}


def test_official_split_wrong_block_size() -> None:
    train = make_train()
    split = make_official(train)
    split["blocks"]["p000"] = 0  # p000 уже был в блоке 0, но меняем другой
    split["blocks"]["p001"] = 0
    report = validate_official_split(split, train=train)
    assert not report["ok"]
    assert any("блок" in e for e in report["errors"])


def test_official_split_wrong_row_count() -> None:
    train = make_train().iloc[:-1]
    split = make_official(make_train())
    report = validate_official_split(split, train=train)
    assert not report["ok"]


def test_valid_pool_split() -> None:
    pool, split = make_pool()
    report = validate_pool_split(split, pool=pool)
    assert report["ok"], report["errors"]
    assert report["fold_sizes"] == {0: 36, 1: 36, 2: 36, 3: 36, 4: 36}
    assert report["fold_classes"] == {0: 18, 1: 18, 2: 18, 3: 18, 4: 18}


def test_pool_split_duplicate_canon() -> None:
    pool, split = make_pool()
    split = pd.concat([split, split.iloc[[0]]], ignore_index=True)
    report = validate_pool_split(split, pool=pool)
    assert not report["ok"]
    assert any("дубл" in e for e in report["errors"])


def test_pool_split_wrong_stats() -> None:
    pool, split = make_pool()
    split.loc[0, "n_rows"] = 999
    report = validate_pool_split(split, pool=pool)
    assert not report["ok"]
    assert any("статистика" in e for e in report["errors"])


def test_pool_split_wrong_model_count() -> None:
    pool, split = make_pool()
    split.loc[0, "n_models"] = 17
    report = validate_pool_split(split, pool=pool)
    assert not report["ok"]
    assert any("статистика" in e for e in report["errors"])


def test_pool_split_wrong_source_top() -> None:
    pool, split = make_pool()
    split.loc[0, "source_top"] = "other/source"
    report = validate_pool_split(split, pool=pool)
    assert not report["ok"]
    assert any("статистика" in e for e in report["errors"])


def test_pool_split_negative_fold() -> None:
    pool, split = make_pool()
    split.loc[0, "fold"] = -1
    report = validate_pool_split(split, pool=pool)
    assert not report["ok"]


@pytest.mark.skipif(
    not (ROOT / "train.csv").exists() or not (ROOT / "splits" / "official_v1.json").exists(),
    reason="нет train.csv или splits/official_v1.json",
)
def test_real_official_split() -> None:
    train = pd.read_csv(
        ROOT / "train.csv", dtype={"id": str, "prompt_id": str, "model": str, "text": str}
    )
    split = load_official_split()
    report = validate_official_split(split, train=train)
    assert report["ok"], report["errors"]
    assert report["n_rows"] == 2699


@pytest.mark.skipif(
    not (ROOT / "research" / "pool" / "ntl_open_pool.parquet").exists()
    or not (ROOT / "splits" / "pool_v1.parquet").exists(),
    reason="нет пула или splits/pool_v1.parquet",
)
def test_real_pool_split() -> None:
    pool = pd.read_parquet(ROOT / "research" / "pool" / "ntl_open_pool.parquet")
    split = load_pool_split()
    report = validate_pool_split(split, pool=pool)
    assert report["ok"], report["errors"]
    assert report["n_prompts"] == 23451
    assert report["fold_sizes"] == {0: 22934, 1: 22934, 2: 22934, 3: 22934, 4: 22934}


def test_official_json_has_no_text() -> None:
    train = make_train()
    split = make_official(train)
    payload = json.dumps(split, ensure_ascii=False)
    assert '"text"' not in payload
    assert set(split["blocks"]) == set(train["prompt_id"])
