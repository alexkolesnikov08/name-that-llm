"""Тесты контракта OOF на синтетических данных."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl.oof import (  # noqa: E402
    OOFValidationError,
    assert_valid_oof,
    prob_columns,
    read_oof,
    validate_oof,
    write_oof,
)

LABELS = ["alpha", "beta", "gamma"]


def make_oof() -> pd.DataFrame:
    rows = []
    for prompt_id, fold in (("p1", 0), ("p2", 1)):
        for j, model in enumerate(LABELS):
            p = np.zeros(len(LABELS))
            p[j] = 1.0
            rows.append(
                {
                    "id": f"{prompt_id}-{j}",
                    "prompt_id": prompt_id,
                    "model": model,
                    "fold": fold,
                    **{f"p_{label}": float(v) for label, v in zip(LABELS, p)},
                }
            )
    return pd.DataFrame(rows)


def make_pool_split() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_canon": ["p1", "p2"],
            "fold": [0, 1],
            "n_rows": [3, 3],
            "n_clean": [3, 3],
            "n_models": [3, 3],
            "n_sources": [1, 1],
            "source_top": ["s", "s"],
        }
    )


def test_valid_oof() -> None:
    frame = make_oof()
    report = validate_oof(
        frame,
        LABELS,
        pool_split=make_pool_split(),
        expected_ids=set(frame["id"]),
    )
    assert report["ok"], report["errors"]
    assert report["n_rows"] == 6
    assert report["n_prompts"] == 2
    assert report["prob_columns"] == prob_columns(LABELS)


def test_prob_columns() -> None:
    assert prob_columns(LABELS) == ["p_alpha", "p_beta", "p_gamma"]


def test_bad_sum() -> None:
    frame = make_oof()
    frame.loc[0, ["p_alpha", "p_beta", "p_gamma"]] = [0.2, 0.2, 0.2]
    report = validate_oof(frame, LABELS)
    assert not report["ok"]
    assert any("суммой" in e for e in report["errors"])
    with pytest.raises(OOFValidationError):
        assert_valid_oof(frame, LABELS)


def test_fold_mismatch() -> None:
    frame = make_oof()
    frame.loc[frame["prompt_id"] == "p2", "fold"] = 0
    report = validate_oof(frame, LABELS, pool_split=make_pool_split())
    assert not report["ok"]
    assert any("не совпадают со сплитом" in e for e in report["errors"])


def test_missing_prompt_in_split() -> None:
    frame = make_oof()
    frame.loc[frame["prompt_id"] == "p2", "prompt_id"] = "p3"
    report = validate_oof(frame, LABELS, pool_split=make_pool_split())
    assert not report["ok"]
    assert any("нет в pool_split" in e for e in report["errors"])


def test_missing_expected_ids() -> None:
    frame = make_oof()
    report = validate_oof(frame, LABELS, expected_ids={"p1-0", "p1-1", "p1-2", "p2-0"})
    assert not report["ok"]
    assert any("лишних" in e for e in report["errors"])


def test_missing_column() -> None:
    frame = make_oof().drop(columns=["p_gamma"])
    report = validate_oof(frame, LABELS)
    assert not report["ok"]
    assert any("нет колонок" in e for e in report["errors"])


def test_extra_prob_column() -> None:
    frame = make_oof()
    frame["p_delta"] = 0.0
    report = validate_oof(frame, LABELS)
    assert not report["ok"]
    assert any("лишние p_" in e for e in report["errors"])


def test_bad_fold_value() -> None:
    frame = make_oof()
    frame.loc[0, "fold"] = 9
    report = validate_oof(frame, LABELS)
    assert not report["ok"]
    assert any("недопустимые" in e for e in report["errors"])


def test_duplicate_id() -> None:
    frame = make_oof()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    report = validate_oof(frame, LABELS)
    assert not report["ok"]
    assert any("дубл" in e for e in report["errors"])


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_roundtrip(tmp_path: Path, suffix: str) -> None:
    frame = make_oof()
    path = tmp_path / f"oof{suffix}"
    write_oof(frame, path)
    loaded = read_oof(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == len(frame)
    assert_valid_oof(loaded, LABELS)
