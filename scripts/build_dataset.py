#!/usr/bin/env python3
"""Собрать публичный Kaggle-датасет Name That LLM v1.

Источники:

* ``research/pool/ntl_open_pool.parquet`` — консервативный внешний пул,
  190 588 строк, 55 источников, 18 моделей;
* ``research/pool_kaggle/prompts.parquet`` — банк 13 021 промпта;
* ``train.csv`` — только для официального сплита и проверок, тексты в
  датасет не попадают;
* ``research/pool_kaggle/sources.csv``, ``removal_summary.csv``,
  ``removed_rows.csv`` — метаданные очистки.

Скрипт детерминирован: при тех же входных файлах получаются те же сплиты,
parquet-файлы и хэши (кроме timestamp в checks.json и отчёте).

Запуск::

    python scripts/build_dataset.py
    python scripts/build_dataset.py --out dataset/build --public-metadata
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl.splits import (  # noqa: E402
    POOL_COLUMNS,
    sha256_file,
    validate_official_split,
    validate_pool_split,
)

DEFAULT_POOL = ROOT / "research" / "pool" / "ntl_open_pool.parquet"
DEFAULT_TRAIN = ROOT / "train.csv"
DEFAULT_PROMPTS = ROOT / "research" / "pool_kaggle" / "prompts.parquet"
DEFAULT_SOURCES = ROOT / "research" / "pool_kaggle" / "sources.csv"
DEFAULT_REMOVAL_SUMMARY = ROOT / "research" / "pool_kaggle" / "removal_summary.csv"
DEFAULT_REMOVED_ROWS = ROOT / "research" / "pool_kaggle" / "removed_rows.csv"
DEFAULT_OUT = ROOT / "dataset" / "build"
DEFAULT_SPLITS = ROOT / "splits"
DEFAULT_REPORT = ROOT / "reports" / "build_dataset_v1.md"

EXPECTED_TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
EXPECTED_POOL_SHA256 = "76fc9f52f6e9c055092f7f4d352e026753982e75c46cb50e906bb27464271d71"
EXPECTED_POOL_ROWS = 190_588
EXPECTED_KEEP_CLEAN = 115_342
EXPECTED_CLEAN_ROWS = 114_670
EXPECTED_EMPTY_CANON_CLEAN = 672
EXPECTED_CLEAN_CANONS = 23_451
EXPECTED_BANK_PROMPTS = 13_021
EXPECTED_BANK_CLEAN_ROWS = 92_132
EXPECTED_BANK_IN_CLEAN = 10_607
EXPECTED_SOURCES = 55

SNAKE = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
SEED = 42

# Порядок классов фиксирован: он же используется в labels.json и OOF.
CLASS_ORDER = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_official_split(train: pd.DataFrame) -> dict:
    """Официальный сплит: сортировка по средней длине text, snake по 5 блокам.

    Длина считается в символах, а не в словах: средняя длина в словах имеет
    ничьи, а в символах уникальна. При равных значениях — tie-break по
    prompt_id, поэтому сплит полностью детерминирован.
    """
    required = {"id", "prompt_id", "text", "model"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"train.csv: нет колонок {sorted(missing)}")

    lengths = train.assign(_len=train["text"].str.len()).groupby("prompt_id")["_len"].mean()
    order = sorted(lengths.index, key=lambda p: (float(lengths[p]), str(p)))
    blocks = {str(prompt_id): SNAKE[i % len(SNAKE)] for i, prompt_id in enumerate(order)}

    return {
        "version": "official_v1",
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "n_prompts": int(len(blocks)),
        "n_rows": int(len(train)),
        "method": (
            "prompts sorted by mean character length of text within prompt "
            "(tie-break prompt_id); snake deal 0,1,2,3,4,4,3,2,1,0 over 5 blocks"
        ),
        "seed": SEED,
        "blocks": dict(sorted(blocks.items())),
        "roles": {"dev": [0, 1, 2, 3], "sealed": [4]},
    }


def make_pool_split(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """5 фолдов по prompt_canon, жадный LPT по числу чистых строк.

    Канонический промпт — единица разбиения, поэтому его строки никогда не
    попадают в разные фолды. Сортировка по убыванию n_clean и выбор самого
    лёгкого фолда дают почти идеальный баланс; при равенстве — меньший номер
    фолда. Пустой prompt_canon в сплит не входит.
    """
    clean = pool[(~pool["flag_kb_proxy"].astype(bool)) & (pool["prompt_canon"].astype(str) != "")]
    if clean.empty:
        raise ValueError("в пуле нет чистых строк с непустым prompt_canon")

    stats = (
        clean.groupby("prompt_canon")
        .agg(n_clean=("id", "size"), n_models=("model", "nunique"), n_sources=("source_dataset", "nunique"))
        .reset_index()
    )
    top = (
        clean.groupby(["prompt_canon", "source_dataset"]).size().rename("n").reset_index()
        .sort_values(["prompt_canon", "n", "source_dataset"], ascending=[True, False, True])
        .drop_duplicates("prompt_canon")
        .set_index("prompt_canon")["source_dataset"]
    )
    stats["source_top"] = stats["prompt_canon"].map(top)

    full_counts = pool[pool["prompt_canon"].astype(str) != ""].groupby("prompt_canon").size()
    stats["n_rows"] = stats["prompt_canon"].map(full_counts).astype(int)

    loads = [0] * 5
    fold_of: dict[str, int] = {}
    for row in stats.sort_values(["n_clean", "prompt_canon"], ascending=[False, True]).itertuples(index=False):
        fold = min(range(5), key=lambda k: (loads[k], k))
        fold_of[str(row.prompt_canon)] = fold
        loads[fold] += int(row.n_clean)

    stats["fold"] = stats["prompt_canon"].map(fold_of).astype(int)
    split = stats[POOL_COLUMNS].sort_values("prompt_canon").reset_index(drop=True)
    for col in ("fold", "n_rows", "n_clean", "n_models", "n_sources"):
        split[col] = split[col].astype("int64")

    if split["prompt_canon"].duplicated().any():
        raise AssertionError("pool split: prompt_canon продублирован")
    if set(split["fold"]) != set(range(5)):
        raise AssertionError(f"pool split: не все фолды присутствуют: {sorted(set(split['fold']))}")
    return split, fold_of


def make_labels(pool: pd.DataFrame) -> dict:
    model_to_lab = pool.drop_duplicates("model").set_index("model")["lab"].to_dict()
    missing = set(CLASS_ORDER) - set(model_to_lab)
    extra = set(model_to_lab) - set(CLASS_ORDER)
    if missing or extra:
        raise ValueError(f"классы не совпадают с CLASS_ORDER: нет {sorted(missing)}, лишние {sorted(extra)}")

    families: dict[str, list[str]] = {}
    for model in CLASS_ORDER:
        families.setdefault(model_to_lab[model], []).append(model)
    class_to_family = {model: model_to_lab[model] for model in CLASS_ORDER}
    return {
        "version": "v1",
        "n_classes": len(CLASS_ORDER),
        "n_families": len(families),
        "classes": CLASS_ORDER,
        "class_index": {model: i for i, model in enumerate(CLASS_ORDER)},
        "families": families,
        "class_to_family": class_to_family,
    }


def make_sources(sources: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    required = {"source_dataset", "source_license", "passages", "models"}
    missing = required - set(sources.columns)
    if missing:
        raise ValueError(f"sources.csv: нет колонок {sorted(missing)}")
    if len(sources) != EXPECTED_SOURCES:
        raise ValueError(f"sources.csv: {len(sources)} источников, ожидается {EXPECTED_SOURCES}")
    if int(sources["passages"].sum()) != EXPECTED_POOL_ROWS:
        raise ValueError("sources.csv: сумма passages не равна 190588")
    if set(sources["source_dataset"]) != set(pool["source_dataset"]):
        raise ValueError("sources.csv: список источников не совпадает с пулом")
    if sources["source_dataset"].duplicated().any():
        raise ValueError("sources.csv: дубли source_dataset")

    out = sources.copy()
    out["source_url"] = "https://www.kaggle.com/datasets/" + out["source_dataset"].astype(str)
    return out[["source_dataset", "source_url", "source_license", "passages", "models"]]


def license_counts(pool: pd.DataFrame) -> dict[str, int]:
    counts = pool["source_license"].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_readme(checks: dict, metadata: dict) -> str:
    counts = checks["counts"]
    folds = checks["pool_split"]["fold_sizes"]
    fold_text = ", ".join(f"{k}: {v:,}" for k, v in sorted(folds.items()))
    return f"""# Name That LLM — clean pool + frozen splits (v1)

A public, prompt-grouped corpus of LLM-generated passages for the Kaggle
competition **Name That LLM**: given a passage, identify which of 18 model
versions wrote it.

This dataset contains **external pool data only**. It does **not** contain the
competition `train.csv` or `test.csv`, and it does not contain any passage text
from them. The official split file maps competition `prompt_id` values to five
blocks; it contains no prompts and no passages.

## What is inside

| File | Rows | Description |
|---|---:|---|
| `ntl_pool_full_v1.parquet` | {counts['full_rows']:,} | Full pool with `keep_clean` and `split_fold` (0–4 or −1). |
| `ntl_pool_keep_clean_v1.parquet` | {counts['keep_clean_rows']:,} | Rows passing the conservative proxy filter (`keep_clean=True`). {counts['empty_canon_clean']} rows have no `prompt_canon` and `split_fold=-1`. |
| `ntl_pool_clean_v1.parquet` | {counts['clean_rows']:,} | Clean, foldable table: `keep_clean=True` and non-empty `prompt_canon`; all rows have a fold. |
| `prompt_bank_v1.parquet` | {counts['bank_prompts']:,} | Prompt bank with `n_clean_rows` and `in_clean`. |
| `official_split_v1.json` | 150 prompts | Competition `prompt_id` → block (0–3 dev, 4 sealed). No text. |
| `pool_split_v1.parquet` | {counts['pool_prompts']:,} | Prompt-level fold table for the clean pool. |
| `labels.json` | 18 classes | Class order, indices, 6 families and `class_to_family`. |
| `sources.csv` | 55 sources | Per-source URL, licence, passages and model count. |
| `removal_summary.csv` | 12 rows | Upstream removal reasons and row counts. |
| `removed_rows.csv` | {counts['removed_rows']:,} | Removed `id` and `reason` only. |
| `ATTRIBUTION.md` | — | All 55 sources, links and licences. |

Counts in one line: **{counts['full_rows']:,}** raw rows, **{counts['keep_clean_rows']:,}**
passing `keep_clean`, **{counts['clean_rows']:,}** foldable clean rows,
**{counts['pool_prompts']:,}** canonical prompts, **18** models, **6** families.

### Why 115,342 vs 114,670

`keep_clean = not flag_kb_proxy` yields **{counts['keep_clean_rows']:,}** rows.
Of these, **{counts['empty_canon_clean']:,}** rows from a single source have an
empty `prompt_canon`: their prompt identity could not be recovered. They remain
in `ntl_pool_keep_clean_v1.parquet` with `split_fold=-1`, but they are excluded
from `ntl_pool_clean_v1.parquet` and from the 5-fold split, because a row with
no prompt identity cannot be assigned to a prompt-disjoint fold.

## Columns

The pool files keep the upstream columns:

| Column | Meaning |
|---|---|
| `id` | Stable row id (`ntlop_*`). |
| `text` | Generated passage. |
| `model` | One of the 18 target classes. |
| `lab` | Model family: `anthropic`, `google`, `openai`, `zai`, `moonshot`, `qwen`. |
| `prompt_group` | Source-local prompt group. Not comparable across sources. |
| `source_dataset` | Kaggle source reference (`owner/slug`). |
| `source_file` | File inside the source dataset. |
| `source_license` | Upstream licence string. |
| `origin_hint` | Upstream provenance hint, when available. |
| `n_words` | Word count. |
| `also_in` | Other sources containing the same passage. |
| `n_sources` | Number of sources containing the passage. |
| `max_cos_train` | Maximum cosine similarity to competition train passages. |
| `max_sim_train_prompt` | Maximum similarity to competition train prompts. |
| `style_score` | Upstream style score. |
| `flag_kb_proxy` | Knowledge-base/proxy generation flag. |
| `flag_gemini37_routing_risk` | Upstream routing-risk flag. |
| `flag_near_train_prompt` | Near-duplicate competition prompt flag. |
| `flag_train_prompt_regen` | Regeneration of a competition train prompt. |
| `prompt_canon` | Canonical prompt identity (`p*` or source-local `s*`). |
| `panel_labels` | Number of models available for the prompt. |
| `prompt_evidence` | How the prompt identity was recovered. |
| `match_weight` | Upstream prompt-match weight. |
| `keep_clean` | **Added.** `not flag_kb_proxy`. |
| `split_fold` | **Added.** 0–4 for rows whose `prompt_canon` is in the clean split, else −1. |

`ntl_pool_clean_v1.parquet` contains the same columns; every row has
`keep_clean=True`, non-empty `prompt_canon` and `split_fold` in 0–4.

## Cleaning

The upstream pool applied conservative filtering before this release:

- removed off-genre sources, API dumps, unverified Gemini 3.7 routing, unavailable
  sources, refusals, licence-risk rows, off-format and truncated passages,
  typography mismatches, and rows overlapping competition train prompts;
- deduplicated passages across sources;
- kept a conservative `flag_kb_proxy=False` subset.

`removal_summary.csv` and `removed_rows.csv` document the removals. This release
does not re-run the upstream cleaning; it freezes the conservative pool and adds
splits. See `reports/build_dataset_v1.md` in the repository for the build report.

## Splits

### Official split (`official_split_v1.json`)

150 competition `prompt_id` values are sorted by mean character length of
`text` within the prompt (tie-break: `prompt_id`) and dealt in a snake pattern
`0,1,2,3,4,4,3,2,1,0` into five blocks of 30:

- blocks **0–3 = dev** (120 prompts, 2,159 rows),
- block **4 = sealed** (30 prompts, 540 rows).

The file contains only `prompt_id → block`. It contains no prompt text, no
passage text and no competition rows. Use the sealed block once, for a final
report; do not tune on it.

### Pool split (`pool_split_v1.parquet`)

Five folds by `prompt_canon`, balanced by clean row count with a deterministic
largest-first greedy assignment. A canonical prompt is never split across folds.
Clean fold sizes: {fold_text}. Every fold contains all 18 classes.

`prompt_canon` is the only valid grouping key for prompt-disjoint validation.
`prompt_group` is source-local and must not be used across sources.

## Usage

```python
import json
import pandas as pd

clean = pd.read_parquet("ntl_pool_clean_v1.parquet")
folds = pd.read_parquet("pool_split_v1.parquet")
labels = json.load(open("labels.json"))

# Prompt-disjoint CV: fit on four folds, validate on the fifth.
for k in range(5):
    train = clean[clean.split_fold != k]
    valid = clean[clean.split_fold == k]
    ...
```

For OOF predictions use one row per clean passage:

```
id, prompt_id, model, fold, p_<class 1>, ..., p_<class 18>
```

`prompt_id` must equal `prompt_canon`, `fold` must match `pool_split_v1.parquet`,
and each `p_*` row must sum to 1. The helper `src/ntl/oof.py` in the repository
reads, writes and validates this contract.

## What is not included

- Competition `train.csv` and `test.csv`, in whole or in part.
- Any competition passage text, prompt text or model labels.
- Any hidden test data or leaderboard-target information.
- Model weights or trained checkpoints.

Only the competition **prompt identifiers** appear in `official_split_v1.json`,
mapped to blocks. No text can be reconstructed from that file.

## Licence and attribution

The dataset is released under **CC BY-SA 4.0**. Upstream sources use CC BY 4.0,
CC BY-SA 4.0 or CC0; see `sources.csv` and `ATTRIBUTION.md` for the complete
list of 55 sources, links and licences. You must preserve attribution and share
derivatives under the same licence.

Original pool: [crtproduction/name-that-llm-open-pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool).

## Links

- Kaggle dataset: <https://www.kaggle.com/datasets/morginalium/name-that-llm-clean-pool-v1>
- Competition: <https://www.kaggle.com/competitions/name-that-llm>
- Original open pool: <https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool>
- Code and report: <https://github.com/alexkolesnikov08/name-that-llm>

## Citation

If you use this dataset, cite the original open pool and this release:

```text
Name That LLM — clean pool + frozen splits (v1)
https://www.kaggle.com/datasets/morginalium/name-that-llm-clean-pool-v1
Based on crtproduction/name-that-llm-open-pool (CC BY-SA 4.0).
```
"""


def build_attribution(sources: pd.DataFrame) -> str:
    spdx = {
        "Attribution 4.0 International (CC BY 4.0)": "CC-BY-4.0",
        "CC-BY-SA-4.0": "CC-BY-SA-4.0",
        "CC0-1.0": "CC0-1.0",
    }
    rows = []
    for i, row in enumerate(sources.itertuples(index=False), 1):
        rows.append(
            [
                i,
                f"[{row.source_dataset}]({row.source_url})",
                row.source_license,
                spdx.get(row.source_license, row.source_license),
                f"{int(row.passages):,}",
                int(row.models),
            ]
        )
    table = markdown_table(
        ["#", "Source", "Upstream licence", "SPDX", "Passages", "Models"],
        rows,
    )
    return f"""# Attribution — Name That LLM v1

This dataset is a cleaned, split and repackaged derivative of **55 public
Kaggle datasets**, assembled in
[crtproduction/name-that-llm-open-pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool).
Please credit the original pool and every upstream source you use.

The release as a whole is distributed under **CC BY-SA 4.0**. The upstream
licences are CC BY 4.0, CC BY-SA 4.0 and CC0 1.0; the table below preserves the
exact licence string from `sources.csv` and adds an SPDX identifier.

## Original pool

- **crtproduction/name-that-llm-open-pool** — <https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool>
- Licence: **CC BY-SA 4.0**
- This release uses its conservative 190,588-row pool and its 13,021-prompt
  bank; we add canonical-prompt folds, an official split and dataset packaging.

## Upstream sources

{table}

## Notes

- The table lists every source that contributed at least one row to the
  190,588-row pool. Some sources contain rows that were later filtered by the
  conservative `flag_kb_proxy` flag; `sources.csv` reports the pool counts.
- Each source's licence applies to its own passages. The combined dataset is
  CC BY-SA 4.0 because it includes CC BY-SA material and because the original
  pool is CC BY-SA 4.0.
- If you redistribute a derivative, keep this file, `sources.csv` and the
  original pool credit.

## Competition data

**The competition `train.csv` and `test.csv` are not included in this
dataset.** No competition passage text, prompt text or labels are included.
`official_split_v1.json` contains only competition `prompt_id → block`
assignments (150 identifiers), which reveal no text. The competition rules
remain in force for the competition data; this dataset is external pool data.
"""


def build_metadata(checks: dict) -> dict:
    c = checks["counts"]
    description = f"""Clean, prompt-grouped pool of LLM-generated passages for **Name That LLM**, with frozen prompt-level splits.

**Contents.** {c['full_rows']:,} raw external-pool rows from 55 public Kaggle datasets; {c['keep_clean_rows']:,} rows pass the conservative proxy filter; {c['clean_rows']:,} foldable clean rows across {c['pool_prompts']:,} canonical prompts; 18 model classes in 6 families. The clean table has five prompt-disjoint folds of {c['clean_rows'] // 5:,} rows each.

**Splits.** `official_split_v1.json` maps the 150 competition prompt IDs to five blocks of 30 (blocks 0–3 dev, block 4 sealed) without any text. `pool_split_v1.parquet` assigns every canonical prompt to one of five folds; a prompt never crosses folds and every fold has all 18 classes.

**Not included.** Competition `train.csv`/`test.csv`, competition passage or prompt text, hidden test data, and model weights. Only prompt IDs appear in the official split.

**Licence.** CC BY-SA 4.0. All 55 upstream sources, links and licences are listed in `ATTRIBUTION.md` and `sources.csv`. Original pool: [crtproduction/name-that-llm-open-pool](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool).

Built by `scripts/build_dataset.py`; see `checks.json`, `MANIFEST.sha256` and the repository report."""
    return {
        "title": "Name That LLM — clean pool + frozen splits (v1)",
        "id": "morginalium/name-that-llm-clean-pool-v1",
        "subtitle": "190k LLM passages, 18 models, frozen prompt-level splits",
        "description": description,
        "isPrivate": True,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "keywords": [
            "nlp",
            "text classification",
            "deep learning",
            "classification",
            "text",
            "artificial intelligence",
        ],
    }


def compute_checks(
    *,
    train: pd.DataFrame,
    pool: pd.DataFrame,
    full: pd.DataFrame,
    clean: pd.DataFrame,
    keep_clean: pd.DataFrame,
    bank: pd.DataFrame,
    official: dict,
    pool_split: pd.DataFrame,
    sources: pd.DataFrame,
    removed: pd.DataFrame,
    removal_summary: pd.DataFrame,
    labels: dict,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    block_series = pd.Series(official["blocks"])
    official_report = validate_official_split(official, train=train, train_sha256=EXPECTED_TRAIN_SHA256)
    pool_report = validate_pool_split(pool_split, pool=pool)
    if not official_report["ok"]:
        errors.extend(f"official: {e}" for e in official_report["errors"])
    if not pool_report["ok"]:
        errors.extend(f"pool: {e}" for e in pool_report["errors"])

    # --- базовые инварианты -------------------------------------------------
    if len(full) != EXPECTED_POOL_ROWS:
        errors.append(f"full: {len(full)} строк, ожидается {EXPECTED_POOL_ROWS}")
    if int(full["keep_clean"].sum()) != EXPECTED_KEEP_CLEAN:
        errors.append(f"keep_clean: {int(full['keep_clean'].sum())}, ожидается {EXPECTED_KEEP_CLEAN}")
    if len(keep_clean) != EXPECTED_KEEP_CLEAN:
        errors.append(f"keep_clean table: {len(keep_clean)}, ожидается {EXPECTED_KEEP_CLEAN}")
    if len(clean) != EXPECTED_CLEAN_ROWS:
        errors.append(f"clean: {len(clean)} строк, ожидается {EXPECTED_CLEAN_ROWS}")
    if clean["prompt_canon"].nunique() != EXPECTED_CLEAN_CANONS:
        errors.append(f"clean: {clean['prompt_canon'].nunique()} canon, ожидается {EXPECTED_CLEAN_CANONS}")
    if int((full["prompt_canon"] == "").sum()) != 1925:
        errors.append("full: неожиданное число пустых prompt_canon")
    if int(((full["prompt_canon"] == "") & (full["split_fold"] != -1)).sum()):
        errors.append("full: у пустого prompt_canon не -1")
    if int((clean["split_fold"] < 0).sum()):
        errors.append("clean: есть строки без фолда")
    if int((clean["prompt_canon"] == "").sum()):
        errors.append("clean: есть пустой prompt_canon")

    # clean должен быть точным подмножеством full
    full_ids = set(full["id"])
    clean_ids = set(clean["id"])
    if not clean_ids <= full_ids:
        errors.append(f"clean: {len(clean_ids - full_ids)} id отсутствуют в full")
    if len(clean_ids) != len(clean):
        errors.append("clean: дубли id")
    merged_fold = full.set_index("id").loc[clean["id"], "split_fold"].to_numpy()
    if not np.array_equal(merged_fold, clean["split_fold"].to_numpy()):
        errors.append("clean/full: split_fold расходятся для одинаковых id")

    # prompt bank
    if len(bank) != EXPECTED_BANK_PROMPTS:
        errors.append(f"prompt bank: {len(bank)} строк, ожидается {EXPECTED_BANK_PROMPTS}")
    if int(bank["n_clean_rows"].sum()) != EXPECTED_BANK_CLEAN_ROWS:
        errors.append(f"prompt bank: n_clean_rows sum={int(bank['n_clean_rows'].sum())}, ожидается {EXPECTED_BANK_CLEAN_ROWS}")
    if int(bank["in_clean"].sum()) != EXPECTED_BANK_IN_CLEAN:
        errors.append(f"prompt bank: in_clean={int(bank['in_clean'].sum())}, ожидается {EXPECTED_BANK_IN_CLEAN}")

    # официальный train не должен пересекаться с пулом
    train_texts = set(train["text"].astype(str))
    pool_texts = set(pool["text"].astype(str))
    text_overlap = len(train_texts & pool_texts)
    if text_overlap:
        errors.append(f"train/pool: {text_overlap} одинаковых текстов")
    prompt_id_overlap = len(set(train["prompt_id"].astype(str)) & set(pool["prompt_canon"].astype(str)))
    if prompt_id_overlap:
        errors.append(f"train/pool: {prompt_id_overlap} одинаковых prompt id")

    # --- распределения ------------------------------------------------------
    class_dist = clean.groupby("model").size().sort_values(ascending=False)
    fold_class = pd.crosstab(clean["model"], clean["split_fold"])
    fold_sizes = {int(k): int(v) for k, v in clean.groupby("split_fold").size().items()}
    fold_models = {int(k): int(v) for k, v in clean.groupby("split_fold")["model"].nunique().items()}
    fold_sources = {int(k): int(v) for k, v in clean.groupby("split_fold")["source_dataset"].nunique().items()}

    source_fold_presence = clean.groupby("source_dataset")["split_fold"].nunique()
    large_sources = clean.groupby("source_dataset").size()
    large_sources = large_sources[large_sources >= 500].index
    unsplit_sources = sorted(
        source_fold_presence[source_fold_presence < 5].index.intersection(large_sources).tolist()
    )
    if unsplit_sources:
        warnings.append(f"крупные источники не во всех 5 фолдах: {unsplit_sources}")

    removed_total = int(
        removal_summary.loc[removal_summary["reason"] != "(kept in v11)", "rows_removed"].sum()
    )

    if len(fold_sizes) != 5:
        errors.append(f"clean: {len(fold_sizes)} фолдов, ожидается 5")
    if any(v != 18 for v in fold_models.values()):
        errors.append(f"clean: не все 18 классов в каждом фолде: {fold_models}")
    if len(set(fold_sizes.values())) > 1:
        warnings.append(f"фолды не равны: {fold_sizes}")

    # --- хэши ---------------------------------------------------------------
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    output_hashes = {path.name: sha256_file(path) for path in outputs.values()}

    expected_inputs = {"pool": EXPECTED_POOL_SHA256, "train": EXPECTED_TRAIN_SHA256}
    for name, expected in expected_inputs.items():
        if input_hashes.get(name) != expected:
            errors.append(f"input hash {name}: {input_hashes.get(name)}, ожидается {expected}")

    return {
        "version": 1,
        "generated_at": utc_now(),
        "generated_by": "scripts/build_dataset.py",
        "status": "ok" if not errors else "failed",
        "counts": {
            "full_rows": int(len(full)),
            "full_columns": int(len(full.columns)),
            "keep_clean_rows": int(full["keep_clean"].sum()),
            "clean_rows": int(len(clean)),
            "empty_canon_clean": int((keep_clean["prompt_canon"] == "").sum()),
            "full_prompts_nonempty": int(full.loc[full["prompt_canon"] != "", "prompt_canon"].nunique()),
            "pool_prompts": int(len(pool_split)),
            "bank_prompts": int(len(bank)),
            "bank_clean_rows": int(bank["n_clean_rows"].sum()),
            "bank_in_clean": int(bank["in_clean"].sum()),
            "train_rows": int(len(train)),
            "train_prompts": int(train["prompt_id"].nunique()),
            "train_models": int(train["model"].nunique()),
            "sources": int(len(sources)),
            "removed_rows": int(len(removed)),
            "removed_summary_rows": removed_total,
        },
        "official_split": {
            "n_prompts": int(official["n_prompts"]),
            "n_rows": int(official["n_rows"]),
            "block_sizes": official_report["block_sizes"],
            "block_rows": official_report["block_rows"],
            "roles": official["roles"],
            "method": official["method"],
            "train_sha256": official["train_sha256"],
            "report": official_report,
        },
        "pool_split": {
            "n_prompts": int(len(pool_split)),
            "n_rows": int(pool_split["n_rows"].sum()),
            "n_clean": int(pool_split["n_clean"].sum()),
            "fold_sizes": fold_sizes,
            "fold_models": fold_models,
            "fold_sources": fold_sources,
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
            "fold_class_distribution": {
                str(fold): {str(k): int(v) for k, v in fold_class[fold].items()} for fold in fold_class.columns
            },
            "source_fold_presence": {str(k): int(v) for k, v in source_fold_presence.items()},
            "report": pool_report,
        },
        "licenses": license_counts(pool),
        "licenses_keep_clean": license_counts(keep_clean),
        "licenses_clean": license_counts(clean),
        "labels": {
            "n_classes": labels["n_classes"],
            "n_families": labels["n_families"],
            "classes": labels["classes"],
            "families": labels["families"],
        },
        "overlap_checks": {
            "canon_in_multiple_folds": int((clean.groupby("prompt_canon")["split_fold"].nunique() > 1).sum()),
            "official_prompt_in_multiple_blocks": int(block_series.index.duplicated().sum()),
            "train_pool_exact_text_overlap": int(text_overlap),
            "train_pool_prompt_id_overlap": int(prompt_id_overlap),
            "full_duplicate_ids": int(full["id"].duplicated().sum()),
            "clean_duplicate_ids": int(clean["id"].duplicated().sum()),
            "clean_rows_without_fold": int((clean["split_fold"] < 0).sum()),
            "empty_canon_with_fold": int(((full["prompt_canon"] == "") & (full["split_fold"] != -1)).sum()),
        },
        "hashes": {"inputs": input_hashes, "outputs": output_hashes},
        "validation": {"errors": errors, "warnings": warnings},
    }


def build_report(checks: dict) -> str:
    c = checks["counts"]
    folds = {int(k): int(v) for k, v in checks["pool_split"]["fold_sizes"].items()}
    fold_models = {int(k): int(v) for k, v in checks["pool_split"]["fold_models"].items()}
    fold_sources = {int(k): int(v) for k, v in checks["pool_split"]["fold_sources"].items()}
    fold_rows = "\n".join(
        f"| {fold} | {size:,} | {fold_models[fold]} | {fold_sources[fold]} |"
        for fold, size in sorted(folds.items())
    )
    class_rows = "\n".join(
        f"| `{model}` | {n:,} |" for model, n in checks["pool_split"]["class_distribution"].items()
    )
    output_rows = "\n".join(
        f"| `{name}` | `{digest[:16]}…` |" for name, digest in sorted(checks["hashes"]["outputs"].items())
    )
    warnings = checks["validation"]["warnings"]
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- нет"
    status = "OK" if checks["status"] == "ok" else "FAILED"
    return f"""# Сборка датасета Name That LLM v1

**Статус:** {status}  
**Сгенерировано:** {checks['generated_at']}  
**Скрипт:** `scripts/build_dataset.py`

## Счётчики

| Что | Значение |
|---|---:|
| Строк в полном пуле | {c['full_rows']:,} |
| `keep_clean=True` | {c['keep_clean_rows']:,} |
| Строк в foldable clean | {c['clean_rows']:,} |
| Пустой `prompt_canon` в clean | {c['empty_canon_clean']} |
| Канонических промптов в clean | {c['pool_prompts']:,} |
| Промптов в банке | {c['bank_prompts']:,} |
| Из них в clean | {c['bank_in_clean']:,} |
| Источников | {c['sources']} |
| Удалённых строк (upstream) | {c['removed_rows']:,} |

## Официальный сплит

- 150 промптов, 5 блоков по 30.
- Строк: {checks['official_split']['n_rows']:,}.
- Блоки 0–3 — dev, блок 4 — sealed.
- Сортировка: средняя длина `text` в символах, snake `0,1,2,3,4,4,3,2,1,0`.
- В JSON только `prompt_id → block`, без текста.

## Фолды пула

| Фолд | Clean-строк | Классов | Источников |
|---:|---:|---:|---:|
{fold_rows}

Проверка: ни один `prompt_canon` не встречается в двух фолдах; все 18 классов
есть в каждом фолде; `split_fold=-1` только у строк вне clean-сплита.

## Распределение классов (clean)

| Класс | Строк |
|---|---:|
{class_rows}

## Лицензии источников

Полный пул:

{chr(10).join(f"- {k}: {v:,}" for k, v in checks['licenses'].items())}

`keep_clean=True` (115 342):

{chr(10).join(f"- {k}: {v:,}" for k, v in checks['licenses_keep_clean'].items())}

Foldable clean (114 670):

{chr(10).join(f"- {k}: {v:,}" for k, v in checks['licenses_clean'].items())}

## Проверки пересечений

{chr(10).join(f"- {k}: {v}" for k, v in checks['overlap_checks'].items())}

## Хэши выходных файлов

| Файл | SHA256 (начало) |
|---|---|
{output_rows}

Полные хэши — в `checks.json` и `MANIFEST.sha256`.

## Предупреждения

{warning_text}

## Замечание о 115 342 vs 114 670

`keep_clean` даёт {c['keep_clean_rows']:,} строк. Из них {c['empty_canon_clean']}
строк имеют пустой `prompt_canon`; они лежат в `ntl_pool_keep_clean_v1.parquet`
с `split_fold=-1`, но не входят в `ntl_pool_clean_v1.parquet` и в фолды, потому
что промпт-дизъюнктный фолд для них невозможен.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--removal-summary", type=Path, default=DEFAULT_REMOVAL_SUMMARY)
    parser.add_argument("--removed-rows", type=Path, default=DEFAULT_REMOVED_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    for path in (args.pool, args.train, args.prompts, args.sources, args.removal_summary, args.removed_rows):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/7] читаю входы")
    pool = pd.read_parquet(args.pool)
    train = pd.read_csv(args.train, dtype={"id": str, "prompt_id": str, "model": str, "text": str})
    bank = pd.read_parquet(args.prompts)
    sources = pd.read_csv(args.sources)
    removal_summary = pd.read_csv(args.removal_summary)
    removed = pd.read_csv(args.removed_rows, dtype={"id": str, "reason": str})

    train_sha = sha256_file(args.train)
    pool_sha = sha256_file(args.pool)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"train.csv SHA256 {train_sha}, ожидается {EXPECTED_TRAIN_SHA256}")
    if pool_sha != EXPECTED_POOL_SHA256:
        raise ValueError(f"pool SHA256 {pool_sha}, ожидается {EXPECTED_POOL_SHA256}")

    print("[2/7] официальный сплит")
    official = make_official_split(train)
    official_report = validate_official_split(official, train=train)
    if not official_report["ok"]:
        raise ValueError("official split: " + "; ".join(official_report["errors"]))

    print("[3/7] фолды пула")
    pool_split, fold_of = make_pool_split(pool)
    pool_report = validate_pool_split(pool_split, pool=pool)
    if not pool_report["ok"]:
        raise ValueError("pool split: " + "; ".join(pool_report["errors"]))

    print("[4/7] собираю таблицы")
    original_columns = list(pool.columns)
    full = pool.copy()
    full["keep_clean"] = ~full["flag_kb_proxy"].astype(bool)
    full["split_fold"] = full["prompt_canon"].map(fold_of).fillna(-1).astype("int16")
    full = full[original_columns + ["keep_clean", "split_fold"]]
    keep_clean = full[full["keep_clean"]].copy()
    clean = full[full["keep_clean"] & (full["prompt_canon"].astype(str) != "")].copy()

    bank = bank.copy()
    bank["n_clean_rows"] = bank["prompt_canon"].map(clean.groupby("prompt_canon").size()).fillna(0).astype("int64")
    bank["in_clean"] = bank["n_clean_rows"] > 0

    labels = make_labels(pool)
    sources_out = make_sources(sources, pool)
    removed_out = removed[["id", "reason"]].copy()
    if removed_out["id"].duplicated().any():
        raise ValueError("removed_rows.csv: дубли id")

    print("[5/7] пишу splits/")
    args.splits_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.splits_dir / "official_v1.json", official)
    pool_split.to_parquet(args.splits_dir / "pool_v1.parquet", index=False, compression="zstd")
    write_json(args.splits_dir / "labels.json", labels)

    print("[6/7] пишу dataset/build/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".build_dataset_", dir=args.out.parent))
    try:
        full.to_parquet(tmp / "ntl_pool_full_v1.parquet", index=False, compression="zstd")
        clean.to_parquet(tmp / "ntl_pool_clean_v1.parquet", index=False, compression="zstd")
        keep_clean.to_parquet(tmp / "ntl_pool_keep_clean_v1.parquet", index=False, compression="zstd")
        bank.to_parquet(tmp / "prompt_bank_v1.parquet", index=False, compression="zstd")
        shutil.copy2(args.splits_dir / "official_v1.json", tmp / "official_split_v1.json")
        shutil.copy2(args.splits_dir / "pool_v1.parquet", tmp / "pool_split_v1.parquet")
        write_json(tmp / "labels.json", labels)
        sources_out.to_csv(tmp / "sources.csv", index=False)
        removal_summary.to_csv(tmp / "removal_summary.csv", index=False)
        removed_out.to_csv(tmp / "removed_rows.csv", index=False)

        # Первый проход checks — по файлам данных без README/ATTRIBUTION, чтобы
        # получить числа для документации.
        outputs_for_hash = {
            p.name: p
            for p in tmp.iterdir()
            if p.is_file() and p.name not in {"checks.json", "MANIFEST.sha256"}
        }
        checks = compute_checks(
            train=train,
            pool=pool,
            full=full,
            clean=clean,
            keep_clean=keep_clean,
            bank=bank,
            official=official,
            pool_split=pool_split,
            sources=sources_out,
            removed=removed_out,
            removal_summary=removal_summary,
            labels=labels,
            inputs={
                "pool": args.pool,
                "train": args.train,
                "prompts": args.prompts,
                "sources": args.sources,
                "removal_summary": args.removal_summary,
                "removed_rows": args.removed_rows,
            },
            outputs=outputs_for_hash,
        )
        if checks["status"] != "ok":
            raise ValueError("checks не прошли: " + "; ".join(checks["validation"]["errors"]))

        metadata = build_metadata(checks)
        (tmp / "README.md").write_text(build_readme(checks, metadata), encoding="utf-8")
        (tmp / "ATTRIBUTION.md").write_text(build_attribution(sources_out), encoding="utf-8")
        write_json(tmp / "dataset-metadata.json", metadata)

        # Второй проход — с хэшами README/ATTRIBUTION/metadata.
        outputs_for_hash = {
            p.name: p
            for p in tmp.iterdir()
            if p.is_file() and p.name not in {"checks.json", "MANIFEST.sha256"}
        }
        checks = compute_checks(
            train=train,
            pool=pool,
            full=full,
            clean=clean,
            keep_clean=keep_clean,
            bank=bank,
            official=official,
            pool_split=pool_split,
            sources=sources_out,
            removed=removed_out,
            removal_summary=removal_summary,
            labels=labels,
            inputs={
                "pool": args.pool,
                "train": args.train,
                "prompts": args.prompts,
                "sources": args.sources,
                "removal_summary": args.removal_summary,
                "removed_rows": args.removed_rows,
            },
            outputs=outputs_for_hash,
        )
        if checks["status"] != "ok":
            raise ValueError("checks не прошли: " + "; ".join(checks["validation"]["errors"]))
        write_json(tmp / "checks.json", checks)

        manifest_files = sorted(
            p.name
            for p in tmp.iterdir()
            if p.is_file() and p.name not in {"MANIFEST.sha256", "dataset-metadata.json"}
        )
        manifest = "\n".join(f"{sha256_file(tmp / name)}  {name}" for name in manifest_files) + "\n"
        (tmp / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")

        if args.out.exists():
            shutil.rmtree(args.out)
        tmp.rename(args.out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(checks), encoding="utf-8")

    print("[7/7] готово")
    print(f"  полный пул:  {len(full):,} строк")
    print(f"  keep_clean:  {int(full['keep_clean'].sum()):,} строк")
    print(f"  clean:       {len(clean):,} строк, {clean['prompt_canon'].nunique():,} промптов")
    print(f"  фолды:       { {int(k): int(v) for k, v in clean.groupby('split_fold').size().items()} }")
    print(f"  банк:        {len(bank):,} промптов")
    print(f"  источники:   {len(sources_out)}")
    print(f"  отчёт:       {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
