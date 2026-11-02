"""Загрузка и валидация замороженных сплитов Name That LLM.

Два сплита:

* ``splits/official_v1.json`` — 150 prompt_id соревнования, 5 блоков по 30.
  Блоки 0–3 — dev, блок 4 — sealed. В файле нет текстов, только id → block.
* ``splits/pool_v1.parquet`` — 5 фолдов по ``prompt_canon`` внешнего пула.
  Один канонический промпт целиком попадает в один фолд.

Модуль не читает данные соревнования, если их не передали явно. Все проверки
возвращают словарь с ``ok`` и списком ``errors``; исключения не бросаются,
чтобы вызывающий код мог напечатать все проблемы сразу.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SPLIT_PATH = ROOT / "splits" / "official_v1.json"
POOL_SPLIT_PATH = ROOT / "splits" / "pool_v1.parquet"

N_PROMPTS = 150
N_ROWS = 2699
N_BLOCKS = 5
N_FOLDS = 5
BLOCK_SIZE = 30
N_MODELS = 18
SEALED_BLOCK = 4

OFFICIAL_KEYS = {
    "version",
    "train_sha256",
    "n_prompts",
    "n_rows",
    "method",
    "seed",
    "blocks",
    "roles",
}
POOL_COLUMNS = [
    "prompt_canon",
    "fold",
    "n_rows",
    "n_clean",
    "n_models",
    "n_sources",
    "source_top",
]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 файла без загрузки его целиком в память."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_official_split(path: str | Path = OFFICIAL_SPLIT_PATH) -> dict:
    """Прочитать официальный JSON-сплит."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pool_split(path: str | Path = POOL_SPLIT_PATH) -> pd.DataFrame:
    """Прочитать parquet со сплитом пула."""
    return pd.read_parquet(path)


def official_block_map(split: dict) -> dict[str, int]:
    """prompt_id → block, с проверкой типов и уникальности."""
    blocks = split.get("blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("official split: blocks должен быть объектом prompt_id → block")
    out = {}
    for prompt_id, block in blocks.items():
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"official split: некорректный prompt_id {prompt_id!r}")
        if not isinstance(block, int) or isinstance(block, bool):
            raise ValueError(f"official split: block для {prompt_id} не int: {block!r}")
        out[prompt_id] = block
    return out


def validate_official_split(
    split: dict,
    train: pd.DataFrame | None = None,
    train_sha256: str | None = None,
) -> dict:
    """Проверить официальный сплит; при наличии — сверить с train.csv.

    Проверки: 150 промптов, по 30 на блок, роли dev/sealed, отсутствие
    пересечений, 2699 строк и раскладка 149×18 + 1×17. Текст не нужен.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = OFFICIAL_KEYS - set(split)
    if missing:
        errors.append(f"нет ключей: {sorted(missing)}")
    extra = set(split) - OFFICIAL_KEYS
    if extra:
        warnings.append(f"лишние ключи (не ошибка): {sorted(extra)}")

    if split.get("n_prompts") != N_PROMPTS:
        errors.append(f"n_prompts={split.get('n_prompts')!r}, ожидается {N_PROMPTS}")
    if split.get("n_rows") != N_ROWS:
        errors.append(f"n_rows={split.get('n_rows')!r}, ожидается {N_ROWS}")

    try:
        blocks = official_block_map(split)
    except ValueError as exc:
        return {"ok": False, "errors": errors + [str(exc)], "warnings": warnings}

    if len(blocks) != N_PROMPTS:
        errors.append(f"blocks: {len(blocks)} промптов, ожидается {N_PROMPTS}")

    block_sizes = {b: 0 for b in range(N_BLOCKS)}
    unknown_blocks = sorted(set(blocks.values()) - set(range(N_BLOCKS)))
    if unknown_blocks:
        errors.append(f"неизвестные блоки: {unknown_blocks}")
    for block in blocks.values():
        if block in block_sizes:
            block_sizes[block] += 1
    for block, size in block_sizes.items():
        if size != BLOCK_SIZE:
            errors.append(f"блок {block}: {size} промптов, ожидается {BLOCK_SIZE}")

    roles = split.get("roles", {})
    if not isinstance(roles, dict):
        errors.append("roles должен быть объектом")
    else:
        dev = roles.get("dev")
        sealed = roles.get("sealed")
        if dev != [0, 1, 2, 3]:
            errors.append(f"roles.dev={dev!r}, ожидается [0, 1, 2, 3]")
        if sealed != [SEALED_BLOCK]:
            errors.append(f"roles.sealed={sealed!r}, ожидается [{SEALED_BLOCK}]")

    n_rows = 0
    block_rows = {b: 0 for b in range(N_BLOCKS)}
    if train is not None:
        required = {"id", "prompt_id", "text", "model"}
        missing_cols = required - set(train.columns)
        if missing_cols:
            errors.append(f"train: нет колонок {sorted(missing_cols)}")
        else:
            if len(train) != N_ROWS:
                errors.append(f"train: {len(train)} строк, ожидается {N_ROWS}")
            if train["id"].duplicated().any():
                errors.append("train: дубли id")
            if train["model"].nunique() != N_MODELS:
                errors.append(f"train: {train['model'].nunique()} моделей, ожидается {N_MODELS}")

            train_prompts = set(train["prompt_id"].astype(str))
            if train_prompts != set(blocks):
                missing_p = sorted(train_prompts - set(blocks))[:10]
                extra_p = sorted(set(blocks) - train_prompts)[:10]
                errors.append(f"train/split: промпты не совпадают; нет в split {missing_p}, лишние {extra_p}")

            sizes = train.groupby("prompt_id").size()
            n_rows = int(sizes.sum())
            if n_rows != N_ROWS:
                errors.append(f"train: сумма строк по промптам {n_rows}, ожидается {N_ROWS}")
            if sorted(sizes.tolist()) != [17] + [18] * 149:
                errors.append("train: ожидается 149 промптов по 18 строк и один по 17")
            for prompt_id, block in blocks.items():
                if block in block_rows:
                    block_rows[block] += int(sizes.get(prompt_id, 0))
            if train_sha256 is not None and split.get("train_sha256") != train_sha256:
                errors.append("train_sha256 не совпадает с фактическим файлом")
            if split.get("train_sha256") and train_sha256 is None:
                warnings.append("train_sha256 не сверен: хэш файла не передан")

    if n_rows and n_rows != N_ROWS:
        errors.append(f"n_rows сплита {split.get('n_rows')}, посчитано {n_rows}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_prompts": len(blocks),
        "n_rows": n_rows or split.get("n_rows"),
        "block_sizes": block_sizes,
        "block_rows": block_rows,
        "roles": {"dev": [0, 1, 2, 3], "sealed": [SEALED_BLOCK]},
    }


def validate_pool_split(pool_split: pd.DataFrame, pool: pd.DataFrame | None = None) -> dict:
    """Проверить сплит пула и, если передан пул, пересчитать статистику.

    Проверки: колонки, уникальность prompt_canon, fold 0–4, непустой canon,
    n_rows >= n_clean >= 1, все 18 классов в каждом фолде, отсутствие canon
    в двух фолдах, согласованность n_rows/n_clean/n_models/n_sources/source_top.
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in POOL_COLUMNS if c not in pool_split.columns]
    if missing_cols:
        return {"ok": False, "errors": [f"нет колонок: {missing_cols}"], "warnings": warnings}

    split = pool_split[POOL_COLUMNS].copy()
    if split["prompt_canon"].isna().any() or (split["prompt_canon"].astype(str) == "").any():
        errors.append("prompt_canon: пустые значения")
    if split["prompt_canon"].duplicated().any():
        errors.append(f"prompt_canon: {int(split['prompt_canon'].duplicated().sum())} дублей")

    if not pd.api.types.is_integer_dtype(split["fold"]):
        errors.append(f"fold: тип {split['fold'].dtype}, ожидается целый")
    bad_folds = sorted(set(split["fold"].dropna().astype(int)) - set(range(N_FOLDS)))
    if bad_folds:
        errors.append(f"fold: недопустимые значения {bad_folds}")

    for col in ("n_rows", "n_clean", "n_models", "n_sources"):
        if not pd.api.types.is_integer_dtype(split[col]):
            errors.append(f"{col}: тип {split[col].dtype}, ожидается целый")
        if (split[col] < 0).any():
            errors.append(f"{col}: отрицательные значения")
    if (split["n_rows"] < split["n_clean"]).any():
        errors.append("n_rows < n_clean хотя бы для одного canon")
    if (split["n_clean"] < 1).any():
        errors.append("n_clean < 1 хотя бы для одного canon")
    if (split["n_models"] < 1).any() or (split["n_sources"] < 1).any():
        errors.append("n_models/n_sources < 1 хотя бы для одного canon")
    if split["source_top"].isna().any() or (split["source_top"].astype(str) == "").any():
        errors.append("source_top: пустые значения")

    fold_sizes = {}
    fold_classes = {}
    fold_sources = {}
    canon_leakage = 0
    if pool is not None:
        required = {"id", "prompt_canon", "model", "source_dataset", "flag_kb_proxy"}
        missing = required - set(pool.columns)
        if missing:
            errors.append(f"pool: нет колонок {sorted(missing)}")
        else:
            clean = pool[(~pool["flag_kb_proxy"]) & (pool["prompt_canon"].astype(str) != "")]
            if set(clean["prompt_canon"]) != set(split["prompt_canon"]):
                only_pool = sorted(set(clean["prompt_canon"]) - set(split["prompt_canon"]))[:10]
                only_split = sorted(set(split["prompt_canon"]) - set(clean["prompt_canon"]))[:10]
                errors.append(
                    "canon: набор не совпадает; нет в split "
                    f"{only_pool}, лишние в split {only_split}"
                )

            merged = clean.merge(split[["prompt_canon", "fold"]], on="prompt_canon", how="left")
            if merged["fold"].isna().any():
                errors.append(f"canon без фолда: {int(merged['fold'].isna().sum())} строк")
            canon_folds = merged.groupby("prompt_canon")["fold"].nunique()
            canon_leakage = int((canon_folds > 1).sum())
            if canon_leakage:
                errors.append(f"canon в нескольких фолдах: {canon_leakage}")

            clean_stats = merged.groupby("prompt_canon").agg(
                n_clean=("id", "size"),
                n_models=("model", "nunique"),
                n_sources=("source_dataset", "nunique"),
            )
            top = (
                merged.groupby(["prompt_canon", "source_dataset"]).size().rename("n").reset_index()
                .sort_values(["prompt_canon", "n", "source_dataset"], ascending=[True, False, True])
                .drop_duplicates("prompt_canon")
                .set_index("prompt_canon")["source_dataset"]
            )
            clean_stats["source_top"] = top
            full_counts = pool[pool["prompt_canon"].astype(str) != ""].groupby("prompt_canon").size()
            clean_stats["n_rows_full"] = clean_stats.index.map(full_counts).astype(int)
            clean_stats = clean_stats.reset_index().merge(
                split[["prompt_canon", "n_rows", "n_clean", "n_models", "n_sources", "source_top", "fold"]],
                on="prompt_canon",
                how="left",
                suffixes=("", "_split"),
            )
            for _, row in clean_stats.iterrows():
                expected = (
                    int(row["n_rows_full"]),
                    int(row["n_clean"]),
                    int(row["n_models"]),
                    int(row["n_sources"]),
                    row["source_top"],
                )
                actual = (
                    int(row["n_rows"]),
                    int(row["n_clean_split"]),
                    int(row["n_models_split"]),
                    int(row["n_sources_split"]),
                    row["source_top_split"],
                )
                if expected != actual:
                    errors.append(f"статистика canon {row['prompt_canon']}: split {actual}, pool {expected}")
                    break

            class_counts = pd.crosstab(merged["model"], merged["fold"])
            for fold in range(N_FOLDS):
                fold_sizes[fold] = int((merged["fold"] == fold).sum())
                fold_classes[fold] = int(merged.loc[merged["fold"] == fold, "model"].nunique())
                fold_sources[fold] = int(merged.loc[merged["fold"] == fold, "source_dataset"].nunique())
                if fold_classes[fold] != N_MODELS:
                    errors.append(f"фолд {fold}: {fold_classes[fold]} классов, ожидается {N_MODELS}")
                if fold in class_counts.columns and (class_counts[fold] == 0).any():
                    empty = sorted(class_counts.index[class_counts[fold] == 0].tolist())
                    errors.append(f"фолд {fold}: нет классов {empty}")

            if len(fold_sizes) == N_FOLDS:
                sizes = list(fold_sizes.values())
                if max(sizes) - min(sizes) > 0.01 * (sum(sizes) / N_FOLDS):
                    warnings.append(f"фолды не идеально сбалансированы: {sizes}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_prompts": int(len(split)),
        "n_rows": int(split["n_rows"].sum()),
        "n_clean": int(split["n_clean"].sum()),
        "fold_sizes": fold_sizes,
        "fold_classes": fold_classes,
        "fold_sources": fold_sources,
        "canon_leakage": canon_leakage,
    }


def assert_valid_official_split(split: dict, train: pd.DataFrame | None = None) -> dict:
    """Как validate_official_split, но бросает ValueError при ошибках."""
    report = validate_official_split(split, train=train)
    if not report["ok"]:
        raise ValueError("official split невалиден: " + "; ".join(report["errors"]))
    return report


def assert_valid_pool_split(pool_split: pd.DataFrame, pool: pd.DataFrame | None = None) -> dict:
    """Как validate_pool_split, но бросает ValueError при ошибках."""
    report = validate_pool_split(pool_split, pool=pool)
    if not report["ok"]:
        raise ValueError("pool split невалиден: " + "; ".join(report["errors"]))
    return report
