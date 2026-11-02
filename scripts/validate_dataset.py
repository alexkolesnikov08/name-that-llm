#!/usr/bin/env python3
"""Проверить собранный датасет Name That LLM v1.

Запуск::

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --dir dataset/build

Код возврата 1, если есть хотя бы одна ошибка. Предупреждения не влияют на
код возврата, но печатаются.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl.splits import (  # noqa: E402
    POOL_COLUMNS,
    load_official_split,
    load_pool_split,
    sha256_file,
    validate_official_split,
    validate_pool_split,
)

DEFAULT_BUILD = ROOT / "dataset" / "build"
DEFAULT_TRAIN = ROOT / "train.csv"
DEFAULT_POOL = ROOT / "research" / "pool" / "ntl_open_pool.parquet"

EXPECTED_FILES = {
    "ntl_pool_full_v1.parquet",
    "ntl_pool_keep_clean_v1.parquet",
    "ntl_pool_clean_v1.parquet",
    "prompt_bank_v1.parquet",
    "official_split_v1.json",
    "pool_split_v1.parquet",
    "labels.json",
    "sources.csv",
    "removal_summary.csv",
    "removed_rows.csv",
    "README.md",
    "ATTRIBUTION.md",
    "MANIFEST.sha256",
    "checks.json",
    "dataset-metadata.json",
}

EXPECTED_COUNTS = {
    "full_rows": 190_588,
    "keep_clean_rows": 115_342,
    "clean_rows": 114_670,
    "empty_canon_clean": 672,
    "pool_prompts": 23_451,
    "bank_prompts": 13_021,
    "bank_clean_rows": 92_132,
    "bank_in_clean": 10_607,
    "sources": 55,
    "removed_rows": 88_508,
    "train_rows": 2_699,
    "train_prompts": 150,
}


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "  " not in line:
            raise ValueError(f"MANIFEST.sha256:{lineno}: нет двух пробелов")
        digest, name = line.split("  ", 1)
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"MANIFEST.sha256:{lineno}: некорректный SHA256")
        if name in entries:
            raise ValueError(f"MANIFEST.sha256:{lineno}: дубль {name}")
        entries[name] = digest
    return entries


def recursive_keys(obj) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(str(key))
            keys |= recursive_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            keys |= recursive_keys(value)
    return keys


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    args = parser.parse_args()

    build = args.dir
    errors: list[str] = []
    warnings: list[str] = []

    if not build.is_dir():
        print(f"ОШИБКА: нет каталога {build}")
        return 1

    # --- состав файлов ------------------------------------------------------
    actual = {p.name for p in build.iterdir() if p.is_file()}
    missing = EXPECTED_FILES - actual
    extra = actual - EXPECTED_FILES
    check(not missing, f"нет файлов: {sorted(missing)}", errors)
    if extra:
        warnings.append(f"лишние файлы (не входят в ожидаемый набор): {sorted(extra)}")

    if errors:
        for error in errors:
            print(f"ОШИБКА: {error}")
        return 1

    # --- manifest и checks --------------------------------------------------
    try:
        manifest = parse_manifest(build / "MANIFEST.sha256")
    except ValueError as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    manifest_names = set(manifest)
    files_for_manifest = actual - {"MANIFEST.sha256", "dataset-metadata.json"}
    check(
        manifest_names == files_for_manifest,
        f"MANIFEST: состав не совпадает; нет {sorted(files_for_manifest - manifest_names)}, "
        f"лишние {sorted(manifest_names - files_for_manifest)}",
        errors,
    )
    for name, expected in manifest.items():
        path = build / name
        if path.is_file() and sha256_file(path) != expected:
            errors.append(f"MANIFEST: хэш не совпадает: {name}")

    checks = json.loads((build / "checks.json").read_text(encoding="utf-8"))
    check(checks.get("status") == "ok", f"checks.status={checks.get('status')!r}", errors)
    check(not checks.get("validation", {}).get("errors"), "checks.validation.errors не пуст", errors)
    for warning in checks.get("validation", {}).get("warnings", []):
        warnings.append(f"checks: {warning}")

    for name, expected in checks.get("hashes", {}).get("outputs", {}).items():
        path = build / name
        if not path.is_file():
            errors.append(f"checks: нет выходного файла {name}")
        elif sha256_file(path) != expected:
            errors.append(f"checks: хэш выхода не совпадает: {name}")

    input_paths = {
        "pool": args.pool,
        "train": args.train,
        "prompts": ROOT / "research" / "pool_kaggle" / "prompts.parquet",
        "sources": ROOT / "research" / "pool_kaggle" / "sources.csv",
        "removal_summary": ROOT / "research" / "pool_kaggle" / "removal_summary.csv",
        "removed_rows": ROOT / "research" / "pool_kaggle" / "removed_rows.csv",
    }
    for name, expected in checks.get("hashes", {}).get("inputs", {}).items():
        path = input_paths.get(name)
        if path is None:
            warnings.append(f"checks: неизвестный вход {name}")
        elif path.exists() and sha256_file(path) != expected:
            errors.append(f"checks: хэш входа не совпадает: {name}")

    # --- таблицы ------------------------------------------------------------
    full = pd.read_parquet(build / "ntl_pool_full_v1.parquet")
    clean = pd.read_parquet(build / "ntl_pool_clean_v1.parquet")
    keep_clean = pd.read_parquet(build / "ntl_pool_keep_clean_v1.parquet")
    bank = pd.read_parquet(build / "prompt_bank_v1.parquet")
    pool_split = pd.read_parquet(build / "pool_split_v1.parquet")
    official = json.loads((build / "official_split_v1.json").read_text(encoding="utf-8"))
    labels = json.loads((build / "labels.json").read_text(encoding="utf-8"))
    sources = pd.read_csv(build / "sources.csv")
    removal_summary = pd.read_csv(build / "removal_summary.csv")
    removed = pd.read_csv(build / "removed_rows.csv", dtype={"id": str, "reason": str})
    metadata = json.loads((build / "dataset-metadata.json").read_text(encoding="utf-8"))

    for key, expected in EXPECTED_COUNTS.items():
        actual_value = checks.get("counts", {}).get(key)
        check(actual_value == expected, f"checks.counts.{key}={actual_value}, ожидается {expected}", errors)

    check(len(full) == 190_588, f"full: {len(full)} строк", errors)
    check(len(clean) == 114_670, f"clean: {len(clean)} строк", errors)
    check(len(keep_clean) == 115_342, f"keep_clean: {len(keep_clean)} строк", errors)
    check(len(bank) == 13_021, f"prompt_bank: {len(bank)} строк", errors)
    check(len(sources) == 55, f"sources: {len(sources)} строк", errors)
    check(len(removed) == 88_508, f"removed_rows: {len(removed)} строк", errors)

    expected_columns = [
        "id", "text", "model", "lab", "prompt_group", "source_dataset", "source_file",
        "source_license", "origin_hint", "n_words", "also_in", "n_sources",
        "max_cos_train", "max_sim_train_prompt", "style_score", "flag_kb_proxy",
        "flag_gemini37_routing_risk", "flag_near_train_prompt", "flag_train_prompt_regen",
        "prompt_canon", "panel_labels", "prompt_evidence", "match_weight",
        "keep_clean", "split_fold",
    ]
    check(list(full.columns) == expected_columns, f"full: колонки {list(full.columns)}", errors)
    check(list(clean.columns) == expected_columns, "clean: набор колонок отличается от full", errors)
    check(list(keep_clean.columns) == expected_columns, "keep_clean: набор колонок отличается от full", errors)
    check(list(pool_split.columns) == POOL_COLUMNS, f"pool_split: колонки {list(pool_split.columns)}", errors)

    if "keep_clean" in full:
        check(full["keep_clean"].dtype == bool, "full: keep_clean не bool", errors)
        check(int(full["keep_clean"].sum()) == 115_342, "full: неверная сумма keep_clean", errors)
        check(
            bool((full["keep_clean"] == ~full["flag_kb_proxy"].astype(bool)).all()),
            "full: keep_clean != ~flag_kb_proxy",
            errors,
        )
    if "split_fold" in full:
        check(
            set(full["split_fold"].unique()) <= {-1, 0, 1, 2, 3, 4},
            f"full: недопустимые split_fold {sorted(set(full['split_fold'].unique()))}",
            errors,
        )
        check(int(((full["prompt_canon"] == "") & (full["split_fold"] != -1)).sum()) == 0,
              "full: пустой prompt_canon с фолдом", errors)
    check(len(full) == full["id"].nunique(), "full: дубли id", errors)
    check(len(clean) == clean["id"].nunique(), "clean: дубли id", errors)
    check((clean["prompt_canon"].astype(str) != "").all(), "clean: пустой prompt_canon", errors)
    check((clean["split_fold"] >= 0).all(), "clean: есть split_fold=-1", errors)
    check(bool(clean["keep_clean"].all()), "clean: есть keep_clean=False", errors)
    check(set(clean["id"]) <= set(full["id"]), "clean: id не из full", errors)
    if set(clean["id"]) <= set(full["id"]):
        fold_map = full.set_index("id")["split_fold"]
        check(
            bool((clean.set_index("id")["split_fold"] == fold_map.loc[clean["id"]].to_numpy()).all()),
            "clean/full: split_fold расходятся",
            errors,
        )
    check(clean["prompt_canon"].nunique() == 23_451, "clean: неверное число canon", errors)

    check(len(bank) == bank["prompt_canon"].nunique(), "prompt_bank: дубли prompt_canon", errors)
    check(int(bank["n_clean_rows"].sum()) == 92_132, "prompt_bank: n_clean_rows sum", errors)
    check(int(bank["in_clean"].sum()) == 10_607, "prompt_bank: in_clean sum", errors)

    # --- сплиты -------------------------------------------------------------
    official_report = validate_official_split(
        official,
        train=pd.read_csv(args.train, dtype={"id": str, "prompt_id": str, "model": str, "text": str})
        if args.train.exists() else None,
        train_sha256=sha256_file(args.train) if args.train.exists() else None,
    )
    if not official_report["ok"]:
        errors.extend(f"official: {e}" for e in official_report["errors"])
    else:
        check(official_report["block_sizes"] == {0: 30, 1: 30, 2: 30, 3: 30, 4: 30},
              f"official: блоки {official_report['block_sizes']}", errors)

    pool = pd.read_parquet(args.pool) if args.pool.exists() else None
    pool_report = validate_pool_split(pool_split, pool=pool)
    if not pool_report["ok"]:
        errors.extend(f"pool: {e}" for e in pool_report["errors"])
    else:
        check(pool_report["fold_sizes"] == {0: 22934, 1: 22934, 2: 22934, 3: 22934, 4: 22934},
              f"pool: фолды {pool_report['fold_sizes']}", errors)

    # официальный JSON не должен содержать текст
    forbidden_keys = {"text", "passage", "prompt_text", "answer", "content"}
    leaked = recursive_keys(official) & forbidden_keys
    check(not leaked, f"official: запрещённые ключи {sorted(leaked)}", errors)
    check(set(official) == {
        "version", "train_sha256", "n_prompts", "n_rows", "method", "seed", "blocks", "roles"
    }, f"official: неожиданные ключи {sorted(set(official))}", errors)
    check(len(official.get("blocks", {})) == 150, "official: не 150 промптов", errors)

    # --- метаданные ---------------------------------------------------------
    check(metadata.get("id") == "morginalium/name-that-llm-clean-pool-v1", "metadata: id", errors)
    check(metadata.get("title") == "Name That LLM — clean pool + frozen splits (v1)", "metadata: title", errors)
    check(metadata.get("isPrivate") is True, "metadata: isPrivate должен быть true в build", errors)
    check(metadata.get("licenses") == [{"name": "CC-BY-SA-4.0"}], "metadata: лицензия", errors)
    check(isinstance(metadata.get("description"), str) and len(metadata["description"]) > 200,
          "metadata: description слишком короткий", errors)
    check(len(metadata.get("keywords", [])) >= 3, "metadata: мало keywords", errors)

    check(len(labels.get("classes", [])) == 18, "labels: не 18 классов", errors)
    check(len(labels.get("families", {})) == 6, "labels: не 6 семейств", errors)
    check(sum(len(v) for v in labels.get("families", {}).values()) == 18,
          "labels: семейства не покрывают 18 классов", errors)
    check(set(labels.get("classes", [])) == set(full["model"].unique()),
          "labels: классы не совпадают с пулом", errors)

    check(len(sources) == sources["source_dataset"].nunique(), "sources: дубли", errors)
    check(int(sources["passages"].sum()) == 190_588, "sources: сумма passages", errors)
    check(all(str(u).startswith("https://www.kaggle.com/datasets/") for u in sources["source_url"]),
          "sources: неверный source_url", errors)
    check(set(sources["source_dataset"]) == set(full["source_dataset"]),
          "sources: источники не совпадают с пулом", errors)

    check(list(removed.columns) == ["id", "reason"], f"removed_rows: колонки {list(removed.columns)}", errors)
    check(len(removed) == removed["id"].nunique(), "removed_rows: дубли id", errors)
    check(int(removal_summary.loc[removal_summary["reason"] != "(kept in v11)", "rows_removed"].sum()) == 88_508,
          "removal_summary: сумма удалённых", errors)
    check("(kept in v11)" in set(removal_summary["reason"]), "removal_summary: нет строки kept", errors)

    # --- лицензии и запрещённые данные --------------------------------------
    clean_licenses = clean["source_license"].value_counts().to_dict()
    check(clean_licenses == {
        "Attribution 4.0 International (CC BY 4.0)": 91_701,
        "CC-BY-SA-4.0": 22_525,
        "CC0-1.0": 444,
    }, f"clean: лицензии {clean_licenses}", errors)
    keep_clean_licenses = keep_clean["source_license"].value_counts().to_dict()
    check(keep_clean_licenses == {
        "Attribution 4.0 International (CC BY 4.0)": 91_701,
        "CC-BY-SA-4.0": 23_197,
        "CC0-1.0": 444,
    }, f"keep_clean: лицензии {keep_clean_licenses}", errors)
    forbidden_files = {"train.csv", "test.csv", "sample_submission.parquet"}
    check(not (actual & forbidden_files), f"в build лежат данные соревнования: {sorted(actual & forbidden_files)}", errors)

    attribution = (build / "ATTRIBUTION.md").read_text(encoding="utf-8")
    check("crtproduction/name-that-llm-open-pool" in attribution, "ATTRIBUTION: нет оригинального пула", errors)
    check("CC BY-SA 4.0" in attribution, "ATTRIBUTION: нет лицензии", errors)
    for ref in sources["source_dataset"]:
        if str(ref) not in attribution:
            errors.append(f"ATTRIBUTION: нет источника {ref}")
            break
    readme = (build / "README.md").read_text(encoding="utf-8")
    check("not included" in readme.lower() or "not contain" in readme.lower(),
          "README: нет заявления об отсутствии competition data", errors)

    # --- итог ---------------------------------------------------------------
    for warning in warnings:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {warning}")
    if errors:
        for error in errors:
            print(f"ОШИБКА: {error}")
        print(f"\nПРОВАЛ: {len(errors)} ошибок")
        return 1

    print("OK: датасет валиден")
    print(f"  файлов: {len(actual)}")
    print(f"  full: {len(full):,} | keep_clean: {len(keep_clean):,} | clean: {len(clean):,}")
    print(f"  промптов: {len(pool_split):,} | фолды: {pool_report['fold_sizes']}")
    print(f"  официальный сплит: {official_report['block_sizes']} блоков, {official_report['n_rows']:,} строк")
    print(f"  источников: {len(sources)} | удалённых строк: {len(removed):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
