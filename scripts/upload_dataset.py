#!/usr/bin/env python3
"""Создать/обновить Kaggle-датасет и при необходимости опубликовать его.

По умолчанию датасет создаётся **приватным**. С флагом ``--public`` скрипт
сначала загружает приватную версию, проверяет её скачиванием и только потом
переводит в public через ``kaggle datasets metadata --update``.

Примеры::

    python scripts/upload_dataset.py --public
    python scripts/upload_dataset.py --version -m "fix split"
    python scripts/upload_dataset.py --verify-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "dataset" / "build"
DEFAULT_REF = "morginalium/name-that-llm-clean-pool-v1"
DEFAULT_MESSAGE = "Name That LLM clean pool + frozen splits v1"


def kaggle_bin() -> str:
    found = shutil.which("kaggle")
    if not found:
        raise SystemExit("kaggle CLI не найден в PATH")
    return found


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, check=False, text=True, capture_output=capture)
    if check and result.returncode != 0:
        detail = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"команда {' '.join(cmd)} завершилась с кодом {result.returncode}\n{detail}")
    return result


def read_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("info", data)


def dataset_exists(ref: str) -> bool:
    result = run([kaggle_bin(), "datasets", "files", "-d", ref], check=False)
    return result.returncode == 0


def wait_ready(ref: str, timeout: int = 1800) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        result = run([kaggle_bin(), "datasets", "status", ref], check=False)
        text = ((result.stdout or "") + (result.stderr or "")).strip()
        last = text
        low = text.lower()
        if "ready" in low:
            return text
        # Пока версия обрабатывается, приватный статус может отдавать 403.
        if "403" in low or "forbidden" in low:
            time.sleep(10)
            continue
        if "error" in low or "failed" in low:
            raise RuntimeError(f"Kaggle status: {text}")
        time.sleep(10)
    raise TimeoutError(f"датасет не стал ready за {timeout}s; последний статус: {last}")


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def validate_local(build: Path, skip: bool) -> None:
    if skip:
        print("  validate: пропущен (--skip-validate)")
        return
    print("  validate: scripts/validate_dataset.py")
    run([sys.executable, str(ROOT / "scripts" / "validate_dataset.py"), "--dir", str(build)])


def upload(build: Path, ref: str, message: str, mode: str) -> str:
    if mode == "auto":
        mode = "version" if dataset_exists(ref) else "create"
    if mode == "create":
        print("  создаю приватный датасет")
        run([kaggle_bin(), "datasets", "create", "-p", str(build), "--keep-tabular", "--quiet"])
    else:
        print("  создаю новую версию")
        run([
            kaggle_bin(), "datasets", "version", "-p", str(build),
            "-m", message, "--keep-tabular", "--quiet",
        ])
    return mode


def verify_download(ref: str, build: Path, expected_files: list[str]) -> list[str]:
    """Скачать датасет и сверить SHA256 всех файлов, кроме metadata.

    Kaggle иногда не отдаёт общий zip для только что созданного приватного
    датасета, хотя отдельные файлы доступны. Поэтому при неудаче полного
    скачивания проверяем файлы по одному.
    """
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ntl_kaggle_verify_") as tmp:
        tmp_path = Path(tmp)
        print(f"  скачиваю обратно в {tmp_path}")
        result = run(
            [kaggle_bin(), "datasets", "download", "-d", ref, "-p", str(tmp_path), "--unzip", "--quiet", "-o"],
            check=False,
        )
        downloaded = {p.name: p for p in tmp_path.iterdir() if p.is_file()}
        if result.returncode != 0 or not downloaded:
            print("  общий zip недоступен, качаю файлы по одному")
            for p in tmp_path.iterdir():
                if p.is_file():
                    p.unlink()
            for name in expected_files:
                if name == "dataset-metadata.json":
                    continue
                one = run(
                    [kaggle_bin(), "datasets", "download", "-d", ref, "-f", name, "-p", str(tmp_path), "--quiet", "-o"],
                    check=False,
                )
                if one.returncode != 0:
                    errors.append(f"не скачивается файл: {name}")
            downloaded = {p.name: p for p in tmp_path.iterdir() if p.is_file()}

        for name in expected_files:
            if name == "dataset-metadata.json":
                continue
            src = build / name
            dst = downloaded.get(name)
            if dst is None:
                errors.append(f"нет в скачанном датасете: {name}")
                continue
            if file_sha256(src) != file_sha256(dst):
                errors.append(f"хэш не совпадает после скачивания: {name}")
        extra = set(downloaded) - set(expected_files) - {"dataset-metadata.json"}
        if extra:
            errors.append(f"лишние файлы в датасете: {sorted(extra)}")
    return errors


def publish(ref: str, build: Path) -> None:
    """Перевести датасет в public через metadata --update."""
    metadata = read_metadata(build / "dataset-metadata.json")
    if metadata.get("id") != ref:
        raise ValueError(f"metadata id={metadata.get('id')!r}, ожидается {ref}")
    public_meta = dict(metadata)
    public_meta["isPrivate"] = False
    public_meta.setdefault("title", metadata.get("title"))
    with tempfile.TemporaryDirectory(prefix="ntl_public_meta_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "dataset-metadata.json").write_text(
            json.dumps(public_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("  перевожу в public")
        run([kaggle_bin(), "datasets", "metadata", ref, "-p", str(tmp_path), "--update"])

        # Проверяем, что Kaggle действительно отдаёт isPrivate=false.
        check_dir = tmp_path / "check"
        check_dir.mkdir()
        run([kaggle_bin(), "datasets", "metadata", ref, "-p", str(check_dir)])
        remote = read_metadata(check_dir / "dataset-metadata.json")
        # Для public Kaggle не отдаёт поле isPrivate вовсе; private -> True.
        if remote.get("isPrivate") is True:
            raise RuntimeError(f"после update isPrivate={remote.get('isPrivate')!r}, ожидается False/None")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--public", action="store_true", help="после проверки перевести в public")
    parser.add_argument("--mode", choices=["auto", "create", "version"], default="auto")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--publish-only", action="store_true", help="только перевести уже загруженный датасет в public")
    args = parser.parse_args()

    build = args.dir
    manifest = build / "MANIFEST.sha256"
    if not manifest.exists():
        raise SystemExit(f"нет {manifest}; сначала запусти scripts/build_dataset.py")
    if args.publish_only:
        publish(args.ref, build)
        print("visibility: public")
        print(f"URL: https://www.kaggle.com/datasets/{args.ref}")
        return 0
    expected_files = [
        line.split("  ", 1)[1]
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Манифест тоже лежит в датасете, но не хэширует сам себя.
    expected_files.append("MANIFEST.sha256")

    if args.verify_only:
        print(f"проверяю {args.ref}")
        errors = verify_download(args.ref, build, expected_files)
        if errors:
            print("  первая попытка не сошлась, повторяю через 15 с")
            time.sleep(15)
            errors = verify_download(args.ref, build, expected_files)
        if errors:
            for error in errors:
                print(f"ОШИБКА: {error}")
            return 1
        print("OK: скачанный датасет совпадает с локальным")
        return 0

    print(f"датасет: {args.ref}")
    validate_local(build, args.skip_validate)
    mode = upload(build, args.ref, args.message, args.mode)
    print(f"  загружено ({mode}), жду processing")
    print(f"  статус: {wait_ready(args.ref)}")

    if not args.skip_verify:
        errors = verify_download(args.ref, build, expected_files)
        if errors:
            print("  первая проверка не сошлась, повторяю через 15 с")
            time.sleep(15)
            errors = verify_download(args.ref, build, expected_files)
        if errors:
            for error in errors:
                print(f"ОШИБКА: {error}")
            print("ПРОВАЛ: скачанная версия не совпадает; public не включаю")
            return 2
        print("  проверка скачивания: OK")

    if args.public:
        publish(args.ref, build)
        print("  visibility: public")
    else:
        print("  visibility: private (--public не передан)")

    print(f"URL: https://www.kaggle.com/datasets/{args.ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
