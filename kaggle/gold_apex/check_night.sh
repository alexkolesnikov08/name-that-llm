#!/usr/bin/env bash
# Проверка ночного прогона: статус сабмита, скор, квота.
set -uo pipefail
REF="${1:-57881538}"
echo "=== submission $REF ==="
kaggle competitions submission "$REF"
echo "=== последние сабмиты ==="
kaggle competitions submissions -c name-that-llm | head -5
echo "=== GPU ==="
kaggle quota | head -3
