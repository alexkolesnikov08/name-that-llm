#!/usr/bin/env bash
# Ночной прогон форка: push -> commit -> submit в соревнование.
# Запускать только по команде пользователя.
set -euo pipefail
cd "$(dirname "$0")/../.."

KERNEL="morginalium/ntl-gold-apex-fork"
COMP="name-that-llm"
DIR="kaggle/gold_apex"

# если передан номер версии — не пушим заново, а сразу отправляем её
if [ $# -ge 1 ]; then
  VER="$1"
  echo "=== submit существующей версии $VER ==="
  kaggle competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -m "gold apex fork v$VER"
  kaggle competitions submission-limits -c "$COMP"
  exit 0
fi

echo "=== квота GPU ==="
kaggle quota

echo "=== push (commit-прогон на 3 строках, ~10 мин) ==="
PUSH_OUT=$(kaggle kernels push -p "$DIR")
echo "$PUSH_OUT"
VER=$(echo "$PUSH_OUT" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | tail -1)
if [ -z "${VER:-}" ]; then
  echo "не смог разобрать номер версии из вывода push" >&2
  exit 1
fi
echo "версия: $VER"

echo "=== ожидание COMPLETE (до 40 мин) ==="
for i in $(seq 1 40); do
  S=$(kaggle kernels status "$KERNEL" 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $S"
  case "$S" in
    *COMPLETE*) break ;;
    *ERROR*|*FAILED*|*CANCEL*) echo "кернел не завершился успешно, сабмит не делаю" >&2; exit 2 ;;
  esac
  sleep 60
done

echo "=== submit версии $VER (полный ре-ран ~8.3 ч) ==="
kaggle competitions submit -c "$COMP" -k "$KERNEL" -v "$VER" -m "gold apex fork v$VER"
echo "=== остаток сабмитов ==="
kaggle competitions submission-limits -c "$COMP"
