# Воспроизведение

Инструкция для человека, который клонировал репозиторий и хочет получить
те же числа. Данные соревнования не входят в git; их нужно скачать с
Kaggle, приняв правила.

## 1. Окружение

- Python **3.12+** (CI — 3.12; локально проверено на 3.13/3.14).
- [uv](https://docs.astral.sh/uv/) для окружения; pip тоже подходит.
- Для baseline/стека: `numpy`, `pandas`, `scikit-learn`, `pyarrow`,
  `pytest`; для стека — `torch`, `transformers`; для RNN — `torch`.
- Kaggle CLI — только для скачивания данных и сабмитов, не для обучения.

```bash
git clone https://github.com/alexkolesnikov08/name-that-llm.git
cd name-that-llm
uv venv --python 3.12
uv pip install numpy pandas scikit-learn pyarrow pytest
# опционально:
uv pip install torch transformers
```

## 2. Данные

### Официальные

1. Примите правила соревнования `name-that-llm` на Kaggle.
2. Скачайте `train.csv`, `test.csv` (и `sample_submission.parquet`) в корень
   репозитория или в отдельный каталог.

```bash
export NTL_DATA_DIR=.        # или путь к каталогу с train.csv/test.csv
```

Локальный `test.csv` — 3 строки-заглушки. Настоящий тест доступен только
в Kaggle rerun; локально сабмит на нём не имеет смысла.

Проверка:

```bash
python - <<'PY'
import pandas as pd
tr = pd.read_csv("train.csv")
assert tr.shape == (2699, 4), tr.shape
assert tr.prompt_id.nunique() == 150
assert tr.model.nunique() == 18
print("ok", tr.shape, tr.prompt_id.nunique(), tr.model.nunique())
PY
```

### Пул

Kaggle dataset `morginalium/name-that-llm-clean-pool-v1` (**публикуется**,
CC BY-SA 4.0). Скачайте и положите `pool_v1.parquet` в `splits/` либо
передайте путь вручную.

Сплиты:

- `splits/official_v1.json` — 5 блоков × 30, блок 4 sealed;
- `splits/pool_v1.parquet` — 5 фолдов по `prompt_canon`, по строке на
  канонический промпт;
- `splits/v2.json` — исторические 30 holdout + 120 dev.

Если официальные артефакты отсутствуют, `splits/v2.json` можно пересобрать:

```bash
PYTHONPATH=src python src/make_split.py
# читает train.csv из NTL_DATA_DIR, пишет splits/v2.json
```

`official_v1.json` — артефакт этапа data/split; его состав фиксируется
отдельно и не должен пересобираться при каждом запуске (иначе sealed
перестаёт быть sealed). Полная сборка и проверка:

```bash
python scripts/build_dataset.py      # из research/pool + train.csv
python scripts/validate_dataset.py   # схема, фолды, хэши; код 1 при ошибке
python scripts/upload_dataset.py --public
```

Скрипты требуют локальные `research/` и `train.csv` (в git их нет).

## 3. Тесты

```bash
PYTHONPATH=src pytest -q
```

Тесты, которым нужны `train.csv`, пул или веса моделей, помечены `skip`
и не ломают CI на чистом клоне. Smoke-тест проверяет импорт пакета и
базовые инварианты без данных.

## 4. Baseline

```bash
NTL_DATA_DIR=. python kaggle/baseline/main.py
```

Ожидаемо: TF-IDF `char_wb` 2–5 + LogReg, ~2 700 строк, CPU, меньше минуты.
На LB этот сабмит дал **0.299**. Файл `submission.parquet` пишется в
текущий каталог.

## 5. 7-view стек

Нужны веса `microsoft/deberta-v3-small` (локально — HF-кэш, на Kaggle —
датасет `eric0045/microsoft-deberta-v3-small-local`).

```bash
export NTL_DATA_DIR=.
export NTL_MODEL_DIR=/path/to/microsoft-deberta-v3-small
python kaggle/stack_v3/main.py
```

Что происходит: char/word TF-IDF, stylo, format, function words, skeleton,
frozen DeBERTa → OOF-вероятности по группам промптов → LogReg-мета.
Ожидаемо: dev BA(18) ≈0.39, LB **0.356**. На CPU может занять десятки
минут; на GPU быстрее.

## 6. Char-BiGRU

```bash
# быстрый smoke на 5k строк
python kaggle/rnn/train_rnn.py --limit 5000 --epochs 1 --max-steps 50

# полный прогон (GPU, ~40–60 мин на T4 при 40k строк)
python kaggle/rnn/train_rnn.py --limit 40000 --epochs 5
```

Скрипт ищет `pool_v1.parquet` в `/kaggle/input/**`, затем в `splits/`,
затем старый `research/pool/ntl_open_pool.parquet`. Официальный train
берётся из `NTL_DATA_DIR`. Лучшая модель сохраняется по BA(2) на
официальном holdout; ожидаемо ≈0.34–0.36.

## 7. Отдельные эксперименты

```bash
PYTHONPATH=src python src/hpo_tfidf.py          # Optuna, нужен runs/
PYTHONPATH=src python src/stack.py              # кэш runs/stack
PYTHONPATH=src python experiments/knn_center_tfidf.py
python experiments/pool_svc_baseline.py         # нужен research/pool
```

Эти команды предполагают локальные `runs/` и `research/`, которых нет в
git (см. `.gitignore`). Числа без них не воспроизводятся один-в-один, но
код и протокол фиксированы.

## 8. Воспроизведение чужого Gold Apex (опционально)

Честная граница: это **не наша модель**. Пайплайн публичный, мы только
обернули его для запуска.

```bash
# 1. достать ноутбук из тега (в main его нет)
git show gold-apex-fork:submit_gold_apex/name-that-llm-gold-apex-fork.ipynb \
  > kaggle/gold_apex/name-that-llm-gold-apex-fork.ipynb

# 2. залить commit-версию (3 строки, проверка входов)
kaggle kernels push -p kaggle/gold_apex

# 3. дождаться COMPLETE
kaggle kernels status morginalium/ntl-gold-apex-fork

# 4. отправить версию в соревнование (полный rerun ~8.3 ч, GPU T4x2)
kaggle competitions submit -c name-that-llm \
  -k morginalium/ntl-gold-apex-fork -v 1 -m "gold apex fork"
```

Требуется GPU-квота и доступ к 15 датасетам/8 kernel-источникам/8 моделям
из `kaggle/gold_apex/kernel-metadata.json`. Публичный исходник заявляет
0.670; наш прогон дал **0.662** (rank 432/4231). Расхождение объясняется
версиями рантайма и входами; алгоритм не менялся.

## 9. Сабмит

```bash
kaggle competitions submit -c name-that-llm \
  -f submission.parquet -m "7-view stack"
```

Формат: `id`, `model`; ровно те `id`, что в тесте. Проверка:

```bash
python - <<'PY'
import pandas as pd
sub = pd.read_parquet("submission.parquet")
test = pd.read_csv("test.csv")
assert list(sub.columns) == ["id", "model"]
assert set(sub.id) == set(test.id)
assert sub.model.isin(pd.read_csv("train.csv").model.unique()).all()
print("ok", len(sub))
PY
```

## 10. Чек-лист воспроизводимости

- [ ] `train.csv` — 2 699 × 4, 150 промптов, 18 классов.
- [ ] `NTL_DATA_DIR` указывает на каталог с данными.
- [ ] Сплит не пересекается по `prompt_id`; для пула — по `prompt_canon`.
- [ ] Сид зафиксирован; BA(2) усреднена по 25 draw-ам.
- [ ] Sealed-блок не использован для выбора.
- [ ] `submission.parquet` совпадает по `id` с тестом.
- [ ] Результат записан в `docs/experiments.md` с CI.

Если какой-то пункт не выполняется, число не считается воспроизведённым.
