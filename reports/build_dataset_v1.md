# Сборка датасета Name That LLM v1

**Статус:** OK  
**Сгенерировано:** 2026-11-02T09:03:42Z  
**Скрипт:** `scripts/build_dataset.py`

## Счётчики

| Что | Значение |
|---|---:|
| Строк в полном пуле | 190,588 |
| `keep_clean=True` | 115,342 |
| Строк в foldable clean | 114,670 |
| Пустой `prompt_canon` в clean | 672 |
| Канонических промптов в clean | 23,451 |
| Промптов в банке | 13,021 |
| Из них в clean | 10,607 |
| Источников | 55 |
| Удалённых строк (upstream) | 88,508 |

## Официальный сплит

- 150 промптов, 5 блоков по 30.
- Строк: 2,699.
- Блоки 0–3 — dev, блок 4 — sealed.
- Сортировка: средняя длина `text` в символах, snake `0,1,2,3,4,4,3,2,1,0`.
- В JSON только `prompt_id → block`, без текста.

## Фолды пула

| Фолд | Clean-строк | Классов | Источников |
|---:|---:|---:|---:|
| 0 | 22,934 | 18 | 27 |
| 1 | 22,934 | 18 | 27 |
| 2 | 22,934 | 18 | 26 |
| 3 | 22,934 | 18 | 28 |
| 4 | 22,934 | 18 | 27 |

Проверка: ни один `prompt_canon` не встречается в двух фолдах; все 18 классов
есть в каждом фолде; `split_fold=-1` только у строк вне clean-сплита.

## Распределение классов (clean)

| Класс | Строк |
|---|---:|
| `gemini-3.8-flash` | 15,454 |
| `gemini-3.6-flash` | 12,092 |
| `glm-5.3-flash` | 9,256 |
| `glm-5.3` | 8,840 |
| `gpt-5.6-luna` | 7,185 |
| `gpt-5.6-terra` | 6,731 |
| `gpt-5.6-sol` | 6,527 |
| `gpt-5.5` | 6,072 |
| `claude-fable-5` | 5,747 |
| `gpt-6-astra` | 5,716 |
| `gemini-3.1-pro-preview` | 5,650 |
| `kimi-k3` | 5,475 |
| `claude-sonnet-5` | 4,121 |
| `claude-opus-4-8` | 3,876 |
| `claude-opus-4-6` | 3,325 |
| `claude-opus-5` | 3,034 |
| `qwen3.8-max` | 2,995 |
| `gemini-3.7-flash` | 2,574 |

## Лицензии источников

Полный пул:

- Attribution 4.0 International (CC BY 4.0): 158,393
- CC-BY-SA-4.0: 28,154
- CC0-1.0: 4,041

`keep_clean=True` (115 342):

- Attribution 4.0 International (CC BY 4.0): 91,701
- CC-BY-SA-4.0: 23,197
- CC0-1.0: 444

Foldable clean (114 670):

- Attribution 4.0 International (CC BY 4.0): 91,701
- CC-BY-SA-4.0: 22,525
- CC0-1.0: 444

## Проверки пересечений

- canon_in_multiple_folds: 0
- official_prompt_in_multiple_blocks: 0
- train_pool_exact_text_overlap: 0
- train_pool_prompt_id_overlap: 0
- full_duplicate_ids: 0
- clean_duplicate_ids: 0
- clean_rows_without_fold: 0
- empty_canon_with_fold: 0

## Хэши выходных файлов

| Файл | SHA256 (начало) |
|---|---|
| `ATTRIBUTION.md` | `66a631e9ecb1a654…` |
| `README.md` | `6b74e2b6f6b8ebf4…` |
| `dataset-metadata.json` | `6d7c7ba623a7b4d5…` |
| `labels.json` | `5f65c86d5c3bc2f2…` |
| `ntl_pool_clean_v1.parquet` | `594a47131c146abb…` |
| `ntl_pool_full_v1.parquet` | `b62bc6ba311d1c87…` |
| `ntl_pool_keep_clean_v1.parquet` | `50a10ecea9bdd7c9…` |
| `official_split_v1.json` | `36f0a2785ccd3121…` |
| `pool_split_v1.parquet` | `29612c50ad5140ab…` |
| `prompt_bank_v1.parquet` | `bf18653d91989896…` |
| `removal_summary.csv` | `0a95a90d321d65ff…` |
| `removed_rows.csv` | `11c579d063008ab6…` |
| `sources.csv` | `53b56f89ae018273…` |

Полные хэши — в `checks.json` и `MANIFEST.sha256`.

## Предупреждения

- нет

## Замечание о 115 342 vs 114 670

`keep_clean` даёт 115,342 строк. Из них 672
строк имеют пустой `prompt_canon`; они лежат в `ntl_pool_keep_clean_v1.parquet`
с `split_fold=-1`, но не входят в `ntl_pool_clean_v1.parquet` и в фолды, потому
что промпт-дизъюнктный фолд для них невозможен.
