# Kaggle-артефакты

Здесь лежат исходники кернелов и их `kernel-metadata.json`. Это
**исторические** прогоны, а не production-пайплайн: данные на Kaggle
монтируются в `/kaggle/input`, локально пути переопределяются через
`NTL_DATA_DIR` / `NTL_MODEL_DIR`.

| каталог | что это | результат |
|---|---|---|
| `baseline/` | TF-IDF `char_wb` + LogReg | LB **0.299** |
| `stack_v3/` | 7-view стек + kNN-центрирование | LB **0.356** |
| `rnn/` | char-BiGRU v2 на публичном пуле | BA(2) **0.341** (v1 — 0.361) |
| `gold_apex/` | обвязка чужого публичного форка | LB **0.662**, не наша модель |

## Запуск

```bash
kaggle kernels push -p kaggle/baseline
kaggle kernels push -p kaggle/stack_v3
kaggle kernels push -p kaggle/rnn
```

`kaggle/gold_apex/` без ноутбука не запускается: сначала достаньте его из
тега, см. `kaggle/gold_apex/README.md`.

## Данные

- Официальный train/test — соревнование `name-that-llm`, в git не входят.
- Пул — `morginalium/name-that-llm-clean-pool-v1` (публикуется,
  CC BY-SA 4.0).
- DeBERTa-small для стека — публичный датасет
  `eric0045/microsoft-deberta-v3-small-local`.

## Лицензии

Код кернелов — MIT (как и репозиторий). Gold Apex — чужой код:
CC BY-SA 4.0 / Apache-2.0, атрибуция в `gold_apex/README.md`. Данные пула —
CC BY-SA 4.0, официальные данные — правила Kaggle.
