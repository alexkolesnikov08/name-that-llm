# Name That LLM — контекст проекта

Портфолио-кейс по Kaggle-соревнованию «Name That LLM»: дан пассаж, нужно
определить, какая из **18 LLM** его написала. Основная метрика —
balanced accuracy, тестовые промпты не пересекаются с train.

Подробности — в `README.md` и `docs/`:

- `docs/data.md` — данные, пул, сплиты, лицензии;
- `docs/validation.md` — почему CV по промптам и что такое BA(2);
- `docs/experiments.md` — все результаты, включая провалы;
- `docs/roadmap.md` — этапы v1-protocol → v2-members → v3-ensemble;
- `docs/reproduction.md` — команды и чек-лист.

## Статус (2026-11-02)

- Официальные данные: 2 699 строк, 150 промптов, 18 классов.
- Публичный пул: 190 588 строк; 115 342 проходят `keep_clean`, из них
  114 670 с непустым `prompt_canon`; 23 451 канонический промпт. Kaggle
  dataset `morginalium/name-that-llm-clean-pool-v1` (опубликован, CC BY-SA 4.0).
- Канонический сплит: `splits/official_v1.json`, 5 блоков × 30, один
  sealed. Исторические `v1`/`v2` оставлены для воспроизводимости.
- Стек 7 вью: **0.3917** dev BA(18), **0.356** LB.
- kNN-центрирование: **0.4712** dev, **0.356** LB — не переносится.
- Char-BiGRU v1: **0.361** BA(2), v2: **0.341**; char-SVC на пуле: **0.347**.
- Потолок переноса пула: **0.36–0.40** BA(2).
- Чужой Gold Apex: 0.670 у автора, наш форк **0.662** LB (rank 432/4231).
  Это baseline, не наша модель; ноутбук — в теге `gold-apex-fork`.

## Что важно не перепутать

- **BA(18/prompt) оптимистична для LB.** Основной локальный протокол для
  новых моделей — BA(2/prompt): 2 случайных ответа на промпт, 25 draw-ов.
  Случайное угадывание = 1/18 = 0.0556.
- **`prompt_group` ≠ `prompt_canon` ≠ `prompt_id`.** Фолды официальных
  данных — по `prompt_id`, пула — по `prompt_canon`.
- **Sealed-блок** не используется для выбора гиперпараметров.
- **Данные не коммитим.** `*.csv`, `*.parquet`, `*.zip`, `runs/`,
  `research/`, `dataset/build/` — в `.gitignore`.
- **Gold Apex — чужой код.** Не выдавать 0.662 за нашу модель.

## Структура

```text
src/ntl/          библиотека: data, metrics, cv, evaluate, features, models
src/              пайплайн: make_split, embeddings, hpo_tfidf, confirm, stack
experiments/      диагностика и отрицательные результаты
kaggle/baseline/  TF-IDF baseline
kaggle/stack_v3/  7-view стек + kNN-центрирование
kaggle/rnn/       char-BiGRU
kaggle/gold_apex/ обвязка чужого форка; ноутбук в теге
splits/           official_v1.json, pool_v1.parquet, исторические v1/v2
tests/            smoke + проверки метрик/CV/сплитов
docs/             данные, валидация, эксперименты, roadmap, reproduction
```

## Команды

```bash
PYTHONPATH=src pytest -q
NTL_DATA_DIR=. python kaggle/baseline/main.py
PYTHONPATH=src python src/stack.py
python kaggle/rnn/train_rnn.py --limit 40000 --epochs 5
```

## Правила

- Публично всё, кроме данных, `runs/`, `research/`, `dataset/build/`.
- Один прогон — одно изменение; порог успеха фиксируется до прогона.
- Результат без сплита, метрики, сида и CI не считается результатом.
- Гипотезы и задачи — GitHub Issues; закрытая issue = есть вердикт.
- Kaggle: инференс без интернета, ≤9 ч, веса только из публичных источников.
