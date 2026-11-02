# Name That LLM

[![CI](https://github.com/alexkolesnikov08/name-that-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/alexkolesnikov08/name-that-llm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY-SA 4.0](https://img.shields.io/badge/data-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

Портфолио-кейс: по фрагменту текста определить, какая из **18 LLM** его
написала. Задача Kaggle «Name That LLM», метрика — **balanced accuracy**
(BA) по 18 классам.

Репозиторий честно разделяет три вещи: наш пайплайн и его ограничения,
воспроизведение чужого публичного baseline и данные, которые мы собрали
для переноса стиля. Наш прогон публичного ноутбука-чемпиона дал **0.662
LB** — это не наша модель, он вынесен в тег и описан отдельно.

## Abstract (EN)

This repository is a case study in authorship attribution for 18 large
language models under a prompt-disjoint test split. The headline result of
our own pipeline is a 7-view linear stack at **0.356 balanced accuracy on
the leaderboard**, against a 0.299 TF-IDF baseline and 0.0556 chance. Local
prompt-grouped validation reached 0.471 after kNN prompt centering, but the
gain did not transfer to the hidden test; we document that failure instead
of hiding it. We also reproduce a public GPU pipeline at **0.662 LB**
(rank 432/4231) as an external reference and explicitly do not claim it as
our model. The repo contains grouped splits, a 115k-row cleaned public pool
of model generations, reusable metrics, and Kaggle kernel sources.

## Задача

Дано:

- `train.csv` — **2 699** пассажей: 150 промптов × 18 моделей (в одном
  промпте отсутствует один ответ, поэтому 149 промптов полные).
- `test.csv` — около 4 000 пассажей на **других промптах**; локально
  лежит только 3-строчная заглушка.
- 18 классов — конкретные версии Claude, Gemini, GLM, GPT, Kimi и Qwen;
  точные имена задаются полем `model` в `train.csv`.
- Метрика Kaggle — balanced accuracy; public/private = 40/60.

Ключевая сложность не в числе классов, а в том, что тестовые промпты не
пересекаются с train. Нельзя выучить «этот промпт → эта модель»: нужно
переносить стиль на новую тему.

## Почему BA(2/prompt), а не BA(18/prompt)

В train на каждый промпт приходится по одному ответу каждой модели, поэтому
удобно считать BA по всем 18 ответам промпта. В тесте на промпт приходится
примерно **два** ответа, и это меняет метрику: внутри промпта ответы связаны
общей темой, а 18-ответная версия даёт модели больше контекста для
«выравнивания» предсказаний.

Мы считаем основной локальной метрикой **BA(2/prompt)**: случайно берём по
2 ответа на промпт, считаем BA, усредняем по 25 розыгрышам. Она ближе к
скрытому тесту; на публичном форуме OOF по 2 ответам коррелирует с LB на
0.91, а по 18 — на 0.13. Случайное угадывание даёт **1/18 = 0.0556**.

Для сравнения с прошлыми экспериментами в таблицах ниже указано, какая
именно метрика использовалась.

## Почему CV по промптам

Фолды режутся по `prompt_id`, а не по строкам. Если перемешать строки, 17
«братьев» промпта попадут в train, и модель запомнит тему, а не стиль.
Групповой сплит моделирует главное условие теста — невиданный промпт.

- `splits/official_v1.json` — канонический сплит: 150 промптов, **5 блоков
  по 30**, один блок sealed. Sealed не используется для выбора
  гиперпараметров и моделей; его метрика — только финальный отчёт.
- `splits/v1.json`, `splits/v2.json` — исторические сплиты; `v2` (30
  locked holdout + 120 dev, 5 повторов) сохранён для воспроизводимости
  старых прогонов.
- `splits/pool_v1.parquet` — 5 фолдов по `prompt_canon` для пула; одна
  строка на канонический промпт, без текстов.

Неопределённость считаем bootstrap по промптам, а не по строкам; для
сравнения моделей — парный bootstrap. Подробнее: `docs/validation.md`.

## Данные

Официальные данные соревнования в репозиторий не включены (правила Kaggle).
Публичный пул генераций:

- Kaggle dataset: [morginalium/name-that-llm-clean-pool-v1](https://www.kaggle.com/datasets/morginalium/name-that-llm-clean-pool-v1)
  — **опубликован** (версия 4). Лицензия данных **CC BY-SA 4.0**.
- **190 588** исходных строк; **115 342** проходят консервативный фильтр
  `keep_clean`, из них **114 670** с непустым `prompt_canon` (foldable),
  **23 451** канонический промпт.
- 18 моделей, разметка `model`, две разные группировки: `prompt_group`
  (группа внутри источника) и `prompt_canon` (канонический промпт после
  слияния источников). Это не взаимозаменяемые ключи.
- Очистка: убраны near-duplicate промпты из train, регенерации train-промптов,
  knowledge-base прокси и строки без надёжной атрибуции.
- Перенос с пула на официальный train ограничен: локальный потолок
  **0.36–0.40** BA(2/prompt). Пул полезен как предобучение, но не заменяет
  официальные данные.

Подробнее о полях и фильтрах: `docs/data.md`.

## Что сделано

1. **Библиотека `src/ntl/`** — загрузка данных, grouped CV, BA и
   bootstrap-CI по промптам, ручные стилометрические/форматные/служебные
   вью, линейные модели.
2. **7-view стек** — char/word TF-IDF, стилометрия, формат, function words,
   скелет текста и frozen DeBERTa-small; OOF-вероятности → LogReg-мета.
3. **Prompt-centering** — вычитание среднего эмбеддингов по промпту;
   oracle-версия и практическая через TF-IDF-кластеры/kNN.
4. **Char-BiGRU** — компактная модель на 1.1M параметров, обучение на
   115k пуле, family-голова, тематический адверсарий и доменная адаптация
   на официальном train.
5. **Char-SVC на пуле** — опорная точка переноса без нейросети.
6. **Воспроизведение публичного Gold Apex** — как внешний baseline, без
   выдачи за собственный результат.
7. **Инфраструктура** — фиксированные сплиты, тесты, CI, Kaggle-метаданные.

## Результаты

Все числа — из сохранённых логов и сабмитов; где метрика локальная, это
указано явно. LB — Kaggle leaderboard.

| Что | Протокол | Результат | Вывод |
|---|---|---|---|
| Случайный выбор | BA(2/prompt) | **0.0556** | точка отсчёта |
| TF-IDF char_wb + LogReg | dev v2, BA(18) | 0.2818 ± 0.0052 | первый baseline |
| TF-IDF HPO | dev v2, BA(18) | 0.2880 ± 0.0056 | потолок ручного TF-IDF |
| TF-IDF baseline | LB | **0.299** | совпал с CV |
| Frozen DeBERTa-small + LogReg | dev v2, BA(18) | 0.3423 ± 0.0055 | эмбеддинги лучше n-грамм |
| 7-view стек | dev v2, BA(18) | 0.3917 ± 0.0029 | +0.049 к лучшей вью |
| 7-view стек | LB | **0.356** | перенос хуже локального |
| Стек + kNN-центрирование | dev v2, BA(18) | 0.4712 ± 0.0026 | лучший локальный результат |
| Стек + kNN-центрирование | holdout, BA(18) | 0.4833 | holdout не спас LB |
| Стек + kNN-центрирование | LB | **0.356** | прирост не перенёсся |
| Char-BiGRU v1 | holdout 30, BA(2/prompt) | **0.361** | лучший наш RNN |
| Char-BiGRU v2 + адверсарий | holdout 30, BA(2/prompt) | **0.341** | адверсарий не помог |
| Char-SVC на пуле | holdout 30, BA(2/prompt) | **0.347** | перенос с пула ограничен |
| Потолок переноса пула | BA(2/prompt) | **0.36–0.40** | выше без новых данных не ушли |
| Публичный Gold Apex (чужой) | LB | 0.670 | источник форка |
| Наш форк Gold Apex (чужой пайплайн) | LB | **0.662**, rank **432/4231** | baseline, не наша модель |

«Holdout 30» в строках RNN/SVC — это 30 промптов, выбранных скриптом
(`--val-prompts 30`), а не sealed-блок `official_v1.json`. Для новых
прогонов используется только канонический сплит.

Контекст лидерборда на момент работы: первое место 0.737, граница бронзы
около 0.662, плато публичных GPU-решений — сотни команд на 0.66. Наш
собственный пайплайн остался на 0.356; это не медальный результат, и мы
не выдаём его за него.

### Честный разбор провала 0.47 → 0.356

Локальный скачок дало kNN-центрирование эмбеддингов: вычитание среднего
соседей по TF-IDF. На dev это +0.08, на holdout — 0.4833. На LB — ноль.
Причины:

- центрирование использует структуру промптов, а на тесте кластеры иные;
- 18 ответов на промпт в CV дают модели больше информации, чем 2 на LB;
- assignment-трюки (Hungarian, Sinkhorn, жёсткое назначение) на публичных
  абляциях теряли 0.009–0.017 LB каждый;
- ручные признаки на 2 699 строках упираются в потолок ≈0.40.

Мы сохранили этот результат как отрицательный: он важнее красивого CV.

## Что здесь чужое

`kaggle/gold_apex/` — обвязка вокруг публичного
`synthreaper/ntl-gold-apex-champion-0-670` (0.670 public LB), собранного
из ноутбуков omerfarukmerey, mercurydust, Dhruv Tibarewal, Roman Tamrazov,
prvsiyan, Lucas Boesen, Kirill Tymchyshyn, sangam21, Joseph Adamski и
других. Алгоритм не менялся; наши правки — только эксплуатация (входы,
guard на 3-строчный тест, fallback сабмита).

Сам ноутбук **удалён из `main`**, чтобы не размывать авторство. Он целиком
лежит в теге `gold-apex-fork`:

```bash
git show gold-apex-fork:submit_gold_apex/name-that-llm-gold-apex-fork.ipynb \
  > /tmp/name-that-llm-gold-apex-fork.ipynb
```

Лицензии исходников: код CC BY-SA 4.0, заимствованные модули Apache-2.0.
Наш форк дал **0.662**, а не 0.670; это ожидаемое расхождение рантайма,
но всё равно чужой пайплайн.

## Структура

```text
src/ntl/              библиотека: data, metrics, cv, evaluate, features, models
src/                  пайплайн: make_split, embeddings, hpo_tfidf, confirm, stack
experiments/          диагностика: views, stack_ablation, prompt_centering,
                      topic_clusters, knn_center_tfidf, pool_svc_baseline, rnn_style*
kaggle/baseline/      TF-IDF baseline, LB 0.299
kaggle/stack_v3/      7-view стек + kNN-центрирование, LB 0.356
kaggle/rnn/           char-BiGRU v2, обучение на пуле
kaggle/gold_apex/     скрипты/метаданные чужого форка; ноутбук — в теге
splits/               official_v1.json, pool_v1.parquet, исторические v1/v2
tests/                smoke + проверки метрик, CV и сплитов
docs/                 данные, валидация, эксперименты, roadmap, reproduction
.github/workflows/    CI: pytest на push/PR
```

## Воспроизвести

Нужны Python 3.12+ и данные соревнования (принять правила Kaggle).
Для базового прогона достаточно CPU; стек и RNN требуют torch.

```bash
uv venv --python 3.12
uv pip install numpy pandas scikit-learn pyarrow pytest

# данные: train.csv/test.csv в корне или в NTL_DATA_DIR
export NTL_DATA_DIR=.

# smoke + unit-тесты (тесты, которым нужны данные, скипаются)
PYTHONPATH=src pytest -q

# baseline: TF-IDF + LogReg
python kaggle/baseline/main.py

# 7-view стек (нужны transformers/torch и веса DeBERTa-small)
python kaggle/stack_v3/main.py

# char-BiGRU на публичном пуле
python kaggle/rnn/train_rnn.py --limit 40000 --epochs 2
```

Полный сценарий, включая проверку сплитов и воспроизведение чужого
Gold Apex: `docs/reproduction.md`. Данные и веса моделей не коммитятся.

## Ссылки

- Kaggle dataset: <https://www.kaggle.com/datasets/morginalium/name-that-llm-clean-pool-v1>
- Kaggle kernels (исторические, могли остаться приватными):
  [baseline](https://www.kaggle.com/code/morginalium/name-that-llm-baseline),
  [stack-v3](https://www.kaggle.com/code/morginalium/name-that-llm-stack-v3),
  [rnn](https://www.kaggle.com/code/morginalium/ntl-rnn-style-train)
- Issues и план работ: <https://github.com/alexkolesnikov08/name-that-llm/issues>
  — новые задачи **#22–#29** (этапы `v1-protocol`, `v2-members`,
  `v3-ensemble`)
- Документация: [`docs/data.md`](docs/data.md), [`docs/validation.md`](docs/validation.md),
  [`docs/experiments.md`](docs/experiments.md), [`docs/roadmap.md`](docs/roadmap.md),
  [`docs/reproduction.md`](docs/reproduction.md)

## Лицензии

- **Код репозитория — MIT**, см. [`LICENSE`](LICENSE).
- **Данные пула — CC BY-SA 4.0**, см. Kaggle dataset card.
- **Чужой Gold Apex** — CC BY-SA 4.0 для кода и Apache-2.0 для
  заимствованных модулей; атрибуция сохранена в `kaggle/gold_apex/README.md`.
- Официальные данные соревнования не распространяются этим репозиторием.
