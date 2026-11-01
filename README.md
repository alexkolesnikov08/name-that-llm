# Name That LLM

Решение Kaggle-соревнования [Name That LLM](https://www.kaggle.com/competitions/name-that-llm):
по пассажу определить, какая из 18 LLM его написала.

## Данные

- `train` — 2 699 пассажей: 150 промптов × 18 моделей.
- `test` — ~4 000 пассажей на промптах, которых нет в train.
- Метрика — balanced accuracy по 18 классам.

## Структура

- `AGENT.md` — контекст, протокол валидации, текущий фокус.
- [Issues](https://github.com/alexkolesnikov08/name-that-llm/issues) — гипотезы
  и результаты экспериментов.
- `src/` — код пайплайна (по мере работы).

Данные соревнования в репозиторий не включены.
