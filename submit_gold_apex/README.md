# Форк публичного Gold Apex

Это не наше решение. Ноутбук — публичный `synthreaper/ntl-gold-apex-champion-0-670`
(0.670 public LB), который сам собран из публичных ноутбуков
omerfarukmerey, mercurydust, Dhruv Tibarewal, Roman Tamrazov, prvsiyan,
Lucas Boesen, Kirill Tymchyshyn, sangam21, Joseph Adamski и других.
Лицензии: код CC BY-SA 4.0, заимствованные модули Apache-2.0.

## Что изменено относительно исходника

Только эксплуатация, алгоритм не тронут:

1. Убраны markdown-раздел с чужими абляциями и график matplotlib.
2. Из ноутбука вырезан блок `metadata.kaggle` — список входов задаётся
   `kernel-metadata.json`, а не тем, что было привязано у автора.
3. В setup-ячейку добавлен guard: если Kaggle выставил флаг
   `KAGGLE_IS_COMPETITION_RERUN`, но подсунул 3-строчный `test.csv`,
   ноутбук падает, а не отправляет заглушки.
4. Финальный запуск `infer.py` обёрнут в `try/except`: при падении пишется
   валидный `submission.parquet` с приором по train, чтобы прогон не остался
   пустым.

## Входы

15 датасетов + 8 kernel-источников + 8 моделей, все публичные:

- базы: Qwen3-8B/14B (base и instruct), Qwen2.5-7B/14B, Llama-3.1-8B-base;
- LoRA-адаптеры и головы: `nlm-l24-adapters`, `-v5`, `nlm-two-stage-lora`,
  `ntl-qwen3-8b-lora-claude-head`, kernel-выходы `lucasboesen/...`,
  `vnton37nguyn/...`;
- пул для SVC: `nlm_pool_v5.parquet` (120k), `nlm_matched_clean.parquet`;
- ATLAS-бандл: `josephadamski91/ntl-atlas-qwen3-8b-v1`;
- энкодеры: DeBERTa-v3-large, UltraBERT, ELECTRA, desklib.

## Режимы

| режим | условие | время |
|---|---|---|
| commit | `test.csv` = 3 строки | ~10 мин, пишет заглушку |
| competition rerun | скрытый тест ~4000 строк | ~8 ч, полный пайплайн |

Бюджет ATLAS считается от старта `infer.py`: `8.3 ч − comp_stack − 10 мин`,
то есть общий прогон упирается в ~8.3 ч при лимите 9 ч.

## Запуск

```bash
# 1. залить версию (commit-прогон на 3 строках, проверка входов)
kaggle kernels push -p submit_gold_apex

# 2. дождаться COMPLETE
kaggle kernels status morginalium/ntl-gold-apex-fork

# 3. отправить версию в соревнование -> полный ре-ран на скрытом тесте
kaggle competitions submit -c name-that-llm \
  -k morginalium/ntl-gold-apex-fork -v 1 \
  -m "gold apex fork"
```

GPU-квота: 30 ч/нед, один прогон ≈ 8.3 ч.
