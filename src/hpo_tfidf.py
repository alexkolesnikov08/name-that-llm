"""Optuna-поиск гиперпараметров TF-IDF + линейной модели.

Оценка — на dev-пуле (120 промптов) по двум повторам 5-fold GroupKFold
(сиды 42–43). Locked holdout не трогаем.

Запуск:
    PYTHONPATH=src python3 src/hpo_tfidf.py --n-trials 40 --timeout 900

Дашборд:
    uvx optuna-dashboard sqlite:///runs/optuna.db
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import optuna

from ntl.data import load_train
from ntl.evaluate import evaluate_config, load_split

ROOT = Path(__file__).resolve().parents[1]
STORAGE = f"sqlite:///{ROOT / 'runs' / 'optuna.db'}"
SEARCH_SEEDS = [42, 43]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", default="tfidf_v1")
    args = parser.parse_args()

    (ROOT / "runs").mkdir(exist_ok=True)
    df = load_train()
    split = load_split()

    def objective(trial: optuna.Trial) -> float:
        view = trial.suggest_categorical("view", ["char_wb", "char", "word", "char+word"])
        cfg = {
            "view": view,
            "ngram_range": trial.suggest_categorical(
                "ngram_range", [(2, 4), (2, 5), (3, 5), (3, 6)]
            ),
            "min_df": trial.suggest_int("min_df", 1, 3),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
            "max_features": trial.suggest_categorical(
                "max_features", [None, 100_000, 300_000]
            ),
            "clf": trial.suggest_categorical("clf", ["logreg", "svc"]),
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        }
        t0 = time.time()
        scores = []
        for step, seed in enumerate(SEARCH_SEEDS):
            res = evaluate_config(cfg, df, split, [seed], n_boot=0)
            scores.append(res["mean"])
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cfg", cfg)
        trial.set_user_attr("per_repeat", scores)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=args.study,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best = study.best_trial
    print(f"best BA={best.value:.4f}")
    print(f"cfg={best.user_attrs['cfg']}")
    print(f"trials={len(study.trials)}")


if __name__ == "__main__":
    main()
