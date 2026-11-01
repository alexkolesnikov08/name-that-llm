"""Подтверждение топ-конфигов HPO на свежих повторах (сиды 44–46).

Тюнинг идёт на сидах 42–43, подтверждение — на 44–46, чтобы выбор не
подгонялся под конкретную разбивку.

Запуск:
    PYTHONPATH=src python3 src/confirm.py --study tfidf_v1 --top 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from ntl.data import load_train
from ntl.evaluate import evaluate_config, load_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="tfidf_v1")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--seeds", default="44,45,46")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    study = optuna.load_study(
        study_name=args.study, storage=f"sqlite:///{ROOT / 'runs' / 'optuna.db'}"
    )
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(done, key=lambda t: t.value, reverse=True)[: args.top]
    df = load_train()
    split = load_split()

    rows = []
    for t in top:
        cfg = t.user_attrs["cfg"]
        res = evaluate_config(cfg, df, split, seeds, n_boot=2000)
        row = {"trial": t.number, "search": t.value, **res, "cfg": cfg}
        rows.append(row)
        print(
            f"trial {t.number:3d} search={t.value:.4f} "
            f"confirm={res['mean']:.4f}±{res['std']:.4f} "
            f"CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}] {cfg}"
        )

    out = ROOT / "runs" / f"confirm_{args.study}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    best = max(rows, key=lambda r: r["mean"])
    print(f"\nbest confirmed: trial {best['trial']} BA={best['mean']:.4f}")
    print(json.dumps(best["cfg"], ensure_ascii=False))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
