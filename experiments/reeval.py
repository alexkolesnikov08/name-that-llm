"""Повторная оценка фиксированных конфигов на сплите v2 (настоящие повторы)."""
from __future__ import annotations

from ntl.data import load_train
from ntl.evaluate import evaluate_config, load_split

CONFIGS = {
    "baseline_char_wb_2_5": {
        "view": "char_wb",
        "ngram_range": (2, 5),
        "min_df": 2,
        "sublinear_tf": True,
        "max_features": 300_000,
        "clf": "logreg",
        "C": 10.0,
    },
    "hpo_best_char_3_6": {
        "view": "char",
        "ngram_range": (3, 6),
        "min_df": 2,
        "sublinear_tf": True,
        "max_features": None,
        "clf": "logreg",
        "C": 9.603593642884443,
    },
}


def main() -> None:
    df = load_train()
    split = load_split()
    for name, cfg in CONFIGS.items():
        res = evaluate_config(cfg, df, split, [42, 43, 44, 45, 46], n_boot=2000)
        print(
            f"{name}: {res['mean']:.4f} ± {res['std']:.4f} "
            f"CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}] "
            f"per_repeat={[round(s, 3) for s in res['per_repeat']]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
