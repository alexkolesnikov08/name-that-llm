"""Статические графики Optuna: история, parallel coordinates, важность."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    study = optuna.load_study(
        study_name="tfidf_v1", storage=f"sqlite:///{ROOT / 'runs' / 'optuna.db'}"
    )
    plots = {
        "history": plot_optimization_history,
        "parallel": plot_parallel_coordinate,
        "importance": plot_param_importances,
        "slice": plot_slice,
    }
    for name, fn in plots.items():
        ax = fn(study)
        fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
        fig.set_size_inches(10, 5)
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
