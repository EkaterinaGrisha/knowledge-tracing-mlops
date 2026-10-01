"""Figures for the report — unified scientific style (seaborn `deep` palette).

A single theme + a fixed per-model colour map are applied across every chart so
the whole report reads as one consistent visual language (white grid, despined
axes, muted colourblind-safe palette).

Figures are built with matplotlib's object-oriented API (no pyplot state, no
GUI backend) and the theme is applied only while a figure is drawn, so
importing this module changes no global settings.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, cast

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.typing import RcKeyType
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

from ..config import resolve_path
from .metrics import DECISION_THRESHOLD

P = ParamSpec("P")

_PALETTE = sns.color_palette("deep")
# fixed colour per model — same colour everywhere it appears
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    "BKT": _PALETTE[0],
    "DKT": _PALETTE[2],
    "DKT+Optuna": _PALETTE[3],
    "AutoML": _PALETTE[1],
}
ACCENT = _PALETTE[0]

_STYLE_OVERRIDES: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": "#3A3A3A",
    "axes.linewidth": 0.8,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}


def _theme() -> dict[RcKeyType, Any]:
    """Matplotlib settings of the report style (seaborn whitegrid + overrides)."""
    theme = {
        **sns.plotting_context("notebook", font_scale=1.05),
        **sns.axes_style("whitegrid"),
        "axes.prop_cycle": cycler(color=_PALETTE),
        **_STYLE_OVERRIDES,
    }
    # seaborn returns plain dicts; every key is a valid matplotlib rc parameter
    return cast("dict[RcKeyType, Any]", theme)


def _themed(plot: Callable[P, Path]) -> Callable[P, Path]:
    """Draw and save the figure with the report style, then restore the settings."""

    @functools.wraps(plot)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Path:
        with mpl.rc_context(_theme()):  # previous settings are restored on exit
            return plot(*args, **kwargs)

    return wrapper


def _color(name: str) -> tuple[float, float, float]:
    return MODEL_COLORS.get(name, _PALETTE[7])


def _output_path(out_dir: Path, name: str) -> Path:
    directory = resolve_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _save(fig: Figure, out_dir: Path, name: str, *, despine: bool = True) -> Path:
    if despine:
        sns.despine(fig=fig)
    fig.tight_layout()
    out = _output_path(out_dir, name)
    fig.savefig(out)
    return out


# ── figures ─────────────────────────────────────────────────────────────────


@_themed
def plot_dataset_overview(long: pd.DataFrame, stats: dict[str, Any], out_dir: Path) -> Path:
    """Histograms of student sequence lengths and per-skill difficulty."""
    fig = Figure(figsize=(11, 4.2))
    axes = fig.subplots(1, 2)
    seq_lengths = long.groupby("user_id").size()
    sns.histplot(seq_lengths, bins=30, color=ACCENT, edgecolor="white", ax=axes[0])
    axes[0].set_title("Длина последовательностей студентов")
    axes[0].set_xlabel("число взаимодействий")
    axes[0].set_ylabel("число студентов")

    rate_by_skill = long.groupby("skill_idx")["correct"].mean()
    sns.histplot(rate_by_skill, bins=20, color=ACCENT, edgecolor="white", ax=axes[1])
    axes[1].set_title("Сложность навыков")
    axes[1].set_xlabel("доля верных ответов по навыку")
    axes[1].set_ylabel("число навыков")

    fig.suptitle(
        f"Датасет: {stats['n_students']} студентов · {stats['n_interactions']} взаимодействий · "
        f"{stats['n_skills']} навыков · общая верность {stats['global_correct_rate']:.2f}",
        fontsize=12,
        y=1.02,
    )
    return _save(fig, out_dir, "dataset_overview.png")


@_themed
def plot_model_comparison(results: dict[str, dict[str, float]], out_dir: Path) -> Path:
    """Grouped bars of AUC / accuracy / F1 per model on the test split."""
    tidy = pd.DataFrame(
        [
            {"Модель": m, "Метрика": metric, "value": results[m][key]}
            for m in results
            for metric, key in (("AUC", "auc"), ("Accuracy", "accuracy"), ("F1", "f1"))
        ]
    )
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    sns.barplot(data=tidy, x="Модель", y="value", hue="Метрика", palette="deep", ax=ax)
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("значение метрики")
    ax.set_xlabel("")
    ax.set_title("Сравнение моделей на тестовой выборке (one-step-ahead)")
    return _save(fig, out_dir, "model_comparison.png")


@_themed
def plot_roc(preds: dict[str, tuple[np.ndarray, np.ndarray]], out_dir: Path) -> Path:
    """ROC curves of all models on the test split."""
    fig = Figure(figsize=(6.5, 6))
    ax = fig.subplots()
    for name, (yt, yp) in preds.items():
        if len(set(yt.tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, label=name, linewidth=2, color=_color(name))
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые моделей (тест)")
    ax.legend(title="Модель")
    return _save(fig, out_dir, "roc_comparison.png")


@_themed
def plot_confusion(yt: np.ndarray, yp: np.ndarray, name: str, out_dir: Path) -> Path:
    """Confusion matrix of one model at the decision threshold."""
    cm = confusion_matrix(yt, (yp > DECISION_THRESHOLD).astype(int))
    fig = Figure(figsize=(5, 4.5))
    ax = fig.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["неверно", "верно"],
        yticklabels=["неверно", "верно"],
        ax=ax,
    )
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истинно")
    ax.set_title(f"Матрица ошибок — {name}")
    return _save(fig, out_dir, "confusion_matrix.png", despine=False)


@_themed
def plot_calibration(yt: np.ndarray, yp: np.ndarray, name: str, out_dir: Path) -> Path:
    """Reliability diagram: predicted probability vs observed correctness."""
    frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy="quantile")
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1, label="идеальная калибровка")
    ax.plot(mean_pred, frac_pos, "o-", color=_color(name), linewidth=2, label=name)
    ax.set_xlabel("средняя предсказанная вероятность")
    ax.set_ylabel("наблюдаемая доля верных")
    ax.set_title(f"Калибровочная кривая — {name}")
    ax.legend(title="Модель")
    return _save(fig, out_dir, "calibration.png")


@_themed
def plot_feature_importance(names: list[str], importances: np.ndarray, out_dir: Path) -> Path:
    """Feature importances of the tabular AutoML model."""
    df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance")
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    sns.barplot(data=df, x="importance", y="feature", color=ACCENT, ax=ax)
    ax.set_title("Важность признаков (AutoML / LightGBM)")
    ax.set_xlabel("важность")
    ax.set_ylabel("")
    return _save(fig, out_dir, "feature_importance.png")


@_themed
def plot_dkt_loss(losses: list[float], out_dir: Path) -> Path:
    """Training loss of DKT per epoch."""
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.plot(range(1, len(losses) + 1), losses, "o-", color=MODEL_COLORS["DKT"], linewidth=2)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("train loss (BCE)")
    ax.set_title("Кривая обучения DKT")
    return _save(fig, out_dir, "dkt_loss.png")
