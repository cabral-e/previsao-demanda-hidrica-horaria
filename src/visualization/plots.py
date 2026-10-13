"""Funções de gráfico do projeto. Estilo centralizado em style.py.

Todas recebem dados já calculados (sem lógica de análise) e devolvem a figura.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualization.style import (
    MODEL_STYLE, OBSERVED, PREDICTED, SHADE, STRIP_FACE, bottom_legend, horizon_label, t, target_label,
)


# ---------------------------------------------------------------- utilitários
def right_strip(ax, text: str, width: float = 0.045) -> None:
    """Faixa vertical à direita do painel com texto a 270° (estilo da referência)."""
    strip = ax.inset_axes([1.0, 0.0, width, 1.0])
    strip.set_facecolor(STRIP_FACE)
    strip.set_xticks([])
    strip.set_yticks([])
    strip.grid(False)
    for spine in strip.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)
    strip.text(0.5, 0.5, text, rotation=270, ha="center", va="center",
               fontsize=13, transform=strip.transAxes)


def add_volume_axis(ax, factor: float, axis: str = "y") -> None:
    """Eixo secundário com o equivalente em m³ (reescala linear da MESMA grandeza)."""
    funcs = (lambda v: v * factor, lambda v: v / factor)
    sec = ax.secondary_yaxis("right", functions=funcs) if axis == "y" \
        else ax.secondary_xaxis("top", functions=funcs)
    (sec.set_ylabel if axis == "y" else sec.set_xlabel)(t("volume"), fontsize=13)


def _date_axis(ax, interval_months: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")


# ---------------------------------------------------------------- série
def plot_series(ts: pd.Series, y: pd.Series, cut: pd.Timestamp | None, factor: float | None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts, y, color="black", linewidth=0.5)
    if cut is not None:
        ax.axvspan(cut, ts.max(), color=SHADE, zorder=0)
        ax.axvline(cut, color="black", linestyle="--", linewidth=0.9)
        ymax = y.max()
        ax.text(ts.min() + (cut - ts.min()) / 2, ymax * 1.02, t("train"), ha="center", fontsize=13)
        ax.text(cut + (ts.max() - cut) / 2, ymax * 1.02, t("test"), ha="center", fontsize=13)
        ax.set_ylim(top=ymax * 1.10)
    ax.set_xlim(ts.min(), ts.max())
    ax.set_ylabel(target_label())
    ax.set_xlabel(t("time"))
    _date_axis(ax)
    if factor:
        add_volume_axis(ax, factor)
    return fig


def plot_window(ts: pd.Series, y: pd.Series, title_strip: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(ts, y, **OBSERVED)
    ax.set_xlim(ts.min(), ts.max())
    ax.set_ylabel(target_label())
    ax.set_xlabel(t("time"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
    if title_strip:
        right_strip(ax, title_strip)
    return fig


# ---------------------------------------------------------------- distribuição
def plot_distribution(y: pd.Series, factor: float | None, bins: int = 60):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(y, bins=bins, color="#bdbdbd", edgecolor="black", linewidth=0.4)
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    ax.axvline(med, color="black", linestyle="-", linewidth=1.2)
    for q in (q1, q3):
        ax.axvline(q, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel(target_label())
    ax.set_ylabel(t("density"))
    if factor:
        add_volume_axis(ax, factor, axis="x")
    handles = [Line2D([], [], color="black", linestyle="-"), Line2D([], [], color="black", linestyle="--")]
    labels = ["Mediana", "Quartis (Q1, Q3)"] if t("train") == "Treino" else ["Median", "Quartiles (Q1, Q3)"]
    bottom_legend(fig, handles, labels, y=-0.01)
    return fig


def plot_boxplot(groups: Sequence[np.ndarray], labels: Sequence[str], xlabel: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(groups, tick_labels=labels, widths=0.6, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
               medianprops={"color": "black", "linewidth": 1.4},
               whiskerprops={"color": "black", "linewidth": 0.8},
               capprops={"color": "black", "linewidth": 0.8},
               flierprops={"marker": "o", "markersize": 2, "markerfacecolor": "none",
                           "markeredgecolor": "#6b6b6b", "markeredgewidth": 0.5})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(target_label())
    ax.grid(axis="x", visible=False)
    return fig


def plot_heatmap(matrix: pd.DataFrame, xlabel: str, ylabel: str, cbar_label: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(cbar_label, fontsize=13)
    cb.outline.set_linewidth(0.8)
    return fig


# ---------------------------------------------------------------- ACF / PACF
def plot_acf_pacf(panels: Sequence[tuple[str, np.ndarray, np.ndarray]], season: int = 24):
    """panels: [(rótulo, valores_lag0..L, banda_conf_95), ...] empilhados."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels)), sharex=True,
                             gridspec_kw={"hspace": 0.06})
    axes = np.atleast_1d(axes)
    for ax, (label, vals, band) in zip(axes, panels):
        lags = np.arange(len(vals))
        ax.vlines(lags[1:], 0, vals[1:], color="black", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.fill_between(lags[1:], -band[1:], band[1:], color="#d9d9d9", step="mid", linewidth=0)
        for k in range(season, len(vals), season):
            ax.axvline(k, color="#d62728", linestyle=":", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        right_strip(ax, label)
    axes[-1].set_xlabel(t("lag"))
    axes[-1].set_xticks(np.arange(0, len(panels[0][1]), season))
    axes[-1].set_xlim(0, len(panels[0][1]) - 0.5)
    fig.supylabel("Autocorrelação" if t("train") == "Treino" else "Autocorrelation", fontsize=16, x=0.04)
    handles = [Patch(color="#d9d9d9"), Line2D([], [], color="#d62728", linestyle=":")]
    labels = (["Intervalo de 95%", f"Múltiplos de {season} h"] if t("train") == "Treino"
              else ["95% interval", f"Multiples of {season} h"])
    bottom_legend(fig, handles, labels, y=0.0)
    return fig


# ---------------------------------------------------------------- decomposição
def plot_components(index: pd.DatetimeIndex, components: Sequence[tuple[str, np.ndarray]], window: slice | None = None):
    """Painéis empilhados (observado, tendência, sazonalidades, resíduo)."""
    n = len(components)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n), sharex=True, gridspec_kw={"hspace": 0.06})
    idx = index if window is None else index[window]
    for ax, (label, vals) in zip(axes, components):
        v = vals if window is None else vals[window]
        ax.plot(idx, v, color="black", linewidth=0.6)
        right_strip(ax, label, width=0.06)
    axes[-1].set_xlabel(t("time"))
    if window is None:
        _date_axis(axes[-1])
    else:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.supylabel(target_label(), fontsize=16, x=0.03)
    return fig


# ---------------------------------------------------------------- perfis
def plot_profiles(x: np.ndarray, curves: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                  xlabel: str, xticks: Sequence | None = None, xticklabels: Sequence | None = None):
    """curves: [(rótulo, média, q25, q75)]; faixa interquartil sombreada."""
    from src.visualization.style import SERIES_STYLES
    fig, ax = plt.subplots(figsize=(10, 4.5))
    handles, labels = [], []
    for style, (label, mean, lo, hi) in zip(SERIES_STYLES, curves):
        ax.fill_between(x, lo, hi, color=style["color"], alpha=0.12, linewidth=0)
        (h,) = ax.plot(x, mean, color=style["color"], linestyle=style["linestyle"],
                       marker=style["marker"], markersize=4, linewidth=1.2)
        handles.append(h)
        labels.append(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(target_label())
    if xticks is not None:
        ax.set_xticks(xticks, xticklabels if xticklabels is not None else xticks)
    if len(curves) > 1:
        bottom_legend(fig, handles, labels, y=-0.01)
    return fig


# ================================================================ resultados (ETAPA 17)
def plot_forecast_panels(panels: Sequence[tuple[str, np.ndarray, np.ndarray]], xtick_step: int,
                         figsize: tuple[float, float] = (10, 7.5)):
    """Previsto × observado empilhado por horizonte (estilo da figura de referência).

    panels: [(rótulo da faixa, observado, previsto), ...] com o mesmo comprimento.
    Eixo x = índice da amostra de teste; mesma escala y em todos os painéis.
    """
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, sharey=True, gridspec_kw={"hspace": 0.06})
    axes = np.atleast_1d(axes)
    for ax, (label, obs, pred) in zip(axes, panels):
        x = np.arange(len(obs))
        ax.plot(x, obs, **OBSERVED, zorder=2)
        ax.plot(x, pred, **PREDICTED, zorder=3)
        right_strip(ax, label)
    last = len(panels[0][1]) - 1
    axes[-1].set_xlim(-0.02 * last, 1.02 * last)
    axes[-1].set_xticks(np.arange(0, last + 1, xtick_step))
    axes[-1].set_xlabel(t("test_samples"))
    fig.supylabel(target_label(), fontsize=16, x=0.04)
    handles = [Line2D([], [], **OBSERVED), Line2D([], [], **PREDICTED)]
    bottom_legend(fig, handles, [t("observed"), t("predicted")], y=0.03)
    return fig


def plot_grouped_bars(values: pd.DataFrame, labels: dict[str, str], ylabel: str, log: bool = False):
    """values: índice = horizonte, colunas = modelos (na ordem desejada)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    hs, models = list(values.index), list(values.columns)
    width = 0.8 / len(models)
    x = np.arange(len(hs))
    for i, m in enumerate(models):
        st = MODEL_STYLE[m]
        ax.bar(x + (i - (len(models) - 1) / 2) * width, values[m].to_numpy(), width * 0.92,
               color=st["color"], hatch=st["hatch"], edgecolor="black", linewidth=0.5, label=labels[m])
    ax.set_xticks(x, [horizon_label(h) for h in hs])
    ax.set_xlabel(t("horizon"))
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
    handles, labs = ax.get_legend_handles_labels()
    bottom_legend(fig, handles, labs, ncol=min(4, len(labs)), y=-0.02)
    return fig


def plot_time_panels(train: pd.DataFrame, pred: pd.DataFrame, total: pd.DataFrame, labels: dict[str, str]):
    """Três painéis (treino, previsão, total), barras por modelo e horizonte, escala log."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (title, df) in zip(axes, ((t("train_time"), train), (t("pred_time"), pred), (t("total_time"), total))):
        hs, models = list(df.index), list(df.columns)
        width = 0.8 / len(models)
        x = np.arange(len(hs))
        for i, m in enumerate(models):
            st = MODEL_STYLE[m]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, df[m].to_numpy(), width * 0.92, color=st["color"],
                   hatch=st["hatch"], edgecolor="black", linewidth=0.5, label=labels[m])
        ax.set_yscale("log")
        ax.set_xticks(x, [str(h) for h in hs])
        ax.set_xlabel(t("horizon"))
        ax.set_title(title, fontsize=14)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t("time_s"))
    handles, labs = axes[0].get_legend_handles_labels()
    bottom_legend(fig, handles, labs, ncol=min(4, len(labs)), y=-0.02)
    return fig


def plot_alpha_sensitivity(curves: dict[int, pd.DataFrame], labels: dict[str, str]):
    """Score × α por horizonte (2×2). curves[h]: índice = α, colunas = modelos."""
    hs = sorted(curves)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True, gridspec_kw={"hspace": 0.08, "wspace": 0.16})
    for ax, h in zip(axes.ravel(), hs):
        df = curves[h]
        for m in df.columns:
            st = MODEL_STYLE[m]
            ax.plot(df.index, df[m], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                    markersize=5, linewidth=1.2, label=labels[m])
        right_strip(ax, horizon_label(h), width=0.07)
    for ax in axes[-1]:
        ax.set_xlabel(t("alpha"))
    fig.supylabel(t("score"), fontsize=16, x=0.03)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    bottom_legend(fig, handles, labs, ncol=min(5, len(labs)), y=0.02)
    return fig


def plot_pareto_panels(points: dict[int, pd.DataFrame], labels: dict[str, str], ref: dict[int, tuple[float, float]] | None = None):
    """Erro × tempo (log) por horizonte (2×2). points[h]: colunas modelo, t, erro, pareto."""
    hs = sorted(points)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), gridspec_kw={"hspace": 0.28, "wspace": 0.38})
    for ax, h in zip(axes.ravel(), hs):
        df = points[h]
        front = df[df["pareto"]].sort_values("t")
        if len(front) > 1:                                       # fronteira em degraus (melhor erro até aquele custo)
            xs, ys = [front["t"].iloc[0]], [front["erro"].iloc[0]]
            for tt, ee in zip(front["t"].iloc[1:], front["erro"].iloc[1:]):
                xs += [tt, tt]
                ys += [ys[-1], ee]
            ax.plot(xs, ys, color="black", linewidth=0.9, linestyle="-", zorder=1)
        for r in df.itertuples(index=False):
            st = MODEL_STYLE[r.modelo]
            ax.scatter(r.t, r.erro, s=70, marker=st["marker"], zorder=3, linewidths=1.2,
                       facecolor=st["color"] if r.pareto else "white", edgecolor=st["color"] if r.pareto else st["color"])
        if ref and h in ref:
            ax.scatter(*ref[h], s=60, marker="D", facecolor="white", edgecolor="#7f7f7f", zorder=2)
            ax.axhline(ref[h][1], color="#7f7f7f", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xscale("log")
        right_strip(ax, horizon_label(h), width=0.07)
    for ax in axes[-1]:
        ax.set_xlabel(t("time_s"))
    fig.supylabel(t("rmse_test"), fontsize=16, x=0.03)
    handles = [Line2D([], [], linestyle="", marker=MODEL_STYLE[m]["marker"], markersize=8, markerfacecolor=MODEL_STYLE[m]["color"],
                      markeredgecolor=MODEL_STYLE[m]["color"]) for m in labels]
    labs = list(labels.values())
    handles += [Line2D([], [], linestyle="", marker="o", markersize=8, markerfacecolor="white", markeredgecolor="black"),
                Line2D([], [], color="black", linewidth=0.9)]
    labs += [t("dominated"), t("pareto")]
    if ref:
        handles.append(Line2D([], [], linestyle=":", color="#7f7f7f", marker="D", markerfacecolor="white"))
        labs.append(t("best_baseline"))
    bottom_legend(fig, handles, labs, ncol=4, y=0.02)
    return fig


def plot_efficiency(curves: pd.DataFrame, labels: dict[str, str]):
    """RMSE de teste × horizonte, uma linha por modelo (custo no rótulo da legenda)."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for m in curves.columns:
        st = MODEL_STYLE[m]
        ax.plot(curves.index, curves[m], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                markersize=6, linewidth=1.3, label=labels[m])
    ax.set_xticks(list(curves.index), [str(h) for h in curves.index])
    ax.set_xlabel(t("horizon"))
    ax.set_ylabel(t("rmse_test"))
    handles, labs = ax.get_legend_handles_labels()
    bottom_legend(fig, handles, labs, ncol=3, y=-0.02)
    return fig
