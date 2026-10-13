"""Estilo gráfico único do projeto (referência: docs/referencia_grafico_previsao.png).

Fundo branco, borda preta fina, grade maior cinza-clara, fonte serifada,
observado em preto contínuo e previsto em vermelho tracejado. Legível em
preto e branco: séries extras diferenciam-se também por linha/marcador.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.utils.paths import FIGURES_DIR, load_config

OBSERVED = {"color": "black", "linestyle": "-", "linewidth": 1.2}
PREDICTED = {"color": "#d62728", "linestyle": "--", "linewidth": 1.0}
GRID_COLOR = "#d9d9d9"
STRIP_FACE = "#ebebeb"   # faixa lateral de rótulo do painel
SHADE = "#f0f0f0"         # sombreamento de regiões (ex.: período de teste)

# Sequência para séries adicionais: cor + linha + marcador (legível em P&B)
SERIES_STYLES = [
    {"color": "black", "linestyle": "-", "marker": "o"},
    {"color": "#d62728", "linestyle": "--", "marker": "s"},
    {"color": "#1f4e9c", "linestyle": "-.", "marker": "^"},
    {"color": "#6b6b6b", "linestyle": ":", "marker": "D"},
    {"color": "#2e7d32", "linestyle": (0, (5, 1, 1, 1)), "marker": "v"},
    {"color": "#8c510a", "linestyle": (0, (1, 1)), "marker": "P"},
    {"color": "#6a3d9a", "linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "X"},
]

# Estilo fixo por modelo (a identidade segue o modelo em todas as figuras, nunca a posição)
MODEL_ORDER = ["persistencia", "sazonal_diaria", "sazonal_semanal", "sarima", "svr", "lightgbm", "lightgbm_rec"]
MODEL_STYLE = {
    "persistencia":    {"color": "#bdbdbd", "linestyle": ":", "marker": "v", "hatch": ""},
    "sazonal_diaria":  {"color": "#969696", "linestyle": ":", "marker": "^", "hatch": "...."},
    "sazonal_semanal": {"color": "#636363", "linestyle": ":", "marker": "D", "hatch": "xx"},
    "melhor_baseline": {"color": "#7f7f7f", "linestyle": ":", "marker": "D", "hatch": "xx"},
    "sarima":          {"color": "#1f4e9c", "linestyle": "-.", "marker": "s", "hatch": "//"},
    "svr":             {"color": "#2e7d32", "linestyle": "--", "marker": "^", "hatch": "\\\\"},
    "lightgbm":        {"color": "black", "linestyle": "-", "marker": "o", "hatch": ""},
    "lightgbm_rec":    {"color": "#d62728", "linestyle": (0, (5, 1, 1, 1)), "marker": "P", "hatch": "--"},
}

_TEXT = {
    "pt": {"observed": "Observado", "predicted": "Previsto", "time": "Data",
           "hour": "Hora do dia", "weekday": "Dia da semana", "month": "Mês",
           "lag": "Defasagem (horas)", "density": "Frequência", "train": "Treino", "test": "Teste",
           "weekdays": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
           "months": ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"],
           "volume": "Volume equivalente (m³)", "test_samples": "Amostras de teste (1 hora)",
           "horizon": "Horizonte (h)", "time_s": "Tempo (s, escala log)", "rmse_test": "RMSE de teste (m)",
           "mae_test": "MAE de teste (m)", "score": "Score composto (menor é melhor)", "alpha": "Peso do erro α",
           "train_time": "Treino", "pred_time": "Previsão", "total_time": "Total", "pareto": "Fronteira de Pareto",
           "dominated": "Dominado", "best_baseline": "Melhor baseline", "hour1": "hora", "hours": "horas"},
    "en": {"observed": "Observed", "predicted": "Predicted", "time": "Date",
           "hour": "Hour of day", "weekday": "Day of week", "month": "Month",
           "lag": "Lag (hours)", "density": "Frequency", "train": "Train", "test": "Test",
           "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
           "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "volume": "Equivalent volume (m³)", "test_samples": "Test samples (1 hour)",
           "horizon": "Horizon (h)", "time_s": "Time (s, log scale)", "rmse_test": "Test RMSE (m)",
           "mae_test": "Test MAE (m)", "score": "Composite score (lower is better)", "alpha": "Error weight α",
           "train_time": "Training", "pred_time": "Prediction", "total_time": "Total", "pareto": "Pareto front",
           "dominated": "Dominated", "best_baseline": "Best baseline", "hour1": "hour", "hours": "hours"},
}


def horizon_label(h: int) -> str:
    return f"{h} {t('hour1') if h == 1 else t('hours')}"


def lang() -> str:
    return load_config()["project"].get("language", "pt")


def t(key: str):
    """Texto no idioma configurado (project.language)."""
    return _TEXT[lang()][key]


def target_label() -> str:
    return load_config()["plots"]["ylabel_target"][lang()]


def decimal_comma() -> bool:
    return bool(load_config()["plots"].get("decimal_comma")) and lang() == "pt"


def num(x: float, fmt: str = ".3g") -> str:
    """Número formatado para texto de figura (vírgula decimal se configurado)."""
    s = format(x, fmt)
    return s.replace(".", ",") if decimal_comma() else s


def _set_numeric_locale() -> bool:
    import locale
    for name in ("pt_BR.UTF-8", "pt_BR.utf8", "pt_BR", "Portuguese_Brazil.1252", "Portuguese_Brazil"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
            return True
        except locale.Error:
            continue
    return False


def apply_style() -> None:
    cfg = load_config()["plots"]
    if decimal_comma():
        mpl.rcParams["axes.formatter.use_locale"] = _set_numeric_locale()
    installed = set(font_manager.get_font_names())
    family = cfg["font_family"] if cfg["font_family"] in installed else cfg["font_fallback"]
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [family, cfg["font_fallback"]],
        "mathtext.fontset": "stix",
        "axes.labelsize": 16, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 14, "axes.titlesize": 14,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": "black", "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.which": "major", "axes.axisbelow": True,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": cfg["dpi"],
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def bottom_legend(fig, handles, labels, ncol: int | None = None, y: float = -0.02) -> None:
    """Legenda horizontal, centralizada abaixo da figura, sem moldura."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(labels), frameon=False, handlelength=2.2)


def save_figure(fig, name: str, subdir: str | None = None) -> list[Path]:
    """Salva PNG (dpi do config) e PDF em results/figures[/subdir]."""
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths
