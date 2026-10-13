"""ETAPA 07 — Baselines ingênuos.

- persistência:       ŷ(T) = y(T − h)     (último valor observado na origem)
- sazonal diária:     ŷ(T) = y(T − 24)    (mesma hora do dia anterior; requer h ≤ 24)
- sazonal semanal:    ŷ(T) = y(T − 168)   (mesma hora da semana anterior; requer h ≤ 168)

Uso: python -m src.models.baseline [--force]
Saídas: results/experiments/registry.csv, results/predictions/exp_*.parquet,
        results/reports/07_baseline.md
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation.protocol import load_eval_data
from src.evaluation.runner import run_experiment
from src.utils.logging_utils import get_logger
from src.utils.paths import REPORTS_DIR, load_config

CONFIG_KEYS = ("data.target", "split", "evaluation", "baselines", "cost", "project.seed")


class SeasonalNaive:
    """ŷ(T) = y(T − lag). Com lag = h, é a persistência."""

    def __init__(self, name: str, lag: int | None):
        self.name, self.lag = name, lag     # lag None = persistência (lag = h)

    def fit(self, y_train: pd.Series, h: int) -> "SeasonalNaive":
        lag = self._lag(h)
        if lag < h:
            raise ValueError(f"{self.name}: lag {lag} < horizonte {h} usaria informação futura")
        return self

    def _lag(self, h: int) -> int:
        return h if self.lag is None else self.lag

    def predict(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray:
        src = targets - pd.Timedelta(hours=self._lag(h))
        return y_hist.reindex(src).to_numpy(dtype=float)

    def params(self) -> dict:
        return {"lag": "h" if self.lag is None else self.lag}


BASELINES = {
    "persistencia": lambda: SeasonalNaive("persistencia", None),
    "sazonal_diaria": lambda: SeasonalNaive("sazonal_diaria", 24),
    "sazonal_semanal": lambda: SeasonalNaive("sazonal_semanal", 168),
}


def build_report(rows: list[dict], data) -> str:
    df = pd.DataFrame(rows)
    L = ["# ETAPA 07 — Baselines", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Protocolo (aprovado; comum a todos os modelos)", "",
         f"- Teste: {len(data.test_targets)} alvos ({data.test_targets.min()} a {data.test_targets.max()}), "
         "idênticos em todos os horizontes; origem = alvo − h.",
         f"- Treino (métricas): {len(data.train_targets)} alvos ({data.train_targets.min()} a "
         f"{data.train_targets.max()}), após aquecimento de 192 h.",
         "- Cada modelo é ajustado uma vez no treino; em cada origem usa as observações reais até a origem.", "",
         "## Definições", "",
         "| Baseline | Previsão | Observação |", "|---|---|---|",
         "| Persistência | ŷ(T) = y(T − h) | último valor conhecido |",
         "| Sazonal diária | ŷ(T) = y(T − 24) | em h = 24 coincide com a persistência |",
         "| Sazonal semanal | ŷ(T) = y(T − 168) | independe de h ≤ 168 |", "",
         "## Resultados (m)", "",
         "| Baseline | h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino (RMSE) | t_previsão (ms) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['modelo']} | {r['horizonte_h']} | {r['mae_treino']:.4f} | {r['rmse_treino']:.4f} | "
                 f"{r['mae_teste']:.4f} | {r['rmse_teste']:.4f} | {r['rmse_teste'] / r['rmse_treino']:.3f} | "
                 f"{1000 * r['t_previsao_s']:.3f} |")
    best = df.loc[df.groupby("horizonte_h")["rmse_teste"].idxmin(), ["horizonte_h", "modelo", "rmse_teste", "mae_teste"]]
    L += ["", "## Melhor baseline por horizonte (RMSE de teste) — referência a ser superada", "",
          "| h | Baseline | RMSE teste | MAE teste |", "|---:|---|---:|---:|"]
    for r in best.itertuples(index=False):
        L.append(f"| {r.horizonte_h} | {r.modelo} | {r.rmse_teste:.4f} | {r.mae_teste:.4f} |")
    L += ["", "Os baselines não têm ajuste (t_treino ≈ 0) e preveem por indexação; o tempo é desprezível.",
          f"Registro: `results/experiments/registry.csv`; previsões: `results/predictions/exp_*.parquet`.", ""]
    return "\n".join(L)


def main(force: bool = False) -> None:
    log = get_logger("baseline", run_name="07_baseline")
    cfg = load_config()
    data = load_eval_data()
    rows = []
    for name, factory in BASELINES.items():
        for h in cfg["models"]["horizons_hours"]:
            rows.append(run_experiment(factory, name, h, data, CONFIG_KEYS, strategy="indexacao", log=log, force=force))
    (REPORTS_DIR / "07_baseline.md").write_text(build_report(rows, data), encoding="utf-8")
    log.info("Relatório: %s", REPORTS_DIR / "07_baseline.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
