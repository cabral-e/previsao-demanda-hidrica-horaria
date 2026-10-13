"""Estratégia RECURSIVA para modelos de aprendizado de máquina (ETAPA 11).

Um único modelo de 1 passo (h = 1) é aplicado h vezes: a cada passo a previsão entra no
histórico como se fosse observação. Vetorizado sobre todas as origens: cada origem mantém um
buffer com as últimas L observações (L = maior janela/defasagem); os atributos são recalculados
do buffer com as MESMAS definições de src/features/build.py (verificado em teste).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build import calendar_features
from src.models.ml_direct import DirectForecaster


class RecursiveForecaster:
    def __init__(self, one_step: DirectForecaster, name: str):
        self.one_step, self.name = one_step, name
        s = one_step.spec
        self.L = max([*s["lags_selecionados"], *s["rolling_mean_windows"], *s["rolling_std_windows"]])

    def fit(self, y_train: pd.Series, h: int = 1) -> "RecursiveForecaster":
        self.one_step.fit(y_train, 1)                      # sempre o modelo de 1 passo
        return self

    def _features(self, buf: np.ndarray, target_ts: pd.DatetimeIndex) -> np.ndarray:
        s = self.one_step.spec
        cols = {}
        for k in s["lags_selecionados"]:
            cols[f"lag_{k}"] = buf[:, -k]
        for w in s["rolling_mean_windows"]:
            cols[f"media_{w}h"] = buf[:, -w:].mean(axis=1)
        for w in s["rolling_std_windows"]:
            cols[f"desvio_{w}h"] = buf[:, -w:].std(axis=1, ddof=1)     # mesmo ddof do pandas
        cal = calendar_features(target_ts, s["calendar_cyclic"])
        for c in cal.columns:
            cols[c] = cal[c].to_numpy()
        return np.column_stack([cols[c] for c in self.one_step.columns_])

    def predict(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray:
        origins = targets - pd.Timedelta(hours=h)
        pos = y_hist.index.get_indexer(origins)
        if (pos < self.L - 1).any() or (pos < 0).any():
            raise ValueError("Origem sem histórico suficiente")
        vals = y_hist.to_numpy(float)
        idx = pos[:, None] + np.arange(-self.L + 1, 1)[None, :]       # buffer: y(t−L+1 .. t), só passado
        buf = vals[idx]
        est = self.one_step.estimator_
        for step in range(1, h + 1):
            target_ts = origins + pd.Timedelta(hours=step)
            pred = np.asarray(est.predict(self._features(buf, pd.DatetimeIndex(target_ts))), float)
            buf = np.concatenate([buf[:, 1:], pred[:, None]], axis=1)
        return pred

    def params(self) -> dict:
        return {**self.one_step.params(), "estrategia": "recursiva (modelo de 1 passo iterado)"}
