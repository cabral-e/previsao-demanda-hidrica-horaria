"""Previsor de aprendizado de máquina em estratégia DIRETA (um modelo por horizonte).

Atributos: especificação da ETAPA 06 (results/reports/06_selecao_atributos.json).
fit(y_train, h): monta pares (atributos na origem t, alvo y(t+h)) usando apenas y_train.
predict(y_hist, targets, h): atributos nas origens T − h, calculados só com y_hist(≤ origem).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import calendar_features, feature_columns, origin_features, supervised
from src.utils.paths import REPORTS_DIR

FEATURE_SPEC = REPORTS_DIR / "06_selecao_atributos.json"


def load_feature_spec() -> dict[str, Any]:
    spec = json.loads(FEATURE_SPEC.read_text(encoding="utf-8"))
    return {k: spec[k] for k in ("lags_selecionados", "rolling_mean_windows", "rolling_std_windows", "calendar_cyclic")}


class DirectForecaster:
    def __init__(self, name: str, estimator_factory: Callable[[], Any], spec: dict[str, Any], hp: dict[str, Any]):
        self.name, self._factory, self.spec, self.hp = name, estimator_factory, spec, hp
        self.estimator_ = None
        self.h_: int | None = None

    def _supervised(self, y: pd.Series, h: int) -> pd.DataFrame:
        s = self.spec
        return supervised(y, h, s["lags_selecionados"], s["rolling_mean_windows"], s["rolling_std_windows"],
                          s["calendar_cyclic"])

    def fit(self, y_train: pd.Series, h: int) -> "DirectForecaster":
        df = self._supervised(y_train, h)            # alvos t+h ainda dentro do treino (dropna)
        self.columns_ = feature_columns(df)
        self.estimator_ = self._factory()
        self.estimator_.fit(df[self.columns_].to_numpy(), df["alvo"].to_numpy())
        self.h_, self.n_fit_ = h, len(df)
        return self

    def features_at(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray:
        s = self.spec
        orig = targets - pd.Timedelta(hours=h)
        X = origin_features(y_hist, s["lags_selecionados"], s["rolling_mean_windows"], s["rolling_std_windows"])
        X = X.reindex(orig)
        cal = calendar_features(targets, s["calendar_cyclic"])
        cal.index = orig
        X = pd.concat([X, cal], axis=1)[self.columns_]
        if X.isna().any().any():
            raise ValueError("Origem sem histórico suficiente para os atributos")
        return X.to_numpy()

    def predict(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray:
        if self.estimator_ is None or h != self.h_:
            raise RuntimeError(f"Modelo ajustado para h={self.h_}, solicitado h={h}")
        return np.asarray(self.estimator_.predict(self.features_at(y_hist, targets, h)), float)

    def params(self) -> dict:
        return {**self.hp, "estrategia": "direta", "n_atributos": len(getattr(self, "columns_", [])),
                "n_amostras_ajuste": getattr(self, "n_fit_", None)}
