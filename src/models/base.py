"""Interface comum dos modelos de previsão.

fit(y_train, h): ajusta usando SOMENTE a série de treino.
predict(y_hist, targets, h): previsão para cada alvo T usando apenas y_hist(≤ T − h).
  `y_hist` pode conter valores posteriores (série completa); o modelo NÃO pode lê-los.
  Isso é verificado por teste automatizado (tests/test_models_leakage.py).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, y_train: pd.Series, h: int) -> "Forecaster": ...

    def predict(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray: ...

    def params(self) -> dict: ...
