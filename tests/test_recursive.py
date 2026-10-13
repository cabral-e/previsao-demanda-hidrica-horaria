"""Testes da estratégia recursiva."""
import numpy as np
import pandas as pd
import pytest

from src.models.lightgbm_model import lgbm_forecaster
from src.models.ml_recursive import RecursiveForecaster

SPEC = {"lags_selecionados": [1, 2, 24, 168], "rolling_mean_windows": [6, 24, 168], "rolling_std_windows": [24, 168],
        "calendar_cyclic": ["hour", "dayofweek"]}
HP = {"n_estimators": 60, "num_leaves": 15, "deterministic": True, "force_row_wise": True}


@pytest.fixture(scope="module")
def series():
    idx = pd.date_range("2019-01-07", periods=24 * 7 * 8, freq="h")
    rng = np.random.default_rng(3)
    t = np.arange(len(idx))
    return pd.Series(3 + 0.6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.05, len(idx)), index=idx)


def test_first_step_equals_direct_h1(series):
    """Com h = 1, a recursão (atributos do buffer em numpy) = modelo direto (atributos em pandas)."""
    direct = lgbm_forecaster(HP, SPEC, 42, 1).fit(series.iloc[: 24 * 42], 1)
    rec = RecursiveForecaster(lgbm_forecaster(HP, SPEC, 42, 1), "rec").fit(series.iloc[: 24 * 42])
    targets = series.index[24 * 45: 24 * 47]
    assert np.allclose(direct.predict(series, targets, 1), rec.predict(series, targets, 1))


@pytest.mark.parametrize("h", [6, 24])
def test_recursive_no_leakage(series, h):
    rec = RecursiveForecaster(lgbm_forecaster(HP, SPEC, 42, 1), "rec").fit(series.iloc[: 24 * 42])
    T = pd.DatetimeIndex([series.index[24 * 50]])
    base = rec.predict(series, T, h)
    pert = series.copy()
    pert[pert.index > T[0] - pd.Timedelta(hours=h)] += 100.0
    assert np.allclose(base, rec.predict(pert, T, h))
