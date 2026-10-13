"""Verificação genérica de vazamento: a previsão de T não pode depender de y(> T − h).

Todo novo modelo deve ser incluído em MODEL_FACTORIES.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.baseline import BASELINES
from src.models.lightgbm_model import lgbm_forecaster
from src.models.svr import svr_forecaster

SPEC = {"lags_selecionados": [1, 2, 24, 168], "rolling_mean_windows": [6, 24], "rolling_std_windows": [24],
        "calendar_cyclic": ["hour", "dayofweek"]}
MODEL_FACTORIES = {
    **BASELINES,
    "svr": lambda: svr_forecaster({"C": 1.0, "gamma": "scale", "epsilon": 0.1}, SPEC),
    "lightgbm": lambda: lgbm_forecaster({"n_estimators": 50, "num_leaves": 15, "deterministic": True,
                                         "force_row_wise": True}, SPEC, 42, 1),
}


@pytest.fixture
def series():
    idx = pd.date_range("2019-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 0.1, len(idx)), index=idx)


@pytest.mark.parametrize("name", list(MODEL_FACTORIES))
@pytest.mark.parametrize("h", [1, 6, 12, 24])
def test_prediction_ignores_values_after_origin(series, name, h):
    train = series.iloc[: 24 * 20]
    model = MODEL_FACTORIES[name]().fit(train, h)
    T = series.index[24 * 25]
    base = model.predict(series, pd.DatetimeIndex([T]), h)
    pert = series.copy()
    pert[pert.index > T - pd.Timedelta(hours=h)] += 100.0
    assert np.allclose(base, model.predict(pert, pd.DatetimeIndex([T]), h))


def test_daily_seasonal_rejects_horizon_above_period(series):
    with pytest.raises(ValueError):
        BASELINES["sazonal_diaria"]().fit(series, 25)


def test_baseline_values(series):
    T = series.index[300]
    assert BASELINES["persistencia"]().predict(series, pd.DatetimeIndex([T]), 6)[0] == series[T - pd.Timedelta(hours=6)]
    assert BASELINES["sazonal_semanal"]().predict(series, pd.DatetimeIndex([T]), 1)[0] == series[T - pd.Timedelta(hours=168)]
