"""Testes do SARIMA: previsão h passos (Kalman) igual à do statsmodels e sem vazamento."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.sarima import SEASON, SarimaForecaster, seasonal_diff, weekly_exog


@pytest.fixture(scope="module")
def series():
    idx = pd.date_range("2019-01-07", periods=24 * 7 * 6, freq="h")
    rng = np.random.default_rng(1)
    t = np.arange(len(idx))
    y = 3 + 0.6 * np.sin(2 * np.pi * t / 24) + 0.2 * np.sin(2 * np.pi * t / 168) + rng.normal(0, 0.05, len(idx))
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def fitted(series):
    return SarimaForecaster((1, 0, 0), (0, 0, 1), {"kind": "fourier", "K": 1}, maxiter=50).fit(series.iloc[: 24 * 28])


@pytest.mark.parametrize("h", [1, 6, 24])
def test_kalman_h_step_matches_statsmodels(series, fitted, h):
    """Previsão vetorizada = forecast do statsmodels com parâmetros fixos aplicado ao histórico até a origem."""
    T = series.index[24 * 32]
    origin = T - pd.Timedelta(hours=h)
    ours = fitted.predict(series, pd.DatetimeIndex([T]), h)[0]

    X = weekly_exog(series.index, "fourier", 1)
    z, Xd = seasonal_diff(series, X)
    n_hist = z.index.get_loc(origin) + 1
    mod = SARIMAX(z.iloc[:n_hist].to_numpy(), exog=Xd.iloc[:n_hist].to_numpy(), order=(1, 0, 0),
                  seasonal_order=(0, 0, 1, SEASON), trend="n")
    res = mod.filter(fitted.params_)
    fut = Xd.iloc[n_hist:n_hist + h].to_numpy()
    z_fc = res.forecast(h, exog=fut)[-1]
    expected = z_fc + series[T - pd.Timedelta(hours=SEASON)]
    assert ours == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("h", [1, 12, 24])
def test_sarima_no_leakage(series, fitted, h):
    T = series.index[24 * 33]
    base = fitted.predict(series, pd.DatetimeIndex([T]), h)
    pert = series.copy()
    pert[pert.index > T - pd.Timedelta(hours=h)] += 100.0
    assert np.allclose(base, fitted.predict(pert, pd.DatetimeIndex([T]), h))


def test_sarima_rejects_horizon_above_season(series, fitted):
    with pytest.raises(ValueError):
        fitted.predict(series, pd.DatetimeIndex([series.index[-1]]), 25)


def test_weekly_exog_shapes():
    idx = pd.date_range("2019-01-07", periods=168, freq="h")
    assert weekly_exog(idx, "fourier", 3).shape == (168, 6)
    dow = weekly_exog(idx, "dow")
    assert dow.shape == (168, 6) and dow.sum(axis=1).max() == 1
