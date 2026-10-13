"""Testes de atributos: convenção das defasagens e ausência de informação futura."""
import numpy as np
import pandas as pd
import pytest

from src.features.build import calendar_features, origin_features, select_by_target_time, supervised


@pytest.fixture
def y():
    idx = pd.date_range("2019-01-01", periods=400, freq="h")
    return pd.Series(np.arange(400, dtype=float), index=idx)


def test_lag_convention(y):
    X = origin_features(y, [1, 24], [], [])
    t = y.index[200]
    assert X.loc[t, "lag_1"] == y[t]
    assert X.loc[t, "lag_24"] == y[t - pd.Timedelta(hours=23)]


def test_rolling_uses_only_past(y):
    X = origin_features(y, [], [6], [6])
    t = y.index[100]
    assert X.loc[t, "media_6h"] == pytest.approx(y.iloc[95:101].mean())
    assert X.loc[t, "desvio_6h"] == pytest.approx(y.iloc[95:101].std())


def test_no_future_leakage(y):
    """Alterar valores após t não pode alterar nenhum atributo na origem t."""
    t_pos = 250
    base = supervised(y, 6, [1, 2, 24, 168], [6, 24], [24], ["hour", "dayofweek"])
    y2 = y.copy()
    y2.iloc[t_pos + 1:] += 1000.0
    pert = supervised(y2, 6, [1, 2, 24, 168], [6, 24], [24], ["hour", "dayofweek"])
    t = y.index[t_pos]
    cols = [c for c in base.columns if c not in ("alvo", "ts_alvo")]
    pd.testing.assert_series_equal(base.loc[t, cols], pert.loc[t, cols])
    assert pert.loc[t, "alvo"] != base.loc[t, "alvo"]       # o alvo (futuro) muda, como esperado


def test_supervised_target_alignment(y):
    df = supervised(y, 12, [1], [], [], [])
    t = df.index[0]
    assert df.loc[t, "alvo"] == y[t + pd.Timedelta(hours=12)]
    assert df.loc[t, "ts_alvo"] == t + pd.Timedelta(hours=12)


def test_calendar_is_of_target_time(y):
    df = supervised(y, 5, [1], [], [], ["hour"])
    t = df.index[10]
    expected = np.sin(2 * np.pi * (t + pd.Timedelta(hours=5)).hour / 24)
    assert df.loc[t, "hour_sin"] == pytest.approx(expected)


def test_calendar_cyclic_bounds():
    c = calendar_features(pd.date_range("2019-01-01", periods=200, freq="h"), ["hour", "dayofweek"])
    assert c.abs().max().max() <= 1.0 + 1e-12
    assert np.allclose(c["hour_sin"] ** 2 + c["hour_cos"] ** 2, 1)


def test_select_by_target_time(y):
    df = supervised(y, 24, [1], [], [], [])
    cut = y.index[300]
    tr = select_by_target_time(df, end_exclusive=cut)
    te = select_by_target_time(df, start=cut)
    assert tr["ts_alvo"].max() < cut <= te["ts_alvo"].min()
