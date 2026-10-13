"""Testes do corte temporal e de funções da EDA (ETAPA 03)."""
import numpy as np
import pandas as pd
import pytest

from src.data.eda import runs, significant_pacf_lags, strength, top_periods
from src.data.split import split_by_cut, temporal_cut, validate_split


def test_temporal_cut_80pct_of_2018_2019():
    cut = temporal_cut("2018-01-01", "2019-12-31 23:59:59", 0.80)
    assert cut == pd.Timestamp("2018-01-01") + pd.Timedelta(hours=14016)


def test_temporal_cut_rejects_bad_fraction():
    with pytest.raises(ValueError):
        temporal_cut("2018-01-01", "2018-01-02", 1.0)


def test_split_is_chronological_and_disjoint():
    ts = pd.date_range("2018-01-01", periods=100, freq="h")
    df = pd.DataFrame({"Timestamp": ts, "y": range(100)})
    tr, te = split_by_cut(df, "Timestamp", ts[80])
    assert len(tr) == 80 and len(te) == 20
    assert tr["Timestamp"].max() < te["Timestamp"].min()
    assert set(tr.index).isdisjoint(te.index)


def test_validate_split_accepts_good_and_rejects_shuffled():
    ts = pd.date_range("2018-01-01", periods=100, freq="h")
    df = pd.DataFrame({"Timestamp": ts, "y": range(100)})
    tr, te = split_by_cut(df, "Timestamp", ts[80])
    assert all(validate_split(df, tr, te, "Timestamp", ts[80]).values())
    with pytest.raises(AssertionError):
        validate_split(df, tr, te.sample(frac=1, random_state=0), "Timestamp", ts[80])
    with pytest.raises(AssertionError):
        validate_split(df, tr.iloc[:-1], te, "Timestamp", ts[80])


def test_runs():
    assert runs(pd.Series([False, True, True, False, True])) == [(1, 2), (4, 4)]
    assert runs(pd.Series([False, False])) == []


def test_strength_bounds():
    rng = np.random.default_rng(0)
    s = np.sin(np.arange(500) * 2 * np.pi / 24)
    assert strength(s, rng.normal(0, 0.01, 500)) > 0.99
    assert strength(np.zeros(500), rng.normal(0, 1, 500)) == 0.0


def test_top_period_detects_daily_cycle():
    x = np.sin(np.arange(24 * 60) * 2 * np.pi / 24)
    assert abs(top_periods(x, k=1)[0]["periodo_h"] - 24) < 0.5


def test_significant_pacf_lags():
    p = np.array([1.0, 0.9, 0.01, -0.5])
    assert significant_pacf_lags(p, n=1000, max_lag=3) == [1, 3]
