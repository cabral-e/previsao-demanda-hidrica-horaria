"""Testes da métrica composta."""
import numpy as np
import pandas as pd

from src.evaluation.composite import minmax, normalize, rank01


def _g():
    return pd.DataFrame({"rmse_teste": [0.5, 0.4, 0.3], "rmse_bb": 0.5, "mae_teste": [0.4, 0.3, 0.2], "mae_bb": 0.4,
                         "t": [0.001, 10.0, 1000.0]})


def test_minmax_and_rank_bounds():
    x = pd.Series([3.0, 1.0, 2.0])
    assert minmax(x).tolist() == [1.0, 0.0, 0.5]
    assert rank01(x).tolist() == [1.0, 0.0, 0.5]
    assert minmax(pd.Series([2.0, 2.0])).tolist() == [0.0, 0.0]       # sem degenerar em divisão por zero


def test_normalization_B_baseline_anchor_and_log_time():
    E, T = normalize(_g(), "B", "rmse", "t")
    assert np.allclose(E, [1.0, 0.8, 0.6])
    assert np.allclose(T, [0.0, 4 / 6, 1.0])                          # log10: −3, 1, 3


def test_normalization_C_ties():
    g = _g().assign(rmse_teste=[0.3, 0.3, 0.5])
    E, _ = normalize(g, "C", "rmse", "t")
    assert E.iloc[0] == E.iloc[1] < E.iloc[2]
