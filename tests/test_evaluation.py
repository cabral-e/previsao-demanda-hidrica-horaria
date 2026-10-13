"""Testes de métricas, custo e registro."""
import time

import numpy as np
import pytest

from src.evaluation import registry
from src.evaluation.cost import PeakRSS, measure
from src.evaluation.metrics import diebold_mariano, holm, mae, newey_west_lag, rmse


def test_dm_identical_errors_not_significant():
    e = np.random.default_rng(0).normal(size=500)
    r = diebold_mariano(e, e.copy(), h=1)
    assert r["p"] == 1.0


def test_dm_detects_clearly_worse_model_and_sign():
    rng = np.random.default_rng(1)
    e_good, e_bad = rng.normal(0, 1, 2000), rng.normal(0, 2, 2000)
    r = diebold_mariano(e_good, e_bad, h=6)
    assert r["dm"] < 0 and r["p"] < 1e-6


def test_dm_equal_accuracy_rarely_rejects():
    rng = np.random.default_rng(2)
    rejections = sum(diebold_mariano(rng.normal(size=1000), rng.normal(size=1000), h=1)["p"] < 0.05 for _ in range(200))
    assert rejections / 200 < 0.10


def test_newey_west_lag():
    assert newey_west_lag(3504, 1) == 8
    assert newey_west_lag(3504, 24) == 23


def test_holm_known_values():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_mae_rmse_known_values():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 1.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))


def test_metrics_reject_nan_and_shape():
    with pytest.raises(ValueError):
        mae(np.array([1.0, 2.0]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_measure_times_and_repeats():
    res = measure(lambda: time.sleep(0.02) or "m", lambda m: [1.0], n_repeats=3, mem_interval_s=0.005)
    assert res["n_repeats"] == 3
    assert res["t_treino_min_s"] >= 0.015
    assert res["t_treino_min_s"] <= res["t_treino_s"] <= res["t_treino_max_s"]
    assert res["t_total_s"] >= res["t_treino_s"]


def test_peak_rss_detects_allocation():
    with PeakRSS(0.001) as mem:
        big = np.ones(30 * 2**20 // 8)   # ~30 MB
        time.sleep(0.05)
        del big
    assert mem.delta_mb > 15


def test_exp_id_format():
    from datetime import datetime
    assert registry.make_exp_id("lightgbm", 6, "v001", datetime(2026, 10, 5, 14, 30)) == \
        "exp_20261005_1430_lightgbm_h06_v001"
