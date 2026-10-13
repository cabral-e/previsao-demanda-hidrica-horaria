"""Testes do tratamento (ETAPA 04)."""
import numpy as np
import pandas as pd
import pytest

from src.data.treatment import (
    best_shift, interpolate_segments, realign_dst, regularize_and_fill, resolve_duplicates, validate,
)


def _dst_like_frame():
    """Grade com a mesma estrutura da base: falta 2018-11-04 00:00, 2018-12-31 21:00 duplicado."""
    grid = pd.date_range("2018-11-03 20:00", "2018-12-31 23:00", freq="h")
    labels = list(grid[grid != pd.Timestamp("2018-11-04 00:00")])
    i = labels.index(pd.Timestamp("2018-12-31 21:00"))
    labels.insert(i, pd.Timestamp("2018-12-31 21:00"))
    return pd.DataFrame({"Timestamp": labels, "LT01": np.arange(len(labels), dtype=float)})


def test_realign_produces_regular_grid_without_changing_values():
    df = _dst_like_frame()
    out = realign_dst(df, "Timestamp")
    assert out["LT01"].tolist() == df["LT01"].tolist()          # sequência intocada
    checks = validate(out, "Timestamp")
    assert all(checks.values())


def test_realign_requires_exact_structure():
    df = _dst_like_frame().drop_duplicates("Timestamp")
    with pytest.raises(ValueError):
        realign_dst(df, "Timestamp")


def test_keep_labels_path():
    df = _dst_like_frame()
    out, n_dup = resolve_duplicates(df, "Timestamp", "first")
    assert n_dup == 1
    out, filled = regularize_and_fill(out, "Timestamp")
    assert filled == [pd.Timestamp("2018-11-04 00:00")]
    assert all(validate(out, "Timestamp").values())


def test_interpolate_segments():
    ts = pd.date_range("2019-01-30 04:00", periods=6, freq="h")
    df = pd.DataFrame({"Timestamp": ts, "LT01": [4.0, 4.0, 2.0, 2.0, 2.0, 5.0]})
    out, n = interpolate_segments(df, "Timestamp", "LT01", [["2019-01-30 06:00", "2019-01-30 08:00"]])
    assert n == 3
    assert np.allclose(out["LT01"], [4.0, 4.0, 4.25, 4.5, 4.75, 5.0])


def test_validate_rejects_nan():
    ts = pd.date_range("2019-01-01", periods=3, freq="h")
    with pytest.raises(AssertionError):
        validate(pd.DataFrame({"Timestamp": ts, "LT01": [1.0, np.nan, 1.0]}), "Timestamp")


def test_best_shift_recovers_known_shift():
    ref = np.sin(np.arange(24) * 2 * np.pi / 24) + (np.arange(24) == 20)
    assert best_shift(np.roll(ref, 1), ref)["shift"] == 1
