"""Testes do filtro temporal e da persistência (ETAPA 02)."""
import pandas as pd
import pytest

from src.data.temporal_filter import filter_period
from src.utils.persistence import is_reusable, write_meta


def _df(stamps):
    return pd.DataFrame({"Timestamp": pd.to_datetime(stamps), "LT01": range(len(stamps))})


def test_filter_is_inclusive_and_drops_2020():
    df = _df(["2017-12-31 23:00", "2018-01-01 00:00", "2019-12-31 23:00", "2020-01-01 00:00"])
    out = filter_period(df, "Timestamp", "2018-01-01 00:00:00", "2019-12-31 23:59:59")
    assert out["Timestamp"].tolist() == [pd.Timestamp("2018-01-01 00:00"), pd.Timestamp("2019-12-31 23:00")]


def test_filter_preserves_order_and_duplicates():
    df = _df(["2018-12-31 20:00", "2018-12-31 21:00", "2018-12-31 21:00", "2018-12-31 22:00"])
    out = filter_period(df, "Timestamp", "2018-01-01", "2019-12-31 23:59:59")
    assert len(out) == 4
    assert out["LT01"].tolist() == [0, 1, 2, 3]


def test_filter_rejects_inverted_period():
    with pytest.raises(ValueError):
        filter_period(_df(["2018-01-01"]), "Timestamp", "2019-01-01", "2018-01-01")


def test_meta_reuse(tmp_path):
    art = tmp_path / "x.parquet"
    art.write_bytes(b"x")
    write_meta(art, {"config_hash": "abc", "input_sha256": "123"})
    assert is_reusable(art, config_hash="abc", input_sha256="123")
    assert not is_reusable(art, config_hash="zzz", input_sha256="123")
