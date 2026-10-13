"""Testes de carregamento e de qualidade temporal (ETAPA 01)."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.inspection import gap_intervals, timestamp_quality
from src.data.load import EXPECTED_COLUMNS, copy_raw, read_raw_excel, sha256
from src.utils.paths import raw_copy_path


def test_sha256_and_copy(tmp_path: Path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")
    dst = tmp_path / "raw" / "a.bin"
    info = copy_raw(src, dst)
    assert info["action"] == "criada"
    assert info["sha256_source"] == sha256(dst)
    assert copy_raw(src, dst)["action"] == "reutilizada"


def test_copy_refuses_divergent_file(tmp_path: Path):
    src, dst = tmp_path / "a.bin", tmp_path / "b.bin"
    src.write_bytes(b"abc")
    dst.write_bytes(b"xyz")
    with pytest.raises(RuntimeError):
        copy_raw(src, dst)


def test_timestamp_quality_detects_gap_and_duplicate():
    ts = pd.Series(pd.to_datetime(
        ["2018-01-01 00:00", "2018-01-01 01:00", "2018-01-01 03:00", "2018-01-01 03:00"]))
    q = timestamp_quality(ts)
    assert q["dominant_step"] == pd.Timedelta(hours=1)
    assert q["missing_timestamps"] == [pd.Timestamp("2018-01-01 02:00")]
    assert q["duplicated_timestamps"] == [pd.Timestamp("2018-01-01 03:00")]


def test_gap_intervals_groups_consecutive_hours():
    step = pd.Timedelta(hours=1)
    missing = [pd.Timestamp("2020-02-29 00:00") + i * step for i in range(24)]
    missing.append(pd.Timestamp("2018-11-04 00:00"))
    assert gap_intervals(missing, step) == [
        (pd.Timestamp("2018-11-04 00:00"), pd.Timestamp("2018-11-04 00:00"), 1),
        (pd.Timestamp("2020-02-29 00:00"), pd.Timestamp("2020-02-29 23:00"), 24),
    ]


@pytest.mark.skipif(not raw_copy_path().is_file(), reason="cópia da base ausente")
def test_read_raw_excel_structure():
    df, info = read_raw_excel()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])
    assert not df.isna().all(axis=1).any()
