"""Engenharia de atributos — sem informação futura.

Convenção (config.yaml → features):
- origem t = último instante observado;
- lag_k = y(t + 1 − k)  (lag_1 = valor em t; lag_24 = mesma hora de t+1 no dia anterior);
- média/desvio móveis em janelas que TERMINAM em t (inclusive);
- calendário (seno/cosseno) do instante-alvo t + h, determinístico e conhecido de antemão.

Uma linha supervisionada para o horizonte h: atributos na origem t, alvo y(t + h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"hour": 24, "dayofweek": 7}


def origin_features(y: pd.Series, lags: list[int], mean_windows: list[int], std_windows: list[int]) -> pd.DataFrame:
    """Atributos da série calculados na origem t (índice = t). Usa só y(≤ t)."""
    if not y.index.is_monotonic_increasing:
        raise ValueError("A série deve estar em ordem cronológica")
    feats = {f"lag_{k}": y.shift(k - 1) for k in sorted(lags)}
    feats |= {f"media_{w}h": y.rolling(w, min_periods=w).mean() for w in mean_windows}
    feats |= {f"desvio_{w}h": y.rolling(w, min_periods=w).std() for w in std_windows}
    return pd.DataFrame(feats, index=y.index)


def calendar_features(index: pd.DatetimeIndex, parts: list[str]) -> pd.DataFrame:
    out = {}
    for p in parts:
        values = getattr(index, p)
        ang = 2 * np.pi * np.asarray(values) / PERIODS[p]
        out[f"{p}_sin"], out[f"{p}_cos"] = np.sin(ang), np.cos(ang)
    return pd.DataFrame(out, index=index)


def supervised(y: pd.Series, h: int, lags: list[int], mean_windows: list[int], std_windows: list[int],
               calendar: list[str]) -> pd.DataFrame:
    """Tabela (origem t) com atributos, `alvo` = y(t+h) e `ts_alvo` = t+h. Linhas incompletas removidas."""
    X = origin_features(y, lags, mean_windows, std_windows)
    target_ts = y.index + pd.Timedelta(hours=h)
    cal = calendar_features(pd.DatetimeIndex(target_ts), calendar)
    cal.index = y.index
    df = pd.concat([X, cal], axis=1)
    df["ts_alvo"] = target_ts
    df["alvo"] = y.shift(-h)
    return df.dropna()


def select_by_target_time(df: pd.DataFrame, start: pd.Timestamp | None = None,
                          end_exclusive: pd.Timestamp | None = None) -> pd.DataFrame:
    """Filtra pelo instante-ALVO (não pela origem): define treino/teste sem vazamento."""
    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["ts_alvo"] >= start
    if end_exclusive is not None:
        m &= df["ts_alvo"] < end_exclusive
    return df.loc[m]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("alvo", "ts_alvo")]
