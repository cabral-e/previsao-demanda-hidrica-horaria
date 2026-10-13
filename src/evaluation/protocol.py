"""Protocolo de avaliação comum a todos os modelos (config.yaml → evaluation).

- Alvos de teste: todas as horas do teste (origem = alvo − h).
- Alvos de treino: horas do treino a partir de início + warmup (iguais em todos os horizontes).
- A série completa (treino + teste) é fornecida aos modelos apenas para LER o passado de
  cada origem; nenhum ajuste usa dados do teste.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.split import TEST_OUT, TRAIN_OUT
from src.utils.paths import load_config


@dataclass(frozen=True)
class EvalData:
    y_train: pd.Series          # série de treino (ajuste)
    y_full: pd.Series           # treino + teste (somente leitura do passado de cada origem)
    cut: pd.Timestamp           # primeiro instante do teste
    train_targets: pd.DatetimeIndex
    test_targets: pd.DatetimeIndex


def load_eval_data() -> EvalData:
    cfg = load_config()
    ts_col, target = cfg["data"]["timestamp_column"], cfg["data"]["target"]
    tr = pd.read_parquet(TRAIN_OUT).set_index(ts_col)[target].asfreq("h")
    te = pd.read_parquet(TEST_OUT).set_index(ts_col)[target].asfreq("h")
    if tr.isna().any() or te.isna().any():
        raise ValueError("Série com lacunas — verifique as ETAPAS 04–05")
    full = pd.concat([tr, te])
    if not full.index.is_monotonic_increasing or full.index.duplicated().any():
        raise ValueError("Série completa fora de ordem ou com duplicatas")
    warmup = pd.Timedelta(hours=cfg["evaluation"]["train_eval_warmup_hours"])
    train_targets = tr.index[tr.index >= tr.index.min() + warmup]
    return EvalData(tr.rename(target), full.rename(target), te.index.min(), train_targets, te.index)


def origins(targets: pd.DatetimeIndex, h: int) -> pd.DatetimeIndex:
    return targets - pd.Timedelta(hours=h)
