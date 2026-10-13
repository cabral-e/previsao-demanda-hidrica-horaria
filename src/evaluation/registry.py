"""Registro de experimentos (seção 8): results/experiments/registry.csv.

Sem sobrescrita silenciosa: mesmo modelo + horizonte + hash de configuração → reutiliza;
configuração diferente → nova versão (v001, v002, ...).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd

from src.utils.paths import EXPERIMENTS_DIR

REGISTRY = EXPERIMENTS_DIR / "registry.csv"
COLUMNS = [
    "exp_id", "timestamp", "modelo", "horizonte_h", "versao", "hiperparametros", "seed", "n_jobs",
    "n_treino", "n_teste", "mae_treino", "rmse_treino", "mae_teste", "rmse_teste",
    "t_treino_s", "t_treino_min_s", "t_treino_max_s", "t_previsao_s", "t_previsao_min_s", "t_previsao_max_s",
    "t_total_s", "t_total_min_s", "t_total_max_s", "memoria_pico_mb", "n_repeats",
    "estrategia", "status", "commit_ou_hash_config",
]


def load() -> pd.DataFrame:
    if not REGISTRY.is_file():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(REGISTRY, encoding="utf-8")


def latest_ok() -> pd.DataFrame:
    """Último experimento com status ok para cada (modelo, horizonte)."""
    reg = load()
    reg = reg[reg["status"] == "ok"]
    return reg.sort_values("timestamp").groupby(["modelo", "horizonte_h"], as_index=False).last()


def find(modelo: str, h: int, cfg_hash: str) -> pd.Series | None:
    reg = load()
    m = (reg["modelo"] == modelo) & (reg["horizonte_h"] == h) & \
        (reg["commit_ou_hash_config"] == cfg_hash) & (reg["status"] == "ok")
    return reg.loc[m].iloc[-1] if m.any() else None


def next_version(modelo: str, h: int) -> str:
    reg = load()
    n = int(((reg["modelo"] == modelo) & (reg["horizonte_h"] == h)).sum())
    return f"v{n + 1:03d}"


def make_exp_id(modelo: str, h: int, versao: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"exp_{when:%Y%m%d_%H%M}_{modelo}_h{h:02d}_{versao}"


def append(row: dict[str, Any]) -> None:
    row = dict(row)
    if isinstance(row.get("hiperparametros"), dict):
        row["hiperparametros"] = json.dumps(row["hiperparametros"], ensure_ascii=False, sort_keys=True)
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Registro incompleto: {missing}")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: row[c] for c in COLUMNS}])
    df.to_csv(REGISTRY, mode="a", header=not REGISTRY.is_file(), index=False, encoding="utf-8")
