"""Caminhos centralizados do projeto.

Único ponto (junto com config/config.yaml) onde caminhos são definidos.
A raiz é deduzida da localização deste arquivo, o que evita depender de
strings com espaços/acentos; o config.yaml fornece a base original.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_FILE: Path = PROJECT_ROOT / "config" / "config.yaml"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

DOCS_DIR: Path = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
MODELS_DIR: Path = RESULTS_DIR / "models"
PREDICTIONS_DIR: Path = RESULTS_DIR / "predictions"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
REPORTS_DIR: Path = RESULTS_DIR / "reports"
LOGS_DIR: Path = RESULTS_DIR / "logs"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"

REQUIRED_DIRS: tuple[Path, ...] = (
    PROJECT_ROOT / "config", DOCS_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    PROJECT_ROOT / "src" / "data", PROJECT_ROOT / "src" / "features",
    PROJECT_ROOT / "src" / "models", PROJECT_ROOT / "src" / "evaluation",
    PROJECT_ROOT / "src" / "visualization", PROJECT_ROOT / "src" / "utils",
    NOTEBOOKS_DIR, TESTS_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    FIGURES_DIR, REPORTS_DIR, LOGS_DIR, EXPERIMENTS_DIR,
)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Lê config/config.yaml (UTF-8)."""
    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def raw_source_path() -> Path:
    """Caminho da base original (somente leitura); relativo à raiz do projeto, se não for absoluto."""
    p = Path(load_config()["paths"]["raw_source"])
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def raw_copy_path() -> Path:
    """Caminho da cópia de trabalho da base em data/raw/."""
    return RAW_DIR / raw_source_path().name


def missing_dirs() -> list[Path]:
    """Diretórios obrigatórios (seção 7) que não existem."""
    return [d for d in REQUIRED_DIRS if not d.is_dir()]
