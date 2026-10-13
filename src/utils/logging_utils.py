"""Configuração de logging: console + arquivo em results/logs/ (UTF-8)."""
from __future__ import annotations

import logging
from datetime import datetime

from src.utils.paths import LOGS_DIR

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, run_name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Retorna um logger que grava em results/logs/<run_name>_<AAAAMMDD_HHMMSS>.log.

    Chamadas repetidas com o mesmo `name` reutilizam os handlers já criados.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{run_name or name}_{stamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Log em %s", log_file)
    return logger
