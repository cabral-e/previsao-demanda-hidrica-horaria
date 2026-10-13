"""Coleta de informações de ambiente (sistema, hardware, bibliotecas).

Uso: python -m src.utils.env_info   (a partir da raiz do projeto)
Gera results/reports/00_ambiente.md e results/reports/00_ambiente.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from importlib import metadata
from typing import Any

import psutil

from src.utils.logging_utils import get_logger
from src.utils.paths import (
    CONFIG_FILE, PROJECT_ROOT, REPORTS_DIR, load_config, missing_dirs, raw_source_path,
)

# nome de distribuição pip -> nome de importação
LIBRARIES: dict[str, str] = {
    "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
    "statsmodels": "statsmodels", "scikit-learn": "sklearn", "lightgbm": "lightgbm",
    "matplotlib": "matplotlib", "openpyxl": "openpyxl", "pyarrow": "pyarrow",
    "psutil": "psutil", "pyyaml": "yaml", "pytest": "pytest",
}

REQUIRED_CONFIG_KEYS: dict[str, list[str]] = {
    "project": ["seed", "language"],
    "paths": ["raw_source", "raw_sheet"],
    "data": ["timestamp_column", "target", "start", "end"],
    "split": ["train_fraction"],
    "models": ["names", "horizons_hours"],
    "cost": ["n_repeats", "n_jobs"],
    "composite_score": ["alpha", "alpha_sensitivity"],
    "plots": ["font_family", "dpi", "figsize", "xtick_step", "zoom_samples"],
}


def system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dist, module in LIBRARIES.items():
        try:
            __import__(module)
            versions[dist] = metadata.version(dist)
        except Exception:  # noqa: BLE001 - registrar qualquer falha de importação
            versions[dist] = None
    return versions


def validate_config() -> list[str]:
    """Retorna a lista de chaves ausentes no config.yaml."""
    cfg = load_config()
    problems = []
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        if section not in cfg:
            problems.append(section)
            continue
        problems += [f"{section}.{k}" for k in keys if k not in cfg[section]]
    return problems


def check_raw_source() -> dict[str, Any]:
    path = raw_source_path()
    info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        info["size_mb"] = round(path.stat().st_size / 1024**2, 3)
        info["modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
        try:
            with path.open("rb") as fh:
                info["readable"] = len(fh.read(1024)) > 0
        except OSError as exc:
            info["readable"] = False
            info["error"] = str(exc)
    return info


def collect() -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "venv_on_project_drive": os.path.splitdrive(sys.prefix)[0].upper()
        == os.path.splitdrive(str(PROJECT_ROOT))[0].upper(),
        "system": system_info(),
        "libraries": library_versions(),
        "missing_dirs": [str(d.relative_to(PROJECT_ROOT)) for d in missing_dirs()],
        "config_file": str(CONFIG_FILE),
        "config_missing_keys": validate_config(),
        "raw_source": check_raw_source(),
    }


def to_markdown(info: dict[str, Any]) -> str:
    s, raw = info["system"], info["raw_source"]
    ok = lambda cond: "OK" if cond else "FALHOU"  # noqa: E731
    lines = [
        "# ETAPA 00 — Ambiente e estrutura",
        "",
        f"Gerado em: {info['generated_at']}",
        "",
        "## Sistema e hardware",
        "",
        "| Item | Valor |", "|---|---|",
        f"| Sistema operacional | {s['os']} |",
        f"| Processador | {s['processor']} |",
        f"| Núcleos físicos / lógicos | {s['cpu_physical_cores']} / {s['cpu_logical_cores']} |",
        f"| RAM total / disponível (GB) | {s['ram_total_gb']} / {s['ram_available_gb']} |",
        f"| Python | {s['python_version']} |",
        f"| Interpretador | `{s['python_executable']}` |",
        "",
        "## Bibliotecas",
        "",
        "| Biblioteca | Versão |", "|---|---|",
        *[f"| {k} | {v or 'NÃO INSTALADA'} |" for k, v in info["libraries"].items()],
        "",
        "## Validações",
        "",
        "| Verificação | Resultado |", "|---|---|",
        f"| Ambiente virtual fora da unidade do projeto | {ok(not info['venv_on_project_drive'])} |",
        f"| Todas as bibliotecas importáveis | {ok(all(info['libraries'].values()))} |",
        f"| Estrutura de pastas (seção 7) | {ok(not info['missing_dirs'])} |",
        f"| Chaves obrigatórias do config.yaml | {ok(not info['config_missing_keys'])} |",
        f"| Base original encontrada | {ok(raw['exists'])} |",
        f"| Base original legível | {ok(raw.get('readable', False))} |",
        "",
        "## Base original (somente leitura)",
        "",
        f"- Caminho: `{raw['path']}`",
        f"- Tamanho: {raw.get('size_mb', '—')} MB",
        f"- Última modificação: {raw.get('modified', '—')}",
    ]
    if info["missing_dirs"]:
        lines += ["", "Pastas ausentes: " + ", ".join(info["missing_dirs"])]
    if info["config_missing_keys"]:
        lines += ["", "Chaves ausentes no config: " + ", ".join(info["config_missing_keys"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    log = get_logger("env_info", run_name="00_ambiente")
    info = collect()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "00_ambiente.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "00_ambiente.md").write_text(to_markdown(info), encoding="utf-8")
    log.info("Python %s em %s", info["system"]["python_version"], info["system"]["python_executable"])
    for lib, ver in info["libraries"].items():
        log.info("%-13s %s", lib, ver or "NÃO INSTALADA")
    log.info("Pastas ausentes: %s", info["missing_dirs"] or "nenhuma")
    log.info("Chaves ausentes no config: %s", info["config_missing_keys"] or "nenhuma")
    log.info("Base original: %s", info["raw_source"])
    log.info("Relatório salvo em %s", REPORTS_DIR / "00_ambiente.md")


if __name__ == "__main__":
    main()
