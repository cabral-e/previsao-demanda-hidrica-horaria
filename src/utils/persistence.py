"""Persistência auditável: hash de configuração e metadados de artefatos.

Cada artefato salvo (ex.: data/interim/x.parquet) ganha um arquivo vizinho
x.meta.json com o hash do bloco de configuração usado e o hash da entrada.
Uma etapa só reutiliza um artefato se esses hashes coincidirem.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.paths import load_config


def config_hash(*keys: str) -> str:
    """SHA-256 (16 hex) dos blocos de configuração indicados por chaves com ponto.

    Ex.: config_hash("data.start", "data.end") muda se qualquer dos dois mudar.
    """
    cfg = load_config()
    selected = {}
    for key in keys:
        node: Any = cfg
        for part in key.split("."):
            node = node[part]
        selected[key] = node
    payload = json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def meta_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.stem + ".meta.json")


def write_meta(artifact: Path, meta: dict[str, Any]) -> None:
    meta = {"artifact": artifact.name, "created_at": datetime.now().isoformat(timespec="seconds"), **meta}
    meta_path(artifact).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_meta(artifact: Path) -> dict[str, Any] | None:
    p = meta_path(artifact)
    if not (artifact.is_file() and p.is_file()):
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def is_reusable(artifact: Path, **expected: str) -> bool:
    """True se o artefato existe e todos os campos esperados batem com o meta."""
    meta = read_meta(artifact)
    return meta is not None and all(meta.get(k) == v for k, v in expected.items())
