"""Preparação para publicação: substitui caminhos absolutos locais por marcadores genéricos.

Uso: python -m src.utils.publish [--check]
  --check  apenas lista as ocorrências (não altera arquivos)

Aplica-se aos arquivos de texto versionáveis (fora de data/, results/models/ e results/predictions/).
Rode novamente após qualquer reexecução das etapas, antes de publicar.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from src.utils.paths import PROJECT_ROOT

TEXT_EXT = {".md", ".json", ".log", ".csv", ".yaml", ".yml", ".txt", ".py"}
SKIP_DIRS = {"data", ".git", "__pycache__", ".pytest_cache"}
SKIP_SUBPATHS = {("results", "models"), ("results", "predictions")}


def _variants(path: str) -> list[str]:
    """Formas em que um caminho Windows aparece: \\, /, \\\\ (JSON) e repr."""
    back = path.replace("/", "\\")
    return sorted({back, back.replace("\\", "/"), back.replace("\\", "\\\\")}, key=len, reverse=True)


def replacements() -> list[tuple[str, str]]:
    home = os.path.expanduser("~")
    pairs = []
    for target, mark in ((str(PROJECT_ROOT), "<PROJETO>"), (str(PROJECT_ROOT.parent), "<PASTA_TCC>"), (home, "<USUARIO>")):
        pairs += [(v, mark) for v in _variants(target)]
    return pairs


def candidate_files() -> list[Path]:
    out = []
    for p in PROJECT_ROOT.rglob("*"):
        rel = p.relative_to(PROJECT_ROOT).parts
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT or rel[0] in SKIP_DIRS:
            continue
        if any(rel[:2] == s for s in SKIP_SUBPATHS) or "__pycache__" in rel:
            continue
        if p.resolve() == Path(__file__).resolve():
            continue
        out.append(p)
    return out


def main(check: bool) -> None:
    pairs = replacements()
    pattern = re.compile("|".join(re.escape(a) for a, _ in pairs), flags=re.IGNORECASE)
    lookup = {a.lower(): b for a, b in pairs}
    total = 0
    for f in candidate_files():
        text = f.read_text(encoding="utf-8", errors="strict")
        n = len(pattern.findall(text))
        if not n:
            continue
        total += n
        print(f"{n:4d}  {f.relative_to(PROJECT_ROOT)}")
        if not check:
            f.write_text(pattern.sub(lambda m: lookup[m.group(0).lower()], text), encoding="utf-8")
    print(f"Total: {total} ocorrência(s) {'encontradas' if check else 'substituídas'}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args().check)
