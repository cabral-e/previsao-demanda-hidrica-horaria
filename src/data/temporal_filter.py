"""ETAPA 02 — Filtro temporal (somente 2018-01-01 a 2019-12-31).

Uso: python -m src.data.temporal_filter   (a partir da raiz do projeto)
Entrada: data/interim/01_raw_loaded.parquet (ETAPA 01)
Saídas:  data/interim/02_filtrado_2018_2019.parquet (+ .meta.json),
         results/reports/02_filtro_temporal.md
Não trata duplicidades nem lacunas (isso é a ETAPA 04): apenas recorta o período.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.inspection import records_per_year, timestamp_quality
from src.data.load import sha256
from src.utils.logging_utils import get_logger
from src.utils.paths import INTERIM_DIR, REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

INPUT = INTERIM_DIR / "01_raw_loaded.parquet"
OUTPUT = INTERIM_DIR / "02_filtrado_2018_2019.parquet"
CONFIG_KEYS = ("data.timestamp_column", "data.start", "data.end")


def filter_period(df: pd.DataFrame, ts_col: str, start: str, end: str) -> pd.DataFrame:
    """Mantém as linhas com start <= timestamp <= end (inclusivo), preservando a ordem."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError(f"Início {start_ts} posterior ao fim {end_ts}")
    mask = df[ts_col].between(start_ts, end_ts, inclusive="both")
    return df.loc[mask].reset_index(drop=True)


def _summary(df: pd.DataFrame, ts_col: str) -> dict:
    return {"inicio": df[ts_col].min(), "fim": df[ts_col].max(), "registros": len(df)}


def build_report(before: dict, after: dict, excluded: dict, per_year: list, tsq: dict,
                 cfg_hash: str, in_hash: str, out_hash: str, reused: bool) -> str:
    rows = "\n".join(f"| {r['ano']} | {r['registros']} | {r['esperado']} | {r['diferenca']} |" for r in per_year)
    gaps = "; ".join(f"{a} a {b} ({n} h)" for a, b, n in tsq["gap_intervals"]) or "nenhuma"
    dups = ", ".join(str(t) for t in tsq["duplicated_timestamps"]) or "nenhum"
    return f"""# ETAPA 02 — Filtro temporal

Gerado em: {datetime.now().isoformat(timespec='seconds')}{' (artefato reutilizado)' if reused else ''}

## Antes × depois

| | Início | Fim | Registros |
|---|---|---|---:|
| Base original | {before['inicio']} | {before['fim']} | {before['registros']} |
| Selecionado (2018–2019) | {after['inicio']} | {after['fim']} | {after['registros']} |
| Excluído | {excluded['inicio']} | {excluded['fim']} | {excluded['registros']} |

### Registros por ano no período selecionado

| ano | registros | esperado | diferenca |
|---|---|---|---|
{rows}

## Anomalias temporais que permanecem (tratamento na ETAPA 04)

- Ordenada crescente: {tsq['is_monotonic_increasing']}
- Lacunas: {gaps}
- Timestamps duplicados: {dups}
- Grade horária completa esperada: {tsq['n_expected_at_dominant_step']} registros; presentes {after['registros']} (inclui 1 duplicado)

## Rastreabilidade

- Entrada: `data/interim/{INPUT.name}` — SHA-256 `{in_hash}`
- Saída: `data/interim/{OUTPUT.name}` — SHA-256 `{out_hash}`
- Hash da configuração ({', '.join(CONFIG_KEYS)}): `{cfg_hash}`
- Critério: `{CONFIG_KEYS[1]}` ≤ Timestamp ≤ `{CONFIG_KEYS[2]}` (inclusivo); nenhuma linha alterada.
"""


def main() -> None:
    log = get_logger("temporal_filter", run_name="02_filtro_temporal")
    cfg = load_config()
    ts_col, start, end = cfg["data"]["timestamp_column"], cfg["data"]["start"], cfg["data"]["end"]

    if not INPUT.is_file():
        raise FileNotFoundError(f"{INPUT} não encontrado — execute a ETAPA 01 (python -m src.data.inspection)")

    in_hash, cfg_hash = sha256(INPUT), config_hash(*CONFIG_KEYS)
    raw = pd.read_parquet(INPUT)

    reused = is_reusable(OUTPUT, input_sha256=in_hash, config_hash=cfg_hash)
    if reused:
        filtered = pd.read_parquet(OUTPUT)
        log.info("Artefato compatível encontrado; reutilizando %s", OUTPUT.name)
    else:
        filtered = filter_period(raw, ts_col, start, end)
        filtered.to_parquet(OUTPUT, index=False)
        log.info("Filtro aplicado: %d -> %d registros", len(raw), len(filtered))

    excluded = raw.loc[~raw[ts_col].isin(filtered[ts_col])]
    tsq = timestamp_quality(filtered[ts_col])
    per_year = records_per_year(filtered[ts_col], tsq["dominant_step"])
    out_hash = sha256(OUTPUT)

    if not reused:
        write_meta(OUTPUT, {"etapa": "02", "input": INPUT.name, "input_sha256": in_hash,
                            "config_hash": cfg_hash, "output_sha256": out_hash,
                            "n_before": len(raw), "n_after": len(filtered), "start": start, "end": end})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "02_filtro_temporal.md").write_text(build_report(
        _summary(raw, ts_col), _summary(filtered, ts_col), _summary(excluded, ts_col),
        per_year, tsq, cfg_hash, in_hash, out_hash, reused), encoding="utf-8")
    log.info("Selecionado %s a %s (%d); excluído %d", filtered[ts_col].min(), filtered[ts_col].max(),
             len(filtered), len(excluded))


if __name__ == "__main__":
    main()
