"""ETAPA 01 — Inspeção da base (sem modelagem e sem alterar dados).

Uso: python -m src.data.inspection   (a partir da raiz do projeto)
Saídas: results/reports/01_inspecao.md, results/reports/01_inspecao.json,
        data/interim/01_raw_loaded.parquet (base lida, sem tratamento)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import copy_raw, read_raw_excel
from src.utils.logging_utils import get_logger
from src.utils.paths import INTERIM_DIR, REPORTS_DIR, load_config

# Transições do horário de verão brasileiro no período (Decreto 9.242/2017).
# Início: relógio salta 00:00 -> 01:00 (hora ausente).
# Fim: relógio volta 00:00 -> 23:00 do dia anterior (hora repetida).
DST_EVENTS: dict[str, str] = {
    "2018-02-17 23:00": "fim do horário de verão 2017/18 (hora repetida esperada)",
    "2018-11-04 00:00": "início do horário de verão 2018/19 (hora ausente esperada)",
    "2019-02-16 23:00": "fim do horário de verão 2018/19 (hora repetida esperada)",
}


def timestamp_quality(ts: pd.Series) -> dict[str, Any]:
    """Ordenação, duplicidades, frequência e lacunas da coluna temporal."""
    diffs = ts.diff().dropna()
    freq_counts = diffs.value_counts().sort_index()
    dominant = freq_counts[freq_counts.index > pd.Timedelta(0)].idxmax()

    full_range = pd.date_range(ts.min(), ts.max(), freq=dominant)
    missing = full_range.difference(pd.DatetimeIndex(ts.unique()))
    dup_mask = ts.duplicated(keep=False)

    return {
        "start": ts.min(), "end": ts.max(),
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
        "n_non_positive_steps": int((diffs <= pd.Timedelta(0)).sum()),
        "step_counts": {str(k): int(v) for k, v in freq_counts.items()},
        "dominant_step": dominant,
        "n_expected_at_dominant_step": len(full_range),
        "missing_timestamps": list(missing),
        "gap_intervals": gap_intervals(list(missing), dominant),
        "duplicated_timestamps": sorted(set(ts[dup_mask])),
        "n_duplicated_rows": int(ts.duplicated().sum()),
    }


def gap_intervals(missing: list[pd.Timestamp], step: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Agrupa timestamps ausentes consecutivos em intervalos (início, fim, n)."""
    out: list[list] = []
    for t in sorted(missing):
        if out and t - out[-1][1] == step:
            out[-1][1] = t
            out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return [tuple(x) for x in out]


def records_per_year(ts: pd.Series, step: pd.Timedelta) -> list[dict[str, Any]]:
    out = []
    for year, count in ts.dt.year.value_counts().sort_index().items():
        start, end = pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1)
        expected = int((end - start) / step)
        out.append({"ano": int(year), "registros": int(count), "esperado": expected,
                    "diferenca": int(count) - expected})
    return out


def column_profile(s: pd.Series) -> dict[str, Any]:
    """Estatísticas básicas, zeros, negativos e maior sequência constante."""
    values = s.to_numpy()
    runs = (s != s.shift()).cumsum()
    longest_run = int(s.groupby(runs).size().max())
    longest_run_value = float(s.groupby(runs).first()[s.groupby(runs).size().idxmax()])
    d = s.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "dtype": str(s.dtype), "n": int(s.size), "nulos": int(s.isna().sum()),
        "unicos": int(s.nunique()), "zeros": int((values == 0).sum()),
        "pct_zeros": round(float((values == 0).mean() * 100), 2),
        "negativos": int((values < 0).sum()),
        "media": float(d["mean"]), "desvio": float(d["std"]), "min": float(d["min"]),
        "p01": float(d["1%"]), "p25": float(d["25%"]), "mediana": float(d["50%"]),
        "p75": float(d["75%"]), "p99": float(d["99%"]), "max": float(d["max"]),
        "maior_seq_constante": longest_run, "valor_seq_constante": longest_run_value,
        "inteiro": bool(np.all(np.mod(values[~np.isnan(values)], 1) == 0)),
    }


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _df_to_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build_report(df: pd.DataFrame, read_info: dict, copy_info: dict, tsq: dict,
                 per_year: list, profiles: dict, ts_col: str, target: str) -> str:
    L: list[str] = ["# ETAPA 01 — Inspeção da base", "",
                    f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
                    "## Arquivo", "",
                    f"- Origem (somente leitura): `{copy_info['source']}`",
                    f"- Cópia de trabalho: `{copy_info['dest']}` ({copy_info['action']})",
                    f"- SHA-256 origem: `{copy_info['sha256_source']}`",
                    f"- SHA-256 cópia:  `{copy_info['sha256_dest']}`",
                    f"- Hashes idênticos: {'OK' if copy_info['sha256_source'] == copy_info['sha256_dest'] else 'FALHOU'}",
                    "", "## Estrutura", "",
                    f"- Faixa usada da aba no Excel (inclui cabeçalho): {read_info['excel_used_rows']} linhas × "
                    f"{read_info['excel_used_cols']} colunas — o excedente são linhas/colunas vazias "
                    "(resíduo de formatação), ignoradas pelo pandas",
                    f"- Lido pelo pandas: {read_info['rows_read']} linhas × {read_info['cols_read']} colunas",
                    f"- Linhas/colunas totalmente vazias removidas adicionalmente: "
                    f"{read_info['empty_rows_removed']} / {read_info['empty_cols_removed']}",
                    f"- **Shape efetivo: {df.shape[0]} linhas × {df.shape[1]} colunas**",
                    f"- Colunas: {', '.join(f'`{c}`' for c in df.columns)}",
                    f"- Valores ausentes (total): {int(df.isna().sum().sum())}",
                    f"- Linhas inteiramente duplicadas: {int(df.duplicated().sum())}",
                    "", "## Coluna temporal", "",
                    f"- Período: {tsq['start']} a {tsq['end']}",
                    f"- Ordenada crescente: {tsq['is_monotonic_increasing']} "
                    f"(passos ≤ 0: {tsq['n_non_positive_steps']})",
                    f"- **Frequência real (passo dominante): {tsq['dominant_step']}**",
                    f"- Distribuição dos passos: {tsq['step_counts']}",
                    f"- Registros esperados no passo dominante: {tsq['n_expected_at_dominant_step']}",
                    f"- Timestamps ausentes: {len(tsq['missing_timestamps'])}, em "
                    f"{len(tsq['gap_intervals'])} intervalo(s): "
                    + "; ".join(f"{a} a {b} ({n} h)" for a, b, n in tsq["gap_intervals"]),
                    f"- Timestamps duplicados: {len(tsq['duplicated_timestamps'])} → "
                    f"{[str(t) for t in tsq['duplicated_timestamps']]} "
                    f"({tsq['n_duplicated_rows']} linha(s) excedente(s))",
                    "", "### Registros por ano", "", _df_to_md(pd.DataFrame(per_year)), ""]

    L += ["### Vizinhança das anomalias temporais", ""]
    anomalies = [(a, b, f"ausente, {n} h") for a, b, n in tsq["gap_intervals"]] + \
                [(t, t, "duplicado") for t in tsq["duplicated_timestamps"]]
    for a, b, kind in sorted(anomalies):
        label = str(a) if a == b else f"{a} a {b}"
        lo, hi = a - pd.Timedelta(hours=2), b + pd.Timedelta(hours=2)
        L += [f"**{label} ({kind})**", "", _df_to_md(df.loc[df[ts_col].between(lo, hi)]), ""]

    L += ["### Transições do horário de verão (verificação)", ""]
    for when, desc in DST_EVENTS.items():
        t = pd.Timestamp(when)
        n = int((df[ts_col] == t).sum())
        L.append(f"- {when} — {desc}: ocorrências do timestamp = {n}")
    L.append("")

    prof = pd.DataFrame(profiles).T.reset_index(names="coluna")
    cols1 = ["coluna", "dtype", "nulos", "unicos", "zeros", "pct_zeros", "negativos", "inteiro",
             "maior_seq_constante", "valor_seq_constante"]
    cols2 = ["coluna", "media", "desvio", "min", "p01", "p25", "mediana", "p75", "p99", "max"]
    L += ["## Perfil das variáveis", "", _df_to_md(prof[cols1]), "", _df_to_md(prof[cols2]), ""]

    corr = df.drop(columns=ts_col).corr().round(3).reset_index(names="")
    L += ["## Correlação de Pearson (base completa, apenas descritiva)", "", _df_to_md(corr), ""]

    t = profiles[target]
    L += [f"## Variável-alvo `{target}`", "",
          f"- Presente na base: OK; tipo `{t['dtype']}`; nulos: {t['nulos']}",
          f"- Faixa: {t['min']:.4f} a {t['max']:.4f}; média {t['media']:.4f}; desvio {t['desvio']:.4f}",
          f"- Zeros: {t['zeros']}; negativos: {t['negativos']}",
          f"- Maior sequência de valor constante: {t['maior_seq_constante']} registros "
          f"(valor {t['valor_seq_constante']:.4f})",
          "- Unidade e significado: **a confirmar pelo usuário**.", ""]

    L += ["## Observações e hipóteses (a validar; nenhuma alteração foi feita nos dados)", "",
          "1. `B1`–`B3` são contínuas entre 0 e ~60, não binárias. Hipótese: frequência do "
          "inversor das bombas (Hz, rede de 60 Hz), com 0 = bomba desligada.",
          "2. `LT01` varia de 0,26 a 4,46, compatível com nível em metros. `PT01` tem correlação "
          "0,961 com `LT01` e `PT01 − LT01` ≈ 0,93 de mediana: provavelmente a pressão no fundo "
          "do reservatório (m.c.a.), quase redundante com o alvo. Usá-la como exógena no mesmo "
          "instante equivale a vazar o alvo.",
          "3. O timestamp duplicado de 2018-12-31 21:00 tem valores **diferentes** nas duas "
          "linhas (não é cópia), formando uma sequência contínua. Junto com a hora ausente em "
          "2018-11-04 00:00 (início do horário de verão), sugere que o registrador aplicou o "
          "horário de verão em 04/11/2018 e foi reajustado em 31/12/2018. Porém, nos fins do "
          "horário de verão (17/02/2018 e 16/02/2019) **não** há hora repetida. Hipótese "
          "não confirmada: tratar na ETAPA 04.",
          "4. O dia 29/02/2020 está ausente por inteiro (24 h); 2020 será excluído na ETAPA 02.",
          ""]
    return "\n".join(L)


def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main() -> None:
    log = get_logger("inspection", run_name="01_inspecao")
    cfg = load_config()
    ts_col, target = cfg["data"]["timestamp_column"], cfg["data"]["target"]

    copy_info = copy_raw()
    log.info("Cópia %s: %s (sha256 %s)", copy_info["action"], copy_info["dest"], copy_info["sha256_dest"][:16])

    df, read_info = read_raw_excel()
    log.info("Lido: %s; removidos %s", df.shape, read_info)
    if target not in df.columns:
        raise ValueError(f"Variável-alvo {target} não encontrada")

    tsq = timestamp_quality(df[ts_col])
    per_year = records_per_year(df[ts_col], tsq["dominant_step"])
    numeric = [c for c in df.columns if c != ts_col]
    profiles = {c: column_profile(df[c].astype("float64")) for c in numeric}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "01_inspecao.md").write_text(
        build_report(df, read_info, copy_info, tsq, per_year, profiles, ts_col, target), encoding="utf-8")
    (REPORTS_DIR / "01_inspecao.json").write_text(json.dumps(
        {"copy": copy_info, "read": read_info, "timestamp": tsq, "per_year": per_year,
         "profiles": profiles, "dtypes": {c: str(t) for c, t in df.dtypes.items()}},
        indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "01_raw_loaded.parquet"
    df.to_parquet(out, index=False)
    log.info("Frequência dominante %s; ausentes %d; duplicados %d",
             tsq["dominant_step"], len(tsq["missing_timestamps"]), len(tsq["duplicated_timestamps"]))
    log.info("Relatório: %s | Parquet: %s", REPORTS_DIR / "01_inspecao.md", out)


if __name__ == "__main__":
    main()
