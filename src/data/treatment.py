"""ETAPA 04 — Tratamento dos dados.

Dois modos:
  python -m src.data.treatment --diagnose   evidências para as decisões (não altera dados)
  python -m src.data.treatment              aplica o tratamento conforme config.yaml → treatment

O modo de aplicação recusa executar enquanto alguma decisão pendente for null.
Entrada: data/interim/02_filtrado_2018_2019.parquet
Saídas:  data/processed/04_tratado.parquet (+ .meta.json), results/reports/04_tratamento.md
         results/reports/04_diagnostico.md (modo --diagnose)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.inspection import timestamp_quality
from src.data.load import sha256
from src.utils.logging_utils import get_logger
from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

INPUT = INTERIM_DIR / "02_filtrado_2018_2019.parquet"
OUTPUT = PROCESSED_DIR / "04_tratado.parquet"
CONFIG_KEYS = ("data.timestamp_column", "data.target", "treatment", "eda.jump_z_threshold")

# Hipótese de horário de verão (ETAPA 01): relógio adiantado de 2018-11-04 01:00
# até a PRIMEIRA ocorrência de 2018-12-31 21:00 (inclusive).
DST_WINDOW_START = pd.Timestamp("2018-11-04 01:00")
DST_WINDOW_END_DUP = pd.Timestamp("2018-12-31 21:00")

# Janelas de rótulo para estimar deslocamento do perfil diário contra a referência
REFERENCE = ("2019-09-01 00:00", "2019-12-31 23:00")   # sem horário de verão legal no Brasil
WINDOWS = {
    "2018-01-01 a 2018-02-17 (HV legal)": ("2018-01-01 00:00", "2018-02-17 23:00"),
    "2018-02-18 a 2018-11-03 (sem HV)": ("2018-02-18 00:00", "2018-11-03 23:00"),
    "2018-11-04 a 2018-12-31 21h (HV legal; suspeita)": ("2018-11-04 01:00", "2018-12-31 20:00"),
    "2019-01-01 a 2019-02-16 (HV legal)": ("2019-01-01 00:00", "2019-02-16 23:00"),
    "2019-02-17 a 2019-08-31 (sem HV)": ("2019-02-17 00:00", "2019-08-31 23:00"),
}


# ======================================================================= diagnóstico
def add_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bombas_ligadas"] = (out[["B1", "B2", "B3"]] > 0).sum(axis=1)
    out["PT01_menos_LT01"] = out["PT01"] - out["LT01"]
    return out


def hourly_profile(df: pd.DataFrame, ts_col: str, col: str) -> np.ndarray:
    return df.groupby(df[ts_col].dt.hour)[col].mean().reindex(range(24)).to_numpy()


def best_shift(profile: np.ndarray, reference: np.ndarray, max_shift: int = 3) -> dict[str, Any]:
    """Deslocamento s (h) que melhor alinha o perfil à referência.

    s > 0: os eventos da janela aparecem s horas DEPOIS (rótulo adiantado em relação
    à referência). Correlação de Pearson circular para s em [-max_shift, max_shift].
    """
    corrs = {s: float(np.corrcoef(np.roll(profile, -s), reference)[0, 1]) for s in range(-max_shift, max_shift + 1)}
    s_best = max(corrs, key=corrs.get)
    return {"shift": s_best, "corr_best": corrs[s_best], "corr_zero": corrs[0], "corrs": corrs}


def dst_evidence(df: pd.DataFrame, ts_col: str, cols: list[str]) -> list[dict[str, Any]]:
    ref = df[df[ts_col].between(*map(pd.Timestamp, REFERENCE))]
    rows = []
    for name, (a, b) in WINDOWS.items():
        win = df[df[ts_col].between(pd.Timestamp(a), pd.Timestamp(b))]
        row = {"janela": name, "n": len(win)}
        for c in cols:
            r = best_shift(hourly_profile(win, ts_col, c), hourly_profile(ref, ts_col, c))
            row[c] = r
        rows.append(row)
    return rows


def realign_structure_check(df: pd.DataFrame, ts_col: str) -> dict[str, Any]:
    """Aplica o realinhamento em cópia e verifica se a grade fica perfeita."""
    out = realign_dst(df, ts_col)
    q = timestamp_quality(out[ts_col])
    return {"n": len(out), "lacunas": len(q["missing_timestamps"]), "duplicados": len(q["duplicated_timestamps"]),
            "ordenado": q["is_monotonic_increasing"], "n_deslocados": int(out.attrs.get("n_shifted", 0))}


def jump_evidence(df: pd.DataFrame, ts_col: str, target: str, z: float, context_h: int = 3) -> list[dict[str, Any]]:
    y = df[target].reset_index(drop=True)
    dy = y.diff()
    idx = dy[dy.abs() > z * dy.std()].index
    cols = [ts_col, "LT01", "PT01", "PT01_menos_LT01", "B1", "B2", "B3", "FT01", "FT02", "FT03"]
    out = []
    for i in idx:
        lo, hi = max(0, i - context_h), min(len(df) - 1, i + context_h)
        seg = df.iloc[lo:hi + 1][cols]
        d_pt = float(df["PT01"].iloc[i] - df["PT01"].iloc[i - 1])
        out.append({"ts": df[ts_col].iloc[i], "dif_LT01": float(dy[i]), "dif_PT01": d_pt, "tabela": seg})
    return out


def mass_balance(df: pd.DataFrame, ts_col: str, factor: float, stamps: list[pd.Timestamp]) -> dict[str, Any]:
    """Balanço de massa da câmara: ΔV (= fator·ΔLT01) × vazão líquida (FT01 − FT02 − FT03).

    Ajusta ΔV = a + b·líquida (média trapezoidal entre leituras) e mede o resíduo nos
    saltos. Resíduo muito acima do usual = variação de nível não explicada pelas vazões.
    """
    d = df.reset_index(drop=True)
    dv = d["LT01"].diff() * factor
    net = d["FT01"] - d["FT02"] - d["FT03"]
    net_avg = (net + net.shift()) / 2
    m = dv.notna() & net_avg.notna()
    b, a = np.polyfit(net_avg[m], dv[m], 1)
    res = dv - (a + b * net_avg)
    sd = float(res[m].std())
    rows = []
    for t in stamps:
        i = d.index[d[ts_col] == t][0]
        rows.append({"ts": t, "dV_obs_m3": float(dv[i]), "dV_balanco_m3": float(a + b * net_avg[i]),
                     "residuo_m3": float(res[i]), "residuo_em_sd": float(abs(res[i]) / sd)})
    return {"a": float(a), "b": float(b), "r": float(np.corrcoef(net_avg[m], dv[m])[0, 1]),
            "res_sd": sd, "res_mediana_abs": float(res[m].abs().median()),
            "n_res_5sd": int((res[m].abs() > 5 * sd).sum()), "saltos": rows}


def sampling_continuity(df: pd.DataFrame, ts_col: str, cols: tuple[str, ...] = ("LT01", "PT01")) -> list[dict[str, Any]]:
    """As variações nas anomalias de rótulo parecem de 1 h ou de 2 h?

    Compara |Δ| observado com a distribuição de |Δ| em 1 h e em 2 h partindo da mesma
    hora do dia (nov–dez, dias regulares). Distingue medição perdida/repetida de rótulo errado.
    """
    d = df.reset_index(drop=True)
    ts = d[ts_col]
    regular = d[~ts.duplicated(keep=False)].set_index(ts_col).asfreq("h")
    dup = ts.index[ts == DST_WINDOW_END_DUP]
    i23 = ts.index[ts == DST_WINDOW_START - pd.Timedelta(hours=2)][0]
    cases = [("Lacuna: 2018-11-03 23h → 2018-11-04 01h (rótulos)", i23, i23 + 1, 23),
             ("Duplicata: 1ª → 2ª ocorrência de 2018-12-31 21h", dup[0], dup[1], 21)]
    out = []
    for label, i0, i1, h in cases:
        for c in cols:
            obs = abs(d[c][i1] - d[c][i0])
            res = {"caso": label, "var": c, "obs": float(obs)}
            for k in (1, 2):
                s = regular[c]
                dd = (s.shift(-k) - s).abs()
                ref = dd[(dd.index.hour == h) & dd.index.month.isin([11, 12])].dropna()
                res[f"mediana_{k}h"] = float(ref.median())
                res[f"pct_{k}h"] = float((ref < obs).mean() * 100)
            out.append(res)
    return out


def offset_anomalies(df: pd.DataFrame, ts_col: str, k: float = 6.0) -> dict[str, Any]:
    """Horas em que PT01 − LT01 foge muito da mediana (sensores discordam)."""
    off = df["PT01_menos_LT01"]
    med = off.median()
    mad = (off - med).abs().median() * 1.4826
    bad = df[(off - med).abs() > k * mad]
    return {"mediana": float(med), "mad": float(mad), "limiar": float(k * mad), "n": len(bad),
            "amostra": bad[[ts_col, "LT01", "PT01", "PT01_menos_LT01"]].head(15)}


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def diagnose_report(dst: list, cols: list[str], struct: dict, jumps: list, offs: dict, z: float,
                    cont: list, mb: dict) -> str:
    L = ["# ETAPA 04 — Diagnóstico para decisões de tratamento", "",
         f"Gerado em: {datetime.now().isoformat(timespec='seconds')}. Nenhum dado foi alterado.", "",
         "## 1. Hipótese de horário de verão", "",
         f"Referência: {REFERENCE[0]} a {REFERENCE[1]} (o Brasil não teve horário de verão em 2019/20).",
         "Para cada janela, estima-se o deslocamento (h) do perfil horário médio que maximiza a correlação "
         "com a referência. `s = +1`: os eventos da janela aparecem 1 h depois (relógio adiantado).", "",
         "| Janela | n | " + " | ".join(f"{c}: s (r com s / r com 0)" for c in cols) + " |",
         "|---|---:|" + "---|" * len(cols)]
    for r in dst:
        cells = [f"**{r[c]['shift']:+d}** ({r[c]['corr_best']:.3f} / {r[c]['corr_zero']:.3f})" for c in cols]
        L.append(f"| {r['janela']} | {r['n']} | " + " | ".join(cells) + " |")
    L += ["", "### Verificação estrutural do realinhamento", "",
          f"Deslocar −1 h os rótulos de {DST_WINDOW_START} até a 1ª ocorrência de {DST_WINDOW_END_DUP} "
          f"({struct['n_deslocados']} registros) resulta em: {struct['n']} registros, "
          f"{struct['lacunas']} lacuna(s), {struct['duplicados']} duplicado(s), ordenado = {struct['ordenado']}.", "",
          "### Continuidade da amostragem nas anomalias", "",
          "Percentil do |Δ| observado na distribuição de variações em 1 h e em 2 h (mesma hora do dia, nov–dez). "
          "Medição perdida → |Δ| típico de 2 h; leitura repetida na mesma hora → |Δ| ≈ 0.", "",
          "| Caso | Var. | |Δ| obs. | mediana 1 h | percentil em 1 h | mediana 2 h | percentil em 2 h |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for c in cont:
        L.append(f"| {c['caso']} | {c['var']} | {c['obs']:.3f} | {c['mediana_1h']:.3f} | {c['pct_1h']:.0f}% | "
                 f"{c['mediana_2h']:.3f} | {c['pct_2h']:.0f}% |")
    L += ["", f"## 2. Saltos abruptos (|ΔLT01| > {z}·σ)", "",
          "PT01 (pressão de sucção, mca) mede o mesmo nível por outro sensor: se PT01 acompanha o salto, "
          "o evento é físico; se não acompanha, é falha do sensor de nível.", ""]
    for j in jumps:
        L += [f"**{j['ts']}** — ΔLT01 = {j['dif_LT01']:+.3f} m; ΔPT01 = {j['dif_PT01']:+.3f} mca", "",
              _md_table(j["tabela"]), ""]
    L += ["### Balanço de massa", "",
          f"ΔV = {mb['a']:.1f} + {mb['b']:.3f}·(FT01 − FT02 − FT03), com ΔV = fator·ΔLT01; r = {mb['r']:.3f}; "
          f"desvio do resíduo = {mb['res_sd']:.1f} m³; mediana |resíduo| = {mb['res_mediana_abs']:.1f} m³; "
          f"horas com |resíduo| > 5 dp: {mb['n_res_5sd']}.", "",
          "| Salto | ΔV observado (m³) | ΔV pelo balanço (m³) | Resíduo (m³) | Resíduo (dp) |",
          "|---|---:|---:|---:|---:|"]
    for s in mb["saltos"]:
        L.append(f"| {s['ts']} | {s['dV_obs_m3']:.0f} | {s['dV_balanco_m3']:.0f} | {s['residuo_m3']:.0f} | "
                 f"{s['residuo_em_sd']:.1f} |")
    L += ["", "## 3. Concordância geral LT01 × PT01", "",
          f"PT01 − LT01: mediana {offs['mediana']:.3f}, MAD robusto {offs['mad']:.3f}. "
          f"Horas com desvio > {offs['limiar']:.3f} (6 MAD): **{offs['n']}**.", ""]
    if offs["n"]:
        L += ["Primeiras ocorrências:", "", _md_table(offs["amostra"]), ""]
    return "\n".join(L)


# ======================================================================= tratamento
def realign_dst(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Desloca −1 h os rótulos da janela suspeita (até a 1ª ocorrência do duplicado)."""
    out = df.reset_index(drop=True).copy()
    ts = out[ts_col]
    dup_positions = ts.index[ts == DST_WINDOW_END_DUP]
    if len(dup_positions) != 2:
        raise ValueError(f"Esperadas 2 ocorrências de {DST_WINDOW_END_DUP}, encontradas {len(dup_positions)}")
    start_pos = ts.index[ts == DST_WINDOW_START]
    if len(start_pos) != 1:
        raise ValueError(f"Início da janela {DST_WINDOW_START} não encontrado de forma única")
    sel = (out.index >= start_pos[0]) & (out.index <= dup_positions[0])
    out.loc[sel, ts_col] = out.loc[sel, ts_col] - pd.Timedelta(hours=1)
    out.attrs["n_shifted"] = int(sel.sum())
    return out


def resolve_duplicates(df: pd.DataFrame, ts_col: str, strategy: str) -> tuple[pd.DataFrame, int]:
    n_dup = int(df[ts_col].duplicated().sum())
    if strategy == "first":
        out = df.drop_duplicates(subset=ts_col, keep="first")
    elif strategy == "mean":
        out = df.groupby(ts_col, as_index=False, sort=False).mean(numeric_only=True)
    else:
        raise ValueError(f"duplicate_strategy inválida: {strategy}")
    return out.reset_index(drop=True), n_dup


def regularize_and_fill(df: pd.DataFrame, ts_col: str) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Reindexa na grade horária e interpola linearmente pontos ausentes isolados."""
    grid = pd.date_range(df[ts_col].min(), df[ts_col].max(), freq="h")
    out = df.set_index(ts_col).reindex(grid)
    missing = list(out.index[out.isna().all(axis=1)])
    out = out.interpolate(method="time", limit=3, limit_area="inside")
    out.index.name = ts_col
    return out.reset_index(), missing


def interpolate_segments(df: pd.DataFrame, ts_col: str, target: str,
                         segments: list[list[str]]) -> tuple[pd.DataFrame, int]:
    """Substitui o alvo nos trechos [início, fim] (inclusivos) por interpolação linear no tempo."""
    out = df.copy()
    mask = pd.Series(False, index=out.index)
    for start, end in segments:
        seg = out[ts_col].between(pd.Timestamp(start), pd.Timestamp(end))
        if not seg.any():
            raise ValueError(f"Trecho {start} a {end} não encontrado")
        mask |= seg
    out.loc[mask, target] = np.nan
    out[target] = out.set_index(ts_col)[target].interpolate(method="time", limit_area="inside").to_numpy()
    return out, int(mask.sum())


def validate(df: pd.DataFrame, ts_col: str) -> dict[str, Any]:
    q = timestamp_quality(df[ts_col])
    checks = {"ordenado": q["is_monotonic_increasing"] and q["n_non_positive_steps"] == 0,
              "sem_duplicados": not q["duplicated_timestamps"], "sem_lacunas": not q["missing_timestamps"],
              "frequencia_1h": q["dominant_step"] == pd.Timedelta(hours=1) and len(q["step_counts"]) == 1,
              "sem_nan": not df.isna().any().any(),
              "sem_inf": bool(np.isfinite(df.drop(columns=ts_col).to_numpy(dtype=float)).all())}
    if not all(checks.values()):
        raise AssertionError(f"Validação do tratamento falhou: {checks}")
    return checks


# ======================================================================= main
def run_diagnose(log, df, ts_col, target, z, factor):
    d = add_diagnostic_columns(df)
    cols = ["LT01", "bombas_ligadas", "FT03"]
    jumps = jump_evidence(d, ts_col, target, z)
    mb = mass_balance(df, ts_col, factor, [j["ts"] for j in jumps])
    rep = diagnose_report(dst_evidence(d, ts_col, cols), cols, realign_structure_check(df, ts_col),
                          jumps, offset_anomalies(d, ts_col), z, sampling_continuity(df, ts_col), mb)
    (REPORTS_DIR / "04_diagnostico.md").write_text(rep, encoding="utf-8")
    log.info("Diagnóstico salvo em %s", REPORTS_DIR / "04_diagnostico.md")


def run_apply(log, df, ts_col, target, tcfg, z, force):
    pending = [k for k in ("dst_strategy", "jump_strategy") if tcfg.get(k) is None]
    if tcfg.get("dst_strategy") == "keep_labels" and tcfg.get("duplicate_strategy") is None:
        pending.append("duplicate_strategy")
    if pending:
        raise SystemExit(f"Decisões pendentes em config.yaml → treatment: {pending}. Rode --diagnose e decida.")

    in_hash, cfg_hash = sha256(INPUT), config_hash(*CONFIG_KEYS)
    if not force and is_reusable(OUTPUT, input_sha256=in_hash, config_hash=cfg_hash):
        log.info("Tratamento já aplicado com mesma entrada e configuração; reutilizando.")
        return

    log_steps: list[str] = []
    n0 = len(df)
    if tcfg["dst_strategy"] == "realign":
        df = realign_dst(df, ts_col)
        log_steps.append(f"Realinhamento −1 h de {df.attrs['n_shifted']} registros "
                         f"({DST_WINDOW_START} até a 1ª ocorrência de {DST_WINDOW_END_DUP}).")
    elif tcfg["dst_strategy"] == "keep_labels":
        df, n_dup = resolve_duplicates(df, ts_col, tcfg["duplicate_strategy"])
        log_steps.append(f"Duplicidades resolvidas ({tcfg['duplicate_strategy']}): {n_dup} linha(s).")
    else:
        raise ValueError(f"dst_strategy inválida: {tcfg['dst_strategy']}")

    df = df.sort_values(ts_col, kind="stable").reset_index(drop=True)
    df, filled = regularize_and_fill(df, ts_col)
    log_steps.append(f"Grade horária regular; pontos interpolados: {len(filled)} {[str(t) for t in filled]}.")

    if tcfg["jump_strategy"] == "interpolate_segments":
        segs = tcfg.get("jump_segments") or []
        df, n_int = interpolate_segments(df, ts_col, target, segs)
        log_steps.append(f"{target} interpolado linearmente em {n_int} registro(s), trechos {segs}.")
    elif tcfg["jump_strategy"] == "keep":
        log_steps.append("Saltos abruptos mantidos.")
    else:
        raise ValueError(f"jump_strategy inválida: {tcfg['jump_strategy']}")
    log_steps.append("Episódios de nível baixo mantidos (decisão do usuário)." if tcfg["keep_low_level_episodes"]
                     else "ATENÇÃO: keep_low_level_episodes = false não implementado.")

    checks = validate(df, ts_col)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    out_hash = sha256(OUTPUT)
    write_meta(OUTPUT, {"etapa": "04", "input": INPUT.name, "input_sha256": in_hash, "config_hash": cfg_hash,
                        "output_sha256": out_hash, "n_before": n0, "n_after": len(df), "passos": log_steps})
    rep = ["# ETAPA 04 — Tratamento", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
           f"Configuração: `{json.dumps(tcfg, ensure_ascii=False)}`", "", "## Transformações (em ordem)", "",
           *[f"{i}. {s}" for i, s in enumerate(log_steps, 1)], "", "## Validações", "",
           *[f"- {k}: {'OK' if v else 'FALHOU'}" for k, v in checks.items()], "",
           f"Registros: {n0} → {len(df)} ({df[ts_col].min()} a {df[ts_col].max()}).", "",
           f"Saída: `data/processed/{OUTPUT.name}` — SHA-256 `{out_hash}`; hash da configuração `{cfg_hash}`."]
    (REPORTS_DIR / "04_tratamento.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    for s in log_steps:
        log.info(s)
    log.info("Validações: %s", checks)


def main(diagnose: bool, force: bool) -> None:
    log = get_logger("treatment", run_name="04_tratamento" + ("_diagnostico" if diagnose else ""))
    cfg = load_config()
    ts_col, target = cfg["data"]["timestamp_column"], cfg["data"]["target"]
    z = cfg["eda"]["jump_z_threshold"]
    if not INPUT.is_file():
        raise FileNotFoundError(f"{INPUT} não encontrado — execute a ETAPA 02")
    df = pd.read_parquet(INPUT)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if diagnose:
        run_diagnose(log, df, ts_col, target, z, cfg["data"]["volume_equivalent"]["factor_m3_per_m"])
    else:
        run_apply(log, df, ts_col, target, cfg["treatment"], z, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.diagnose, a.force)
