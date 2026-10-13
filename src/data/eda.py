"""ETAPA 03 — Análise exploratória de LT01 em duas camadas.

- Descritiva: período completo 2018–2019 (série, distribuição, padrões, outliers).
- Inferencial: SOMENTE treino (antes do corte de split.py) — ADF/KPSS, ACF/PACF,
  decomposição MSTL e periodograma. São as análises que orientam escolhas de
  modelo; usar o teste nelas seria vazamento de informação.

Uso: python -m src.data.eda [--force]
Entrada: data/interim/02_filtrado_2018_2019.parquet (ETAPA 02)
Saídas:  results/reports/03_eda.md, results/reports/03_eda.json (+ .meta.json),
         results/figures/eda/*.png|pdf

Observação: o tratamento de duplicidade/lacuna só ocorre na ETAPA 04. Aqui a
série é usada na ordem das linhas (2 de 17.520 pontos afetados), o que tem
efeito desprezível nas estatísticas.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src.data.load import sha256
from src.data.split import configured_cut, split_by_cut
from src.utils.logging_utils import get_logger
from src.utils.paths import FIGURES_DIR, INTERIM_DIR, REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta
from src.visualization import plots
from src.visualization.style import apply_style, save_figure, t

INPUT = INTERIM_DIR / "02_filtrado_2018_2019.parquet"
OUTPUT = REPORTS_DIR / "03_eda.json"
FIG_SUBDIR = "eda"
CONFIG_KEYS = ("data.target", "data.start", "data.end", "split", "eda", "plots", "project.language")


# ======================================================================= descritiva
def describe(y: pd.Series) -> dict[str, float]:
    d = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    out = {k: float(v) for k, v in d.items()}
    out.update(skew=float(y.skew()), kurtosis=float(y.kurt()), cv=float(y.std() / y.mean()))
    return out


def profile(df: pd.DataFrame, key: pd.Series, target: str) -> pd.DataFrame:
    g = df[target].groupby(key)
    return pd.DataFrame({"media": g.mean(), "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                         "mediana": g.median(), "n": g.size()})


def runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Intervalos [ini, fim] de posições consecutivas onde mask é True."""
    m = mask.to_numpy()
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def outliers(df: pd.DataFrame, ts_col: str, target: str, k: float, z: float) -> dict[str, Any]:
    y = df[target].reset_index(drop=True)
    ts = df[ts_col].reset_index(drop=True)
    q1, q3 = y.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    episodes = []
    for a, b in runs(y < lo):
        seg = y.iloc[a:b + 1]
        episodes.append({"inicio": ts[a], "fim": ts[b], "horas": int(b - a + 1),
                         "minimo": float(seg.min()), "ts_minimo": ts[seg.idxmin()]})
    dy = y.diff()
    thr = z * dy.std()
    jumps = dy[dy.abs() > thr]
    top_jumps = [{"ts": ts[i], "de": float(y[i - 1]), "para": float(y[i]), "dif": float(dy[i])}
                 for i in jumps.abs().sort_values(ascending=False).index[:10]]
    return {"iqr_k": k, "limite_inferior": float(lo), "limite_superior": float(hi),
            "n_abaixo": int((y < lo).sum()), "n_acima": int((y > hi).sum()),
            "episodios_abaixo": sorted(episodes, key=lambda e: e["minimo"]),
            "dif_desvio": float(dy.std()), "salto_limiar": float(thr), "n_saltos": int(len(jumps)),
            "maiores_saltos": top_jumps}


# ======================================================================= inferencial (treino)
def stationarity(x: np.ndarray) -> dict[str, Any]:
    adf = adfuller(x, regression="c", autolag="AIC", result_object=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", InterpolationWarning)
        kp = kpss(x, regression="c", nlags="auto")
    bound = next((str(i.message) for i in w if issubclass(i.category, InterpolationWarning)), None)
    adf_stat, adf_p = float(adf[0]), float(adf[1])
    kp_stat, kp_p = float(kp[0]), float(kp[1])
    return {"adf_stat": adf_stat, "adf_p": adf_p, "adf_lags": int(adf[2]),
            "adf_crit_5": float(adf[4]["5%"]),
            "kpss_stat": kp_stat, "kpss_p": kp_p, "kpss_lags": int(kp[2]),
            "kpss_crit_5": float(kp[3]["5%"]), "kpss_p_limite_tabela": bound is not None,
            "adf_rejeita_raiz_unitaria": adf_p < 0.05, "kpss_rejeita_estacionariedade": kp_p < 0.05}


def stationarity_verdict(r: dict[str, Any]) -> str:
    a, k = r["adf_rejeita_raiz_unitaria"], r["kpss_rejeita_estacionariedade"]
    if a and not k:
        return "estacionária (ADF e KPSS concordam)"
    if not a and k:
        return "não estacionária (ADF e KPSS concordam)"
    if a and k:
        return "conflito: ADF rejeita raiz unitária, KPSS rejeita estacionariedade (indício de estacionariedade em torno de tendência/memória longa)"
    return "inconclusivo (nenhum teste rejeita)"


def acf_pacf(x: np.ndarray, nlags: int) -> dict[str, np.ndarray]:
    a, a_ci = acf(x, nlags=nlags, alpha=0.05, fft=True, result_object=False)
    p, p_ci = pacf(x, nlags=nlags, alpha=0.05, method="ywm")
    return {"acf": a, "acf_band": a_ci[:, 1] - a, "pacf": p, "pacf_band": p_ci[:, 1] - p}


def significant_pacf_lags(p: np.ndarray, n: int, max_lag: int = 48) -> list[int]:
    band = 1.96 / np.sqrt(n)
    return [int(k) for k in range(1, min(max_lag, len(p) - 1) + 1) if abs(p[k]) > band]


def strength(component: np.ndarray, resid: np.ndarray) -> float:
    """Força (Wang, Smith & Hyndman, 2006): max(0, 1 − Var(R)/Var(C+R))."""
    return float(max(0.0, 1 - np.var(resid) / np.var(component + resid)))


def top_periods(x: np.ndarray, k: int = 6) -> list[dict[str, float]]:
    f, pxx = periodogram(x - x.mean(), fs=1.0, detrend="linear")
    mask = f > 0
    f, pxx = f[mask], pxx[mask]
    order = np.argsort(pxx)[::-1]
    out, seen = [], []
    for i in order:
        period = 1 / f[i]
        if any(abs(period - s) / s < 0.05 for s in seen):
            continue
        seen.append(period)
        out.append({"periodo_h": float(period), "potencia_rel": float(pxx[i] / pxx.sum())})
        if len(out) == k:
            break
    return out


# ======================================================================= figuras
def make_figures(df, train, ts_col, target, cut, factor, comps_idx, comps, ac_level, ac_diff, periods):
    saved = []
    y = df[target]
    saved += save_figure(plots.plot_series(df[ts_col], y, cut, factor), "03_serie_lt01", FIG_SUBDIR)
    w = train[train[ts_col].between("2019-03-04", "2019-03-17 23:00")]
    saved += save_figure(plots.plot_window(w[ts_col], w[target], "2 semanas (treino)"), "03_janela_2_semanas", FIG_SUBDIR)
    saved += save_figure(plots.plot_distribution(y, factor), "03_distribuicao_lt01", FIG_SUBDIR)

    hour, wd, month = df[ts_col].dt.hour, df[ts_col].dt.dayofweek, df[ts_col].dt.month
    saved += save_figure(plots.plot_boxplot([y[hour == h].to_numpy() for h in range(24)],
                                            [str(h) for h in range(24)], t("hour")), "03_boxplot_hora", FIG_SUBDIR)
    saved += save_figure(plots.plot_boxplot([y[wd == d].to_numpy() for d in range(7)],
                                            t("weekdays"), t("weekday")), "03_boxplot_dia_semana", FIG_SUBDIR)
    saved += save_figure(plots.plot_boxplot([y[month == m].to_numpy() for m in range(1, 13)],
                                            t("months"), t("month")), "03_boxplot_mes", FIG_SUBDIR)

    heat = y.groupby([wd, hour]).mean().unstack()
    heat.index = t("weekdays")
    saved += save_figure(plots.plot_heatmap(heat, t("hour"), t("weekday"), "Média de LT01 (m)"),
                         "03_heatmap_hora_dia", FIG_SUBDIR)

    is_we = wd >= 5
    curves = []
    for label, m in (("Dias úteis", ~is_we), ("Fim de semana", is_we)):
        p = profile(df.loc[m], hour[m], target)
        curves.append((label, p["media"].to_numpy(), p["q25"].to_numpy(), p["q75"].to_numpy()))
    saved += save_figure(plots.plot_profiles(np.arange(24), curves, t("hour"), xticks=range(0, 24, 2)),
                         "03_perfil_horario", FIG_SUBDIR)

    saved += save_figure(plots.plot_acf_pacf([("ACF", ac_level["acf"], ac_level["acf_band"]),
                                              ("PACF", ac_level["pacf"], ac_level["pacf_band"])]),
                         "03_acf_pacf_nivel", FIG_SUBDIR)
    saved += save_figure(plots.plot_acf_pacf([("ACF (Δ1)", ac_diff["acf"], ac_diff["acf_band"]),
                                              ("PACF (Δ1)", ac_diff["pacf"], ac_diff["pacf_band"])]),
                         "03_acf_pacf_diferenca", FIG_SUBDIR)

    labels = ["Observado", "Tendência", *[f"Sazonal {p} h" for p in periods], "Resíduo"]
    components = list(zip(labels, comps))
    saved += save_figure(plots.plot_components(comps_idx, components), "03_decomposicao_mstl", FIG_SUBDIR)
    saved += save_figure(plots.plot_components(comps_idx, components, window=slice(-21 * 24, None)),
                         "03_decomposicao_mstl_3_semanas", FIG_SUBDIR)
    return saved


# ======================================================================= relatório
def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def build_report(r: dict[str, Any]) -> str:
    d, tr, te = r["descr_total"], r["descr_treino"], r["descr_teste"]
    L = ["# ETAPA 03 — Análise exploratória (LT01)", "",
         f"Gerado em: {r['gerado_em']}", "",
         "## Desenho da análise", "",
         f"- **Descritiva**: período completo ({r['periodo'][0]} a {r['periodo'][1]}, {r['n_total']} registros).",
         f"- **Inferencial** (orienta escolhas de modelo): **somente treino**, antes do corte "
         f"`{r['corte']}` (train_fraction = {r['train_fraction']}; {r['n_treino']} treino / {r['n_teste']} teste).",
         "- Duplicidade (2018-12-31 21:00) e lacuna (2018-11-04 00:00) ainda não tratadas (ETAPA 04): "
         "série usada na ordem das linhas; efeito desprezível (2 de 17.520 pontos).",
         f"- Figuras: `results/figures/{FIG_SUBDIR}/` (PNG {r['dpi']} dpi + PDF).", "",
         "## 1. Estatística descritiva", "",
         "| Estatística | Total | Treino | Teste |", "|---|---:|---:|---:|"]
    for k, lbl in [("count", "n"), ("mean", "média"), ("std", "desvio"), ("min", "mín"), ("1%", "p01"),
                   ("5%", "p05"), ("25%", "p25"), ("50%", "mediana"), ("75%", "p75"), ("95%", "p95"),
                   ("99%", "p99"), ("max", "máx"), ("skew", "assimetria"), ("kurtosis", "curtose (excesso)"),
                   ("cv", "coef. variação")]:
        nd = 0 if k == "count" else 4
        L.append(f"| {lbl} | {_f(d[k], nd)} | {_f(tr[k], nd)} | {_f(te[k], nd)} |")
    L += ["", f"Equivalente em volume (× {r['fator_m3']} m³/m): média total ≈ {d['mean'] * r['fator_m3']:.0f} m³, "
          f"faixa ≈ {d['min'] * r['fator_m3']:.0f} a {d['max'] * r['fator_m3']:.0f} m³.", ""]

    L += ["### Por ano", "", "| Ano | média | desvio | mín | máx |", "|---|---:|---:|---:|---:|"]
    for y, s in r["por_ano"].items():
        L.append(f"| {y} | {_f(s['mean'])} | {_f(s['std'])} | {_f(s['min'])} | {_f(s['max'])} |")

    amp = r["amplitudes"]
    L += ["", "## 2. Padrões sazonais (descritivo, período completo)", "",
          "Amplitude = máx − mín das médias por grupo. Indica quanto cada calendário explica.", "",
          "| Agrupamento | Amplitude das médias (m) | Grupo de menor média | Grupo de maior média |",
          "|---|---:|---|---|"]
    for k, v in amp.items():
        L.append(f"| {k} | {_f(v['amplitude'])} | {v['min_grupo']} ({_f(v['min'])}) | {v['max_grupo']} ({_f(v['max'])}) |")
    L += ["", "Perfil horário médio (m): " + ", ".join(f"{h}h={v:.2f}" for h, v in r["perfil_hora"].items()), "",
          f"Diferença média dias úteis − fim de semana, por hora: máx |Δ| = {_f(r['delta_util_fds_max'])} m "
          f"(na hora {r['delta_util_fds_hora']}).", ""]

    o = r["outliers"]
    L += ["## 3. Outliers e eventos (descritivo — nada removido)", "",
          f"- Regra IQR (k = {o['iqr_k']}): limites [{_f(o['limite_inferior'])}, {_f(o['limite_superior'])}] m.",
          f"- Abaixo do limite inferior: **{o['n_abaixo']}** registros, em {len(o['episodios_abaixo'])} episódio(s); "
          f"acima do superior: **{o['n_acima']}** (o nível é fisicamente limitado pela altura da câmara).",
          f"- Saltos horários |Δ| > {r['jump_z']}·σ(Δ) = {_f(o['salto_limiar'])} m: **{o['n_saltos']}**.", "",
          "Episódios abaixo do limite inferior (10 mais profundos):", "",
          "| Início | Fim | Horas | Mínimo (m) | Momento do mínimo |", "|---|---|---:|---:|---|"]
    for e in o["episodios_abaixo"][:10]:
        L.append(f"| {e['inicio']} | {e['fim']} | {e['horas']} | {_f(e['minimo'])} | {e['ts_minimo']} |")
    L += ["", "Maiores saltos horários:", "", "| Timestamp | De | Para | Δ |", "|---|---:|---:|---:|"]
    for j in o["maiores_saltos"]:
        L.append(f"| {j['ts']} | {_f(j['de'])} | {_f(j['para'])} | {_f(j['dif'])} |")

    L += ["", "## 4. Estacionariedade (somente treino)", "",
          "ADF: H0 = raiz unitária. KPSS: H0 = estacionária (em nível). α = 5%.", "",
          "| Série | ADF estat. | ADF p | KPSS estat. | KPSS p | Conclusão |", "|---|---:|---:|---:|---:|---|"]
    for name, s in r["estacionariedade"].items():
        if s["kpss_p_limite_tabela"]:
            kp = "> 0,10" if s["kpss_p"] >= 0.1 else "< 0,01"
        else:
            kp = f"{s['kpss_p']:.3f}"
        adf_p = "< 1e-16" if s["adf_p"] < 1e-16 else f"{s['adf_p']:.2e}"
        L.append(f"| {name} | {s['adf_stat']:.3f} | {adf_p} | {s['kpss_stat']:.3f} | {kp} | {s['conclusao']} |")
    L += ["", "KPSS \"> 0,10\": estatística abaixo do menor valor crítico tabelado (p-valor real maior que 0,10).", ""]

    a = r["autocorrelacao"]
    L += ["## 5. Autocorrelação (somente treino)", "",
          "| Defasagem (h) | ACF nível | ACF Δ1 |", "|---:|---:|---:|"]
    for k in a["lags_destaque"]:
        L.append(f"| {k} | {a['acf_nivel'][k]:.3f} | {a['acf_diff'][k]:.3f} |")
    L += ["", f"- PACF do nível: defasagens significativas (1–48): {a['pacf_sig_nivel']}",
          f"- PACF de Δ1: defasagens significativas (1–48): {a['pacf_sig_diff']}", ""]

    L += ["## 6. Decomposição MSTL e espectro (somente treino)", "",
          f"Períodos: {r['periodos']} h. Força = max(0, 1 − Var(R)/Var(C+R)) (Wang, Smith & Hyndman, 2006).", "",
          "| Componente | Força |", "|---|---:|"]
    for k, v in r["forcas"].items():
        L.append(f"| {k} | {v:.3f} |")
    L += ["", "Periodograma — maiores picos:", "", "| Período (h) | Potência relativa |", "|---:|---:|"]
    for p in r["periodograma"]:
        L.append(f"| {p['periodo_h']:.1f} | {p['potencia_rel']:.3f} |")
    L += [""]
    return "\n".join(L)


# ======================================================================= main
def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, pd.Timedelta)):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.generic,)):
        return o.item()
    raise TypeError(type(o))


def main(force: bool = False) -> None:
    log = get_logger("eda", run_name="03_eda")
    cfg = load_config()
    ts_col, target = cfg["data"]["timestamp_column"], cfg["data"]["target"]
    e = cfg["eda"]
    factor = cfg["data"]["volume_equivalent"]["factor_m3_per_m"] if cfg["plots"].get("show_volume_equivalent") else None

    if not INPUT.is_file():
        raise FileNotFoundError(f"{INPUT} não encontrado — execute a ETAPA 02")
    in_hash, cfg_hash = sha256(INPUT), config_hash(*CONFIG_KEYS)
    if not force and is_reusable(OUTPUT, input_sha256=in_hash, config_hash=cfg_hash) \
            and (FIGURES_DIR / FIG_SUBDIR).is_dir():
        log.info("EDA já calculada com a mesma entrada e configuração; nada a fazer (use --force).")
        return

    apply_style()
    df = pd.read_parquet(INPUT)
    cut = configured_cut()
    train, test = split_by_cut(df, ts_col, cut)
    y_tr = train[target].to_numpy(dtype="float64")
    log.info("Corte %s: treino %d, teste %d", cut, len(train), len(test))

    # --- descritiva
    hour, wd, month = df[ts_col].dt.hour, df[ts_col].dt.dayofweek, df[ts_col].dt.month
    amplitudes = {}
    for name, key, names in (("Hora do dia", hour, {h: f"{h}h" for h in range(24)}),
                             ("Dia da semana", wd, dict(enumerate(t("weekdays")))),
                             ("Mês", month, {i + 1: m for i, m in enumerate(t("months"))})):
        m = df[target].groupby(key).mean()
        amplitudes[name] = {"amplitude": float(m.max() - m.min()), "min_grupo": names[m.idxmin()],
                            "min": float(m.min()), "max_grupo": names[m.idxmax()], "max": float(m.max())}
    is_we = wd >= 5
    delta = df.loc[~is_we, target].groupby(hour[~is_we]).mean() - df.loc[is_we, target].groupby(hour[is_we]).mean()

    # --- inferencial (treino)
    d1 = np.diff(y_tr)
    d24 = y_tr[24:] - y_tr[:-24]
    d1_24 = np.diff(d24)
    stat = {}
    for name, x in (("Nível", y_tr), ("Δ1 (1ª diferença)", d1), ("Δ24 (sazonal)", d24), ("Δ1Δ24", d1_24)):
        s = stationarity(x)
        s["conclusao"] = stationarity_verdict(s)
        stat[name] = s
        log.info("Estacionariedade %s: ADF p=%.3g, KPSS p=%.3g -> %s", name, s["adf_p"], s["kpss_p"], s["conclusao"])

    ac_level = acf_pacf(y_tr, e["acf_max_lag"])
    ac_diff = acf_pacf(d1, e["acf_max_lag"])

    periods = tuple(e["stl_periods"])
    res = MSTL(y_tr, periods=periods).fit()
    seas = np.asarray(res.seasonal).reshape(len(y_tr), -1)
    trend, resid = np.asarray(res.trend), np.asarray(res.resid)
    forcas = {"Tendência": strength(trend, resid)}
    for i, p in enumerate(periods):
        forcas[f"Sazonal {p} h"] = strength(seas[:, i], resid)
    log.info("Forças MSTL: %s", forcas)

    lags_dest = [k for k in (1, 2, 3, 6, 12, 23, 24, 25, 48, 72, 144, 167, 168) if k <= e["acf_max_lag"]]
    result = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "periodo": [df[ts_col].min(), df[ts_col].max()], "n_total": len(df),
        "corte": cut, "train_fraction": cfg["split"]["train_fraction"],
        "n_treino": len(train), "n_teste": len(test), "dpi": cfg["plots"]["dpi"],
        "fator_m3": factor or cfg["data"]["volume_equivalent"]["factor_m3_per_m"],
        "descr_total": describe(df[target]), "descr_treino": describe(train[target]),
        "descr_teste": describe(test[target]),
        "por_ano": {int(y): describe(g) for y, g in df[target].groupby(df[ts_col].dt.year)},
        "amplitudes": amplitudes,
        "perfil_hora": {int(h): float(v) for h, v in df[target].groupby(hour).mean().items()},
        "delta_util_fds_max": float(delta.abs().max()), "delta_util_fds_hora": int(delta.abs().idxmax()),
        "outliers": outliers(df, ts_col, target, e["outlier_iqr_k"], e["jump_z_threshold"]),
        "jump_z": e["jump_z_threshold"],
        "estacionariedade": stat,
        "autocorrelacao": {"lags_destaque": lags_dest,
                           "acf_nivel": {k: float(ac_level["acf"][k]) for k in lags_dest},
                           "acf_diff": {k: float(ac_diff["acf"][k]) for k in lags_dest},
                           "pacf_sig_nivel": significant_pacf_lags(ac_level["pacf"], len(y_tr)),
                           "pacf_sig_diff": significant_pacf_lags(ac_diff["pacf"], len(d1))},
        "periodos": list(periods), "forcas": forcas, "periodograma": top_periods(y_tr),
    }

    comps = [y_tr, trend, *[seas[:, i] for i in range(len(periods))], resid]
    figs = make_figures(df, train, ts_col, target, cut, factor, pd.DatetimeIndex(train[ts_col]),
                        comps, ac_level, ac_diff, periods)
    result["figuras"] = [str(p.relative_to(FIGURES_DIR.parent.parent)) for p in figs]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    (REPORTS_DIR / "03_eda.md").write_text(build_report(result), encoding="utf-8")
    write_meta(OUTPUT, {"etapa": "03", "input": INPUT.name, "input_sha256": in_hash, "config_hash": cfg_hash,
                        "corte": str(cut)})
    log.info("EDA concluída: %d figuras; relatório %s", len(figs), REPORTS_DIR / "03_eda.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recalcula mesmo com artefato compatível")
    main(force=ap.parse_args().force)
