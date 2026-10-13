"""ETAPA 15 — Trade-off erro × tempo e fronteira de Pareto.

Por horizonte, cada modelo é um ponto (t_total, RMSE de teste). Um modelo a DOMINA b se não é
pior em nenhum critério e é estritamente melhor em pelo menos um. Tolerâncias:
- tempo: só conta como "estritamente menor" se os intervalos [mín, máx] das 3 repetições não se
  sobrepõem (diferenças dentro do ruído de medição são empate);
- erro: empate apenas se os valores forem idênticos (ex.: LightGBM direto e recursivo em h = 1).
A fronteira complementa o score composto; não substitui MAE/RMSE.

Uso: python -m src.evaluation.pareto
Saídas: results/metrics/15_pareto.csv|parquet, results/reports/15_pareto.md
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.cost_report import SHARED_FIT, tuning_costs
from src.evaluation.metrics_report import BASELINES, LABELS, TRAINABLE
from src.utils.logging_utils import get_logger
from src.utils.paths import METRICS_DIR, REPORTS_DIR

EPS_ERR = 1e-12


def _t_less(a: pd.Series, b: pd.Series, tcol: str) -> bool:
    """a estritamente mais rápido que b, além do ruído (intervalos não se sobrepõem)."""
    return a[f"{tcol}_max"] < b[f"{tcol}_min"]


def _t_leq(a: pd.Series, b: pd.Series, tcol: str) -> bool:
    """a não é mais lento que b (mais rápido ou empatado dentro do ruído)."""
    return not _t_less(b, a, tcol)


def dominates(a: pd.Series, b: pd.Series, err: str, tcol: str) -> bool:
    e_leq = a[err] <= b[err] + EPS_ERR
    e_less = a[err] < b[err] - EPS_ERR
    return e_leq and _t_leq(a, b, tcol) and (e_less or _t_less(a, b, tcol))


def _same_forecaster(a: pd.Series, b: pd.Series) -> bool:
    """Previsões idênticas (RMSE e MAE iguais) = mesmo previsor (ex.: LightGBM direto e recursivo em h = 1);
    diferenças de tempo entre eles são só de medição e não geram dominância."""
    cols = [c for c in ("rmse_teste", "mae_teste") if c in a.index]
    return len(cols) == 2 and all(abs(a[c] - b[c]) <= EPS_ERR for c in cols)


def pareto(g: pd.DataFrame, err: str, tcol: str) -> pd.DataFrame:
    g = g.reset_index(drop=True).copy()
    dom_by = []
    for i, b in g.iterrows():
        dom_by.append([g.loc[j, "modelo"] for j, a in g.iterrows()
                       if j != i and not _same_forecaster(a, b) and dominates(a, b, err, tcol)])
    g["dominado_por"] = [", ".join(LABELS[m] for m in d) for d in dom_by]
    g["pareto"] = [len(d) == 0 for d in dom_by]
    return g


def marginal_tradeoff(front: pd.DataFrame, err: str, tcol: str) -> list[dict]:
    """Ao longo da fronteira (do mais rápido ao mais preciso): quanto erro se ganha por década de tempo."""
    f = front.sort_values(tcol).reset_index(drop=True)
    out = []
    for i in range(1, len(f)):
        a, b = f.iloc[i - 1], f.iloc[i]
        dlog = np.log10(b[tcol]) - np.log10(a[tcol])
        derr = a[err] - b[err]
        out.append({"de": a["modelo"], "para": b["modelo"], "fator_tempo": b[tcol] / a[tcol],
                    "reducao_erro": derr, "reducao_erro_pct": 100 * derr / a[err],
                    "reducao_por_decada": derr / dlog if dlog > 0 else np.nan})
    return out


def build_points(reg: pd.DataFrame, tun: dict) -> pd.DataFrame:
    pts = []
    for h, g in reg.groupby("horizonte_h"):
        bb = g[g["modelo"].isin(BASELINES)].sort_values("rmse_teste").iloc[0]["modelo"]
        sel = g[g["modelo"].isin([bb, *TRAINABLE])].copy()
        sel["papel"] = np.where(sel["modelo"] == bb, "baseline", "modelo")
        sel["t_tuning_s"] = sel["modelo"].map(lambda m: tun.get(m, 0.0)).fillna(0.0)
        for k in ("", "_min", "_max"):
            sel[f"t_com_tuning{k}"] = sel[f"t_total{k}_s" if k else "t_total_s"] + sel["t_tuning_s"]
            sel[f"t_sem_tuning{k}"] = sel[f"t_total{k}_s" if k else "t_total_s"]
        pts.append(sel)
    return pd.concat(pts, ignore_index=True)


def aggregate_4h(reg: pd.DataFrame) -> pd.DataFrame:
    """Visão por modelo: custo para servir os 4 horizontes × RMSE médio de teste nos 4 horizontes."""
    rows = []
    for m, g in reg[reg["modelo"].isin([*BASELINES, *TRAINABLE])].groupby("modelo"):
        if m in SHARED_FIT:
            tr = g["t_treino_s"].iloc[0], g["t_treino_min_s"].iloc[0], g["t_treino_max_s"].iloc[0]
        else:
            tr = g["t_treino_s"].sum(), g["t_treino_min_s"].sum(), g["t_treino_max_s"].sum()
        rows.append({"modelo": m, "rmse_medio": g["rmse_teste"].mean(), "mae_medio": g["mae_teste"].mean(),
                     "t_sem_tuning": tr[0] + g["t_previsao_s"].sum(),
                     "t_sem_tuning_min": tr[1] + g["t_previsao_min_s"].sum(),
                     "t_sem_tuning_max": tr[2] + g["t_previsao_max_s"].sum()})
    return pd.DataFrame(rows)


def build_report(res: pd.DataFrame, marg: pd.DataFrame, agg: pd.DataFrame) -> str:
    L = ["# ETAPA 15 — Trade-off erro × tempo e fronteira de Pareto", "",
         f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "Ponto = (t_total, RMSE de teste) por modelo e horizonte. **a domina b** se não é pior em nenhum critério e é "
         "estritamente melhor em pelo menos um. Tempo: diferenças dentro do intervalo [mín, máx] das 3 repetições contam "
         "como empate. Pareto em estimativas pontuais: diferenças de erro não significativas pelo DM (ETAPA 12) devem "
         "ser lidas com cautela. Conjunto principal: 4 treináveis; com baseline: sensibilidade (decisão da ETAPA 14).", ""]
    main = res[(res.conjunto == "treinaveis") & (res.erro == "rmse_teste") & (res.variante == "sem_tuning")]
    L += ["## Fronteira principal (treináveis, RMSE, sem tuning)", "",
          "| h | Modelo | t_total (s) | RMSE teste | Pareto | Dominado por |", "|---:|---|---:|---:|---|---|"]
    for r in main.sort_values(["horizonte_h", "t"]).itertuples(index=False):
        L.append(f"| {r.horizonte_h} | {LABELS[r.modelo]} | {r.t:.4g} | {r.erro_valor:.4f} | "
                 f"{'**sim**' if r.pareto else 'não'} | {r.dominado_por or '—'} |")
    L += ["", "## Ganho marginal ao longo da fronteira (treináveis, RMSE, sem tuning)", "",
          "| h | De | Para | Tempo × | Redução de RMSE (m) | Redução (%) | Redução por década de tempo (m) |",
          "|---:|---|---|---:|---:|---:|---:|"]
    mm = marg[(marg.conjunto == "treinaveis") & (marg.erro == "rmse_teste") & (marg.variante == "sem_tuning")]
    if mm.empty:
        L.append("| — | fronteira com um único modelo em todos os horizontes | | | | | |")
    for r in mm.itertuples(index=False):
        L.append(f"| {r.horizonte_h} | {LABELS[r.de]} | {LABELS[r.para]} | {r.fator_tempo:.2f} | {r.reducao_erro:.4f} | "
                 f"{r.reducao_erro_pct:.1f} | {r.reducao_por_decada:.4f} |")

    L += ["", "## Robustez — modelos na fronteira por conjunto, erro e variante", "",
          "| h | Conjunto | Erro | Sem tuning | Com tuning |", "|---:|---|---|---|---|"]
    for (h, conj, err), g in res.groupby(["horizonte_h", "conjunto", "erro"]):
        cell = lambda v: ", ".join(LABELS[m] for m in g[(g.variante == v) & g.pareto].sort_values("t")["modelo"])  # noqa: E731
        L.append(f"| {h} | {conj} | {err.split('_')[0].upper()} | {cell('sem_tuning')} | {cell('com_tuning')} |")

    L += ["", "## Visão agregada: servir os 4 horizontes", "",
          "Custo total para servir os 4 horizontes (ajustes distintos + previsões dos 4 h) × RMSE médio de teste.", "",
          "| Modelo | t 4 h (s) | RMSE médio | MAE médio | Pareto (treináveis) | Pareto (com baselines) |", "|---|---:|---:|---:|---|---|"]
    for r in agg.sort_values("t_sem_tuning").itertuples(index=False):
        L.append(f"| {LABELS[r.modelo]} | {r.t_sem_tuning:.4g} | {r.rmse_medio:.4f} | {r.mae_medio:.4f} | "
                 f"{'—' if r.modelo in BASELINES else ('**sim**' if r.pareto_treinaveis else 'não')} | "
                 f"{'**sim**' if r.pareto_todos else 'não'} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    log = get_logger("pareto", run_name="15_pareto")
    reg = registry.latest_ok()
    pts = build_points(reg, tuning_costs())
    rows, margs = [], []
    for h, g_all in pts.groupby("horizonte_h"):
        for conj in ("treinaveis", "com_baseline"):
            g = g_all if conj == "com_baseline" else g_all[g_all.papel == "modelo"]
            for err in ("rmse_teste", "mae_teste"):
                for var in ("sem_tuning", "com_tuning"):
                    p = pareto(g, err, f"t_{var}")
                    p = p.assign(conjunto=conj, erro=err, variante=var, t=p[f"t_{var}"], erro_valor=p[err])
                    rows.append(p[["horizonte_h", "modelo", "papel", "conjunto", "erro", "variante", "t", "erro_valor",
                                   "pareto", "dominado_por"]])
                    for m in marginal_tradeoff(p[p.pareto], err, f"t_{var}"):
                        margs.append({"horizonte_h": int(h), "conjunto": conj, "erro": err, "variante": var, **m})
    res = pd.concat(rows, ignore_index=True)
    marg = pd.DataFrame(margs)

    agg = aggregate_4h(reg)
    agg_tr = pareto(agg[agg.modelo.isin(TRAINABLE)], "rmse_medio", "t_sem_tuning")
    agg_all = pareto(agg, "rmse_medio", "t_sem_tuning")
    agg = agg.merge(agg_tr[["modelo", "pareto"]].rename(columns={"pareto": "pareto_treinaveis"}), on="modelo", how="left") \
             .merge(agg_all[["modelo", "pareto"]].rename(columns={"pareto": "pareto_todos"}), on="modelo")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in ((res, "15_pareto"), (marg, "15_pareto_marginal"), (agg, "15_pareto_4h")):
        df.to_csv(METRICS_DIR / f"{name}.csv", index=False, encoding="utf-8")
        df.to_parquet(METRICS_DIR / f"{name}.parquet", index=False)
    (REPORTS_DIR / "15_pareto.md").write_text(build_report(res, marg, agg), encoding="utf-8")
    m = res[(res.conjunto == "treinaveis") & (res.erro == "rmse_teste") & (res.variante == "sem_tuning") & res.pareto]
    for h, g in m.groupby("horizonte_h"):
        log.info("h=%d fronteira (treináveis): %s", h, list(g.sort_values("t")["modelo"]))


if __name__ == "__main__":
    main()
