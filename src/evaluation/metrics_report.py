"""ETAPA 12 — Métricas consolidadas e teste de Diebold-Mariano.

Fonte única: results/experiments/registry.csv (último experimento ok por modelo × horizonte)
e os arquivos de previsão correspondentes.

Uso: python -m src.evaluation.metrics_report
Saídas: results/metrics/12_metricas.csv|parquet, results/metrics/12_diebold_mariano.csv|parquet,
        results/reports/12_metricas.md
"""
from __future__ import annotations

from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.metrics import diebold_mariano, holm
from src.utils.logging_utils import get_logger
from src.utils.paths import METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR

BASELINES = ("persistencia", "sazonal_diaria", "sazonal_semanal")
TRAINABLE = ("sarima", "svr", "lightgbm", "lightgbm_rec")
LABELS = {"persistencia": "Persistência", "sazonal_diaria": "Sazonal diária", "sazonal_semanal": "Sazonal semanal",
          "sarima": "SARIMA", "svr": "SVR", "lightgbm": "LightGBM (direto)", "lightgbm_rec": "LightGBM (recursivo)"}
ALPHA = 0.05


def metrics_table(reg: pd.DataFrame) -> pd.DataFrame:
    t = reg[["modelo", "horizonte_h", "exp_id", "mae_treino", "rmse_treino", "mae_teste", "rmse_teste"]].copy()
    t["razao_rmse_teste_treino"] = t["rmse_teste"] / t["rmse_treino"]
    t["razao_mae_teste_treino"] = t["mae_teste"] / t["mae_treino"]
    t["rank_rmse_teste"] = t.groupby("horizonte_h")["rmse_teste"].rank(method="min").astype(int)
    t["rank_mae_teste"] = t.groupby("horizonte_h")["mae_teste"].rank(method="min").astype(int)
    order = {m: i for i, m in enumerate([*BASELINES, *TRAINABLE])}
    return t.sort_values(["horizonte_h", "modelo"], key=lambda s: s.map(order) if s.name == "modelo" else s) \
        .reset_index(drop=True)


def test_errors(exp_id: str) -> pd.Series:
    p = pd.read_parquet(PREDICTIONS_DIR / f"{exp_id}.parquet")
    p = p[p["conjunto"] == "teste"]
    return pd.Series((p["y"] - p["y_prev"]).to_numpy(), index=pd.DatetimeIndex(p["ts_alvo"]))


def dm_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in t.groupby("horizonte_h"):
        base = g[g["modelo"].isin(BASELINES)].sort_values("rmse_teste").iloc[0]
        models = [base["modelo"], *[m for m in TRAINABLE if m in set(g["modelo"])]]
        errs = {m: test_errors(g.loc[g["modelo"] == m, "exp_id"].iloc[0]) for m in models}
        ref_idx = errs[models[0]].index
        if any(not e.index.equals(ref_idx) for e in errs.values()):
            raise AssertionError(f"h={h}: alvos de teste diferentes entre modelos")
        block = []
        for m1, m2 in combinations(models, 2):
            sq = diebold_mariano(errs[m1].to_numpy(), errs[m2].to_numpy(), int(h), "squared")
            ab = diebold_mariano(errs[m1].to_numpy(), errs[m2].to_numpy(), int(h), "absolute")
            block.append({"horizonte_h": int(h), "modelo_1": m1, "modelo_2": m2,
                          "rmse_1": float(np.sqrt(np.mean(errs[m1] ** 2))), "rmse_2": float(np.sqrt(np.mean(errs[m2] ** 2))),
                          "dm_quad": sq["dm"], "p_quad": sq["p"], "dm_abs": ab["dm"], "p_abs": ab["p"],
                          "lag_nw": sq["lag_nw"]})
        for key in ("quad", "abs"):
            adj = holm([b[f"p_{key}"] for b in block])
            for b, a in zip(block, adj):
                b[f"p_{key}_holm"] = a
        for b in block:
            if b["p_quad_holm"] < ALPHA:
                b["conclusao"] = f"{b['modelo_1'] if b['dm_quad'] < 0 else b['modelo_2']} melhor"
            else:
                b["conclusao"] = "sem diferença significativa"
        rows += block
    return pd.DataFrame(rows)


def best_group(t: pd.DataFrame, dm: pd.DataFrame) -> dict[int, list[str]]:
    """Modelos não significativamente piores (Holm, perda quadrática) que o de menor RMSE no horizonte."""
    out = {}
    for h, g in t.groupby("horizonte_h"):
        best = g.sort_values("rmse_teste").iloc[0]["modelo"]
        grp = [best]
        d = dm[dm["horizonte_h"] == h]
        for r in d.itertuples(index=False):
            if best in (r.modelo_1, r.modelo_2) and r.p_quad_holm >= ALPHA:
                grp.append(r.modelo_2 if r.modelo_1 == best else r.modelo_1)
        out[int(h)] = grp
    return out


def build_report(t: pd.DataFrame, dm: pd.DataFrame, groups: dict[int, list[str]]) -> str:
    L = ["# ETAPA 12 — Métricas", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "MAE = (1/n) Σ |y − ŷ|; RMSE = √((1/n) Σ (y − ŷ)²); em metros. Teste: 3.504 alvos; treino: 13.824 alvos "
         "(mesmos para todos). O desempenho de **teste** tem prioridade; a razão teste/treino indica sobreajuste "
         "(> 1: erro maior fora da amostra).", "",
         "## Tabela consolidada", "",
         "| Modelo | h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino (RMSE) | Rank RMSE | Rank MAE |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples(index=False):
        L.append(f"| {LABELS[r.modelo]} | {r.horizonte_h} | {r.mae_treino:.4f} | {r.rmse_treino:.4f} | {r.mae_teste:.4f} | "
                 f"{r.rmse_teste:.4f} | {r.razao_rmse_teste_treino:.3f} | {r.rank_rmse_teste} | {r.rank_mae_teste} |")
    L += ["", "## Melhor por horizonte", "",
          "| h | Menor RMSE teste | Menor MAE teste | Grupo dos melhores (DM, Holm 5%) |", "|---:|---|---|---|"]
    def winners(g: pd.DataFrame, col: str) -> str:
        best = g[col].min()
        names = [LABELS[m] for m in g.loc[np.isclose(g[col], best, rtol=0, atol=1e-12), "modelo"]]
        return f"{' = '.join(names)} ({best:.4f})" + (" — empate exato" if len(names) > 1 else "")

    for h, g in t.groupby("horizonte_h"):
        L.append(f"| {h} | {winners(g, 'rmse_teste')} | {winners(g, 'mae_teste')} | "
                 f"{', '.join(LABELS[m] for m in groups[int(h)])} |")
    L += ["", "Em h = 1 o LightGBM recursivo é o próprio modelo direto de 1 passo (previsões idênticas)."]
    L += ["", "\"Grupo dos melhores\": o modelo de menor RMSE e os que **não** diferem significativamente dele.", "",
          "## Teste de Diebold-Mariano", "",
          "Pares entre SARIMA, SVR, LightGBM direto, LightGBM recursivo e o melhor baseline de cada horizonte. "
          "Perda quadrática (principal) e absoluta (verificação). Variância HAC de Newey-West (Bartlett) com "
          "defasagem max(h − 1, 8); correção de Harvey-Leybourne-Newbold; t(n − 1) bicaudal; Holm por horizonte "
          f"(10 pares); α = {ALPHA}. DM < 0 ⇒ modelo 1 com perda menor.", "",
          "| h | Modelo 1 | Modelo 2 | RMSE 1 | RMSE 2 | DM (quad.) | p Holm (quad.) | DM (abs.) | p Holm (abs.) | Conclusão (quad.) |",
          "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in dm.itertuples(index=False):
        concl = r.conclusao
        for m in LABELS:
            concl = concl.replace(f"{m} melhor", f"{LABELS[m]} melhor") if concl.startswith(m) else concl
        L.append(f"| {r.horizonte_h} | {LABELS[r.modelo_1]} | {LABELS[r.modelo_2]} | {r.rmse_1:.4f} | {r.rmse_2:.4f} | "
                 f"{r.dm_quad:.2f} | {r.p_quad_holm:.3g} | {r.dm_abs:.2f} | {r.p_abs_holm:.3g} | {concl} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    log = get_logger("metrics_report", run_name="12_metricas")
    reg = registry.latest_ok()
    t = metrics_table(reg)
    dm = dm_table(t)
    groups = best_group(t, dm)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in ((t, "12_metricas"), (dm, "12_diebold_mariano")):
        df.to_csv(METRICS_DIR / f"{name}.csv", index=False, encoding="utf-8")
        df.to_parquet(METRICS_DIR / f"{name}.parquet", index=False)
    (REPORTS_DIR / "12_metricas.md").write_text(build_report(t, dm, groups), encoding="utf-8")
    for h, g in groups.items():
        log.info("h=%d grupo dos melhores: %s", h, g)
    log.info("Relatório: %s", REPORTS_DIR / "12_metricas.md")


if __name__ == "__main__":
    main()
