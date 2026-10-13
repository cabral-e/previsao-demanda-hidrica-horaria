"""ETAPA 16 — Consolidação: tabela final (seção 13) e respostas às 12 perguntas.

Fontes: registry.csv (v mais recente), 12_metricas / 12_diebold_mariano, 14_score, 15_pareto.

Uso: python -m src.evaluation.consolidate
Saídas: results/metrics/resultados_finais.csv|parquet, results/reports/16_consolidacao.md
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.metrics_report import BASELINES, LABELS, TRAINABLE, best_group, dm_table, metrics_table
from src.utils.logging_utils import get_logger
from src.utils.paths import METRICS_DIR, REPORTS_DIR, load_config

STRATEGY = {"persistencia": "indexação", "sazonal_diaria": "indexação", "sazonal_semanal": "indexação",
            "sarima": "recursiva (Kalman)", "svr": "direta", "lightgbm": "direta", "lightgbm_rec": "recursiva"}


def final_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    factor = cfg["data"]["volume_equivalent"]["factor_m3_per_m"]
    sc = cfg["composite_score"]
    reg = registry.latest_ok()
    reg = reg[reg["modelo"].isin([*BASELINES, *TRAINABLE])]
    t = metrics_table(reg)
    dm = dm_table(t)
    groups = best_group(t, dm)

    score = pd.read_parquet(METRICS_DIR / "14_score.parquet")
    base = score[(score.normalizacao == sc["main_normalization"]) & (score.erro == "rmse") &
                 (score.variante == "sem_tuning") & np.isclose(score.alpha, sc["alpha"])]
    s_main = base[base.conjunto == sc["main_set"]][["horizonte_h", "modelo", "score"]].rename(columns={"score": "score"})
    s_sens = base[base.conjunto == "com_baseline"][["horizonte_h", "modelo", "score"]].rename(columns={"score": "score_com_baseline"})
    par = pd.read_parquet(METRICS_DIR / "15_pareto.parquet")
    par = par[(par.conjunto == sc["main_set"]) & (par.erro == "rmse_teste") & (par.variante == "sem_tuning")][
        ["horizonte_h", "modelo", "pareto"]]

    out = reg.merge(t[["modelo", "horizonte_h", "razao_rmse_teste_treino", "rank_rmse_teste"]], on=["modelo", "horizonte_h"]) \
             .merge(s_main, on=["horizonte_h", "modelo"], how="left") \
             .merge(s_sens, on=["horizonte_h", "modelo"], how="left") \
             .merge(par, on=["horizonte_h", "modelo"], how="left")
    out["grupo_melhores_dm"] = [m in groups[int(h)] for m, h in zip(out["modelo"], out["horizonte_h"])]
    out["estrategia"] = out["modelo"].map(STRATEGY)
    out["mae_teste_m3"] = out["mae_teste"] * factor
    out["rmse_teste_m3"] = out["rmse_teste"] * factor
    out["Modelo"] = out["modelo"].map(LABELS)
    order = {m: i for i, m in enumerate([*BASELINES, *TRAINABLE])}
    out = out.sort_values(["horizonte_h", "modelo"], key=lambda s: s.map(order) if s.name == "modelo" else s)
    cols = {"Modelo": "Modelo", "horizonte_h": "Horizonte (h)", "mae_treino": "MAE Treino (m)", "rmse_treino": "RMSE Treino (m)",
            "mae_teste": "MAE Teste (m)", "rmse_teste": "RMSE Teste (m)", "t_treino_s": "Tempo Treino (s)",
            "t_previsao_s": "Tempo Previsão (s)", "t_total_s": "Tempo Total (s)", "score": "Score",
            "score_com_baseline": "Score (com baseline)", "memoria_pico_mb": "Memória (MB)",
            "razao_rmse_teste_treino": "RMSE Teste/Treino", "mae_teste_m3": "MAE Teste (m³ eq.)",
            "rmse_teste_m3": "RMSE Teste (m³ eq.)", "estrategia": "Estratégia", "pareto": "Pareto",
            "grupo_melhores_dm": "Grupo dos melhores (DM)", "exp_id": "exp_id", "versao": "Versão"}
    final = out[list(cols)].rename(columns=cols).reset_index(drop=True)
    return final, dm


def answers(f: pd.DataFrame) -> list[str]:
    tr = f[f["Modelo"].isin([LABELS[m] for m in TRAINABLE])]

    def by(col: str, df: pd.DataFrame = f) -> list[tuple]:
        """(h, vencedor(es) com empates exatos, valor)."""
        out = []
        for h, g in df.groupby("Horizonte (h)"):
            v = g[col].min()
            out.append((h, " = ".join(g.loc[np.isclose(g[col], v, rtol=0, atol=1e-12), "Modelo"]), v))
        return out

    gains = []
    for h, g in f.groupby("Horizonte (h)"):
        bb = g[~g["Modelo"].isin(tr["Modelo"])]["RMSE Teste (m)"].min()
        gains += list(100 * (1 - g[g["Modelo"].isin(tr["Modelo"])]["RMSE Teste (m)"] / bb))
    sc_rows = []
    for h, g in tr.groupby("Horizonte (h)"):
        best = g["Score"].min()
        same_err = g.loc[np.isclose(g["RMSE Teste (m)"], g.loc[g["Score"].idxmin(), "RMSE Teste (m)"], atol=1e-12)
                         & np.isclose(g["MAE Teste (m)"], g.loc[g["Score"].idxmin(), "MAE Teste (m)"], atol=1e-12), "Modelo"]
        sc_rows.append((h, " = ".join(same_err) if len(same_err) > 1 else g.loc[g["Score"].idxmin(), "Modelo"]))
    mean = f.groupby("Modelo")[["MAE Teste (m)", "RMSE Teste (m)"]].mean()
    tot4 = pd.read_parquet(METRICS_DIR / "15_pareto_4h.parquet").assign(Modelo=lambda d: d["modelo"].map(LABELS))
    ratio = tr.groupby("Modelo")["RMSE Teste/Treino"].agg(["min", "max"])
    best_h = {}
    for h, g in f.groupby("Horizonte (h)"):
        ties = g.loc[np.isclose(g["RMSE Teste (m)"], g["RMSE Teste (m)"].min(), atol=1e-12), "Modelo"].tolist()
        grp = g.loc[g["Grupo dos melhores (DM)"], "Modelo"].tolist()
        best_h[int(h)] = (" = ".join(ties), g["RMSE Teste (m)"].min(), ", ".join(grp))

    L = ["## Respostas às perguntas da seção 13", "",
         f"1. **Menor MAE**: média nos 4 horizontes — {mean['MAE Teste (m)'].idxmin()} ({mean['MAE Teste (m)'].min():.4f} m). "
         "Por horizonte: " + "; ".join(f"{h} h: {m} ({v:.4f})" for h, m, v in by("MAE Teste (m)")) + ".",
         f"2. **Menor RMSE**: média — {mean['RMSE Teste (m)'].idxmin()} ({mean['RMSE Teste (m)'].min():.4f} m). "
         "Por horizonte: " + "; ".join(f"{h} h: {m} ({v:.4f})" for h, m, v in by("RMSE Teste (m)")) + ". "
         "Em 1 h, LightGBM direto e recursivo são o mesmo modelo (empate exato).",
         "3. **Menor tempo**: baselines (~0,005 s para os 4 horizontes). Entre os treináveis, "
         f"{tot4[tot4.modelo.isin(TRAINABLE)].sort_values('t_sem_tuning').iloc[0]['Modelo']} "
         f"({tot4[tot4.modelo.isin(TRAINABLE)]['t_sem_tuning'].min():.1f} s para os 4 horizontes; "
         f"SARIMA {tot4.loc[tot4.modelo == 'sarima', 't_sem_tuning'].iloc[0]:.0f} s).",
         "4. **Melhor generalização**: em erro de teste, o LightGBM. Pela razão teste/treino (sobreajuste), o mais "
         "estável é o SARIMA " + f"({ratio.loc['SARIMA', 'min']:.2f}–{ratio.loc['SARIMA', 'max']:.2f}); os modelos de "
         "aprendizado de máquina ficam entre " + f"{ratio.drop('SARIMA')['min'].min():.2f} e {ratio.drop('SARIMA')['max'].max():.2f}"
         " — ajustam-se mais ao treino, mas ainda generalizam melhor em erro absoluto. O teste é um período mais regular "
         "(razões < 1 nos baselines), o que deve ser considerado.",
         *[f"{4 + i}. **Melhor em {h} h**: {b[0]} (RMSE {b[1]:.4f} m); sem diferença significativa (DM, Holm 5%): {b[2]}."
           for i, (h, b) in enumerate(sorted(best_h.items()), start=1)],
         "9. **O modelo mais complexo é necessariamente melhor?** Não. O de maior custo computacional (SARIMA) é "
         "dominado em todos os horizontes; o SVR (2º mais caro) também. O melhor desempenho veio de um modelo flexível "
         "e barato (LightGBM). A resposta depende de como \"complexidade\" é definida no texto (custo × capacidade).",
         "10. **O ganho de acurácia justifica o custo?** Em relação ao melhor baseline de cada horizonte, sim: os "
         f"treináveis reduzem o RMSE de {min(gains):.0f}% a {max(gains):.0f}%, com custo de segundos no LightGBM. "
         "Entre os treináveis, o custo adicional do SVR e do SARIMA não "
         "compra acurácia. O LightGBM recursivo paga 5–10× mais tempo de previsão por ~2,5% de ganho em 6–12 h, "
         "sem diferença significativa na perda quadrática.",
         "11. **Melhor custo-benefício**: " + ", ".join(f"{h} h: {m}" for h, m in sc_rows) +
         " (score principal, estável em α de 0,1 a 0,9; na fronteira de Pareto em todos os horizontes).",
         "12. **O resultado muda com o horizonte?** Sim, em magnitude: a vantagem dos modelos de aprendizado de máquina "
         "diminui com h (em 24 h, os quatro treináveis são estatisticamente equivalentes); a estratégia recursiva reduz "
         "o erro em 6–12 h e deixa de reduzir o RMSE em 24 h. A ordem de custo-benefício (LightGBM à frente) não muda.", ""]
    return L


def build_report(f: pd.DataFrame) -> str:
    L = ["# ETAPA 16 — Consolidação", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "Tabela final em `results/metrics/resultados_finais.csv` e `.parquet` (todas as colunas). "
         "Score = principal (B, somente treináveis, α = 0,5, RMSE, sem tuning); baselines sem score principal "
         "(ver \"Score (com baseline)\", análise de sensibilidade). Tempos: remedição controlada (ETAPA 13).", "",
         "## Tabela final", "",
         "| Modelo | h | MAE Treino | RMSE Treino | MAE Teste | RMSE Teste | Tempo Treino (s) | Tempo Previsão (s) | Tempo Total (s) | Score |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in f.itertuples(index=False):
        sc = "—" if pd.isna(r.Score) else f"{r.Score:.3f}"
        L.append(f"| {r.Modelo} | {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4g} | {r[7]:.4g} | "
                 f"{r[8]:.4g} | {sc} |")
    L += ["", "## Colunas complementares", "",
          "| Modelo | h | RMSE Teste (m³ eq.) | Memória (MB) | Teste/Treino | Estratégia | Pareto | Grupo dos melhores (DM) | Score (com baseline) |",
          "|---|---:|---:|---:|---:|---|---|---|---:|"]
    for _, r in f.iterrows():
        par = "—" if pd.isna(r["Pareto"]) else ("sim" if r["Pareto"] else "não")
        L.append(f"| {r['Modelo']} | {r['Horizonte (h)']} | {r['RMSE Teste (m³ eq.)']:.1f} | {r['Memória (MB)']:.1f} | "
                 f"{r['RMSE Teste/Treino']:.3f} | {r['Estratégia']} | {par} | {'sim' if r['Grupo dos melhores (DM)'] else 'não'} | "
                 f"{'—' if pd.isna(r['Score (com baseline)']) else format(r['Score (com baseline)'], '.3f')} |")
    L += ["", *answers(f)]
    return "\n".join(L)


def main() -> None:
    log = get_logger("consolidate", run_name="16_consolidacao")
    f, _ = final_table()
    expected = len([*BASELINES, *TRAINABLE]) * len(load_config()["models"]["horizons_hours"])
    if len(f) != expected or f[["MAE Teste (m)", "RMSE Teste (m)", "Tempo Total (s)"]].isna().any().any():
        raise AssertionError(f"Tabela final incompleta: {len(f)} linhas (esperado {expected})")
    if f.loc[f["Modelo"].isin([LABELS[m] for m in TRAINABLE]), "Score"].isna().any():
        raise AssertionError("Score ausente para modelo treinável")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    f.to_csv(METRICS_DIR / "resultados_finais.csv", index=False, encoding="utf-8")
    f.to_parquet(METRICS_DIR / "resultados_finais.parquet", index=False)
    (REPORTS_DIR / "16_consolidacao.md").write_text(build_report(f), encoding="utf-8")
    log.info("Tabela final: %d linhas; relatório %s", len(f), REPORTS_DIR / "16_consolidacao.md")


if __name__ == "__main__":
    main()
