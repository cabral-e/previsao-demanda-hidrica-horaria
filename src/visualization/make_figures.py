"""ETAPA 17 — Geração de todas as figuras de resultados (seção 11).

Fontes: registry.csv (última versão), previsões em results/predictions/, métricas da ETAPA 12–16.
As figuras da EDA (série, distribuição, ACF/PACF) já estão em results/figures/eda/ (ETAPA 03).

Uso: python -m src.visualization.make_figures
Saídas: results/figures/resultados/*.png|pdf e results/reports/17_figuras.md
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.metrics_report import BASELINES, LABELS, TRAINABLE
from src.utils.logging_utils import get_logger
from src.utils.paths import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, load_config
from src.visualization import plots
from src.visualization.style import MODEL_ORDER, apply_style, horizon_label, num, save_figure, t

SUB = "resultados"
FULL_XTICK = 500          # decisão do usuário: teste inteiro a cada 500 amostras; versão ampliada a cada 100


def test_predictions(exp_id: str) -> pd.DataFrame:
    p = pd.read_parquet(PREDICTIONS_DIR / f"{exp_id}.parquet")
    return p[p["conjunto"] == "teste"].sort_values("ts_alvo").reset_index(drop=True)


def forecast_figures(reg: pd.DataFrame, cfg: dict, log) -> list[str]:
    pc = cfg["plots"]
    out = []
    best_bl = {h: g[g.modelo.isin(BASELINES)].sort_values("rmse_teste").iloc[0]["modelo"] for h, g in reg.groupby("horizonte_h")}
    groups = {**{m: {h: m for h in best_bl} for m in TRAINABLE}, "baseline": best_bl}
    for name, per_h in groups.items():
        panels_full, panels_zoom = [], []
        for h in sorted(per_h):
            m = per_h[h]
            row = reg[(reg.modelo == m) & (reg.horizonte_h == h)].iloc[0]
            p = test_predictions(row["exp_id"])
            label = horizon_label(h) if name != "baseline" else f"{horizon_label(h)} — {LABELS[m]}"
            panels_full.append((label, p["y"].to_numpy(), p["y_prev"].to_numpy()))
            z = pc["zoom_samples"]
            panels_zoom.append((label, p["y"].to_numpy()[:z], p["y_prev"].to_numpy()[:z]))
        fname = f"17_previsao_{name}"
        out += [str(x) for x in save_figure(plots.plot_forecast_panels(panels_full, FULL_XTICK, tuple(pc["figsize"])),
                                            f"{fname}_teste_completo", SUB)]
        out += [str(x) for x in save_figure(plots.plot_forecast_panels(panels_zoom, pc["xtick_step"], tuple(pc["figsize"])),
                                            f"{fname}_primeiras_{pc['zoom_samples']}", SUB)]
        log.info("Figura de previsão: %s", name)
    return out


def main() -> None:
    log = get_logger("make_figures", run_name="17_figuras")
    cfg = load_config()
    apply_style()
    reg = registry.latest_ok()
    reg = reg[reg.modelo.isin([*BASELINES, *TRAINABLE])]
    labels = {m: LABELS[m] for m in MODEL_ORDER if m in set(reg.modelo)}
    files = forecast_figures(reg, cfg, log)

    piv = lambda col: reg.pivot(index="horizonte_h", columns="modelo", values=col)[list(labels)]  # noqa: E731
    files += [str(x) for x in save_figure(plots.plot_grouped_bars(piv("mae_teste"), labels, t("mae_test")), "17_mae_modelo_horizonte", SUB)]
    files += [str(x) for x in save_figure(plots.plot_grouped_bars(piv("rmse_teste"), labels, t("rmse_test")), "17_rmse_modelo_horizonte", SUB)]

    tr_labels = {m: LABELS[m] for m in MODEL_ORDER if m in TRAINABLE}
    tp = lambda col: reg[reg.modelo.isin(TRAINABLE)].pivot(index="horizonte_h", columns="modelo", values=col)[list(tr_labels)]  # noqa: E731
    files += [str(x) for x in save_figure(plots.plot_time_panels(tp("t_treino_s"), tp("t_previsao_s"), tp("t_total_s"), tr_labels),
                                          "17_tempo_computacional", SUB)]

    sc = cfg["composite_score"]
    s = pd.read_parquet(METRICS_DIR / "14_score.parquet")
    s = s[(s.normalizacao == sc["main_normalization"]) & (s.erro == "rmse") & (s.variante == "sem_tuning")]
    main_s = s[(s.conjunto == sc["main_set"]) & np.isclose(s.alpha, sc["alpha"])].pivot(index="horizonte_h", columns="modelo", values="score")[list(tr_labels)]
    files += [str(x) for x in save_figure(plots.plot_grouped_bars(main_s, tr_labels, t("score")), "17_score_principal", SUB)]
    curves = {h: g.pivot(index="alpha", columns="modelo", values="score")[list(tr_labels)]
              for h, g in s[s.conjunto == sc["main_set"]].groupby("horizonte_h")}
    files += [str(x) for x in save_figure(plots.plot_alpha_sensitivity(curves, tr_labels), "17_score_sensibilidade_alpha", SUB)]
    sb = s[s.conjunto == "com_baseline"].copy()
    sb["modelo"] = np.where(sb.modelo.isin(BASELINES), "melhor_baseline", sb.modelo)
    bl_labels = {**tr_labels, "melhor_baseline": t("best_baseline")}
    curves_b = {h: g.pivot(index="alpha", columns="modelo", values="score")[list(bl_labels)] for h, g in sb.groupby("horizonte_h")}
    files += [str(x) for x in save_figure(plots.plot_alpha_sensitivity(curves_b, bl_labels), "17_score_sensibilidade_com_baseline", SUB)]

    par = pd.read_parquet(METRICS_DIR / "15_pareto.parquet")
    pm = par[(par.conjunto == sc["main_set"]) & (par.erro == "rmse_teste") & (par.variante == "sem_tuning")]
    points = {h: g.drop(columns="erro").rename(columns={"erro_valor": "erro"})[["modelo", "t", "erro", "pareto"]]
              for h, g in pm.groupby("horizonte_h")}
    pb = par[(par.conjunto == "com_baseline") & (par.papel == "baseline") & (par.erro == "rmse_teste") & (par.variante == "sem_tuning")]
    ref = {h: (float(g["t"].iloc[0]), float(g["erro_valor"].iloc[0])) for h, g in pb.groupby("horizonte_h")}
    files += [str(x) for x in save_figure(plots.plot_pareto_panels(points, tr_labels, ref), "17_tradeoff_pareto", SUB)]

    agg = pd.read_parquet(METRICS_DIR / "15_pareto_4h.parquet").set_index("modelo")
    eff = piv("rmse_teste")[list(tr_labels)].copy()
    best = reg[reg.modelo.isin(BASELINES)].groupby("horizonte_h")["rmse_teste"].min()
    eff["melhor_baseline"] = best
    eff_labels = {m: f"{LABELS[m]} — {num(agg.loc[m, 't_sem_tuning'])} s" for m in tr_labels}
    eff_labels["melhor_baseline"] = f"{t('best_baseline')} — ~0 s"
    files += [str(x) for x in save_figure(plots.plot_efficiency(eff, eff_labels), "17_curva_eficiencia", SUB)]

    eda = sorted(p.name for p in (FIGURES_DIR / "eda").glob("*.png"))
    pngs = sorted({f.split("\\")[-1] for f in files if f.endswith(".png")})
    L = ["# ETAPA 17 — Visualizações", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "Estilo da referência (`docs/referencia_grafico_previsao.png`): fundo branco, borda preta, grade clara, fonte "
         "serifada, observado em preto contínuo e previsto em vermelho tracejado, legenda abaixo. PNG 300 dpi + PDF.", "",
         f"## Resultados — `results/figures/{SUB}/` ({len(pngs)} figuras)", "", *[f"- `{p}`" for p in pngs], "",
         "## EDA (ETAPA 03) — `results/figures/eda/`", "", *[f"- `{p}`" for p in eda], "",
         "Notas: previsão × observado em duas janelas (teste completo, marcações a cada 500; primeiras 1.300 amostras, "
         "a cada 100). A figura \"baseline\" mostra o melhor baseline de cada horizonte (indicado na faixa). "
         "Curva de eficiência = RMSE de teste × horizonte; custo para servir os 4 horizontes na legenda. "
         "ACF e PACF estão na mesma figura (painéis empilhados).", ""]
    (REPORTS_DIR / "17_figuras.md").write_text("\n".join(L), encoding="utf-8")
    log.info("%d figuras de resultados geradas", len(pngs))


if __name__ == "__main__":
    main()
