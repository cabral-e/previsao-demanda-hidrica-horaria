"""ETAPA 13 — Consolidação do custo computacional.

Fonte: registry.csv (medição mais recente = remedição controlada) e os JSON de tuning.
Também compara com a medição anterior (estabilidade) e confere que as métricas de erro
não mudaram entre medições (determinismo).

Uso: python -m src.evaluation.cost_report
Saídas: results/metrics/13_custos.csv|parquet, results/reports/13_custo.md
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.metrics_report import LABELS
from src.utils.logging_utils import get_logger
from src.utils.paths import METRICS_DIR, REPORTS_DIR, load_config

SHARED_FIT = {"sarima", "lightgbm_rec"}                 # um ajuste serve aos 4 horizontes
TUNING_FILES = {"sarima": "08_sarima_selecao.json", "svr": "09_svr_tuning.json", "lightgbm": "10_lightgbm_tuning.json"}


def tuning_costs() -> dict[str, float]:
    out = {}
    for m, f in TUNING_FILES.items():
        p = REPORTS_DIR / f
        out[m] = float(json.loads(p.read_text(encoding="utf-8"))["custo_tuning_s"]) if p.is_file() else np.nan
    probe = REPORTS_DIR / "09_svr_sondagem.json"
    if probe.is_file():
        out["svr_sondagem"] = sum(r["t_fit_s"] + r["t_pred_s"] for r in json.loads(probe.read_text(encoding="utf-8"))["linhas"])
    sel = REPORTS_DIR / "06_selecao_atributos.json"
    if sel.is_file():
        out["selecao_atributos"] = float(json.loads(sel.read_text(encoding="utf-8"))["custo_selecao_s"])
    out["lightgbm_rec"] = out.get("lightgbm", np.nan)        # reutiliza o tuning de h = 1 do LightGBM
    return out


def previous_vs_latest() -> pd.DataFrame:
    reg = registry.load()
    reg = reg[reg["status"] == "ok"].sort_values("timestamp")
    rows = []
    for (m, h), g in reg.groupby(["modelo", "horizonte_h"]):
        if len(g) < 2:
            continue
        a, b = g.iloc[-2], g.iloc[-1]
        rows.append({"modelo": m, "horizonte_h": h, "versao_ant": a["versao"], "versao_atual": b["versao"],
                     "t_treino_ant": a["t_treino_s"], "t_treino_atual": b["t_treino_s"],
                     "t_prev_ant": a["t_previsao_s"], "t_prev_atual": b["t_previsao_s"],
                     "mem_ant": a["memoria_pico_mb"], "mem_atual": b["memoria_pico_mb"],
                     "erro_identico": bool(np.isclose(a["rmse_teste"], b["rmse_teste"], rtol=0, atol=1e-12)
                                           and np.isclose(a["mae_teste"], b["mae_teste"], rtol=0, atol=1e-12)
                                           and np.isclose(a["rmse_treino"], b["rmse_treino"], rtol=0, atol=1e-12))})
    return pd.DataFrame(rows)


def cost_table(reg: pd.DataFrame, n_test: int) -> pd.DataFrame:
    t = reg[["modelo", "horizonte_h", "exp_id", "versao", "t_treino_s", "t_treino_min_s", "t_treino_max_s",
             "t_previsao_s", "t_previsao_min_s", "t_previsao_max_s", "t_total_s", "t_total_min_s", "t_total_max_s",
             "memoria_pico_mb", "n_repeats", "n_jobs", "rmse_teste", "mae_teste"]].copy()
    t["ms_por_previsao"] = 1000 * t["t_previsao_s"] / n_test
    t["dispersao_treino_rel"] = (t["t_treino_max_s"] - t["t_treino_min_s"]) / t["t_treino_s"].replace(0, np.nan)
    return t


def serve_all(t: pd.DataFrame, tun: dict[str, float]) -> pd.DataFrame:
    rows = []
    for m, g in t.groupby("modelo"):
        train = g["t_treino_s"].iloc[0] if m in SHARED_FIT else g["t_treino_s"].sum()
        pred = g["t_previsao_s"].sum()
        rows.append({"modelo": m, "n_modelos_ajustados": 1 if m in SHARED_FIT else (0 if m in
                     ("persistencia", "sazonal_diaria", "sazonal_semanal") else len(g)),
                     "t_treino_4h_s": train, "t_previsao_4h_s": pred, "t_total_4h_s": train + pred,
                     "memoria_max_mb": g["memoria_pico_mb"].max(), "t_tuning_s": tun.get(m, 0.0)})
    return pd.DataFrame(rows).sort_values("t_total_4h_s")


def build_report(t, sa, comp, tun, state) -> str:
    cfg = load_config()
    L = ["# ETAPA 13 — Custo computacional", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Como foi medido", "",
         f"- Tempo: `time.perf_counter`; {cfg['cost']['n_repeats']} execuções independentes; mediana [mín–máx].",
         f"- Threads: n_jobs = {cfg['cost']['n_jobs']} em todos os modelos (libsvm e statsmodels são monothread).",
         "- Tempo de previsão: todas as 3.504 previsões de teste do horizonte (inclui montar atributos/filtrar).",
         "- Memória: pico de RSS do processo acima do nível inicial (psutil), em execução separada da cronometragem.",
         "- Tuning registrado à parte; não entra no tempo de treino.",
         "- Remedição controlada em lote único (aplicativos pesados fechados); medição anterior preservada no registro."]
    if state:
        s = state["inicio"]
        L += [f"- Estado da máquina no início do lote: RAM livre {s['ram_livre_gb']} GB de {s['ram_total_gb']} GB; "
              f"CPU {s['cpu_percent_1s']}% (1 s); na tomada: {s['na_tomada']}; {s['n_processos']} processos. "
              f"Maiores: {', '.join(f'{n} ({m} MB)' for n, m in s['top_processos_mb'][:5])}."]
    L += ["", "## Custo por modelo e horizonte", "",
          "| Modelo | h | t_treino (s) | t_previsão (s) | t_total (s) | ms/previsão | Memória (MB) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples(index=False):
        L.append(f"| {LABELS[r.modelo]} | {r.horizonte_h} | {r.t_treino_s:.4g} [{r.t_treino_min_s:.4g}–{r.t_treino_max_s:.4g}] | "
                 f"{r.t_previsao_s:.4g} [{r.t_previsao_min_s:.4g}–{r.t_previsao_max_s:.4g}] | {r.t_total_s:.4g} | "
                 f"{r.ms_por_previsao:.4f} | {r.memoria_pico_mb:.1f} |")
    L += ["", "SARIMA e LightGBM recursivo: um único ajuste serve aos 4 horizontes; o t_treino de cada h é o desse "
          "ajuste (convenção da ETAPA 13).", "",
          "## Custo para servir os 4 horizontes", "",
          "| Modelo | Modelos ajustados | Treino (s) | Previsão, soma dos 4 h (s) | Total (s) | Memória máx. (MB) | Tuning (s, à parte) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in sa.itertuples(index=False):
        L.append(f"| {LABELS[r.modelo]} | {r.n_modelos_ajustados} | {r.t_treino_4h_s:.4g} | {r.t_previsao_4h_s:.4g} | "
                 f"{r.t_total_4h_s:.4g} | {r.memoria_max_mb:.1f} | {r.t_tuning_s:.4g} |")
    L += ["", f"Outros custos de preparação: seleção de defasagens (ETAPA 06) {tun.get('selecao_atributos', np.nan):.1f} s "
          f"(comum a SVR e LightGBM); sondagem do SVR {tun.get('svr_sondagem', np.nan):.1f} s. "
          "LightGBM recursivo reutiliza o tuning do LightGBM (h = 1).", ""]
    if len(comp):
        L += ["## Estabilidade entre medições (anterior × remedição controlada)", "",
              f"Métricas de erro idênticas em todas as {len(comp)} combinações: "
              f"**{'sim' if comp['erro_identico'].all() else 'NÃO'}** (modelos determinísticos).", "",
              "| Modelo | h | t_treino ant. → atual (s) | t_previsão ant. → atual (s) | Memória ant. → atual (MB) |",
              "|---|---:|---|---|---|"]
        for r in comp.itertuples(index=False):
            L.append(f"| {LABELS[r.modelo]} | {r.horizonte_h} | {r.t_treino_ant:.4g} → {r.t_treino_atual:.4g} | "
                     f"{r.t_prev_ant:.4g} → {r.t_prev_atual:.4g} | {r.mem_ant:.1f} → {r.mem_atual:.1f} |")
        L.append("")
    return "\n".join(L)


def main() -> None:
    log = get_logger("cost_report", run_name="13_custo")
    reg = registry.latest_ok()
    n_test = int(reg["n_teste"].iloc[0])
    t = cost_table(reg, n_test)
    tun = tuning_costs()
    sa = serve_all(t, tun)
    comp = previous_vs_latest()
    if len(comp) and not comp["erro_identico"].all():
        raise AssertionError("Métricas de erro mudaram entre medições — investigar determinismo")
    state_p = REPORTS_DIR / "13_estado_maquina.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.is_file() else None
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in ((t, "13_custos"), (sa, "13_custo_4_horizontes")):
        df.to_csv(METRICS_DIR / f"{name}.csv", index=False, encoding="utf-8")
        df.to_parquet(METRICS_DIR / f"{name}.parquet", index=False)
    (REPORTS_DIR / "13_custo.md").write_text(build_report(t, sa, comp, tun, state), encoding="utf-8")
    log.info("Relatório: %s", REPORTS_DIR / "13_custo.md")


if __name__ == "__main__":
    main()
