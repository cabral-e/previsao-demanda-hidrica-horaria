"""ETAPA 06 — Avaliação das defasagens candidatas (23 e 25 h) SOMENTE no treino.

Validação cronológica interna: o treino é dividido em ajuste (primeiros 80% da grade)
e validação (últimos 20%), pelo instante-alvo. O conjunto de teste NÃO é lido.
Compara base × base + candidatas com Ridge e LightGBM (parâmetros padrão, sem tuning),
em estratégia direta para cada horizonte. Aplica a regra fixada em
config.yaml → features.lag_eval antes da execução.

Uso: python -m src.features.lag_evaluation [--force]
Saídas: results/reports/06_atributos.md, results/reports/06_selecao_atributos.json (+ .meta.json)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load import sha256
from src.data.split import temporal_cut
from src.features.build import feature_columns, select_by_target_time, supervised
from src.utils.logging_utils import get_logger
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

TRAIN = PROCESSED_DIR / "05_treino.parquet"
OUTPUT = REPORTS_DIR / "06_selecao_atributos.json"
CONFIG_KEYS = ("features", "project.seed", "cost.n_jobs", "data.target")


def make_model(name: str, seed: int, n_jobs: int):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))   # escalonador ajustado só no 'ajuste'
    if name == "lightgbm":
        return LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1)
    raise ValueError(name)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(y: pd.Series, inner_cut: pd.Timestamp, lags: list[int], fcfg: dict, h: int,
             model_name: str, seed: int, n_jobs: int) -> dict:
    df = supervised(y, h, lags, fcfg["rolling_mean_windows"], fcfg["rolling_std_windows"], fcfg["calendar_cyclic"])
    fit = select_by_target_time(df, end_exclusive=inner_cut)
    val = select_by_target_time(df, start=inner_cut)
    cols = feature_columns(df)
    model = make_model(model_name, seed, n_jobs)
    t0 = time.perf_counter()
    model.fit(fit[cols].to_numpy(), fit["alvo"].to_numpy())
    pred = model.predict(val[cols].to_numpy())
    elapsed = time.perf_counter() - t0
    return {"rmse": rmse(val["alvo"].to_numpy(), pred),
            "mae": float(np.mean(np.abs(val["alvo"].to_numpy() - pred))),
            "n_ajuste": len(fit), "n_val": len(val), "n_atributos": len(cols), "tempo_s": elapsed}


def main(force: bool = False) -> None:
    log = get_logger("lag_evaluation", run_name="06_atributos")
    cfg = load_config()
    fcfg, ev = cfg["features"], cfg["features"]["lag_eval"]
    ts_col, target, seed, n_jobs = cfg["data"]["timestamp_column"], cfg["data"]["target"], \
        cfg["project"]["seed"], cfg["cost"]["n_jobs"]

    in_hash, cfg_hash = sha256(TRAIN), config_hash(*CONFIG_KEYS)
    if not force and is_reusable(OUTPUT, input_sha256=in_hash, config_hash=cfg_hash):
        log.info("Avaliação já realizada com mesma entrada e configuração; reutilizando.")
        return

    train = pd.read_parquet(TRAIN)
    y = train.set_index(ts_col)[target].asfreq("h")
    if y.isna().any():
        raise ValueError("Série de treino com lacunas — verifique a ETAPA 04")
    inner_cut = temporal_cut(str(y.index.min()), str(y.index.max()), ev["inner_train_fraction"])
    log.info("Validação interna: ajuste < %s <= validação (somente treino, até %s)", inner_cut, y.index.max())

    base, extra = list(fcfg["lags_base"]), list(fcfg["lags_candidates_extra"])
    sets = {"base": base, "base+23,25": sorted(base + extra)}
    rows, t_total = [], time.perf_counter()
    for model_name in ev["models"]:
        for h in ev["horizons"]:
            res = {name: evaluate(y, inner_cut, lags, fcfg, h, model_name, seed, n_jobs) for name, lags in sets.items()}
            gain = 1 - res["base+23,25"]["rmse"] / res["base"]["rmse"]
            rows.append({"modelo": model_name, "h": h, **{f"rmse_{k}": v["rmse"] for k, v in res.items()},
                         **{f"mae_{k}": v["mae"] for k, v in res.items()}, "ganho_rmse": gain,
                         "n_ajuste": res["base"]["n_ajuste"], "n_val": res["base"]["n_val"],
                         "tempo_s": sum(v["tempo_s"] for v in res.values())})
            log.info("%s h=%d: RMSE base %.4f | +23,25 %.4f | ganho %+.2f%%", model_name, h,
                     res["base"]["rmse"], res["base+23,25"]["rmse"], 100 * gain)
    t_total = time.perf_counter() - t_total

    tab = pd.DataFrame(rows)
    mean_gain = float(tab["ganho_rmse"].mean())
    include = mean_gain >= ev["include_if_mean_rmse_gain_at_least"]
    selected = sets["base+23,25"] if include else sets["base"]
    n_feat = len(selected) + len(fcfg["rolling_mean_windows"]) + len(fcfg["rolling_std_windows"]) \
        + 2 * len(fcfg["calendar_cyclic"])

    result = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "corte_interno": str(inner_cut),
              "treino_fim": str(y.index.max()), "ganho_medio_rmse": mean_gain,
              "limiar": ev["include_if_mean_rmse_gain_at_least"], "incluir_23_25": include,
              "lags_selecionados": selected, "rolling_mean_windows": fcfg["rolling_mean_windows"],
              "rolling_std_windows": fcfg["rolling_std_windows"], "calendar_cyclic": fcfg["calendar_cyclic"],
              "use_month": fcfg["use_month"], "n_atributos": n_feat,
              "custo_selecao_s": t_total, "tabela": rows}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_meta(OUTPUT, {"etapa": "06", "input": TRAIN.name, "input_sha256": in_hash, "config_hash": cfg_hash})

    L = ["# ETAPA 06 — Engenharia de atributos", "", f"Gerado em: {result['gerado_em']}", "",
         "## Conjunto de atributos", "",
         "Origem t = último instante observado; alvo y(t + h). Nenhum atributo usa valor posterior a t.", "",
         "| Grupo | Definição | Itens |", "|---|---|---|",
         f"| Defasagens | lag_k = LT01(t + 1 − k) | {', '.join(map(str, selected))} |",
         f"| Médias móveis | média de LT01 em [t − w + 1, t] | w = {', '.join(map(str, fcfg['rolling_mean_windows']))} h |",
         f"| Desvios móveis | desvio de LT01 em [t − w + 1, t] | w = {', '.join(map(str, fcfg['rolling_std_windows']))} h |",
         f"| Calendário cíclico | seno e cosseno de {', '.join(fcfg['calendar_cyclic'])} do instante-alvo t + h | sem mês |",
         "", f"Total: **{n_feat} atributos**. Univariado (sem exógenas).", "",
         "## Avaliação das defasagens 23 e 25 h (somente treino)", "",
         f"- Ajuste: alvos até {inner_cut - pd.Timedelta(hours=1)}; validação: alvos de {inner_cut} a {y.index.max()}.",
         "- O conjunto de teste não foi lido nesta etapa.",
         "- Modelos com parâmetros padrão (Ridge α = 1 com padronização ajustada no período de ajuste; "
         f"LightGBM padrão, seed {seed}, n_jobs {n_jobs}); estratégia direta por horizonte.",
         f"- Regra fixada antes da execução: incluir se o ganho relativo médio de RMSE ≥ "
         f"{100 * ev['include_if_mean_rmse_gain_at_least']:.0f}%.", "",
         "| Modelo | h | RMSE base | RMSE +23,25 | Ganho | MAE base | MAE +23,25 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['modelo']} | {r['h']} | {r['rmse_base']:.4f} | {r['rmse_base+23,25']:.4f} | "
                 f"{100 * r['ganho_rmse']:+.2f}% | {r['mae_base']:.4f} | {r['mae_base+23,25']:.4f} |")
    L += ["", f"**Ganho médio: {100 * mean_gain:+.2f}% → {'incluir' if include else 'não incluir'} 23 e 25 h.**", "",
          f"Custo da seleção (registrado à parte do custo dos modelos finais): {t_total:.1f} s "
          f"({len(rows) * 2} ajustes).", ""]
    (REPORTS_DIR / "06_atributos.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Ganho médio %.2f%% -> incluir 23/25: %s; %d atributos", 100 * mean_gain, include, n_feat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
