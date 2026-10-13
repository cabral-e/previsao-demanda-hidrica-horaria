"""ETAPA 09 — SVR (kernel RBF), estratégia direta.

Pipeline: padronização dos atributos E do alvo ajustadas SOMENTE nos dados de ajuste
(o ε passa a ser medido em desvios-padrão do alvo, comparável entre horizontes).

Uso:
  python -m src.models.svr --probe    desempenho e custo com parâmetros padrão (validação interna do treino)
  python -m src.models.svr --tune     busca controlada (após aprovação do espaço)
  python -m src.models.svr            ajuste final + avaliação
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data.split import temporal_cut
from src.evaluation.metrics import mae, rmse
from src.evaluation.protocol import load_eval_data
from src.features.build import select_by_target_time
from src.models.ml_direct import DirectForecaster, load_feature_spec
from src.utils.logging_utils import get_logger
from src.utils.paths import REPORTS_DIR, load_config

PROBE_OUT = REPORTS_DIR / "09_svr_sondagem.json"


def make_svr(C: float = 1.0, gamma: Any = "scale", epsilon: float = 0.1, cache_size: int = 200):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon,
                                                      cache_size=cache_size)),
        transformer=StandardScaler())


def svr_forecaster(hp: dict[str, Any], spec: dict[str, Any]) -> DirectForecaster:
    return DirectForecaster("svr", lambda: make_svr(**hp), spec, {"kernel": "rbf", **hp})


def inner_split(y_train: pd.Series, fraction: float) -> pd.Timestamp:
    return temporal_cut(str(y_train.index.min()), str(y_train.index.max()), fraction)


def validate_inner(hp: dict, spec: dict, y_train: pd.Series, inner_cut: pd.Timestamp, h: int,
                   max_fit_samples: int | None = None) -> dict[str, Any]:
    """Ajusta com alvos < corte interno e valida nos alvos ≥ corte (tudo dentro do treino)."""
    f = svr_forecaster(hp, spec)
    df = f._supervised(y_train, h)
    fit_df = select_by_target_time(df, end_exclusive=inner_cut)
    if max_fit_samples:
        fit_df = fit_df.iloc[-max_fit_samples:]                       # mais recentes (cronológico)
    val_df = select_by_target_time(df, start=inner_cut)
    cols = [c for c in df.columns if c not in ("alvo", "ts_alvo")]
    est = make_svr(**hp)
    t0 = time.perf_counter()
    est.fit(fit_df[cols].to_numpy(), fit_df["alvo"].to_numpy())
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = est.predict(val_df[cols].to_numpy())
    t_pred = time.perf_counter() - t1
    n_sv = int(est.regressor_[-1].support_.shape[0])
    return {"hp": hp, "h": h, "n_ajuste": len(fit_df), "n_val": len(val_df), "n_vetores_suporte": n_sv,
            "rmse_val": rmse(val_df["alvo"].to_numpy(), pred), "mae_val": mae(val_df["alvo"].to_numpy(), pred),
            "t_fit_s": t_fit, "t_pred_s": t_pred}


def run_probe(log) -> None:
    cfg = load_config()
    data = load_eval_data()
    spec = load_feature_spec()
    inner_cut = inner_split(data.y_train, cfg["features"]["lag_eval"]["inner_train_fraction"])
    default = {"C": 1.0, "gamma": "scale", "epsilon": 0.1}
    rows = []
    for n in (2000, 4000, None):                                     # curva de custo × amostras
        r = validate_inner(default, spec, data.y_train, inner_cut, 1, n)
        rows.append(r)
        log.info("h=1 n=%d: RMSE val %.4f | fit %.1fs | pred %.2fs | SV %d", r["n_ajuste"], r["rmse_val"],
                 r["t_fit_s"], r["t_pred_s"], r["n_vetores_suporte"])
    for h in (6, 24):
        r = validate_inner(default, spec, data.y_train, inner_cut, h, None)
        rows.append(r)
        log.info("h=%d n=%d: RMSE val %.4f | fit %.1fs | pred %.2fs | SV %d", h, r["n_ajuste"], r["rmse_val"],
                 r["t_fit_s"], r["t_pred_s"], r["n_vetores_suporte"])
    PROBE_OUT.write_text(json.dumps({"gerado_em": datetime.now().isoformat(timespec="seconds"),
                                     "corte_interno": str(inner_cut), "linhas": rows}, indent=2,
                                    ensure_ascii=False, default=str), encoding="utf-8")


TUNING_OUT = REPORTS_DIR / "09_svr_tuning.json"
TUNING_KEYS = ("svr", "features", "data.target", "split")
CONFIG_KEYS = ("svr", "features", "data.target", "split", "evaluation", "cost", "project.seed")


def gamma_scale(spec: dict) -> float:
    """Valor de gamma='scale' do sklearn (1 / (n_atributos · Var(X))) com X padronizado = 1/n_atributos."""
    n_feat = len(spec["lags_selecionados"]) + len(spec["rolling_mean_windows"]) + \
        len(spec["rolling_std_windows"]) + 2 * len(spec["calendar_cyclic"])
    return 1.0 / n_feat


def run_tuning(log, force: bool) -> dict:
    from src.data.load import sha256
    from src.data.split import TRAIN_OUT
    from src.utils.persistence import config_hash, is_reusable, write_meta

    cfg = load_config()
    sc = cfg["svr"]
    in_hash, cfg_hash = sha256(TRAIN_OUT), config_hash(*TUNING_KEYS)
    if not force and is_reusable(TUNING_OUT, input_sha256=in_hash, config_hash=cfg_hash):
        log.info("Tuning já realizado com mesma entrada e configuração; reutilizando.")
        return json.loads(TUNING_OUT.read_text(encoding="utf-8"))

    data = load_eval_data()
    spec = load_feature_spec()
    inner_cut = inner_split(data.y_train, sc["inner_train_fraction"])
    g0 = gamma_scale(spec)
    p1 = sc["tuning"]["phase1"]
    t_all = time.perf_counter()
    rows, best = [], {}
    for h in cfg["models"]["horizons_hours"]:
        cands = []
        for C in p1["C"]:
            for m in p1["gamma_scale_multipliers"]:
                hp = {"C": float(C), "gamma": float(m * g0), "epsilon": float(p1["epsilon"]), "cache_size": sc["cache_size_mb"]}
                r = validate_inner(hp, spec, data.y_train, inner_cut, h) | {"fase": 1}
                cands.append(r)
                log.info("h=%d C=%g γ=%.4f ε=%g: RMSE %.4f | fit %.1fs | SV %d", h, C, hp["gamma"], hp["epsilon"],
                         r["rmse_val"], r["t_fit_s"], r["n_vetores_suporte"])
        b1 = min(cands, key=lambda r: r["rmse_val"])
        for eps in sc["tuning"]["phase2_epsilon"]:
            hp = {**b1["hp"], "epsilon": float(eps)}
            r = validate_inner(hp, spec, data.y_train, inner_cut, h) | {"fase": 2}
            cands.append(r)
            log.info("h=%d ε=%g: RMSE %.4f | fit %.1fs | SV %d", h, eps, r["rmse_val"], r["t_fit_s"], r["n_vetores_suporte"])
        bh = min(cands, key=lambda r: r["rmse_val"])
        best[h] = bh["hp"]
        rows += cands
        log.info("h=%d escolhido %s (RMSE val %.4f)", h, bh["hp"], bh["rmse_val"])
    t_all = time.perf_counter() - t_all
    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "corte_interno": str(inner_cut),
           "gamma_scale": g0, "escolhidos": {str(h): v for h, v in best.items()}, "candidatos": rows,
           "custo_tuning_s": t_all, "n_ajustes": len(rows)}
    TUNING_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_meta(TUNING_OUT, {"etapa": "09", "input": TRAIN_OUT.name, "input_sha256": in_hash, "config_hash": cfg_hash})
    log.info("Tuning concluído: %d ajustes, %.1f s", len(rows), t_all)
    return out


def build_report(tun: dict, rows: list[dict]) -> str:
    probe = json.loads(PROBE_OUT.read_text(encoding="utf-8")) if PROBE_OUT.is_file() else None
    L = ["# ETAPA 09 — SVR (RBF), estratégia direta", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Configuração", "",
         "- Um modelo por horizonte; atributos da ETAPA 06 (17) calculados na origem t, alvo y(t + h).",
         "- Padronização dos atributos e do alvo ajustada somente nos dados de ajuste (Pipeline + TransformedTargetRegressor).",
         "- ε em desvios-padrão do alvo; cache do kernel 200 MB; 1 thread (libsvm).", ""]
    if probe:
        L += ["## Sondagem com parâmetros padrão (C = 1, γ = scale, ε = 0,1; validação interna)", "",
              "| h | n ajuste | RMSE val | t ajuste (s) | t previsão (s) | vetores de suporte |", "|---:|---:|---:|---:|---:|---:|"]
        for r in probe["linhas"]:
            L.append(f"| {r['h']} | {r['n_ajuste']} | {r['rmse_val']:.4f} | {r['t_fit_s']:.1f} | {r['t_pred_s']:.2f} | "
                     f"{r['n_vetores_suporte']} |")
        L.append("")
    L += ["## Tuning controlado (somente treino)", "",
          f"Validação interna: alvos a partir de {tun['corte_interno']}; γ_scale = {tun['gamma_scale']:.4f}. "
          f"{tun['n_ajustes']} ajustes; custo total (*tuning*, à parte): **{tun['custo_tuning_s']:.1f} s**.", "",
          "| h | Fase | C | γ | ε | RMSE val | MAE val | SV | t ajuste (s) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(tun["candidatos"], key=lambda r: (r["h"], r["rmse_val"])):
        hp = r["hp"]
        L.append(f"| {r['h']} | {r['fase']} | {hp['C']:g} | {hp['gamma']:.4f} | {hp['epsilon']:g} | {r['rmse_val']:.4f} | "
                 f"{r['mae_val']:.4f} | {r['n_vetores_suporte']} | {r['t_fit_s']:.1f} |")
    L += ["", "## Escolhidos e resultados no teste (m)", "",
          "| h | C | γ | ε | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino | t_treino (s) | t_previsão (s) | Memória (MB) |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        hp = tun["escolhidos"][str(r["horizonte_h"])]
        L.append(f"| {r['horizonte_h']} | {hp['C']:g} | {hp['gamma']:.4f} | {hp['epsilon']:g} | {r['mae_treino']:.4f} | "
                 f"{r['rmse_treino']:.4f} | {r['mae_teste']:.4f} | {r['rmse_teste']:.4f} | "
                 f"{r['rmse_teste'] / r['rmse_treino']:.3f} | {r['t_treino_s']:.2f} [{r['t_treino_min_s']:.2f}–"
                 f"{r['t_treino_max_s']:.2f}] | {r['t_previsao_s']:.2f} | {r['memoria_pico_mb']:.1f} |")
    L += ["", "Tempos: mediana de 3 repetições [mín–máx]; um ajuste por horizonte (estratégia direta).", ""]
    return "\n".join(L)


def run_final(log, force: bool) -> None:
    from src.evaluation.runner import run_experiment

    cfg = load_config()
    tun = run_tuning(log, force=False)
    spec = load_feature_spec()
    data = load_eval_data()
    rows = []
    for h in cfg["models"]["horizons_hours"]:
        hp = tun["escolhidos"][str(h)]
        rows.append(run_experiment(lambda hp=hp: svr_forecaster(hp, spec), "svr", h, data, CONFIG_KEYS,
                                   strategy="direta", log=log, force=force,
                                   extra_hash={"hp": json.dumps(hp, sort_keys=True), "spec": json.dumps(spec, sort_keys=True)}))
    (REPORTS_DIR / "09_svr.md").write_text(build_report(tun, rows), encoding="utf-8")
    log.info("Relatório: %s", REPORTS_DIR / "09_svr.md")


def main(probe: bool, tune: bool, force: bool) -> None:
    log = get_logger("svr", run_name="09_svr" + ("_sondagem" if probe else "_tuning" if tune else ""))
    if probe:
        run_probe(log)
    elif tune:
        run_tuning(log, force)
    else:
        run_final(log, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.probe, a.tune, a.force)
