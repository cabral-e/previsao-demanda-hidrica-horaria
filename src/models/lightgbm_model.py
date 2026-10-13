"""ETAPA 10 — LightGBM, estratégia direta.

Tuning controlado por horizonte na validação cronológica interna do treino: grade pequena
(num_leaves × min_child_samples × reg_lambda) com learning_rate fixo e número de árvores por
parada antecipada NA VALIDAÇÃO INTERNA. O ajuste final no treino inteiro usa esse número de
árvores, sem parada antecipada (nenhum dado fora do ajuste é consultado).

Uso:
  python -m src.models.lightgbm_model --tune
  python -m src.models.lightgbm_model
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any

from lightgbm import LGBMRegressor, early_stopping

from src.data.load import sha256
from src.data.split import TRAIN_OUT, temporal_cut
from src.evaluation.metrics import mae, rmse
from src.evaluation.protocol import load_eval_data
from src.evaluation.runner import run_experiment
from src.features.build import feature_columns, select_by_target_time
from src.models.ml_direct import DirectForecaster, load_feature_spec
from src.utils.logging_utils import get_logger
from src.utils.paths import REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

TUNING_OUT = REPORTS_DIR / "10_lightgbm_tuning.json"
TUNING_KEYS = ("lightgbm", "features", "data.target", "split", "project.seed", "cost.n_jobs")
CONFIG_KEYS = ("lightgbm", "features", "data.target", "split", "evaluation", "cost", "project.seed")


def make_lgbm(hp: dict[str, Any], seed: int, n_jobs: int) -> LGBMRegressor:
    return LGBMRegressor(**hp, random_state=seed, n_jobs=n_jobs, verbose=-1)


def lgbm_forecaster(hp: dict[str, Any], spec: dict[str, Any], seed: int, n_jobs: int) -> DirectForecaster:
    return DirectForecaster("lightgbm", lambda: make_lgbm(hp, seed, n_jobs), spec, {**hp, "seed": seed, "n_jobs": n_jobs})


def run_tuning(log, force: bool) -> dict:
    cfg = load_config()
    lc, seed, n_jobs = cfg["lightgbm"], cfg["project"]["seed"], cfg["cost"]["n_jobs"]
    in_hash, cfg_hash = sha256(TRAIN_OUT), config_hash(*TUNING_KEYS)
    if not force and is_reusable(TUNING_OUT, input_sha256=in_hash, config_hash=cfg_hash):
        log.info("Tuning já realizado com mesma entrada e configuração; reutilizando.")
        return json.loads(TUNING_OUT.read_text(encoding="utf-8"))

    data = load_eval_data()
    spec = load_feature_spec()
    y = data.y_train
    inner_cut = temporal_cut(str(y.index.min()), str(y.index.max()), lc["inner_train_fraction"])
    t_all = time.perf_counter()
    rows, best = [], {}
    for h in cfg["models"]["horizons_hours"]:
        f = lgbm_forecaster({}, spec, seed, n_jobs)
        df = f._supervised(y, h)
        cols = feature_columns(df)
        fit_df, val_df = select_by_target_time(df, end_exclusive=inner_cut), select_by_target_time(df, start=inner_cut)
        cands = []
        g = lc["tuning"]
        for nl in g["num_leaves"]:
            for mcs in g["min_child_samples"]:
                for lam in g["reg_lambda"]:
                    hp = {**lc["fixed"], "num_leaves": nl, "min_child_samples": mcs, "reg_lambda": float(lam),
                          "n_estimators": lc["max_estimators"]}
                    est = make_lgbm(hp, seed, n_jobs)
                    t0 = time.perf_counter()
                    est.fit(fit_df[cols], fit_df["alvo"], eval_set=[(val_df[cols], val_df["alvo"])],
                            eval_metric="l2", callbacks=[early_stopping(lc["early_stopping_rounds"], verbose=False)])
                    t_fit = time.perf_counter() - t0
                    pred = est.predict(val_df[cols], num_iteration=est.best_iteration_)
                    r = {"h": h, "hp": hp, "best_iteration": int(est.best_iteration_),
                         "rmse_val": rmse(val_df["alvo"].to_numpy(), pred), "mae_val": mae(val_df["alvo"].to_numpy(), pred),
                         "t_fit_s": t_fit, "n_ajuste": len(fit_df), "n_val": len(val_df)}
                    cands.append(r)
                    log.info("h=%d leaves=%d mcs=%d λ=%g: RMSE %.4f | árvores %d | %.1fs", h, nl, mcs, lam,
                             r["rmse_val"], r["best_iteration"], t_fit)
        b = min(cands, key=lambda r: r["rmse_val"])
        best[h] = {**b["hp"], "n_estimators": b["best_iteration"]}
        rows += cands
        log.info("h=%d escolhido %s (RMSE val %.4f)", h, best[h], b["rmse_val"])
    t_all = time.perf_counter() - t_all
    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"), "corte_interno": str(inner_cut),
           "escolhidos": {str(h): v for h, v in best.items()}, "candidatos": rows,
           "custo_tuning_s": t_all, "n_ajustes": len(rows)}
    TUNING_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_meta(TUNING_OUT, {"etapa": "10", "input": TRAIN_OUT.name, "input_sha256": in_hash, "config_hash": cfg_hash})
    log.info("Tuning concluído: %d ajustes, %.1f s", len(rows), t_all)
    return out


def build_report(tun: dict, rows: list[dict], lc: dict) -> str:
    L = ["# ETAPA 10 — LightGBM, estratégia direta", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Configuração", "",
         "- Um modelo por horizonte; atributos da ETAPA 06 (17); objetivo L2; seed única; 1 thread; modo determinístico.",
         f"- Fixos: {json.dumps(lc['fixed'], ensure_ascii=False)}.",
         f"- Número de árvores: parada antecipada (paciência {lc['early_stopping_rounds']}, máx. {lc['max_estimators']}) "
         "somente na validação interna do treino; o ajuste final usa esse número, sem parada antecipada.", "",
         "## Tuning controlado (somente treino)", "",
         f"Validação interna: alvos a partir de {tun['corte_interno']}. {tun['n_ajustes']} ajustes; "
         f"custo total (*tuning*, à parte): **{tun['custo_tuning_s']:.1f} s**.", "",
         "| h | num_leaves | min_child_samples | λ (L2) | árvores | RMSE val | MAE val | t ajuste (s) |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(tun["candidatos"], key=lambda r: (r["h"], r["rmse_val"])):
        hp = r["hp"]
        L.append(f"| {r['h']} | {hp['num_leaves']} | {hp['min_child_samples']} | {hp['reg_lambda']:g} | "
                 f"{r['best_iteration']} | {r['rmse_val']:.4f} | {r['mae_val']:.4f} | {r['t_fit_s']:.1f} |")
    L += ["", "## Escolhidos e resultados no teste (m)", "",
          "| h | leaves | mcs | λ | árvores | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino | t_treino (s) | t_previsão (s) | Memória (MB) |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        hp = tun["escolhidos"][str(r["horizonte_h"])]
        L.append(f"| {r['horizonte_h']} | {hp['num_leaves']} | {hp['min_child_samples']} | {hp['reg_lambda']:g} | "
                 f"{hp['n_estimators']} | {r['mae_treino']:.4f} | {r['rmse_treino']:.4f} | {r['mae_teste']:.4f} | "
                 f"{r['rmse_teste']:.4f} | {r['rmse_teste'] / r['rmse_treino']:.3f} | {r['t_treino_s']:.2f} "
                 f"[{r['t_treino_min_s']:.2f}–{r['t_treino_max_s']:.2f}] | {r['t_previsao_s']:.3f} | {r['memoria_pico_mb']:.1f} |")
    L += ["", "Tempos: mediana de 3 repetições [mín–máx]; um ajuste por horizonte (estratégia direta).", ""]
    return "\n".join(L)


def run_final(log, force: bool) -> None:
    cfg = load_config()
    seed, n_jobs = cfg["project"]["seed"], cfg["cost"]["n_jobs"]
    tun = run_tuning(log, force=False)
    spec = load_feature_spec()
    data = load_eval_data()
    rows = []
    for h in cfg["models"]["horizons_hours"]:
        hp = tun["escolhidos"][str(h)]
        rows.append(run_experiment(lambda hp=hp: lgbm_forecaster(hp, spec, seed, n_jobs), "lightgbm", h, data,
                                   CONFIG_KEYS, strategy="direta", log=log, force=force,
                                   extra_hash={"hp": json.dumps(hp, sort_keys=True), "spec": json.dumps(spec, sort_keys=True)}))
    (REPORTS_DIR / "10_lightgbm.md").write_text(build_report(tun, rows, cfg["lightgbm"]), encoding="utf-8")
    log.info("Relatório: %s", REPORTS_DIR / "10_lightgbm.md")


def run_recursive(log, force: bool) -> list[dict]:
    """ETAPA 11 — variante recursiva: modelo de 1 passo (hiperparâmetros escolhidos para h = 1) iterado."""
    from src.evaluation.runner import run_shared_experiment
    from src.models.ml_recursive import RecursiveForecaster

    cfg = load_config()
    seed, n_jobs = cfg["project"]["seed"], cfg["cost"]["n_jobs"]
    tun = run_tuning(log, force=False)
    spec = load_feature_spec()
    hp = tun["escolhidos"]["1"]
    factory = lambda: RecursiveForecaster(lgbm_forecaster(hp, spec, seed, n_jobs), "lightgbm_rec")  # noqa: E731
    return run_shared_experiment(factory, "lightgbm_rec", cfg["models"]["horizons_hours"], load_eval_data(),
                                 CONFIG_KEYS, strategy="recursiva", log=log, force=force,
                                 extra_hash={"hp": json.dumps(hp, sort_keys=True), "spec": json.dumps(spec, sort_keys=True)})


def main(tune: bool, force: bool, recursive: bool = False) -> None:
    log = get_logger("lightgbm", run_name="10_lightgbm" + ("_tuning" if tune else "_recursivo" if recursive else ""))
    if tune:
        run_tuning(log, force)
    elif recursive:
        run_recursive(log, force)
    else:
        run_final(log, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--recursive", action="store_true", help="variante recursiva (ETAPA 11)")
    a = ap.parse_args()
    main(a.tune, a.force, a.recursive)
