"""Executor comum de experimentos: ajuste, previsão, custo, métricas, persistência e registro."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import sha256
from src.data.split import TEST_OUT, TRAIN_OUT
from src.evaluation import registry
from src.evaluation.cost import measure, measure_shared
from src.evaluation.metrics import mae, rmse, validate_predictions
from src.evaluation.protocol import EvalData, origins
from src.models.base import Forecaster
from src.utils.paths import PREDICTIONS_DIR, load_config
from src.utils.persistence import config_hash


def experiment_hash(config_keys: tuple[str, ...], extra: dict | None = None) -> str:
    """Hash da configuração relevante + dados de entrada + parâmetros específicos."""
    parts = [config_hash(*config_keys), sha256(TRAIN_OUT), sha256(TEST_OUT), repr(sorted((extra or {}).items()))]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _finalize(res: dict[str, Any], modelo: str, h: int, data: EvalData, strategy: str, cfg_h: str, log) -> dict:
    """Valida, calcula métricas de treino/teste, salva previsões e registra o experimento."""
    cfg = load_config()
    model, pred_test = res.pop("model"), np.asarray(res.pop("pred"), float)
    if not np.allclose(pred_test, np.asarray(res.pop("pred_mem_run"), float)):
        raise AssertionError(f"{modelo} h={h}: previsões diferentes entre execuções (não determinístico)")

    pred_train = np.asarray(model.predict(data.y_train, data.train_targets, h), float)
    y_test = data.y_full.reindex(data.test_targets).to_numpy()
    y_train = data.y_train.reindex(data.train_targets).to_numpy()
    validate_predictions(y_test, pred_test)
    validate_predictions(y_train, pred_train)
    if len(pred_test) != len(data.test_targets):
        raise AssertionError("Número de previsões de teste diferente do esperado")

    versao = registry.next_version(modelo, h)
    exp_id = registry.make_exp_id(modelo, h, versao)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([
        pd.DataFrame({"conjunto": "treino", "ts_alvo": data.train_targets,
                      "origem": origins(data.train_targets, h), "y": y_train, "y_prev": pred_train}),
        pd.DataFrame({"conjunto": "teste", "ts_alvo": data.test_targets,
                      "origem": origins(data.test_targets, h), "y": y_test, "y_prev": pred_test}),
    ], ignore_index=True).to_parquet(PREDICTIONS_DIR / f"{exp_id}.parquet", index=False)

    row = {
        "exp_id": exp_id, "timestamp": datetime.now().isoformat(timespec="seconds"), "modelo": modelo,
        "horizonte_h": h, "versao": versao, "hiperparametros": model.params(), "seed": cfg["project"]["seed"],
        "n_jobs": cfg["cost"]["n_jobs"], "n_treino": len(data.y_train), "n_teste": len(data.test_targets),
        "mae_treino": mae(y_train, pred_train), "rmse_treino": rmse(y_train, pred_train),
        "mae_teste": mae(y_test, pred_test), "rmse_teste": rmse(y_test, pred_test),
        **res, "estrategia": strategy, "status": "ok", "commit_ou_hash_config": cfg_h,
    }
    registry.append(row)
    log.info("%s h=%d [%s]: MAE teste %.4f | RMSE teste %.4f | t_treino %.3fs | t_prev %.3fs | mem %.1f MB",
             modelo, h, exp_id, row["mae_teste"], row["rmse_teste"], row["t_treino_s"], row["t_previsao_s"],
             row["memoria_pico_mb"])
    return row


def run_experiment(factory: Callable[[], Forecaster], modelo: str, h: int, data: EvalData,
                   config_keys: tuple[str, ...], strategy: str, log, force: bool = False,
                   extra_hash: dict | None = None) -> dict[str, Any]:
    """Um ajuste por horizonte (ex.: baselines, estratégia direta)."""
    cfg = load_config()
    cfg_h = experiment_hash(config_keys, {"h": h, "modelo": modelo, **(extra_hash or {})})
    if not force:
        existing = registry.find(modelo, h, cfg_h)
        if existing is not None:
            log.info("%s h=%d: reutilizando %s", modelo, h, existing["exp_id"])
            return existing.to_dict()
    res = measure(lambda: factory().fit(data.y_train, h),
                  lambda m: m.predict(data.y_full, data.test_targets, h),
                  cfg["cost"]["n_repeats"], cfg["evaluation"]["memory_sampling_interval_s"])
    return _finalize(res, modelo, h, data, strategy, cfg_h, log)


def run_shared_experiment(factory: Callable[[], Forecaster], modelo: str, horizons: list[int], data: EvalData,
                          config_keys: tuple[str, ...], strategy: str, log, force: bool = False,
                          extra_hash: dict | None = None) -> list[dict[str, Any]]:
    """Um único ajuste serve a todos os horizontes (ex.: SARIMA recursivo)."""
    cfg = load_config()
    hashes = {h: experiment_hash(config_keys, {"h": h, "modelo": modelo, **(extra_hash or {})}) for h in horizons}
    if not force:
        found = {h: registry.find(modelo, h, hashes[h]) for h in horizons}
        if all(v is not None for v in found.values()):
            for h, v in found.items():
                log.info("%s h=%d: reutilizando %s", modelo, h, v["exp_id"])
            return [v.to_dict() for v in found.values()]
    predicts = {h: (lambda m, h=h: m.predict(data.y_full, data.test_targets, h)) for h in horizons}
    res = measure_shared(lambda: factory().fit(data.y_train, max(horizons)), predicts,
                         cfg["cost"]["n_repeats"], cfg["evaluation"]["memory_sampling_interval_s"])
    return [_finalize(res[h], modelo, h, data, strategy, hashes[h], log) for h in horizons]
