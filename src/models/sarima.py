"""ETAPA 08 — SARIMA com calendário semanal exógeno.

Modelo: SARIMA(p,0,q)(P,1,Q)24 com regressão no calendário semanal, implementado com
diferença sazonal explícita: z(t) = y(t) − y(t−24), ajustando SARIMA(p,0,q)(P,0,Q)24 em z
com as exógenas também diferenciadas (formulação equivalente, estado ~2× menor).

Previsão (estratégia recursiva, natural do modelo): parâmetros fixos do treino; o filtro de
Kalman é atualizado com as observações reais até cada origem t, e a previsão h passos à frente é
ẑ(t+h|t) = Z·T^(h−1)·a(t+1|t) + d(t+h), com d = β'Δx (exógena determinística).
Nível: ŷ(T) = ẑ(T) + y(T−24), conhecido na origem para h ≤ 24.

Uso:
  python -m src.models.sarima --select   seleção da ordem e da exógena (últimas 26 semanas do treino)
  python -m src.models.sarima            ajuste final + avaliação (registro, previsões, custo)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace import kalman_filter as kf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.load import sha256
from src.data.split import TRAIN_OUT
from src.evaluation.protocol import load_eval_data
from src.evaluation.runner import run_shared_experiment
from src.utils.logging_utils import get_logger
from src.utils.paths import REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

SEASON = 24
SELECTION_OUT = REPORTS_DIR / "08_sarima_selecao.json"
SELECTION_KEYS = ("sarima", "data.target", "split")
CONFIG_KEYS = ("sarima", "data.target", "split", "evaluation", "cost", "project.seed")
# Na previsão só precisamos das médias preditas do estado e das variáveis do sistema.
FILTER_MEMORY = (kf.MEMORY_NO_FILTERED | kf.MEMORY_NO_PREDICTED_COV | kf.MEMORY_NO_FORECAST_COV
                 | kf.MEMORY_NO_SMOOTHING | kf.MEMORY_NO_GAIN | kf.MEMORY_NO_STD_FORECAST)


# ======================================================================= exógenas
def weekly_exog(index: pd.DatetimeIndex, kind: str, K: int | None = None) -> pd.DataFrame:
    """Calendário semanal determinístico (conhecido de antemão)."""
    if kind == "fourier":
        how = index.dayofweek * 24 + index.hour                     # hora da semana 0..167
        cols = {}
        for k in range(1, K + 1):
            ang = 2 * np.pi * k * how / 168
            cols[f"sem_sin{k}"], cols[f"sem_cos{k}"] = np.sin(ang), np.cos(ang)
        return pd.DataFrame(cols, index=index)
    if kind == "dow":
        return pd.DataFrame({f"dow_{d}": (index.dayofweek == d).astype(float) for d in range(1, 7)}, index=index)
    raise ValueError(f"Exógena desconhecida: {kind}")


def exog_label(spec: dict) -> str:
    return f"fourier_K{spec['K']}" if spec["kind"] == "fourier" else spec["kind"]


def seasonal_diff(y: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    z, Xd = y.diff(SEASON), X.diff(SEASON)
    return z.iloc[SEASON:], Xd.iloc[SEASON:]


# ======================================================================= modelo
class SarimaForecaster:
    name = "sarima"

    def __init__(self, order: tuple[int, int, int], seasonal: tuple[int, int, int], exog: dict, maxiter: int = 200):
        self.order, self.seasonal, self.exog, self.maxiter = tuple(order), tuple(seasonal), dict(exog), maxiter
        self.params_: np.ndarray | None = None
        self.converged_: bool | None = None

    def _model(self, y: pd.Series) -> SARIMAX:
        X = weekly_exog(y.index, self.exog["kind"], self.exog.get("K"))
        z, Xd = seasonal_diff(y, X)
        return SARIMAX(z.to_numpy(), exog=Xd.to_numpy(), order=self.order,
                       seasonal_order=(*self.seasonal, SEASON), trend="n",
                       enforce_stationarity=True, enforce_invertibility=True)

    def fit(self, y_train: pd.Series, h: int = 1) -> "SarimaForecaster":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            res = self._model(y_train).fit(method="lbfgs", maxiter=self.maxiter, disp=False, low_memory=True)
        self.params_ = np.asarray(res.params)
        self.param_names_ = list(res.param_names)
        self.converged_ = bool(res.mle_retvals.get("converged", False)) and \
            not any(issubclass(i.category, ConvergenceWarning) for i in w)
        self.llf_, self.aic_, self.bic_ = float(res.llf), float(res.aic), float(res.bic)
        return self

    def predict(self, y_hist: pd.Series, targets: pd.DatetimeIndex, h: int) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Modelo não ajustado")
        if not 1 <= h <= SEASON:
            raise ValueError(f"h = {h} fora de [1, {SEASON}] (reconstrução usa y(T−24))")
        mod = self._model(y_hist)
        mod.ssm.set_conserve_memory(FILTER_MEMORY)
        fres = mod.filter(self.params_, return_ssm=True)            # parâmetros fixos (sem reajuste)
        a_pred = np.asarray(fres.predicted_state)                    # a(i|i−1), colunas 0..n
        mod.update(self.params_)
        Z = np.asarray(mod.ssm["design"])[:, :, 0] if mod.ssm["design"].ndim == 3 else np.asarray(mod.ssm["design"])
        T = np.asarray(mod.ssm["transition"])[:, :, 0] if mod.ssm["transition"].ndim == 3 else np.asarray(mod.ssm["transition"])
        d = np.asarray(mod.ssm["obs_intercept"])
        d = d[0] if d.ndim == 2 else np.full(a_pred.shape[1], float(d[0]))

        z_index = y_hist.index[SEASON:]                              # índice de z / colunas do filtro
        pos = z_index.get_indexer(targets - pd.Timedelta(hours=h))   # posição da origem t em z
        if (pos < 0).any():
            raise ValueError("Origem sem histórico suficiente (precisa de ≥ 24 h antes)")
        M = Z @ np.linalg.matrix_power(T, h - 1)                     # 1 × m
        z_hat = (M @ a_pred[:, pos + 1]).ravel() + d[pos + h]
        y_lag = y_hist.reindex(targets - pd.Timedelta(hours=SEASON)).to_numpy(float)
        return z_hat + y_lag

    def params(self) -> dict:
        return {"order": list(self.order), "seasonal_order": [*self.seasonal, SEASON], "exog": exog_label(self.exog),
                "diferenciacao": "sazonal explícita (D=1)", "converged": self.converged_,
                "coef": dict(zip(getattr(self, "param_names_", []), np.round(self.params_, 6).tolist()))
                if self.params_ is not None else {}}


# ======================================================================= seleção
def ljung_box(model: SarimaForecaster, y: pd.Series, lags: list[int]) -> dict[str, float]:
    mod = model._model(y)
    res = mod.filter(model.params_)
    resid = np.asarray(res.forecasts_error[0])[2 * SEASON:]          # descarta aquecimento do filtro
    lb = acorr_ljungbox(resid, lags=lags)
    return {f"lb_p_{k}": float(lb.loc[k, "lb_pvalue"]) for k in lags}


def fit_candidate(y: pd.Series, order, seasonal, exog: dict, maxiter: int, lb_lags: list[int]) -> dict[str, Any]:
    m = SarimaForecaster(order, seasonal, exog, maxiter)
    t0 = time.perf_counter()
    try:
        m.fit(y)
        ok, err = True, None
    except Exception as exc:  # noqa: BLE001 — registrar falha e seguir
        ok, err = False, repr(exc)
    elapsed = time.perf_counter() - t0
    row = {"order": list(order), "seasonal": list(seasonal), "exog": exog_label(exog), "exog_spec": exog,
           "tempo_s": elapsed, "ok": ok, "erro": err}
    if ok:
        row |= {"aic": m.aic_, "bic": m.bic_, "llf": m.llf_, "converged": m.converged_,
                "n_params": len(m.params_), **ljung_box(m, y, lb_lags)}
    return row


def run_selection(log, force: bool) -> dict:
    cfg = load_config()
    sc = cfg["sarima"]
    in_hash, cfg_hash = sha256(TRAIN_OUT), config_hash(*SELECTION_KEYS)
    if not force and is_reusable(SELECTION_OUT, input_sha256=in_hash, config_hash=cfg_hash):
        log.info("Seleção já realizada com mesma entrada e configuração; reutilizando.")
        return json.loads(SELECTION_OUT.read_text(encoding="utf-8"))

    data = load_eval_data()
    n_sel = sc["selection_weeks"] * 168
    y_sel = data.y_train.iloc[-(n_sel + SEASON):]                    # +24 h para a diferença sazonal
    log.info("Seleção em %s a %s (%d h, somente treino)", y_sel.index[SEASON], y_sel.index[-1], n_sel)

    crit, lb_lags, maxiter = sc["criterion"], sc["ljungbox_lags"], sc["maxiter"]
    t_all = time.perf_counter()
    order_rows = []
    c = sc["candidates"]
    for p in c["p"]:
        for q in c["q"]:
            for P in c["P"]:
                for Q in c["Q"]:
                    r = fit_candidate(y_sel, (p, 0, q), (P, 0, Q), sc["exog_for_order_selection"], maxiter, lb_lags)
                    order_rows.append(r)
                    log.info("(%d,0,%d)(%d,1,%d)24 %s: %s", p, q, P, Q, r["exog"],
                             f"BIC {r['bic']:.1f} AIC {r['aic']:.1f} conv {r['converged']} {r['tempo_s']:.1f}s"
                             if r["ok"] else f"FALHOU {r['erro']}")
    valid = [r for r in order_rows if r["ok"] and r["converged"]]
    if not valid:
        raise RuntimeError("Nenhum candidato convergiu")
    best = min(valid, key=lambda r: r[crit])

    exog_rows = []
    for spec in sc["exog_candidates"]:
        if exog_label(spec) == best["exog"]:
            exog_rows.append(best)
            continue
        r = fit_candidate(y_sel, tuple(best["order"]), tuple(best["seasonal"]), spec, maxiter, lb_lags)
        exog_rows.append(r)
        log.info("exógena %s: %s", r["exog"], f"BIC {r['bic']:.1f} conv {r['converged']}" if r["ok"] else r["erro"])
    valid_ex = [r for r in exog_rows if r["ok"] and r["converged"]]
    best_ex = min(valid_ex, key=lambda r: r[crit])
    t_all = time.perf_counter() - t_all

    out = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
           "janela_selecao": [str(y_sel.index[SEASON]), str(y_sel.index[-1])], "n_selecao": n_sel,
           "criterio": crit, "candidatos_ordem": order_rows, "candidatos_exog": exog_rows,
           "escolhido": {"order": best_ex["order"], "seasonal": best_ex["seasonal"], "exog": best_ex["exog_spec"]},
           "custo_tuning_s": t_all}
    SELECTION_OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_meta(SELECTION_OUT, {"etapa": "08", "input": TRAIN_OUT.name, "input_sha256": in_hash, "config_hash": cfg_hash})
    log.info("Escolhido: %s; custo total da seleção %.1f s", out["escolhido"], t_all)
    return out


# ======================================================================= relatório
def _cand_table(rows: list[dict], crit: str) -> list[str]:
    L = ["| Ordem | Exógena | AIC | BIC | Conv. | LB p (24) | LB p (48) | Parâm. | Tempo (s) |",
         "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: r.get(crit, np.inf)):
        o, s = r["order"], r["seasonal"]
        name = f"({o[0]},0,{o[2]})({s[0]},1,{s[2]})24"
        if not r["ok"]:
            L.append(f"| {name} | {r['exog']} | — | — | falhou | — | — | — | {r['tempo_s']:.1f} |")
            continue
        L.append(f"| {name} | {r['exog']} | {r['aic']:.1f} | {r['bic']:.1f} | {'sim' if r['converged'] else 'não'} | "
                 f"{r['lb_p_24']:.3g} | {r['lb_p_48']:.3g} | {r['n_params']} | {r['tempo_s']:.1f} |")
    return L


def _readable_coef(coef: dict, exog: dict) -> dict:
    """x1..xk (nomes do statsmodels para exógenas sem rótulo) → nomes das colunas do calendário."""
    cols = list(weekly_exog(pd.DatetimeIndex(["2019-01-07"]), exog["kind"], exog.get("K")).columns)
    return {(f"Δ24 {cols[int(k[1:]) - 1]}" if k.startswith("x") and k[1:].isdigit() else k): v for k, v in coef.items()}


def build_report(sel: dict, rows: list[dict], final_params: dict) -> str:
    e = sel["escolhido"]
    o, s = e["order"], e["seasonal"]
    final_params = {**final_params, "coef": _readable_coef(final_params.get("coef", {}), e["exog"])}
    L = ["# ETAPA 08 — SARIMA", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Justificativa (EDA, somente treino)", "",
         "- Nível estacionário (ADF e KPSS concordam) → d = 0.",
         "- Sazonalidade diária dominante (ACF em 24/48/72 h; 32% da potência espectral em 24 h; força 0,66) → s = 24.",
         "- Diferenciação sazonal D = 1 (decisão do usuário).",
         "- Efeito semanal relevante (força 0,46) → calendário semanal como exógena (SARIMA com s = 168 é inviável).", "",
         "## Seleção controlada (somente treino)", "",
         f"Janela: {sel['janela_selecao'][0]} a {sel['janela_selecao'][1]} ({sel['n_selecao']} h). "
         f"Critério: menor {sel['criterio'].upper()} entre os que convergiram. "
         f"Custo total da seleção (*tuning*, registrado à parte): **{sel['custo_tuning_s']:.1f} s**.", "",
         "### Ordem (exógena fixa: Fourier K = 2)", "", *_cand_table(sel["candidatos_ordem"], sel["criterio"]), "",
         "### Exógena semanal (ordem escolhida)", "", *_cand_table(sel["candidatos_exog"], sel["criterio"]), "",
         "Ljung-Box com ~4.400 resíduos tem poder muito alto: rejeições indicam autocorrelação residual "
         "remanescente, mas não invalidam a comparação relativa entre candidatos.", "",
         f"## Modelo final: SARIMA({o[0]},0,{o[2]})({s[0]},1,{s[2]})24 + {exog_label(e['exog'])}", "",
         "Ajustado uma vez no treino inteiro (14.016 h); parâmetros fixos no teste; filtro atualizado com as "
         "observações até cada origem; previsões de 1, 6, 12 e 24 h a partir do mesmo ajuste.", "",
         f"Convergência: {final_params.get('converged')}.", "", "Coeficientes:", "",
         "| Parâmetro | Valor |", "|---|---:|", *[f"| {k} | {v:.6f} |" for k, v in final_params.get("coef", {}).items()], "",
         "## Resultados (m)", "",
         "| h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino | t_treino (s) | t_previsão (s) | Memória (MB) |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['horizonte_h']} | {r['mae_treino']:.4f} | {r['rmse_treino']:.4f} | {r['mae_teste']:.4f} | "
                 f"{r['rmse_teste']:.4f} | {r['rmse_teste'] / r['rmse_treino']:.3f} | {r['t_treino_s']:.2f} "
                 f"[{r['t_treino_min_s']:.2f}–{r['t_treino_max_s']:.2f}] | {r['t_previsao_s']:.3f} | "
                 f"{r['memoria_pico_mb']:.1f} |")
    L += ["", "Tempos: mediana de 3 repetições [mín–máx]. O t_treino é o mesmo ajuste para todos os h "
          "(um modelo serve aos 4 horizontes).", ""]
    return "\n".join(L)


def main(select: bool, force: bool) -> None:
    log = get_logger("sarima", run_name="08_sarima" + ("_selecao" if select else ""))
    sel = run_selection(log, force and select)
    if select:
        return
    cfg = load_config()
    e = sel["escolhido"]
    factory = lambda: SarimaForecaster(tuple(e["order"]), tuple(e["seasonal"]), e["exog"], cfg["sarima"]["maxiter"])  # noqa: E731
    data = load_eval_data()
    rows = run_shared_experiment(factory, "sarima", cfg["models"]["horizons_hours"], data, CONFIG_KEYS,
                                 strategy="recursiva (Kalman, parâmetros fixos)", log=log, force=force,
                                 extra_hash={"escolhido": json.dumps(e, sort_keys=True)})
    hp = rows[0]["hiperparametros"]
    hp = json.loads(hp) if isinstance(hp, str) else hp
    (REPORTS_DIR / "08_sarima.md").write_text(build_report(sel, rows, hp), encoding="utf-8")
    log.info("Relatório: %s", REPORTS_DIR / "08_sarima.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.select, a.force)
