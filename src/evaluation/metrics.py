"""Métricas de erro (ETAPA 12) e validação das previsões."""
from __future__ import annotations

import numpy as np


def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Tamanhos diferentes: {y_true.shape} × {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Previsões vazias")
    if not np.isfinite(y_pred).all():
        raise ValueError(f"Previsões com NaN/Inf: {int((~np.isfinite(y_pred)).sum())}")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE = (1/n) Σ |y_i − ŷ_i|"""
    validate_predictions(y_true, y_pred)
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = sqrt((1/n) Σ (y_i − ŷ_i)²)"""
    validate_predictions(y_true, y_pred)
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def newey_west_lag(n: int, h: int) -> int:
    """max(h − 1, ⌊4·(n/100)^(2/9)⌋) — cobre a sobreposição das origens e a autocorrelação residual."""
    return max(h - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int, loss: str = "squared") -> dict[str, float]:
    """Teste de Diebold-Mariano (1995) com variância HAC de Newey-West (Bartlett) e correção
    de Harvey, Leybourne & Newbold (1997). H0: mesma acurácia esperada. Bicaudal, t(n−1).

    d_t = L(e1_t) − L(e2_t): estatística negativa ⇒ modelo 1 tem perda menor.
    """
    from scipy import stats

    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if e1.shape != e2.shape:
        raise ValueError("Séries de erro com tamanhos diferentes")
    f = np.square if loss == "squared" else np.abs
    d = f(e1) - f(e2)
    n = d.size
    dbar = d.mean()
    dc = d - dbar
    L = newey_west_lag(n, h)
    var = np.dot(dc, dc) / n
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * np.dot(dc[k:], dc[:-k]) / n
    if var <= 0:
        return {"dm": 0.0, "p": 1.0, "lag_nw": L, "d_medio": float(dbar)}
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return {"dm": stat, "p": p, "lag_nw": L, "d_medio": float(dbar)}


def holm(pvalues: list[float]) -> list[float]:
    """p-valores ajustados por Holm-Bonferroni (controle do FWER)."""
    p = np.asarray(pvalues, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
