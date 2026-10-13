"""Medição de custo computacional (ETAPA 13) — medida, nunca estimada.

- Tempo: time.perf_counter, `n_repeats` execuções independentes; reporta mediana, mín e máx.
- Memória: pico de RSS do processo (psutil) acima do nível inicial, amostrado por uma thread
  em execução SEPARADA da cronometragem (a amostragem não contamina os tempos).
  RSS inclui memória nativa (LightGBM, libsvm), que tracemalloc não enxerga.
"""
from __future__ import annotations

import gc
import statistics
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil


class PeakRSS:
    """Context manager: pico de RSS (MB) acima do RSS na entrada."""

    def __init__(self, interval_s: float = 0.005):
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.base_mb = self.peak_mb = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            time.sleep(self.interval_s)

    def __enter__(self):
        gc.collect()
        self.base_mb = self.peak_mb = self._proc.memory_info().rss / 2**20
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.base_mb)


def measure_shared(fit: Callable[[], Any], predicts: dict[int, Callable[[Any], Any]], n_repeats: int,
                   mem_interval_s: float) -> dict[int, dict[str, Any]]:
    """Um ajuste serve a todos os horizontes (ex.: SARIMA recursivo).

    Em cada repetição: ajusta uma vez e prevê cada horizonte. O tempo de treino de cada
    horizonte é o do ajuste completo (o modelo precisa ser treinado para prever aquele h).
    Memória (execução separada): max(pico do ajuste, pico da previsão de h) − RSS inicial.
    """
    t_fit: list[float] = []
    t_pred: dict[int, list[float]] = {h: [] for h in predicts}
    model = None
    preds: dict[int, Any] = {}
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        model = fit()
        t_fit.append(time.perf_counter() - t0)
        for h, predict in predicts.items():
            t1 = time.perf_counter()
            preds[h] = predict(model)
            t_pred[h].append(time.perf_counter() - t1)
    del model
    gc.collect()
    proc = psutil.Process()
    base = proc.memory_info().rss / 2**20
    with PeakRSS(mem_interval_s) as mem_fit:
        model = fit()
    mem_h, preds_mem = {}, {}
    for h, predict in predicts.items():
        with PeakRSS(mem_interval_s) as m:
            preds_mem[h] = predict(model)
        mem_h[h] = max(0.0, max(mem_fit.peak_mb, m.peak_mb) - base)

    out = {}
    for h in predicts:
        tot = [a + b for a, b in zip(t_fit, t_pred[h])]
        out[h] = {
            "t_treino_s": statistics.median(t_fit), "t_treino_min_s": min(t_fit), "t_treino_max_s": max(t_fit),
            "t_previsao_s": statistics.median(t_pred[h]), "t_previsao_min_s": min(t_pred[h]),
            "t_previsao_max_s": max(t_pred[h]),
            "t_total_s": statistics.median(tot), "t_total_min_s": min(tot), "t_total_max_s": max(tot),
            "memoria_pico_mb": mem_h[h], "n_repeats": n_repeats,
            "model": model, "pred": preds[h], "pred_mem_run": preds_mem[h],
        }
    return out


def measure(fit: Callable[[], Any], predict: Callable[[Any], Any], n_repeats: int,
            mem_interval_s: float) -> dict[str, Any]:
    """Executa fit→predict `n_repeats` vezes (tempos) + 1 vez com amostragem de memória.

    `fit()` devolve um modelo novo e ajustado; `predict(model)` devolve as previsões.
    Devolve tempos (mediana/mín/máx), memória e o último modelo/previsão (para métricas).
    """
    t_fit, t_pred = [], []
    model = pred = None
    for _ in range(n_repeats):
        gc.collect()
        t0 = time.perf_counter()
        model = fit()
        t1 = time.perf_counter()
        pred = predict(model)
        t2 = time.perf_counter()
        t_fit.append(t1 - t0)
        t_pred.append(t2 - t1)
    del model
    gc.collect()
    with PeakRSS(mem_interval_s) as mem:
        model = fit()
        pred_mem = predict(model)
    t_total = [a + b for a, b in zip(t_fit, t_pred)]
    return {
        "t_treino_s": statistics.median(t_fit), "t_treino_min_s": min(t_fit), "t_treino_max_s": max(t_fit),
        "t_previsao_s": statistics.median(t_pred), "t_previsao_min_s": min(t_pred), "t_previsao_max_s": max(t_pred),
        "t_total_s": statistics.median(t_total), "t_total_min_s": min(t_total), "t_total_max_s": max(t_total),
        "memoria_pico_mb": mem.delta_mb, "n_repeats": n_repeats,
        "model": model, "pred": pred, "pred_mem_run": pred_mem,
    }
