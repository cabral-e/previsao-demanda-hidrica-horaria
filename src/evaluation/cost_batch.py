"""ETAPA 13 — Remedição controlada de custos (lote único, mesmo estado da máquina).

Reexecuta o ajuste final e a previsão de TODOS os modelos com --force (novas versões no registro;
as anteriores são preservadas). O tuning NÃO é refeito (reutilizado). Registra o estado da máquina
antes e depois de cada modelo.

Uso: python -m src.evaluation.cost_batch
Saída: results/reports/13_estado_maquina.json + novas linhas em registry.csv
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import psutil

from src.utils.logging_utils import get_logger
from src.utils.paths import REPORTS_DIR

STATE_OUT = REPORTS_DIR / "13_estado_maquina.json"


def machine_state() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    bat = psutil.sensors_battery()
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["name"], p.info["memory_info"].rss / 2**20))
        except (psutil.Error, TypeError):
            continue
    procs.sort(key=lambda x: -x[1])
    return {"ts": datetime.now().isoformat(timespec="seconds"),
            "ram_livre_gb": round(vm.available / 2**30, 2), "ram_total_gb": round(vm.total / 2**30, 2),
            "cpu_percent_1s": psutil.cpu_percent(interval=1.0),
            "cpu_freq_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None,
            "na_tomada": None if bat is None else bool(bat.power_plugged),
            "bateria_pct": None if bat is None else bat.percent,
            "n_processos": len(procs), "top_processos_mb": [(n, round(m)) for n, m in procs[:8]]}


def main() -> None:
    log = get_logger("cost_batch", run_name="13_remedicao")
    from src.models import baseline, lightgbm_model, sarima, svr

    steps = [
        ("baselines", lambda: baseline.main(force=True)),
        ("lightgbm", lambda: lightgbm_model.run_final(log, force=True)),
        ("lightgbm_rec", lambda: lightgbm_model.run_recursive(log, force=True)),
        ("svr", lambda: svr.run_final(log, force=True)),
        ("sarima", lambda: sarima.main(select=False, force=True)),
    ]
    states = {"inicio": machine_state(), "etapas": []}
    log.info("Estado inicial: %s", states["inicio"])
    for name, fn in steps:
        before = machine_state()
        t0 = time.perf_counter()
        fn()
        states["etapas"].append({"modelo": name, "antes": before, "depois": machine_state(),
                                 "duracao_lote_s": time.perf_counter() - t0})
        log.info("%s concluído em %.1f s (antes: RAM livre %.2f GB, CPU %.0f%%, tomada %s)", name,
                 states["etapas"][-1]["duracao_lote_s"], before["ram_livre_gb"], before["cpu_percent_1s"],
                 before["na_tomada"])
        STATE_OUT.write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")
    states["fim"] = machine_state()
    STATE_OUT.write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Remedição concluída.")


if __name__ == "__main__":
    main()
