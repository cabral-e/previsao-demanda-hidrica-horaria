"""Testes da fronteira de Pareto."""
import pandas as pd

from src.evaluation.pareto import dominates, pareto


def _pts():
    # rápido-ruim, médio-bom, lento-pior (dominado), empate de tempo dentro do ruído
    return pd.DataFrame({
        "modelo": ["persistencia", "lightgbm", "sarima", "lightgbm_rec"],
        "err": [0.5, 0.3, 0.4, 0.3],
        "t": [0.001, 1.0, 300.0, 1.05], "t_min": [0.001, 0.95, 290.0, 0.99], "t_max": [0.001, 1.06, 310.0, 1.1],
    })


def test_dominance_rules():
    p = _pts()
    assert dominates(p.iloc[1], p.iloc[2], "err", "t")          # melhor e mais rápido
    assert not dominates(p.iloc[0], p.iloc[1], "err", "t")      # mais rápido, mas pior
    assert not dominates(p.iloc[1], p.iloc[3], "err", "t")      # mesmo erro, tempo dentro do ruído → empate


def test_identical_forecasters_do_not_dominate_each_other():
    p = pd.DataFrame({"modelo": ["lightgbm", "lightgbm_rec"], "rmse_teste": [0.08, 0.08], "mae_teste": [0.05, 0.05],
                      "t": [2.85, 2.75], "t_min": [2.83, 2.74], "t_max": [2.93, 2.81]})
    assert pareto(p, "rmse_teste", "t")["pareto"].all()


def test_pareto_front():
    res = pareto(_pts(), "err", "t")
    front = set(res.loc[res.pareto, "modelo"])
    assert front == {"persistencia", "lightgbm", "lightgbm_rec"}
    assert "LightGBM (direto)" in res.loc[res.modelo == "sarima", "dominado_por"].iloc[0]
