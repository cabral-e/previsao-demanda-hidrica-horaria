"""ETAPA 11 — Horizontes e estratégias multi-step: documentação e verificação de justiça.

Verifica, a partir dos arquivos de previsão, que todos os modelos usam exatamente os mesmos
alvos e origens (treino e teste) em cada horizonte, e converte horizontes em passos pela
frequência real (1 h, ETAPA 01).

Uso: python -m src.evaluation.strategy_check
Saída: results/reports/11_horizontes_estrategias.md
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.utils.logging_utils import get_logger
from src.utils.paths import PREDICTIONS_DIR, REPORTS_DIR, load_config

STRATEGIES = {
    "persistencia": ("indexação", "ŷ(T) = y(T − h)", "—"),
    "sazonal_diaria": ("indexação", "ŷ(T) = y(T − 24)", "—"),
    "sazonal_semanal": ("indexação", "ŷ(T) = y(T − 168)", "—"),
    "sarima": ("recursiva (Kalman)", "1 ajuste; previsões de 1 a 24 passos do mesmo modelo", "1"),
    "svr": ("direta", "1 modelo por horizonte", "4"),
    "lightgbm": ("direta", "1 modelo por horizonte", "4"),
    "lightgbm_rec": ("recursiva", "modelo de 1 passo iterado h vezes", "1"),
}


def main() -> None:
    log = get_logger("strategy_check", run_name="11_estrategias")
    cfg = load_config()
    reg = registry.latest_ok()
    checks, ref = [], {}
    for r in reg.itertuples(index=False):
        p = pd.read_parquet(PREDICTIONS_DIR / f"{r.exp_id}.parquet")
        for conj in ("treino", "teste"):
            part = p[p["conjunto"] == conj]
            key = (r.horizonte_h, conj)
            arr = (part["ts_alvo"].to_numpy(), part["origem"].to_numpy())
            steps_ok = bool(((part["ts_alvo"] - part["origem"]) == pd.Timedelta(hours=int(r.horizonte_h))).all())
            if key not in ref:
                ref[key] = arr
            same = len(arr[0]) == len(ref[key][0]) and np.array_equal(arr[0], ref[key][0]) and np.array_equal(arr[1], ref[key][1])
            checks.append({"modelo": r.modelo, "h": int(r.horizonte_h), "conjunto": conj, "n": len(part),
                           "mesmos_pontos": same, "passos_ok": steps_ok, "sem_nan": bool(part["y_prev"].notna().all())})
    chk = pd.DataFrame(checks)
    all_ok = bool(chk[["mesmos_pontos", "passos_ok", "sem_nan"]].all().all())

    rec = reg[reg["modelo"].isin(["lightgbm", "lightgbm_rec"])].pivot(index="horizonte_h", columns="modelo",
                                                                       values=["rmse_teste", "mae_teste", "t_previsao_s"])
    L = ["# ETAPA 11 — Horizontes e estratégias multi-step", "",
         f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Horizontes em passos", "",
         "Frequência real verificada na ETAPA 01: 1 h. Portanto, horizonte de h horas = h passos.", "",
         "| Horizonte | Passos |", "|---|---:|",
         *[f"| {h} h | {h} |" for h in cfg["models"]["horizons_hours"]], "",
         "## Estratégia por modelo", "",
         "| Modelo | Estratégia | Como | Modelos para os 4 horizontes |", "|---|---|---|---:|",
         *[f"| {m} | {s[0]} | {s[1]} | {s[2]} |" for m, s in STRATEGIES.items() if m in set(reg["modelo"])], "",
         "Critérios: custo (nº de modelos a treinar), propagação de erro (recursiva acumula erro dos passos "
         "anteriores; direta não) e complexidade. SARIMA é recursivo por natureza; SVR e LightGBM diretos "
         "(aprovado nas ETAPAS 09–10); a variante LightGBM recursiva isola o efeito da estratégia.", "",
         "## Protocolo comum (aprovado na ETAPA 07)", "",
         "- Teste: todas as 3.504 horas como alvos em cada horizonte; origem = alvo − h.",
         "- Treino: mesmos 13.824 alvos para todos os modelos e horizontes.",
         "- Ajuste único no treino; em cada origem, só observações reais até a origem.", "",
         "## Verificação de justiça (a partir dos arquivos de previsão)", "",
         f"**Resultado: {'OK — todos os modelos usam exatamente os mesmos alvos e origens' if all_ok else 'FALHOU'}.**", "",
         "| Modelo | h | Conjunto | n | Mesmos pontos | alvo − origem = h | Sem NaN |", "|---|---:|---|---:|---|---|---|"]
    for c in checks:
        ok = lambda v: "OK" if v else "FALHOU"  # noqa: E731
        L.append(f"| {c['modelo']} | {c['h']} | {c['conjunto']} | {c['n']} | {ok(c['mesmos_pontos'])} | "
                 f"{ok(c['passos_ok'])} | {ok(c['sem_nan'])} |")
    L += ["", "## Direta × recursiva (LightGBM, mesmos hiperparâmetros de h = 1 na recursiva)", "",
          "| h | RMSE direta | RMSE recursiva | MAE direta | MAE recursiva | t_previsão direta (s) | t_previsão recursiva (s) |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for h in rec.index:
        g = lambda m, c: rec.loc[h, (m, c)]  # noqa: E731
        L.append(f"| {h} | {g('rmse_teste', 'lightgbm'):.4f} | {g('rmse_teste', 'lightgbm_rec'):.4f} | "
                 f"{g('mae_teste', 'lightgbm'):.4f} | {g('mae_teste', 'lightgbm_rec'):.4f} | "
                 f"{g('t_previsao_s', 'lightgbm'):.3f} | {g('t_previsao_s', 'lightgbm_rec'):.3f} |")
    L.append("")
    (REPORTS_DIR / "11_horizontes_estrategias.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Verificação de justiça: %s (%d combinações)", "OK" if all_ok else "FALHOU", len(checks))
    if not all_ok:
        raise AssertionError("Pontos de avaliação diferentes entre modelos")


if __name__ == "__main__":
    main()
