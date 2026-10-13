"""ETAPA 14 — Métrica composta (erro + tempo), por horizonte; menor é melhor.

Score = α·E_norm + (1 − α)·T_norm. Normalizações (config.yaml → composite_score):
  B (principal): E = RMSE / RMSE(melhor baseline); T = mín-máx de log10(t_total) no horizonte
  A: E = mín-máx(RMSE); T = idem
  C: E, T = (posição − 1)/(n − 1)   (posições médias em empates)
Variantes de tempo: "sem_tuning" (t_treino + t_previsão do h) e "com_tuning" (+ custo de tuning).
Erro principal: RMSE de teste; MAE como verificação.

Uso: python -m src.evaluation.composite
Saídas: results/metrics/14_score.csv|parquet, results/reports/14_score.md
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation import registry
from src.evaluation.cost_report import tuning_costs
from src.evaluation.metrics_report import BASELINES, LABELS, TRAINABLE
from src.utils.logging_utils import get_logger
from src.utils.paths import METRICS_DIR, REPORTS_DIR, load_config


def minmax(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else x * 0.0


def rank01(x: pd.Series) -> pd.Series:
    n = len(x)
    return (x.rank(method="average") - 1) / (n - 1) if n > 1 else x * 0.0


def candidates(reg: pd.DataFrame, tun: dict[str, float]) -> pd.DataFrame:
    """Por horizonte: 4 treináveis + melhor baseline (menor RMSE de teste)."""
    out = []
    for h, g in reg.groupby("horizonte_h"):
        bb = g[g["modelo"].isin(BASELINES)].sort_values("rmse_teste").iloc[0]
        sel = pd.concat([g[g["modelo"] == bb["modelo"]], g[g["modelo"].isin(TRAINABLE)]])
        sel = sel.assign(papel=np.where(sel["modelo"] == bb["modelo"], "baseline", "modelo"),
                         rmse_bb=bb["rmse_teste"], mae_bb=bb["mae_teste"])
        out.append(sel)
    c = pd.concat(out, ignore_index=True)
    c["t_tuning_s"] = c["modelo"].map(lambda m: tun.get(m, 0.0)).fillna(0.0)
    c["t_sem_tuning"] = c["t_total_s"]
    c["t_com_tuning"] = c["t_total_s"] + c["t_tuning_s"]
    return c


def normalize(g: pd.DataFrame, norm: str, err: str, tcol: str) -> tuple[pd.Series, pd.Series]:
    e = g[f"{err}_teste"]
    logt = np.log10(g[tcol])
    if norm == "B":
        return e / g[f"{err}_bb"], minmax(logt)
    if norm == "A":
        return minmax(e), minmax(logt)
    if norm == "C":
        return rank01(e), rank01(g[tcol])
    raise ValueError(norm)


CONJUNTOS = {"treinaveis": "PRINCIPAL — somente os 4 treináveis (baseline apenas como âncora do erro em B)",
             "com_baseline": "SENSIBILIDADE — 4 treináveis + melhor baseline (definição inicial)"}


def scores(c: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    rows = []
    for h, g_all in c.groupby("horizonte_h"):
        for conj in CONJUNTOS:
            g = g_all if conj == "com_baseline" else g_all[g_all["papel"] == "modelo"]
            for norm in ("A", "B", "C"):
                for err in ("rmse", "mae"):
                    for var, tcol in (("sem_tuning", "t_sem_tuning"), ("com_tuning", "t_com_tuning")):
                        E, T = normalize(g, norm, err, tcol)
                        for a in alphas:
                            s = a * E + (1 - a) * T
                            rk = s.rank(method="min").astype(int)
                            for i, m in enumerate(g["modelo"]):
                                rows.append({"horizonte_h": int(h), "conjunto": conj, "normalizacao": norm, "erro": err,
                                             "variante": var, "alpha": round(a, 2), "modelo": m,
                                             "E_norm": float(E.iloc[i]), "T_norm": float(T.iloc[i]),
                                             "score": float(s.iloc[i]), "rank": int(rk.iloc[i])})
    return pd.DataFrame(rows)


def winners(s: pd.DataFrame) -> pd.DataFrame:
    """Vencedor(es) por combinação; empates exatos aparecem juntos."""
    w = s[s["rank"] == 1].groupby(["horizonte_h", "conjunto", "normalizacao", "erro", "variante", "alpha"])["modelo"] \
        .apply(lambda x: " = ".join(LABELS[m] for m in sorted(x))).reset_index()
    return w


def build_report(c: pd.DataFrame, s_all: pd.DataFrame, w_all: pd.DataFrame, alpha0: float) -> str:
    L = ["# ETAPA 14 — Métrica composta (erro + tempo)", "", f"Gerado em: {datetime.now().isoformat(timespec='seconds')}", "",
         "## Achado metodológico: efeito estrutural do baseline", "",
         "Na normalização B, o melhor baseline tem E_norm = 1 (por definição) e T_norm = 0 (é o mais rápido), logo "
         "score = α·1 + (1 − α)·0 = α em qualquer horizonte (0,5 em α = 0,5): o baseline vence sempre que nenhum "
         "modelo tenha α·E + (1 − α)·T < α. Como os baselines levam milissegundos, a escala "
         "log-mín-máx do tempo fica dominada por eles. O resultado com baseline reflete essa construção, não "
         "evidência de que não modelar seja preferível (todos os treináveis superam o baseline com significância, "
         "ETAPA 12). Decisão do usuário: conjunto principal = somente treináveis; com baseline = sensibilidade.", ""]
    for conj, desc in CONJUNTOS.items():
        s, w = s_all[s_all.conjunto == conj], w_all[w_all.conjunto == conj]
        L += [f"# Conjunto: {desc}", ""]
        sec = _section(c, s, w, alpha0)
        if conj == "treinaveis":
            sec = [x.replace("entre os 5 modelos do horizonte", "entre os 4 modelos treináveis do horizonte")
                    .replace("SARIMA, SVR, LightGBM direto, LightGBM recursivo e o melhor baseline.",
                             "SARIMA, SVR, LightGBM direto e LightGBM recursivo (baseline só como âncora de E em B).")
                   .replace("## Definição (aprovada)", "## Definição") for x in sec]
        L += sec
    L += ["Nota: em h = 1, LightGBM direto e recursivo têm previsões idênticas; a ordem entre eles decorre apenas "
          "da diferença de tempo de medição (empate prático).", ""]
    return "\n".join(L)


def _section(c: pd.DataFrame, s: pd.DataFrame, w: pd.DataFrame, alpha0: float) -> list[str]:
    L = [
         "## Definição (aprovada)", "",
         "Score = α · E_norm + (1 − α) · T_norm, por horizonte; **menor é melhor**. α = 0,5 (pesos iguais) é o ponto "
         "de partida, sem preferência a priori entre acurácia e custo; a sensibilidade a α (0,1 a 0,9) mostra se o "
         "ranking depende dessa escolha.", "",
         "- **B (principal)**: E_norm = RMSE / RMSE do melhor baseline (baseline = 1; previsão perfeita = 0); "
         "T_norm = mín-máx de log10(t_total) entre os 5 modelos do horizonte (mais rápido = 0; mais lento = 1).",
         "- **A (robustez)**: E_norm = mín-máx do RMSE; T_norm igual a B.",
         "- **C (robustez)**: posições normalizadas (0 = melhor; 1 = pior) em erro e em tempo.",
         "- Tempo: t_total = treino + previsão do horizonte (convenção da ETAPA 13); variante com *tuning* somada.",
         "- Modelos por horizonte: SARIMA, SVR, LightGBM direto, LightGBM recursivo e o melhor baseline.", "",
         f"## Score principal (B, RMSE, sem tuning, α = {alpha0})", "",
         "| h | Modelo | RMSE teste | E_norm | t_total (s) | T_norm | Score | Posição |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    main = s[(s.normalizacao == "B") & (s.erro == "rmse") & (s.variante == "sem_tuning") & (s.alpha == alpha0)]
    main = main.merge(c[["horizonte_h", "modelo", "rmse_teste", "t_total_s"]], on=["horizonte_h", "modelo"])
    for r in main.sort_values(["horizonte_h", "score"]).itertuples(index=False):
        L.append(f"| {r.horizonte_h} | {LABELS[r.modelo]} | {r.rmse_teste:.4f} | {r.E_norm:.3f} | {r.t_total_s:.4g} | "
                 f"{r.T_norm:.3f} | **{r.score:.3f}** | {r.rank} |")

    L += ["", "## Sensibilidade a α — vencedor por horizonte (B, RMSE, sem tuning)", "",
          "| h | " + " | ".join(f"α={a:.1f}" for a in sorted(s.alpha.unique())) + " |",
          "|---:|" + "---|" * s.alpha.nunique()]
    wb = w[(w.normalizacao == "B") & (w.erro == "rmse") & (w.variante == "sem_tuning")]
    for h, g in wb.groupby("horizonte_h"):
        L.append(f"| {h} | " + " | ".join(g.sort_values("alpha")["modelo"]) + " |")

    L += ["", "## Ranking completo por α (B, RMSE, sem tuning)", ""]
    sb = s[(s.normalizacao == "B") & (s.erro == "rmse") & (s.variante == "sem_tuning")]
    for h, g in sb.groupby("horizonte_h"):
        piv = g.pivot(index="modelo", columns="alpha", values="rank")
        L += [f"**h = {h}**", "", "| Modelo | " + " | ".join(f"{a:.1f}" for a in piv.columns) + " |",
              "|---|" + "---:|" * len(piv.columns)]
        for m, r in piv.sort_values(alpha0).iterrows():
            L.append(f"| {LABELS[m]} | " + " | ".join(str(int(v)) for v in r.values) + " |")
        L.append("")

    L += ["## Robustez — vencedor em α = 0,5 por normalização, erro e variante", "",
          "| h | Normalização | Erro | Sem tuning | Com tuning |", "|---:|---|---|---|---|"]
    w5 = w[w.alpha == alpha0]
    for (h, n, e), g in w5.groupby(["horizonte_h", "normalizacao", "erro"]):
        st = g.loc[g.variante == "sem_tuning", "modelo"].iloc[0]
        ct = g.loc[g.variante == "com_tuning", "modelo"].iloc[0]
        L.append(f"| {h} | {n} | {e.upper()} | {st} | {ct} |")

    stab = []
    for keys, g in w.groupby(["horizonte_h", "normalizacao", "erro", "variante"]):
        ref = g.loc[g.alpha == alpha0, "modelo"].iloc[0]
        stab.append((*keys, ref, (g["modelo"] == ref).mean()))
    st = pd.DataFrame(stab, columns=["h", "norm", "erro", "variante", "vencedor_05", "frac"])
    L += ["", "## Estabilidade do vencedor ao longo de α", "",
          "Fração dos 9 valores de α em que o vencedor é o mesmo de α = 0,5.", "",
          "| h | Normalização | Erro | Variante | Vencedor (α = 0,5) | Estabilidade |", "|---:|---|---|---|---|---:|"]
    for r in st.itertuples(index=False):
        L.append(f"| {r.h} | {r.norm} | {r.erro.upper()} | {r.variante} | {r.vencedor_05} | {100 * r.frac:.0f}% |")
    L.append("")
    return L


def main() -> None:
    log = get_logger("composite", run_name="14_score")
    cfg = load_config()["composite_score"]
    reg = registry.latest_ok()
    c = candidates(reg, tuning_costs())
    s = scores(c, cfg["alpha_sensitivity"])
    w = winners(s)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(METRICS_DIR / "14_score.csv", index=False, encoding="utf-8")
    s.to_parquet(METRICS_DIR / "14_score.parquet", index=False)
    (REPORTS_DIR / "14_score.md").write_text(build_report(c, s, w, cfg["alpha"]), encoding="utf-8")
    sel = w[(w.normalizacao == cfg["main_normalization"]) & (w.erro == "rmse") & (w.variante == "sem_tuning")]
    for (conj, h), g in sel.groupby(["conjunto", "horizonte_h"]):
        log.info("[%s] h=%d vencedores por α: %s", conj, h, list(g.sort_values("alpha")["modelo"]))


if __name__ == "__main__":
    main()
