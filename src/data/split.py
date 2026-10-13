"""ETAPA 05 — Divisão temporal treino/teste (cronológica, sem embaralhar).

O ponto de corte é definido sobre a grade regular do período configurado
(não sobre o número de linhas), para ser o mesmo na EDA (ETAPA 03) e aqui.

Uso: python -m src.data.split [--force]
Entrada: data/processed/04_tratado.parquet (ETAPA 04)
Saídas:  data/processed/05_treino.parquet, data/processed/05_teste.parquet (+ .meta.json),
         results/reports/05_divisao.md
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

from src.data.load import sha256
from src.utils.logging_utils import get_logger
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, load_config
from src.utils.persistence import config_hash, is_reusable, write_meta

INPUT = PROCESSED_DIR / "04_tratado.parquet"
TRAIN_OUT = PROCESSED_DIR / "05_treino.parquet"
TEST_OUT = PROCESSED_DIR / "05_teste.parquet"
CONFIG_KEYS = ("data.timestamp_column", "data.start", "data.end", "split")


def temporal_cut(start: str, end: str, train_fraction: float, freq: str = "h") -> pd.Timestamp:
    """Primeiro timestamp do teste: start + round(fraction * n_grade) passos."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction deve estar em (0, 1): {train_fraction}")
    grid = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
    n_train = int(round(train_fraction * len(grid)))
    return grid[n_train]


def configured_cut() -> pd.Timestamp:
    cfg = load_config()
    return temporal_cut(cfg["data"]["start"], cfg["data"]["end"],
                        cfg["split"]["train_fraction"], cfg["split"].get("freq", "h"))


def split_by_cut(df: pd.DataFrame, ts_col: str, cut: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino: ts < cut; teste: ts >= cut. Não há sobreposição."""
    is_train = df[ts_col] < cut
    return df.loc[is_train].copy(), df.loc[~is_train].copy()


def validate_split(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, ts_col: str,
                   cut: pd.Timestamp) -> dict[str, bool]:
    step = pd.Timedelta(hours=1)
    checks = {
        "tamanhos_somam": len(train) + len(test) == len(df),
        "sem_sobreposicao": train[ts_col].max() < test[ts_col].min(),
        "treino_antes_do_corte": bool((train[ts_col] < cut).all()),
        "teste_a_partir_do_corte": bool((test[ts_col] >= cut).all()),
        "teste_comeca_no_corte": test[ts_col].min() == cut,
        "contiguos": test[ts_col].min() - train[ts_col].max() == step,
        "ordem_preservada": bool(train[ts_col].is_monotonic_increasing and test[ts_col].is_monotonic_increasing),
        "nao_vazios": len(train) > 0 and len(test) > 0,
    }
    if not all(checks.values()):
        raise AssertionError(f"Divisão inválida: {checks}")
    return checks


def _stats(s: pd.Series) -> str:
    return f"{s.mean():.4f} | {s.std():.4f} | {s.min():.4f} | {s.median():.4f} | {s.max():.4f}"


def main(force: bool = False) -> None:
    log = get_logger("split", run_name="05_divisao")
    cfg = load_config()
    ts_col, target = cfg["data"]["timestamp_column"], cfg["data"]["target"]
    if not INPUT.is_file():
        raise FileNotFoundError(f"{INPUT} não encontrado — execute a ETAPA 04")

    in_hash, cfg_hash = sha256(INPUT), config_hash(*CONFIG_KEYS)
    if not force and all(is_reusable(p, input_sha256=in_hash, config_hash=cfg_hash) for p in (TRAIN_OUT, TEST_OUT)):
        log.info("Divisão já existente com mesma entrada e configuração; reutilizando.")
        return

    df = pd.read_parquet(INPUT)
    cut = configured_cut()
    train, test = split_by_cut(df, ts_col, cut)
    checks = validate_split(df, train, test, ts_col, cut)

    for part, path in (("treino", TRAIN_OUT), ("teste", TEST_OUT)):
        data = train if part == "treino" else test
        data.to_parquet(path, index=False)
        write_meta(path, {"etapa": "05", "parte": part, "input": INPUT.name, "input_sha256": in_hash,
                          "config_hash": cfg_hash, "output_sha256": sha256(path), "corte": str(cut),
                          "n": len(data), "inicio": str(data[ts_col].min()), "fim": str(data[ts_col].max())})

    n = len(df)
    rep = f"""# ETAPA 05 — Divisão temporal

Gerado em: {datetime.now().isoformat(timespec='seconds')}

Divisão **cronológica**, sem embaralhar. Corte calculado sobre a grade horária do período
(`train_fraction` = {cfg['split']['train_fraction']}); o mesmo corte usado na EDA (ETAPA 03).

| Conjunto | Início | Fim | Registros | Fração | Dias |
|---|---|---|---:|---:|---:|
| Treino | {train[ts_col].min()} | {train[ts_col].max()} | {len(train)} | {len(train) / n:.4f} | {len(train) / 24:.0f} |
| Teste | {test[ts_col].min()} | {test[ts_col].max()} | {len(test)} | {len(test) / n:.4f} | {len(test) / 24:.0f} |

**Corte (primeiro instante do teste): {cut}**

## {target} por conjunto

| Conjunto | média | desvio | mín | mediana | máx |
|---|---:|---:|---:|---:|---:|
| Treino | {_stats(train[target])} |
| Teste | {_stats(test[target])} |

## Validações

{chr(10).join(f"- {k}: {'OK' if v else 'FALHOU'}" for k, v in checks.items())}

## Notas para as etapas seguintes

- Atributos defasados no início do teste usam as últimas observações do treino (até 168 h).
  Isso é legítimo: na origem de cada previsão, esse passado já é conhecido. Nenhum valor do
  teste entra no ajuste de modelos, escalonadores ou na seleção de hiperparâmetros.
- Qualquer validação para ajuste de hiperparâmetros será feita **dentro do treino**, também
  de forma cronológica (ETAPAS 09–10).

## Rastreabilidade

- Entrada: `data/processed/{INPUT.name}` — SHA-256 `{in_hash}`
- Hash da configuração ({', '.join(CONFIG_KEYS)}): `{cfg_hash}`
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "05_divisao.md").write_text(rep, encoding="utf-8")
    log.info("Corte %s: treino %d (%s a %s), teste %d (%s a %s)", cut, len(train), train[ts_col].min(),
             train[ts_col].max(), len(test), test[ts_col].min(), test[ts_col].max())
    log.info("Validações: %s", checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
