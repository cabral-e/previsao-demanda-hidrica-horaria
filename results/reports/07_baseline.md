# ETAPA 07 — Baselines

Gerado em: 2026-10-05T18:10:06

## Protocolo (aprovado; comum a todos os modelos)

- Teste: 3504 alvos (2019-08-08 00:00:00 a 2019-12-31 23:00:00), idênticos em todos os horizontes; origem = alvo − h.
- Treino (métricas): 13824 alvos (2018-01-09 00:00:00 a 2019-08-07 23:00:00), após aquecimento de 192 h.
- Cada modelo é ajustado uma vez no treino; em cada origem usa as observações reais até a origem.

## Definições

| Baseline | Previsão | Observação |
|---|---|---|
| Persistência | ŷ(T) = y(T − h) | último valor conhecido |
| Sazonal diária | ŷ(T) = y(T − 24) | em h = 24 coincide com a persistência |
| Sazonal semanal | ŷ(T) = y(T − 168) | independe de h ≤ 168 |

## Resultados (m)

| Baseline | h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino (RMSE) | t_previsão (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistencia | 1 | 0.1442 | 0.1919 | 0.1487 | 0.1968 | 1.025 | 1.290 |
| persistencia | 6 | 0.6123 | 0.7754 | 0.5971 | 0.7682 | 0.991 | 1.452 |
| persistencia | 12 | 0.7739 | 0.9522 | 0.7448 | 0.9221 | 0.968 | 1.276 |
| persistencia | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 1.224 |
| sazonal_diaria | 1 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 1.270 |
| sazonal_diaria | 6 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 1.218 |
| sazonal_diaria | 12 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 1.243 |
| sazonal_diaria | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 1.266 |
| sazonal_semanal | 1 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 1.223 |
| sazonal_semanal | 6 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 1.291 |
| sazonal_semanal | 12 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 1.208 |
| sazonal_semanal | 24 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 1.215 |

## Melhor baseline por horizonte (RMSE de teste) — referência a ser superada

| h | Baseline | RMSE teste | MAE teste |
|---:|---|---:|---:|
| 1 | persistencia | 0.1968 | 0.1487 |
| 6 | sazonal_semanal | 0.5205 | 0.3708 |
| 12 | sazonal_semanal | 0.5205 | 0.3708 |
| 24 | sazonal_semanal | 0.5205 | 0.3708 |

Os baselines não têm ajuste (t_treino ≈ 0) e preveem por indexação; o tempo é desprezível.
Registro: `results/experiments/registry.csv`; previsões: `results/predictions/exp_*.parquet`.
