# ETAPA 06 — Engenharia de atributos

Gerado em: 2026-10-05T15:58:35

## Conjunto de atributos

Origem t = último instante observado; alvo y(t + h). Nenhum atributo usa valor posterior a t.

| Grupo | Definição | Itens |
|---|---|---|
| Defasagens | lag_k = LT01(t + 1 − k) | 1, 2, 3, 6, 12, 24, 48, 168 |
| Médias móveis | média de LT01 em [t − w + 1, t] | w = 6, 24, 168 h |
| Desvios móveis | desvio de LT01 em [t − w + 1, t] | w = 24, 168 h |
| Calendário cíclico | seno e cosseno de hour, dayofweek do instante-alvo t + h | sem mês |

Total: **17 atributos**. Univariado (sem exógenas).

## Avaliação das defasagens 23 e 25 h (somente treino)

- Ajuste: alvos até 2019-04-13 04:00:00; validação: alvos de 2019-04-13 05:00:00 a 2019-08-07 23:00:00.
- O conjunto de teste não foi lido nesta etapa.
- Modelos com parâmetros padrão (Ridge α = 1 com padronização ajustada no período de ajuste; LightGBM padrão, seed 42, n_jobs 1); estratégia direta por horizonte.
- Regra fixada antes da execução: incluir se o ganho relativo médio de RMSE ≥ 1%.

| Modelo | h | RMSE base | RMSE +23,25 | Ganho | MAE base | MAE +23,25 |
|---|---:|---:|---:|---:|---:|---:|
| ridge | 1 | 0.1234 | 0.1158 | +6.14% | 0.0891 | 0.0814 |
| ridge | 6 | 0.4062 | 0.4063 | -0.02% | 0.3119 | 0.3118 |
| ridge | 12 | 0.4534 | 0.4535 | -0.02% | 0.3472 | 0.3471 |
| ridge | 24 | 0.4863 | 0.4852 | +0.23% | 0.3665 | 0.3654 |
| lightgbm | 1 | 0.0964 | 0.0972 | -0.88% | 0.0583 | 0.0591 |
| lightgbm | 6 | 0.3079 | 0.3118 | -1.27% | 0.2171 | 0.2203 |
| lightgbm | 12 | 0.3888 | 0.3884 | +0.12% | 0.2890 | 0.2873 |
| lightgbm | 24 | 0.4576 | 0.4530 | +1.00% | 0.3361 | 0.3326 |

**Ganho médio: +0.66% → não incluir 23 e 25 h.**

Custo da seleção (registrado à parte do custo dos modelos finais): 5.1 s (16 ajustes).
