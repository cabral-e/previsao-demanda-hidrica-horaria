# ETAPA 14 — Métrica composta (erro + tempo)

Gerado em: 2026-10-05T19:39:12

## Achado metodológico: efeito estrutural do baseline

Na normalização B, o melhor baseline tem E_norm = 1 (por definição) e T_norm = 0 (é o mais rápido), logo score = α·1 + (1 − α)·0 = α em qualquer horizonte (0,5 em α = 0,5): o baseline vence sempre que nenhum modelo tenha α·E + (1 − α)·T < α. Como os baselines levam milissegundos, a escala log-mín-máx do tempo fica dominada por eles. O resultado com baseline reflete essa construção, não evidência de que não modelar seja preferível (todos os treináveis superam o baseline com significância, ETAPA 12). Decisão do usuário: conjunto principal = somente treináveis; com baseline = sensibilidade.

# Conjunto: PRINCIPAL — somente os 4 treináveis (baseline apenas como âncora do erro em B)

## Definição

Score = α · E_norm + (1 − α) · T_norm, por horizonte; **menor é melhor**. α = 0,5 (pesos iguais) é o ponto de partida, sem preferência a priori entre acurácia e custo; a sensibilidade a α (0,1 a 0,9) mostra se o ranking depende dessa escolha.

- **B (principal)**: E_norm = RMSE / RMSE do melhor baseline (baseline = 1; previsão perfeita = 0); T_norm = mín-máx de log10(t_total) entre os 4 modelos treináveis do horizonte (mais rápido = 0; mais lento = 1).
- **A (robustez)**: E_norm = mín-máx do RMSE; T_norm igual a B.
- **C (robustez)**: posições normalizadas (0 = melhor; 1 = pior) em erro e em tempo.
- Tempo: t_total = treino + previsão do horizonte (convenção da ETAPA 13); variante com *tuning* somada.
- Modelos por horizonte: SARIMA, SVR, LightGBM direto e LightGBM recursivo (baseline só como âncora de E em B).

## Score principal (B, RMSE, sem tuning, α = 0.5)

| h | Modelo | RMSE teste | E_norm | t_total (s) | T_norm | Score | Posição |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM (recursivo) | 0.0803 | 0.408 | 2.755 | 0.000 | **0.204** | 1 |
| 1 | LightGBM (direto) | 0.0803 | 0.408 | 2.851 | 0.007 | **0.207** | 2 |
| 1 | SVR | 0.0916 | 0.466 | 33.11 | 0.508 | **0.487** | 3 |
| 1 | SARIMA | 0.1162 | 0.590 | 369.2 | 1.000 | **0.795** | 4 |
| 6 | LightGBM (direto) | 0.2795 | 0.537 | 0.8701 | 0.000 | **0.269** | 1 |
| 6 | LightGBM (recursivo) | 0.2724 | 0.523 | 4.781 | 0.282 | **0.402** | 2 |
| 6 | SVR | 0.2990 | 0.574 | 23.55 | 0.545 | **0.560** | 3 |
| 6 | SARIMA | 0.3747 | 0.720 | 369.2 | 1.000 | **0.860** | 4 |
| 12 | LightGBM (direto) | 0.3620 | 0.695 | 0.7826 | 0.000 | **0.348** | 1 |
| 12 | LightGBM (recursivo) | 0.3524 | 0.677 | 7.34 | 0.364 | **0.520** | 2 |
| 12 | SVR | 0.3650 | 0.701 | 18.49 | 0.514 | **0.608** | 3 |
| 12 | SARIMA | 0.4038 | 0.776 | 369.1 | 1.000 | **0.888** | 4 |
| 24 | LightGBM (direto) | 0.3899 | 0.749 | 0.4948 | 0.000 | **0.375** | 1 |
| 24 | LightGBM (recursivo) | 0.3947 | 0.758 | 11.91 | 0.481 | **0.620** | 2 |
| 24 | SVR | 0.3968 | 0.762 | 23.71 | 0.585 | **0.674** | 3 |
| 24 | SARIMA | 0.4002 | 0.769 | 369.2 | 1.000 | **0.884** | 4 |

## Sensibilidade a α — vencedor por horizonte (B, RMSE, sem tuning)

| h | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | α=0.6 | α=0.7 | α=0.8 | α=0.9 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) |
| 6 | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |
| 12 | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |
| 24 | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |

## Ranking completo por α (B, RMSE, sem tuning)

**h = 1**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (recursivo) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LightGBM (direto) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| SVR | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| SARIMA | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

**h = 6**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (direto) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LightGBM (recursivo) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| SVR | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| SARIMA | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

**h = 12**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (direto) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LightGBM (recursivo) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| SVR | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| SARIMA | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

**h = 24**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (direto) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| LightGBM (recursivo) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| SVR | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| SARIMA | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |

## Robustez — vencedor em α = 0,5 por normalização, erro e variante

| h | Normalização | Erro | Sem tuning | Com tuning |
|---:|---|---|---|---|
| 1 | A | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | A | RMSE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | B | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | B | RMSE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | C | RMSE | LightGBM (recursivo) | LightGBM (recursivo) |
| 6 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 6 | A | RMSE | LightGBM (direto) | LightGBM (recursivo) |
| 6 | B | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 6 | B | RMSE | LightGBM (direto) | LightGBM (direto) |
| 6 | C | MAE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 6 | C | RMSE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 12 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 12 | A | RMSE | LightGBM (direto) | LightGBM (recursivo) |
| 12 | B | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 12 | B | RMSE | LightGBM (direto) | LightGBM (direto) |
| 12 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 12 | C | RMSE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 24 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 24 | A | RMSE | LightGBM (direto) | LightGBM (direto) |
| 24 | B | MAE | LightGBM (direto) | LightGBM (direto) |
| 24 | B | RMSE | LightGBM (direto) | LightGBM (direto) |
| 24 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 24 | C | RMSE | LightGBM (direto) | LightGBM (direto) |

## Estabilidade do vencedor ao longo de α

Fração dos 9 valores de α em que o vencedor é o mesmo de α = 0,5.

| h | Normalização | Erro | Variante | Vencedor (α = 0,5) | Estabilidade |
|---:|---|---|---|---|---:|
| 1 | A | MAE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | A | MAE | sem_tuning | LightGBM (recursivo) | 100% |
| 1 | A | RMSE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | A | RMSE | sem_tuning | LightGBM (recursivo) | 100% |
| 1 | B | MAE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | B | MAE | sem_tuning | LightGBM (recursivo) | 100% |
| 1 | B | RMSE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | B | RMSE | sem_tuning | LightGBM (recursivo) | 100% |
| 1 | C | MAE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | C | MAE | sem_tuning | LightGBM (recursivo) | 100% |
| 1 | C | RMSE | com_tuning | LightGBM (recursivo) | 100% |
| 1 | C | RMSE | sem_tuning | LightGBM (recursivo) | 100% |
| 6 | A | MAE | com_tuning | LightGBM (recursivo) | 89% |
| 6 | A | MAE | sem_tuning | LightGBM (direto) | 67% |
| 6 | A | RMSE | com_tuning | LightGBM (recursivo) | 78% |
| 6 | A | RMSE | sem_tuning | LightGBM (direto) | 89% |
| 6 | B | MAE | com_tuning | LightGBM (recursivo) | 67% |
| 6 | B | MAE | sem_tuning | LightGBM (direto) | 100% |
| 6 | B | RMSE | com_tuning | LightGBM (direto) | 56% |
| 6 | B | RMSE | sem_tuning | LightGBM (direto) | 100% |
| 6 | C | MAE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | MAE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | RMSE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | RMSE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 12 | A | MAE | com_tuning | LightGBM (recursivo) | 89% |
| 12 | A | MAE | sem_tuning | LightGBM (direto) | 56% |
| 12 | A | RMSE | com_tuning | LightGBM (recursivo) | 89% |
| 12 | A | RMSE | sem_tuning | LightGBM (direto) | 67% |
| 12 | B | MAE | com_tuning | LightGBM (recursivo) | 56% |
| 12 | B | MAE | sem_tuning | LightGBM (direto) | 89% |
| 12 | B | RMSE | com_tuning | LightGBM (direto) | 67% |
| 12 | B | RMSE | sem_tuning | LightGBM (direto) | 100% |
| 12 | C | MAE | com_tuning | LightGBM (recursivo) | 67% |
| 12 | C | MAE | sem_tuning | LightGBM (recursivo) | 67% |
| 12 | C | RMSE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 12 | C | RMSE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 24 | A | MAE | com_tuning | LightGBM (recursivo) | 78% |
| 24 | A | MAE | sem_tuning | LightGBM (direto) | 78% |
| 24 | A | RMSE | com_tuning | LightGBM (direto) | 100% |
| 24 | A | RMSE | sem_tuning | LightGBM (direto) | 100% |
| 24 | B | MAE | com_tuning | LightGBM (direto) | 89% |
| 24 | B | MAE | sem_tuning | LightGBM (direto) | 100% |
| 24 | B | RMSE | com_tuning | LightGBM (direto) | 100% |
| 24 | B | RMSE | sem_tuning | LightGBM (direto) | 100% |
| 24 | C | MAE | com_tuning | LightGBM (recursivo) | 67% |
| 24 | C | MAE | sem_tuning | LightGBM (recursivo) | 67% |
| 24 | C | RMSE | com_tuning | LightGBM (direto) | 100% |
| 24 | C | RMSE | sem_tuning | LightGBM (direto) | 100% |

# Conjunto: SENSIBILIDADE — 4 treináveis + melhor baseline (definição inicial)

## Definição (aprovada)

Score = α · E_norm + (1 − α) · T_norm, por horizonte; **menor é melhor**. α = 0,5 (pesos iguais) é o ponto de partida, sem preferência a priori entre acurácia e custo; a sensibilidade a α (0,1 a 0,9) mostra se o ranking depende dessa escolha.

- **B (principal)**: E_norm = RMSE / RMSE do melhor baseline (baseline = 1; previsão perfeita = 0); T_norm = mín-máx de log10(t_total) entre os 5 modelos do horizonte (mais rápido = 0; mais lento = 1).
- **A (robustez)**: E_norm = mín-máx do RMSE; T_norm igual a B.
- **C (robustez)**: posições normalizadas (0 = melhor; 1 = pior) em erro e em tempo.
- Tempo: t_total = treino + previsão do horizonte (convenção da ETAPA 13); variante com *tuning* somada.
- Modelos por horizonte: SARIMA, SVR, LightGBM direto, LightGBM recursivo e o melhor baseline.

## Score principal (B, RMSE, sem tuning, α = 0.5)

| h | Modelo | RMSE teste | E_norm | t_total (s) | T_norm | Score | Posição |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Persistência | 0.1968 | 1.000 | 0.001308 | 0.000 | **0.500** | 1 |
| 1 | LightGBM (recursivo) | 0.0803 | 0.408 | 2.755 | 0.610 | **0.509** | 2 |
| 1 | LightGBM (direto) | 0.0803 | 0.408 | 2.851 | 0.612 | **0.510** | 3 |
| 1 | SVR | 0.0916 | 0.466 | 33.11 | 0.808 | **0.637** | 4 |
| 1 | SARIMA | 0.1162 | 0.590 | 369.2 | 1.000 | **0.795** | 5 |
| 6 | Sazonal semanal | 0.5205 | 1.000 | 0.001306 | 0.000 | **0.500** | 1 |
| 6 | LightGBM (direto) | 0.2795 | 0.537 | 0.8701 | 0.518 | **0.527** | 2 |
| 6 | LightGBM (recursivo) | 0.2724 | 0.523 | 4.781 | 0.654 | **0.589** | 3 |
| 6 | SVR | 0.2990 | 0.574 | 23.55 | 0.781 | **0.678** | 4 |
| 6 | SARIMA | 0.3747 | 0.720 | 369.2 | 1.000 | **0.860** | 5 |
| 12 | Sazonal semanal | 0.5205 | 1.000 | 0.001224 | 0.000 | **0.500** | 1 |
| 12 | LightGBM (direto) | 0.3620 | 0.695 | 0.7826 | 0.512 | **0.604** | 2 |
| 12 | LightGBM (recursivo) | 0.3524 | 0.677 | 7.34 | 0.689 | **0.683** | 3 |
| 12 | SVR | 0.3650 | 0.701 | 18.49 | 0.763 | **0.732** | 4 |
| 12 | SARIMA | 0.4038 | 0.776 | 369.1 | 1.000 | **0.888** | 5 |
| 24 | Sazonal semanal | 0.5205 | 1.000 | 0.001231 | 0.000 | **0.500** | 1 |
| 24 | LightGBM (direto) | 0.3899 | 0.749 | 0.4948 | 0.475 | **0.612** | 2 |
| 24 | LightGBM (recursivo) | 0.3947 | 0.758 | 11.91 | 0.728 | **0.743** | 3 |
| 24 | SVR | 0.3968 | 0.762 | 23.71 | 0.782 | **0.772** | 4 |
| 24 | SARIMA | 0.4002 | 0.769 | 369.2 | 1.000 | **0.884** | 5 |

## Sensibilidade a α — vencedor por horizonte (B, RMSE, sem tuning)

| h | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 | α=0.6 | α=0.7 | α=0.8 | α=0.9 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | Persistência | Persistência | Persistência | Persistência | Persistência | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) | LightGBM (recursivo) |
| 6 | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |
| 12 | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |
| 24 | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | Sazonal semanal | LightGBM (direto) | LightGBM (direto) | LightGBM (direto) |

## Ranking completo por α (B, RMSE, sem tuning)

**h = 1**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistência | 1 | 1 | 1 | 1 | 1 | 3 | 4 | 5 | 5 |
| LightGBM (recursivo) | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| LightGBM (direto) | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 2 |
| SVR | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 3 |
| SARIMA | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 |

**h = 6**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sazonal semanal | 1 | 1 | 1 | 1 | 1 | 3 | 4 | 5 | 5 |
| LightGBM (direto) | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| LightGBM (recursivo) | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 2 |
| SVR | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 3 |
| SARIMA | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 |

**h = 12**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sazonal semanal | 1 | 1 | 1 | 1 | 1 | 1 | 3 | 4 | 5 |
| LightGBM (direto) | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| LightGBM (recursivo) | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 |
| SVR | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 3 |
| SARIMA | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 |

**h = 24**

| Modelo | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sazonal semanal | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 4 | 5 |
| LightGBM (direto) | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 |
| LightGBM (recursivo) | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 |
| SVR | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 3 |
| SARIMA | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 |

## Robustez — vencedor em α = 0,5 por normalização, erro e variante

| h | Normalização | Erro | Sem tuning | Com tuning |
|---:|---|---|---|---|
| 1 | A | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | A | RMSE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | B | MAE | LightGBM (recursivo) | Persistência |
| 1 | B | RMSE | Persistência | Persistência |
| 1 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 1 | C | RMSE | LightGBM (recursivo) | LightGBM (recursivo) |
| 6 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 6 | A | RMSE | LightGBM (direto) | LightGBM (recursivo) |
| 6 | B | MAE | Sazonal semanal | Sazonal semanal |
| 6 | B | RMSE | Sazonal semanal | Sazonal semanal |
| 6 | C | MAE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 6 | C | RMSE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 12 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 12 | A | RMSE | LightGBM (direto) | LightGBM (recursivo) |
| 12 | B | MAE | Sazonal semanal | Sazonal semanal |
| 12 | B | RMSE | Sazonal semanal | Sazonal semanal |
| 12 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 12 | C | RMSE | LightGBM (direto) = LightGBM (recursivo) | LightGBM (direto) = LightGBM (recursivo) |
| 24 | A | MAE | LightGBM (direto) | LightGBM (recursivo) |
| 24 | A | RMSE | LightGBM (direto) | LightGBM (direto) |
| 24 | B | MAE | Sazonal semanal | Sazonal semanal |
| 24 | B | RMSE | Sazonal semanal | Sazonal semanal |
| 24 | C | MAE | LightGBM (recursivo) | LightGBM (recursivo) |
| 24 | C | RMSE | LightGBM (direto) | LightGBM (direto) |

## Estabilidade do vencedor ao longo de α

Fração dos 9 valores de α em que o vencedor é o mesmo de α = 0,5.

| h | Normalização | Erro | Variante | Vencedor (α = 0,5) | Estabilidade |
|---:|---|---|---|---|---:|
| 1 | A | MAE | com_tuning | LightGBM (recursivo) | 56% |
| 1 | A | MAE | sem_tuning | LightGBM (recursivo) | 67% |
| 1 | A | RMSE | com_tuning | LightGBM (recursivo) | 56% |
| 1 | A | RMSE | sem_tuning | LightGBM (recursivo) | 67% |
| 1 | B | MAE | com_tuning | Persistência | 56% |
| 1 | B | MAE | sem_tuning | LightGBM (recursivo) | 56% |
| 1 | B | RMSE | com_tuning | Persistência | 56% |
| 1 | B | RMSE | sem_tuning | Persistência | 56% |
| 1 | C | MAE | com_tuning | LightGBM (recursivo) | 78% |
| 1 | C | MAE | sem_tuning | LightGBM (recursivo) | 78% |
| 1 | C | RMSE | com_tuning | LightGBM (recursivo) | 78% |
| 1 | C | RMSE | sem_tuning | LightGBM (recursivo) | 78% |
| 6 | A | MAE | com_tuning | LightGBM (recursivo) | 56% |
| 6 | A | MAE | sem_tuning | LightGBM (direto) | 33% |
| 6 | A | RMSE | com_tuning | LightGBM (recursivo) | 56% |
| 6 | A | RMSE | sem_tuning | LightGBM (direto) | 56% |
| 6 | B | MAE | com_tuning | Sazonal semanal | 67% |
| 6 | B | MAE | sem_tuning | Sazonal semanal | 56% |
| 6 | B | RMSE | com_tuning | Sazonal semanal | 67% |
| 6 | B | RMSE | sem_tuning | Sazonal semanal | 56% |
| 6 | C | MAE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | MAE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | RMSE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 6 | C | RMSE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 12 | A | MAE | com_tuning | LightGBM (recursivo) | 56% |
| 12 | A | MAE | sem_tuning | LightGBM (direto) | 22% |
| 12 | A | RMSE | com_tuning | LightGBM (recursivo) | 56% |
| 12 | A | RMSE | sem_tuning | LightGBM (direto) | 44% |
| 12 | B | MAE | com_tuning | Sazonal semanal | 67% |
| 12 | B | MAE | sem_tuning | Sazonal semanal | 67% |
| 12 | B | RMSE | com_tuning | Sazonal semanal | 78% |
| 12 | B | RMSE | sem_tuning | Sazonal semanal | 67% |
| 12 | C | MAE | com_tuning | LightGBM (recursivo) | 67% |
| 12 | C | MAE | sem_tuning | LightGBM (recursivo) | 67% |
| 12 | C | RMSE | com_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 12 | C | RMSE | sem_tuning | LightGBM (direto) = LightGBM (recursivo) | 11% |
| 24 | A | MAE | com_tuning | LightGBM (recursivo) | 56% |
| 24 | A | MAE | sem_tuning | LightGBM (direto) | 67% |
| 24 | A | RMSE | com_tuning | LightGBM (direto) | 56% |
| 24 | A | RMSE | sem_tuning | LightGBM (direto) | 67% |
| 24 | B | MAE | com_tuning | Sazonal semanal | 78% |
| 24 | B | MAE | sem_tuning | Sazonal semanal | 67% |
| 24 | B | RMSE | com_tuning | Sazonal semanal | 78% |
| 24 | B | RMSE | sem_tuning | Sazonal semanal | 67% |
| 24 | C | MAE | com_tuning | LightGBM (recursivo) | 67% |
| 24 | C | MAE | sem_tuning | LightGBM (recursivo) | 67% |
| 24 | C | RMSE | com_tuning | LightGBM (direto) | 78% |
| 24 | C | RMSE | sem_tuning | LightGBM (direto) | 78% |

Nota: em h = 1, LightGBM direto e recursivo têm previsões idênticas; a ordem entre eles decorre apenas da diferença de tempo de medição (empate prático).
