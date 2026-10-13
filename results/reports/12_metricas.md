# ETAPA 12 — Métricas

Gerado em: 2026-10-05T18:05:08

MAE = (1/n) Σ |y − ŷ|; RMSE = √((1/n) Σ (y − ŷ)²); em metros. Teste: 3.504 alvos; treino: 13.824 alvos (mesmos para todos). O desempenho de **teste** tem prioridade; a razão teste/treino indica sobreajuste (> 1: erro maior fora da amostra).

## Tabela consolidada

| Modelo | h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino (RMSE) | Rank RMSE | Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistência | 1 | 0.1442 | 0.1919 | 0.1487 | 0.1968 | 1.025 | 5 | 5 |
| Sazonal diária | 1 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 7 | 7 |
| Sazonal semanal | 1 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 6 | 6 |
| SARIMA | 1 | 0.0742 | 0.1078 | 0.0795 | 0.1162 | 1.077 | 4 | 4 |
| SVR | 1 | 0.0535 | 0.0839 | 0.0589 | 0.0916 | 1.092 | 3 | 3 |
| LightGBM (direto) | 1 | 0.0395 | 0.0604 | 0.0522 | 0.0803 | 1.330 | 1 | 1 |
| LightGBM (recursivo) | 1 | 0.0395 | 0.0604 | 0.0522 | 0.0803 | 1.330 | 1 | 1 |
| Persistência | 6 | 0.6123 | 0.7754 | 0.5971 | 0.7682 | 0.991 | 7 | 7 |
| Sazonal diária | 6 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 6 | 6 |
| Sazonal semanal | 6 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 5 | 5 |
| SARIMA | 6 | 0.2869 | 0.3744 | 0.2780 | 0.3747 | 1.001 | 4 | 4 |
| SVR | 6 | 0.1937 | 0.2750 | 0.2071 | 0.2990 | 1.087 | 3 | 3 |
| LightGBM (direto) | 6 | 0.1707 | 0.2312 | 0.1953 | 0.2795 | 1.209 | 2 | 2 |
| LightGBM (recursivo) | 6 | 0.1590 | 0.2292 | 0.1838 | 0.2724 | 1.189 | 1 | 1 |
| Persistência | 12 | 0.7739 | 0.9522 | 0.7448 | 0.9221 | 0.968 | 7 | 7 |
| Sazonal diária | 12 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 6 | 6 |
| Sazonal semanal | 12 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 5 | 5 |
| SARIMA | 12 | 0.3323 | 0.4300 | 0.2997 | 0.4038 | 0.939 | 4 | 4 |
| SVR | 12 | 0.2222 | 0.3145 | 0.2509 | 0.3650 | 1.161 | 3 | 2 |
| LightGBM (direto) | 12 | 0.2380 | 0.3180 | 0.2545 | 0.3620 | 1.138 | 2 | 3 |
| LightGBM (recursivo) | 12 | 0.2243 | 0.3144 | 0.2384 | 0.3524 | 1.121 | 1 | 1 |
| Persistência | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 6 | 6 |
| Sazonal diária | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 0.967 | 6 | 6 |
| Sazonal semanal | 24 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 0.837 | 5 | 5 |
| SARIMA | 24 | 0.3464 | 0.4553 | 0.2949 | 0.4002 | 0.879 | 4 | 4 |
| SVR | 24 | 0.2450 | 0.3603 | 0.2769 | 0.3968 | 1.101 | 3 | 2 |
| LightGBM (direto) | 24 | 0.2843 | 0.3783 | 0.2786 | 0.3899 | 1.030 | 1 | 3 |
| LightGBM (recursivo) | 24 | 0.2894 | 0.4070 | 0.2763 | 0.3947 | 0.970 | 2 | 1 |

## Melhor por horizonte

| h | Menor RMSE teste | Menor MAE teste | Grupo dos melhores (DM, Holm 5%) |
|---:|---|---|---|
| 1 | LightGBM (direto) = LightGBM (recursivo) (0.0803) — empate exato | LightGBM (direto) = LightGBM (recursivo) (0.0522) — empate exato | LightGBM (recursivo), LightGBM (direto) |
| 6 | LightGBM (recursivo) (0.2724) | LightGBM (recursivo) (0.1838) | LightGBM (recursivo), LightGBM (direto) |
| 12 | LightGBM (recursivo) (0.3524) | LightGBM (recursivo) (0.2384) | LightGBM (recursivo), SVR, LightGBM (direto) |
| 24 | LightGBM (direto) (0.3899) | LightGBM (recursivo) (0.2763) | LightGBM (direto), SARIMA, SVR, LightGBM (recursivo) |

Em h = 1 o LightGBM recursivo é o próprio modelo direto de 1 passo (previsões idênticas).

"Grupo dos melhores": o modelo de menor RMSE e os que **não** diferem significativamente dele.

## Teste de Diebold-Mariano

Pares entre SARIMA, SVR, LightGBM direto, LightGBM recursivo e o melhor baseline de cada horizonte. Perda quadrática (principal) e absoluta (verificação). Variância HAC de Newey-West (Bartlett) com defasagem max(h − 1, 8); correção de Harvey-Leybourne-Newbold; t(n − 1) bicaudal; Holm por horizonte (10 pares); α = 0.05. DM < 0 ⇒ modelo 1 com perda menor.

| h | Modelo 1 | Modelo 2 | RMSE 1 | RMSE 2 | DM (quad.) | p Holm (quad.) | DM (abs.) | p Holm (abs.) | Conclusão (quad.) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Persistência | SARIMA | 0.1968 | 0.1162 | 17.41 | 2.32e-64 | 18.65 | 3.02e-73 | SARIMA melhor |
| 1 | Persistência | SVR | 0.1968 | 0.0916 | 24.60 | 1.62e-122 | 28.34 | 2.13e-158 | SVR melhor |
| 1 | Persistência | LightGBM (direto) | 0.1968 | 0.0803 | 25.72 | 9.34e-133 | 29.22 | 3.42e-167 | LightGBM (direto) melhor |
| 1 | Persistência | LightGBM (recursivo) | 0.1968 | 0.0803 | 25.72 | 9.34e-133 | 29.22 | 3.42e-167 | LightGBM (recursivo) melhor |
| 1 | SARIMA | SVR | 0.1162 | 0.0916 | 10.73 | 7.4e-26 | 14.45 | 2.07e-45 | SVR melhor |
| 1 | SARIMA | LightGBM (direto) | 0.1162 | 0.0803 | 14.41 | 5.12e-45 | 18.37 | 2.96e-71 | LightGBM (direto) melhor |
| 1 | SARIMA | LightGBM (recursivo) | 0.1162 | 0.0803 | 14.41 | 5.12e-45 | 18.37 | 2.96e-71 | LightGBM (recursivo) melhor |
| 1 | SVR | LightGBM (direto) | 0.0916 | 0.0803 | 8.76 | 8.59e-18 | 7.58 | 1.32e-13 | LightGBM (direto) melhor |
| 1 | SVR | LightGBM (recursivo) | 0.0916 | 0.0803 | 8.76 | 8.59e-18 | 7.58 | 1.32e-13 | LightGBM (recursivo) melhor |
| 1 | LightGBM (direto) | LightGBM (recursivo) | 0.0803 | 0.0803 | 0.00 | 1 | 0.00 | 1 | sem diferença significativa |
| 6 | Sazonal semanal | SARIMA | 0.5205 | 0.3747 | 5.73 | 4.38e-08 | 5.93 | 1.34e-08 | SARIMA melhor |
| 6 | Sazonal semanal | SVR | 0.5205 | 0.2990 | 7.88 | 2.21e-14 | 10.54 | 6.64e-25 | SVR melhor |
| 6 | Sazonal semanal | LightGBM (direto) | 0.5205 | 0.2795 | 8.46 | 2.25e-16 | 11.54 | 1.83e-29 | LightGBM (direto) melhor |
| 6 | Sazonal semanal | LightGBM (recursivo) | 0.5205 | 0.2724 | 8.63 | 6.31e-17 | 12.39 | 1.28e-33 | LightGBM (recursivo) melhor |
| 6 | SARIMA | SVR | 0.3747 | 0.2990 | 9.54 | 2.01e-20 | 11.89 | 4.03e-31 | SVR melhor |
| 6 | SARIMA | LightGBM (direto) | 0.3747 | 0.2795 | 11.53 | 3.43e-29 | 13.79 | 3.41e-41 | LightGBM (direto) melhor |
| 6 | SARIMA | LightGBM (recursivo) | 0.3747 | 0.2724 | 10.80 | 8.27e-26 | 14.12 | 4.82e-43 | LightGBM (recursivo) melhor |
| 6 | SVR | LightGBM (direto) | 0.2990 | 0.2795 | 4.09 | 0.000131 | 3.57 | 0.000707 | LightGBM (direto) melhor |
| 6 | SVR | LightGBM (recursivo) | 0.2990 | 0.2724 | 3.99 | 0.000134 | 5.28 | 4.14e-07 | LightGBM (recursivo) melhor |
| 6 | LightGBM (direto) | LightGBM (recursivo) | 0.2795 | 0.2724 | 1.62 | 0.105 | 3.58 | 0.000707 | sem diferença significativa |
| 12 | Sazonal semanal | SARIMA | 0.5205 | 0.4038 | 4.55 | 2.8e-05 | 4.29 | 7.32e-05 | SARIMA melhor |
| 12 | Sazonal semanal | SVR | 0.5205 | 0.3650 | 5.52 | 2.7e-07 | 6.99 | 2.63e-11 | SVR melhor |
| 12 | Sazonal semanal | LightGBM (direto) | 0.5205 | 0.3620 | 5.72 | 1.01e-07 | 6.98 | 2.63e-11 | LightGBM (direto) melhor |
| 12 | Sazonal semanal | LightGBM (recursivo) | 0.5205 | 0.3524 | 5.90 | 3.89e-08 | 7.74 | 1.29e-13 | LightGBM (recursivo) melhor |
| 12 | SARIMA | SVR | 0.4038 | 0.3650 | 4.35 | 5.7e-05 | 6.58 | 3.22e-10 | SVR melhor |
| 12 | SARIMA | LightGBM (direto) | 0.4038 | 0.3620 | 4.96 | 4.42e-06 | 6.52 | 3.95e-10 | LightGBM (direto) melhor |
| 12 | SARIMA | LightGBM (recursivo) | 0.4038 | 0.3524 | 5.53 | 2.7e-07 | 7.42 | 1.27e-12 | LightGBM (recursivo) melhor |
| 12 | SVR | LightGBM (direto) | 0.3650 | 0.3620 | 0.58 | 0.563 | -0.95 | 0.344 | sem diferença significativa |
| 12 | SVR | LightGBM (recursivo) | 0.3650 | 0.3524 | 1.54 | 0.37 | 2.08 | 0.0748 | sem diferença significativa |
| 12 | LightGBM (direto) | LightGBM (recursivo) | 0.3620 | 0.3524 | 1.11 | 0.533 | 2.58 | 0.0294 | sem diferença significativa |
| 24 | Sazonal semanal | SARIMA | 0.5205 | 0.4002 | 4.13 | 0.000315 | 4.14 | 0.000244 | SARIMA melhor |
| 24 | Sazonal semanal | SVR | 0.5205 | 0.3968 | 3.90 | 0.000682 | 4.59 | 3.68e-05 | SVR melhor |
| 24 | Sazonal semanal | LightGBM (direto) | 0.5205 | 0.3899 | 4.22 | 0.00025 | 4.79 | 1.53e-05 | LightGBM (direto) melhor |
| 24 | Sazonal semanal | LightGBM (recursivo) | 0.5205 | 0.3947 | 4.14 | 0.000315 | 4.85 | 1.27e-05 | LightGBM (recursivo) melhor |
| 24 | SARIMA | SVR | 0.4002 | 0.3968 | 0.27 | 1 | 2.23 | 0.129 | sem diferença significativa |
| 24 | SARIMA | LightGBM (direto) | 0.4002 | 0.3899 | 1.19 | 1 | 2.58 | 0.0598 | sem diferença significativa |
| 24 | SARIMA | LightGBM (recursivo) | 0.4002 | 0.3947 | 0.53 | 1 | 2.08 | 0.15 | sem diferença significativa |
| 24 | SVR | LightGBM (direto) | 0.3968 | 0.3899 | 1.01 | 1 | -0.34 | 1 | sem diferença significativa |
| 24 | SVR | LightGBM (recursivo) | 0.3968 | 0.3947 | 0.17 | 1 | 0.07 | 1 | sem diferença significativa |
| 24 | LightGBM (direto) | LightGBM (recursivo) | 0.3899 | 0.3947 | -0.46 | 1 | 0.28 | 1 | sem diferença significativa |
