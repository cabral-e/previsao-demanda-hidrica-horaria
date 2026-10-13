# ETAPA 15 — Trade-off erro × tempo e fronteira de Pareto

Gerado em: 2026-10-05T18:57:12

Ponto = (t_total, RMSE de teste) por modelo e horizonte. **a domina b** se não é pior em nenhum critério e é estritamente melhor em pelo menos um. Tempo: diferenças dentro do intervalo [mín, máx] das 3 repetições contam como empate. Pareto em estimativas pontuais: diferenças de erro não significativas pelo DM (ETAPA 12) devem ser lidas com cautela. Conjunto principal: 4 treináveis; com baseline: sensibilidade (decisão da ETAPA 14).

## Fronteira principal (treináveis, RMSE, sem tuning)

| h | Modelo | t_total (s) | RMSE teste | Pareto | Dominado por |
|---:|---|---:|---:|---|---|
| 1 | LightGBM (recursivo) | 2.755 | 0.0803 | **sim** | — |
| 1 | LightGBM (direto) | 2.851 | 0.0803 | **sim** | — |
| 1 | SVR | 33.11 | 0.0916 | não | LightGBM (direto), LightGBM (recursivo) |
| 1 | SARIMA | 369.2 | 0.1162 | não | LightGBM (direto), LightGBM (recursivo), SVR |
| 6 | LightGBM (direto) | 0.8701 | 0.2795 | **sim** | — |
| 6 | LightGBM (recursivo) | 4.781 | 0.2724 | **sim** | — |
| 6 | SVR | 23.55 | 0.2990 | não | LightGBM (direto), LightGBM (recursivo) |
| 6 | SARIMA | 369.2 | 0.3747 | não | LightGBM (direto), LightGBM (recursivo), SVR |
| 12 | LightGBM (direto) | 0.7826 | 0.3620 | **sim** | — |
| 12 | LightGBM (recursivo) | 7.34 | 0.3524 | **sim** | — |
| 12 | SVR | 18.49 | 0.3650 | não | LightGBM (direto), LightGBM (recursivo) |
| 12 | SARIMA | 369.1 | 0.4038 | não | LightGBM (direto), LightGBM (recursivo), SVR |
| 24 | LightGBM (direto) | 0.4948 | 0.3899 | **sim** | — |
| 24 | LightGBM (recursivo) | 11.91 | 0.3947 | não | LightGBM (direto) |
| 24 | SVR | 23.71 | 0.3968 | não | LightGBM (direto), LightGBM (recursivo) |
| 24 | SARIMA | 369.2 | 0.4002 | não | LightGBM (direto), LightGBM (recursivo), SVR |

## Ganho marginal ao longo da fronteira (treináveis, RMSE, sem tuning)

| h | De | Para | Tempo × | Redução de RMSE (m) | Redução (%) | Redução por década de tempo (m) |
|---:|---|---|---:|---:|---:|---:|
| 1 | LightGBM (recursivo) | LightGBM (direto) | 1.03 | 0.0000 | 0.0 | 0.0000 |
| 6 | LightGBM (direto) | LightGBM (recursivo) | 5.49 | 0.0071 | 2.5 | 0.0096 |
| 12 | LightGBM (direto) | LightGBM (recursivo) | 9.38 | 0.0096 | 2.6 | 0.0098 |

## Robustez — modelos na fronteira por conjunto, erro e variante

| h | Conjunto | Erro | Sem tuning | Com tuning |
|---:|---|---|---|---|
| 1 | com_baseline | MAE | Persistência, LightGBM (recursivo), LightGBM (direto) | Persistência, LightGBM (recursivo), LightGBM (direto) |
| 1 | com_baseline | RMSE | Persistência, LightGBM (recursivo), LightGBM (direto) | Persistência, LightGBM (recursivo), LightGBM (direto) |
| 1 | treinaveis | MAE | LightGBM (recursivo), LightGBM (direto) | LightGBM (recursivo), LightGBM (direto) |
| 1 | treinaveis | RMSE | LightGBM (recursivo), LightGBM (direto) | LightGBM (recursivo), LightGBM (direto) |
| 6 | com_baseline | MAE | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) |
| 6 | com_baseline | RMSE | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) |
| 6 | treinaveis | MAE | LightGBM (direto), LightGBM (recursivo) | LightGBM (direto), LightGBM (recursivo) |
| 6 | treinaveis | RMSE | LightGBM (direto), LightGBM (recursivo) | LightGBM (direto), LightGBM (recursivo) |
| 12 | com_baseline | MAE | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) |
| 12 | com_baseline | RMSE | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) |
| 12 | treinaveis | MAE | LightGBM (direto), LightGBM (recursivo) | LightGBM (direto), LightGBM (recursivo) |
| 12 | treinaveis | RMSE | LightGBM (direto), LightGBM (recursivo) | LightGBM (direto), LightGBM (recursivo) |
| 24 | com_baseline | MAE | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) | Sazonal semanal, LightGBM (direto), LightGBM (recursivo) |
| 24 | com_baseline | RMSE | Sazonal semanal, LightGBM (direto) | Sazonal semanal, LightGBM (direto) |
| 24 | treinaveis | MAE | LightGBM (direto), LightGBM (recursivo) | LightGBM (direto), LightGBM (recursivo) |
| 24 | treinaveis | RMSE | LightGBM (direto) | LightGBM (direto) |

## Visão agregada: servir os 4 horizontes

Custo total para servir os 4 horizontes (ajustes distintos + previsões dos 4 h) × RMSE médio de teste.

| Modelo | t 4 h (s) | RMSE médio | MAE médio | Pareto (treináveis) | Pareto (com baselines) |
|---|---:|---:|---:|---|---|
| Sazonal semanal | 0.005001 | 0.5205 | 0.3708 | — | **sim** |
| Sazonal diária | 0.005057 | 0.5665 | 0.3954 | — | não |
| Persistência | 0.00531 | 0.6134 | 0.4715 | — | não |
| LightGBM (direto) | 4.993 | 0.2779 | 0.1952 | **sim** | **sim** |
| LightGBM (recursivo) | 19.71 | 0.2749 | 0.1877 | **sim** | **sim** |
| SVR | 98.73 | 0.2881 | 0.1984 | não | não |
| SARIMA | 370.3 | 0.3237 | 0.2380 | não | não |
