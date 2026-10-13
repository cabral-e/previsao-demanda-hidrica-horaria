# ETAPA 11 — Horizontes e estratégias multi-step

Gerado em: 2026-10-05T18:05:12

## Horizontes em passos

Frequência real verificada na ETAPA 01: 1 h. Portanto, horizonte de h horas = h passos.

| Horizonte | Passos |
|---|---:|
| 1 h | 1 |
| 6 h | 6 |
| 12 h | 12 |
| 24 h | 24 |

## Estratégia por modelo

| Modelo | Estratégia | Como | Modelos para os 4 horizontes |
|---|---|---|---:|
| persistencia | indexação | ŷ(T) = y(T − h) | — |
| sazonal_diaria | indexação | ŷ(T) = y(T − 24) | — |
| sazonal_semanal | indexação | ŷ(T) = y(T − 168) | — |
| sarima | recursiva (Kalman) | 1 ajuste; previsões de 1 a 24 passos do mesmo modelo | 1 |
| svr | direta | 1 modelo por horizonte | 4 |
| lightgbm | direta | 1 modelo por horizonte | 4 |
| lightgbm_rec | recursiva | modelo de 1 passo iterado h vezes | 1 |

Critérios: custo (nº de modelos a treinar), propagação de erro (recursiva acumula erro dos passos anteriores; direta não) e complexidade. SARIMA é recursivo por natureza; SVR e LightGBM diretos (aprovado nas ETAPAS 09–10); a variante LightGBM recursiva isola o efeito da estratégia.

## Protocolo comum (aprovado na ETAPA 07)

- Teste: todas as 3.504 horas como alvos em cada horizonte; origem = alvo − h.
- Treino: mesmos 13.824 alvos para todos os modelos e horizontes.
- Ajuste único no treino; em cada origem, só observações reais até a origem.

## Verificação de justiça (a partir dos arquivos de previsão)

**Resultado: OK — todos os modelos usam exatamente os mesmos alvos e origens.**

| Modelo | h | Conjunto | n | Mesmos pontos | alvo − origem = h | Sem NaN |
|---|---:|---|---:|---|---|---|
| lightgbm | 1 | treino | 13824 | OK | OK | OK |
| lightgbm | 1 | teste | 3504 | OK | OK | OK |
| lightgbm | 6 | treino | 13824 | OK | OK | OK |
| lightgbm | 6 | teste | 3504 | OK | OK | OK |
| lightgbm | 12 | treino | 13824 | OK | OK | OK |
| lightgbm | 12 | teste | 3504 | OK | OK | OK |
| lightgbm | 24 | treino | 13824 | OK | OK | OK |
| lightgbm | 24 | teste | 3504 | OK | OK | OK |
| lightgbm_rec | 1 | treino | 13824 | OK | OK | OK |
| lightgbm_rec | 1 | teste | 3504 | OK | OK | OK |
| lightgbm_rec | 6 | treino | 13824 | OK | OK | OK |
| lightgbm_rec | 6 | teste | 3504 | OK | OK | OK |
| lightgbm_rec | 12 | treino | 13824 | OK | OK | OK |
| lightgbm_rec | 12 | teste | 3504 | OK | OK | OK |
| lightgbm_rec | 24 | treino | 13824 | OK | OK | OK |
| lightgbm_rec | 24 | teste | 3504 | OK | OK | OK |
| persistencia | 1 | treino | 13824 | OK | OK | OK |
| persistencia | 1 | teste | 3504 | OK | OK | OK |
| persistencia | 6 | treino | 13824 | OK | OK | OK |
| persistencia | 6 | teste | 3504 | OK | OK | OK |
| persistencia | 12 | treino | 13824 | OK | OK | OK |
| persistencia | 12 | teste | 3504 | OK | OK | OK |
| persistencia | 24 | treino | 13824 | OK | OK | OK |
| persistencia | 24 | teste | 3504 | OK | OK | OK |
| sarima | 1 | treino | 13824 | OK | OK | OK |
| sarima | 1 | teste | 3504 | OK | OK | OK |
| sarima | 6 | treino | 13824 | OK | OK | OK |
| sarima | 6 | teste | 3504 | OK | OK | OK |
| sarima | 12 | treino | 13824 | OK | OK | OK |
| sarima | 12 | teste | 3504 | OK | OK | OK |
| sarima | 24 | treino | 13824 | OK | OK | OK |
| sarima | 24 | teste | 3504 | OK | OK | OK |
| sazonal_diaria | 1 | treino | 13824 | OK | OK | OK |
| sazonal_diaria | 1 | teste | 3504 | OK | OK | OK |
| sazonal_diaria | 6 | treino | 13824 | OK | OK | OK |
| sazonal_diaria | 6 | teste | 3504 | OK | OK | OK |
| sazonal_diaria | 12 | treino | 13824 | OK | OK | OK |
| sazonal_diaria | 12 | teste | 3504 | OK | OK | OK |
| sazonal_diaria | 24 | treino | 13824 | OK | OK | OK |
| sazonal_diaria | 24 | teste | 3504 | OK | OK | OK |
| sazonal_semanal | 1 | treino | 13824 | OK | OK | OK |
| sazonal_semanal | 1 | teste | 3504 | OK | OK | OK |
| sazonal_semanal | 6 | treino | 13824 | OK | OK | OK |
| sazonal_semanal | 6 | teste | 3504 | OK | OK | OK |
| sazonal_semanal | 12 | treino | 13824 | OK | OK | OK |
| sazonal_semanal | 12 | teste | 3504 | OK | OK | OK |
| sazonal_semanal | 24 | treino | 13824 | OK | OK | OK |
| sazonal_semanal | 24 | teste | 3504 | OK | OK | OK |
| svr | 1 | treino | 13824 | OK | OK | OK |
| svr | 1 | teste | 3504 | OK | OK | OK |
| svr | 6 | treino | 13824 | OK | OK | OK |
| svr | 6 | teste | 3504 | OK | OK | OK |
| svr | 12 | treino | 13824 | OK | OK | OK |
| svr | 12 | teste | 3504 | OK | OK | OK |
| svr | 24 | treino | 13824 | OK | OK | OK |
| svr | 24 | teste | 3504 | OK | OK | OK |

## Direta × recursiva (LightGBM, mesmos hiperparâmetros de h = 1 na recursiva)

| h | RMSE direta | RMSE recursiva | MAE direta | MAE recursiva | t_previsão direta (s) | t_previsão recursiva (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0803 | 0.0803 | 0.0522 | 0.0522 | 0.402 | 0.406 |
| 6 | 0.2795 | 0.2724 | 0.1953 | 0.1838 | 0.097 | 2.365 |
| 12 | 0.3620 | 0.3524 | 0.2545 | 0.2384 | 0.087 | 4.795 |
| 24 | 0.3899 | 0.3947 | 0.2786 | 0.2763 | 0.051 | 9.608 |
