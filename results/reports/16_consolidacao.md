# ETAPA 16 — Consolidação

Gerado em: 2026-10-05T19:12:47

Tabela final em `results/metrics/resultados_finais.csv` e `.parquet` (todas as colunas). Score = principal (B, somente treináveis, α = 0,5, RMSE, sem tuning); baselines sem score principal (ver "Score (com baseline)", análise de sensibilidade). Tempos: remedição controlada (ETAPA 13).

## Tabela final

| Modelo | h | MAE Treino | RMSE Treino | MAE Teste | RMSE Teste | Tempo Treino (s) | Tempo Previsão (s) | Tempo Total (s) | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistência | 1 | 0.1442 | 0.1919 | 0.1487 | 0.1968 | 1.87e-05 | 0.00129 | 0.001308 | — |
| Sazonal diária | 1 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 1.62e-05 | 0.00127 | 0.001285 | — |
| Sazonal semanal | 1 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 1.5e-05 | 0.001223 | 0.001237 | — |
| SARIMA | 1 | 0.0742 | 0.1078 | 0.0795 | 0.1162 | 368.7 | 0.4028 | 369.2 | 0.795 |
| SVR | 1 | 0.0535 | 0.0839 | 0.0589 | 0.0916 | 28.81 | 4.306 | 33.11 | 0.487 |
| LightGBM (direto) | 1 | 0.0395 | 0.0604 | 0.0522 | 0.0803 | 2.443 | 0.4038 | 2.851 | 0.207 |
| LightGBM (recursivo) | 1 | 0.0395 | 0.0604 | 0.0522 | 0.0803 | 2.354 | 0.4075 | 2.755 | 0.204 |
| Persistência | 6 | 0.6123 | 0.7754 | 0.5971 | 0.7682 | 1.99e-05 | 0.001452 | 0.001472 | — |
| Sazonal diária | 6 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 1.62e-05 | 0.001218 | 0.001232 | — |
| Sazonal semanal | 6 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 1.87e-05 | 0.001291 | 0.001306 | — |
| SARIMA | 6 | 0.2869 | 0.3744 | 0.2780 | 0.3747 | 368.7 | 0.4001 | 369.2 | 0.860 |
| SVR | 6 | 0.1937 | 0.2750 | 0.2071 | 0.2990 | 18.59 | 4.833 | 23.55 | 0.560 |
| LightGBM (direto) | 6 | 0.1707 | 0.2312 | 0.1953 | 0.2795 | 0.7695 | 0.09974 | 0.8701 | 0.269 |
| LightGBM (recursivo) | 6 | 0.1590 | 0.2292 | 0.1838 | 0.2724 | 2.354 | 2.426 | 4.781 | 0.402 |
| Persistência | 12 | 0.7739 | 0.9522 | 0.7448 | 0.9221 | 1.47e-05 | 0.001276 | 0.001291 | — |
| Sazonal diária | 12 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 1.46e-05 | 0.001243 | 0.001257 | — |
| Sazonal semanal | 12 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 1.51e-05 | 0.001208 | 0.001224 | — |
| SARIMA | 12 | 0.3323 | 0.4300 | 0.2997 | 0.4038 | 368.7 | 0.4077 | 369.1 | 0.888 |
| SVR | 12 | 0.2222 | 0.3145 | 0.2509 | 0.3650 | 13.05 | 5.425 | 18.49 | 0.608 |
| LightGBM (direto) | 12 | 0.2380 | 0.3180 | 0.2545 | 0.3620 | 0.6861 | 0.09653 | 0.7826 | 0.348 |
| LightGBM (recursivo) | 12 | 0.2243 | 0.3144 | 0.2384 | 0.3524 | 2.354 | 4.956 | 7.34 | 0.520 |
| Persistência | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 1.55e-05 | 0.001224 | 0.001245 | — |
| Sazonal diária | 24 | 0.4179 | 0.5858 | 0.3954 | 0.5665 | 1.49e-05 | 0.001266 | 0.001281 | — |
| Sazonal semanal | 24 | 0.4547 | 0.6221 | 0.3708 | 0.5205 | 1.51e-05 | 0.001215 | 0.001231 | — |
| SARIMA | 24 | 0.3464 | 0.4553 | 0.2949 | 0.4002 | 368.7 | 0.4005 | 369.2 | 0.884 |
| SVR | 24 | 0.2450 | 0.3603 | 0.2769 | 0.3968 | 16.72 | 6.991 | 23.71 | 0.674 |
| LightGBM (direto) | 24 | 0.2843 | 0.3783 | 0.2786 | 0.3899 | 0.4351 | 0.05916 | 0.4948 | 0.375 |
| LightGBM (recursivo) | 24 | 0.2894 | 0.4070 | 0.2763 | 0.3947 | 2.354 | 9.562 | 11.91 | 0.620 |

## Colunas complementares

| Modelo | h | RMSE Teste (m³ eq.) | Memória (MB) | Teste/Treino | Estratégia | Pareto | Grupo dos melhores (DM) | Score (com baseline) |
|---|---:|---:|---:|---:|---|---|---|---:|
| Persistência | 1 | 109.2 | 0.0 | 1.025 | indexação | — | não | 0.500 |
| Sazonal diária | 1 | 314.5 | 0.0 | 0.967 | indexação | — | não | — |
| Sazonal semanal | 1 | 289.0 | 0.0 | 0.837 | indexação | — | não | — |
| SARIMA | 1 | 64.5 | 20.6 | 1.077 | recursiva (Kalman) | não | não | 0.795 |
| SVR | 1 | 50.9 | 207.9 | 1.092 | direta | não | não | 0.637 |
| LightGBM (direto) | 1 | 44.6 | 9.7 | 1.330 | direta | sim | sim | 0.510 |
| LightGBM (recursivo) | 1 | 44.6 | 13.4 | 1.330 | recursiva | sim | sim | 0.509 |
| Persistência | 6 | 426.5 | 0.0 | 0.991 | indexação | — | não | — |
| Sazonal diária | 6 | 314.5 | 0.0 | 0.967 | indexação | — | não | — |
| Sazonal semanal | 6 | 289.0 | 0.0 | 0.837 | indexação | — | não | 0.500 |
| SARIMA | 6 | 208.0 | 20.6 | 1.001 | recursiva (Kalman) | não | não | 0.860 |
| SVR | 6 | 166.0 | 212.6 | 1.087 | direta | não | não | 0.678 |
| LightGBM (direto) | 6 | 155.2 | 5.8 | 1.209 | direta | sim | sim | 0.527 |
| LightGBM (recursivo) | 6 | 151.2 | 14.1 | 1.189 | recursiva | sim | sim | 0.589 |
| Persistência | 12 | 512.0 | 0.0 | 0.968 | indexação | — | não | — |
| Sazonal diária | 12 | 314.5 | 0.0 | 0.967 | indexação | — | não | — |
| Sazonal semanal | 12 | 289.0 | 0.0 | 0.837 | indexação | — | não | 0.500 |
| SARIMA | 12 | 224.2 | 20.6 | 0.939 | recursiva (Kalman) | não | não | 0.888 |
| SVR | 12 | 202.7 | 143.6 | 1.161 | direta | não | sim | 0.732 |
| LightGBM (direto) | 12 | 201.0 | 9.3 | 1.138 | direta | sim | sim | 0.604 |
| LightGBM (recursivo) | 12 | 195.6 | 14.1 | 1.121 | recursiva | sim | sim | 0.683 |
| Persistência | 24 | 314.5 | 0.0 | 0.967 | indexação | — | não | — |
| Sazonal diária | 24 | 314.5 | 0.0 | 0.967 | indexação | — | não | — |
| Sazonal semanal | 24 | 289.0 | 0.0 | 0.837 | indexação | — | não | 0.500 |
| SARIMA | 24 | 222.2 | 20.6 | 0.879 | recursiva (Kalman) | não | sim | 0.884 |
| SVR | 24 | 220.3 | 164.1 | 1.101 | direta | não | sim | 0.772 |
| LightGBM (direto) | 24 | 216.5 | 7.1 | 1.030 | direta | sim | sim | 0.612 |
| LightGBM (recursivo) | 24 | 219.1 | 14.1 | 0.970 | recursiva | não | sim | 0.743 |

## Respostas às perguntas da seção 13

1. **Menor MAE**: média nos 4 horizontes — LightGBM (recursivo) (0.1877 m). Por horizonte: 1 h: LightGBM (direto) = LightGBM (recursivo) (0.0522); 6 h: LightGBM (recursivo) (0.1838); 12 h: LightGBM (recursivo) (0.2384); 24 h: LightGBM (recursivo) (0.2763).
2. **Menor RMSE**: média — LightGBM (recursivo) (0.2749 m). Por horizonte: 1 h: LightGBM (direto) = LightGBM (recursivo) (0.0803); 6 h: LightGBM (recursivo) (0.2724); 12 h: LightGBM (recursivo) (0.3524); 24 h: LightGBM (direto) (0.3899). Em 1 h, LightGBM direto e recursivo são o mesmo modelo (empate exato).
3. **Menor tempo**: baselines (~0,005 s para os 4 horizontes). Entre os treináveis, LightGBM (direto) (5.0 s para os 4 horizontes; SARIMA 370 s).
4. **Melhor generalização**: em erro de teste, o LightGBM. Pela razão teste/treino (sobreajuste), o mais estável é o SARIMA (0.88–1.08); os modelos de aprendizado de máquina ficam entre 0.97 e 1.33 — ajustam-se mais ao treino, mas ainda generalizam melhor em erro absoluto. O teste é um período mais regular (razões < 1 nos baselines), o que deve ser considerado.
5. **Melhor em 1 h**: LightGBM (direto) = LightGBM (recursivo) (RMSE 0.0803 m); sem diferença significativa (DM, Holm 5%): LightGBM (direto), LightGBM (recursivo).
6. **Melhor em 6 h**: LightGBM (recursivo) (RMSE 0.2724 m); sem diferença significativa (DM, Holm 5%): LightGBM (direto), LightGBM (recursivo).
7. **Melhor em 12 h**: LightGBM (recursivo) (RMSE 0.3524 m); sem diferença significativa (DM, Holm 5%): SVR, LightGBM (direto), LightGBM (recursivo).
8. **Melhor em 24 h**: LightGBM (direto) (RMSE 0.3899 m); sem diferença significativa (DM, Holm 5%): SARIMA, SVR, LightGBM (direto), LightGBM (recursivo).
9. **O modelo mais complexo é necessariamente melhor?** Não. O de maior custo computacional (SARIMA) é dominado em todos os horizontes; o SVR (2º mais caro) também. O melhor desempenho veio de um modelo flexível e barato (LightGBM). A resposta depende de como "complexidade" é definida no texto (custo × capacidade).
10. **O ganho de acurácia justifica o custo?** Em relação ao melhor baseline de cada horizonte, sim: os treináveis reduzem o RMSE de 22% a 59%, com custo de segundos no LightGBM. Entre os treináveis, o custo adicional do SVR e do SARIMA não compra acurácia. O LightGBM recursivo paga 5–10× mais tempo de previsão por ~2,5% de ganho em 6–12 h, sem diferença significativa na perda quadrática.
11. **Melhor custo-benefício**: 1 h: LightGBM (direto) = LightGBM (recursivo), 6 h: LightGBM (direto), 12 h: LightGBM (direto), 24 h: LightGBM (direto) (score principal, estável em α de 0,1 a 0,9; na fronteira de Pareto em todos os horizontes).
12. **O resultado muda com o horizonte?** Sim, em magnitude: a vantagem dos modelos de aprendizado de máquina diminui com h (em 24 h, os quatro treináveis são estatisticamente equivalentes); a estratégia recursiva reduz o erro em 6–12 h e deixa de reduzir o RMSE em 24 h. A ordem de custo-benefício (LightGBM à frente) não muda.
