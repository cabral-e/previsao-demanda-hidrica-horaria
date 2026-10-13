# ETAPA 08 — SARIMA

Gerado em: 2026-10-05T18:46:01

## Justificativa (EDA, somente treino)

- Nível estacionário (ADF e KPSS concordam) → d = 0.
- Sazonalidade diária dominante (ACF em 24/48/72 h; 32% da potência espectral em 24 h; força 0,66) → s = 24.
- Diferenciação sazonal D = 1 (decisão do usuário).
- Efeito semanal relevante (força 0,46) → calendário semanal como exógena (SARIMA com s = 168 é inviável).

## Seleção controlada (somente treino)

Janela: 2019-02-07 00:00:00 a 2019-08-07 23:00:00 (4368 h). Critério: menor BIC entre os que convergiram. Custo total da seleção (*tuning*, registrado à parte): **890.3 s**.

### Ordem (exógena fixa: Fourier K = 2)

| Ordem | Exógena | AIC | BIC | Conv. | LB p (24) | LB p (48) | Parâm. | Tempo (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| (2,0,1)(1,1,1)24 | fourier_K2 | -7299.0 | -7235.1 | sim | 1.84e-09 | 2.15e-12 | 10 | 68.8 |
| (2,0,1)(0,1,1)24 | fourier_K2 | -7227.9 | -7170.5 | sim | 3.85e-21 | 3.01e-23 | 9 | 62.8 |
| (2,0,0)(1,1,1)24 | fourier_K2 | -7219.6 | -7162.2 | sim | 3.5e-24 | 5.97e-24 | 9 | 96.4 |
| (1,0,1)(1,1,1)24 | fourier_K2 | -7176.6 | -7119.2 | sim | 1.57e-43 | 8.03e-52 | 9 | 91.1 |
| (2,0,0)(0,1,1)24 | fourier_K2 | -7144.5 | -7093.4 | sim | 2.64e-37 | 1.09e-36 | 8 | 59.0 |
| (1,0,1)(0,1,1)24 | fourier_K2 | -7090.5 | -7039.4 | sim | 4.41e-70 | 1.76e-80 | 8 | 53.7 |
| (1,0,0)(1,1,1)24 | fourier_K2 | -5871.6 | -5820.5 | sim | 0 | 0 | 8 | 70.7 |
| (1,0,0)(0,1,1)24 | fourier_K2 | -5740.7 | -5696.0 | sim | 0 | 0 | 7 | 40.2 |

### Exógena semanal (ordem escolhida)

| Ordem | Exógena | AIC | BIC | Conv. | LB p (24) | LB p (48) | Parâm. | Tempo (s) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| (2,0,1)(1,1,1)24 | fourier_K3 | -7317.4 | -7240.8 | sim | 4.34e-11 | 3.49e-14 | 12 | 109.6 |
| (2,0,1)(1,1,1)24 | fourier_K2 | -7299.0 | -7235.1 | sim | 1.84e-09 | 2.15e-12 | 10 | 68.8 |
| (2,0,1)(1,1,1)24 | dow | -7267.6 | -7191.0 | sim | 3.52e-09 | 4.36e-12 | 12 | 216.8 |

Ljung-Box com ~4.400 resíduos tem poder muito alto: rejeições indicam autocorrelação residual remanescente, mas não invalidam a comparação relativa entre candidatos.

## Modelo final: SARIMA(2,0,1)(1,1,1)24 + fourier_K3

Ajustado uma vez no treino inteiro (14.016 h); parâmetros fixos no teste; filtro atualizado com as observações até cada origem; previsões de 1, 6, 12 e 24 h a partir do mesmo ajuste.

Convergência: True.

Coeficientes:

| Parâmetro | Valor |
|---|---:|
| Δ24 sem_sin1 | 0.073374 |
| Δ24 sem_cos1 | 0.184671 |
| Δ24 sem_sin2 | -0.030157 |
| Δ24 sem_cos2 | 0.160767 |
| Δ24 sem_sin3 | -0.061709 |
| Δ24 sem_cos3 | 0.082641 |
| ar.L1 | 1.247918 |
| ar.L2 | -0.310830 |
| ma.L1 | 0.261037 |
| ar.S.L24 | 0.114801 |
| ma.S.L24 | -0.967663 |
| sigma2 | 0.011720 |

## Resultados (m)

| h | MAE treino | RMSE treino | MAE teste | RMSE teste | Teste/treino | t_treino (s) | t_previsão (s) | Memória (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0742 | 0.1078 | 0.0795 | 0.1162 | 1.077 | 368.74 [367.35–370.23] | 0.403 | 20.6 |
| 6 | 0.2869 | 0.3744 | 0.2780 | 0.3747 | 1.001 | 368.74 [367.35–370.23] | 0.400 | 20.6 |
| 12 | 0.3323 | 0.4300 | 0.2997 | 0.4038 | 0.939 | 368.74 [367.35–370.23] | 0.408 | 20.6 |
| 24 | 0.3464 | 0.4553 | 0.2949 | 0.4002 | 0.879 | 368.74 [367.35–370.23] | 0.401 | 20.6 |

Tempos: mediana de 3 repetições [mín–máx]. O t_treino é o mesmo ajuste para todos os h (um modelo serve aos 4 horizontes).
