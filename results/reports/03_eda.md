# ETAPA 03 — Análise exploratória (LT01)

Gerado em: 2026-10-05T19:42:01

## Desenho da análise

- **Descritiva**: período completo (2018-01-01 00:00:00 a 2019-12-31 23:00:00, 17520 registros).
- **Inferencial** (orienta escolhas de modelo): **somente treino**, antes do corte `2019-08-08 00:00:00` (train_fraction = 0.8; 14016 treino / 3504 teste).
- Duplicidade (2018-12-31 21:00) e lacuna (2018-11-04 00:00) ainda não tratadas (ETAPA 04): série usada na ordem das linhas; efeito desprezível (2 de 17.520 pontos).
- Figuras: `results/figures/eda/` (PNG 300 dpi + PDF).

## 1. Estatística descritiva

| Estatística | Total | Treino | Teste |
|---|---:|---:|---:|
| n | 17520 | 14016 | 3504 |
| média | 3.5126 | 3.4866 | 3.6163 |
| desvio | 0.6701 | 0.6836 | 0.6022 |
| mín | 0.2941 | 0.2941 | 1.3752 |
| p01 | 1.6345 | 1.5926 | 1.9000 |
| p05 | 2.1971 | 2.1332 | 2.4336 |
| p25 | 3.0770 | 3.0403 | 3.2204 |
| mediana | 3.7148 | 3.6804 | 3.8334 |
| p75 | 4.0474 | 4.0380 | 4.0789 |
| p95 | 4.2648 | 4.2620 | 4.2713 |
| p99 | 4.3565 | 4.3555 | 4.3637 |
| máx | 4.4451 | 4.4451 | 4.4043 |
| assimetria | -0.9748 | -0.9486 | -1.0219 |
| curtose (excesso) | 0.3218 | 0.2661 | 0.2875 |
| coef. variação | 0.1908 | 0.1961 | 0.1665 |

Equivalente em volume (× 555.2 m³/m): média total ≈ 1950 m³, faixa ≈ 163 a 2468 m³.

### Por ano

| Ano | média | desvio | mín | máx |
|---|---:|---:|---:|---:|
| 2018 | 3.4997 | 0.6802 | 0.9570 | 4.4019 |
| 2019 | 3.5254 | 0.6597 | 0.2941 | 4.4451 |

## 2. Padrões sazonais (descritivo, período completo)

Amplitude = máx − mín das médias por grupo. Indica quanto cada calendário explica.

| Agrupamento | Amplitude das médias (m) | Grupo de menor média | Grupo de maior média |
|---|---:|---|---|
| Hora do dia | 1.2891 | 22h (2.6965) | 9h (3.9856) |
| Dia da semana | 0.4698 | Sáb (3.3011) | Dom (3.7710) |
| Mês | 0.4971 | Abr (3.1866) | Nov (3.6837) |

Perfil horário médio (m): 0h=2.88, 1h=3.06, 2h=3.27, 3h=3.49, 4h=3.69, 5h=3.85, 6h=3.94, 7h=3.96, 8h=3.98, 9h=3.99, 10h=3.95, 11h=3.89, 12h=3.81, 13h=3.74, 14h=3.68, 15h=3.65, 16h=3.65, 17h=3.65, 18h=3.59, 19h=3.34, 20h=3.04, 21h=2.76, 22h=2.70, 23h=2.75

Diferença média dias úteis − fim de semana, por hora: máx |Δ| = 0.8502 m (na hora 22).

## 3. Outliers e eventos (descritivo — nada removido)

- Regra IQR (k = 1.5): limites [1.6213, 5.5032] m.
- Abaixo do limite inferior: **167** registros, em 37 episódio(s); acima do superior: **0** (o nível é fisicamente limitado pela altura da câmara).
- Saltos horários |Δ| > 5·σ(Δ) = 0.9683 m: **3**.

Episódios abaixo do limite inferior (10 mais profundos):

| Início | Fim | Horas | Mínimo (m) | Momento do mínimo |
|---|---|---:|---:|---|
| 2019-06-01 15:00:00 | 2019-06-02 00:00:00 | 10 | 0.2941 | 2019-06-01 20:00:00 |
| 2019-01-22 13:00:00 | 2019-01-22 23:00:00 | 11 | 0.7131 | 2019-01-22 16:00:00 |
| 2018-05-04 19:00:00 | 2018-05-05 01:00:00 | 7 | 0.9570 | 2018-05-04 20:00:00 |
| 2019-04-04 20:00:00 | 2019-04-05 02:00:00 | 7 | 0.9837 | 2019-04-04 21:00:00 |
| 2019-05-01 21:00:00 | 2019-05-02 02:00:00 | 6 | 1.0276 | 2019-05-01 23:00:00 |
| 2018-12-14 16:00:00 | 2018-12-15 02:00:00 | 11 | 1.0448 | 2018-12-14 21:00:00 |
| 2018-03-16 21:00:00 | 2018-03-17 00:00:00 | 4 | 1.1835 | 2018-03-16 22:00:00 |
| 2018-04-13 20:00:00 | 2018-04-14 01:00:00 | 6 | 1.1871 | 2018-04-13 21:00:00 |
| 2019-03-15 21:00:00 | 2019-03-16 01:00:00 | 5 | 1.2921 | 2019-03-15 22:00:00 |
| 2018-02-28 20:00:00 | 2018-03-01 00:00:00 | 5 | 1.3101 | 2018-02-28 20:00:00 |

Maiores saltos horários:

| Timestamp | De | Para | Δ |
|---|---:|---:|---:|
| 2019-01-30 09:00:00 | 2.4667 | 4.0045 | 1.5378 |
| 2019-01-30 06:00:00 | 3.8965 | 2.4564 | -1.4402 |
| 2019-04-28 12:00:00 | 4.1068 | 2.6755 | -1.4313 |

## 4. Estacionariedade (somente treino)

ADF: H0 = raiz unitária. KPSS: H0 = estacionária (em nível). α = 5%.

| Série | ADF estat. | ADF p | KPSS estat. | KPSS p | Conclusão |
|---|---:|---:|---:|---:|---|
| Nível | -11.912 | < 1e-16 | 0.263 | > 0,10 | estacionária (ADF e KPSS concordam) |
| Δ1 (1ª diferença) | -26.421 | < 1e-16 | 0.004 | > 0,10 | estacionária (ADF e KPSS concordam) |
| Δ24 (sazonal) | -22.929 | < 1e-16 | 0.005 | > 0,10 | estacionária (ADF e KPSS concordam) |
| Δ1Δ24 | -30.217 | < 1e-16 | 0.003 | > 0,10 | estacionária (ADF e KPSS concordam) |

KPSS "> 0,10": estatística abaixo do menor valor crítico tabelado (p-valor real maior que 0,10).

## 5. Autocorrelação (somente treino)

| Defasagem (h) | ACF nível | ACF Δ1 |
|---:|---:|---:|
| 1 | 0.960 | 0.681 |
| 2 | 0.866 | 0.363 |
| 3 | 0.743 | 0.133 |
| 6 | 0.359 | -0.226 |
| 12 | 0.034 | -0.146 |
| 23 | 0.620 | 0.521 |
| 24 | 0.633 | 0.620 |
| 25 | 0.597 | 0.510 |
| 48 | 0.457 | 0.528 |
| 72 | 0.424 | 0.515 |
| 144 | 0.454 | 0.577 |
| 167 | 0.556 | 0.548 |
| 168 | 0.582 | 0.657 |

- PACF do nível: defasagens significativas (1–48): [1, 2, 3, 4, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 31, 32, 33, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48]
- PACF de Δ1: defasagens significativas (1–48): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 31, 34, 35, 37, 39, 40, 41, 42, 43, 44, 45, 47, 48]

## 6. Decomposição MSTL e espectro (somente treino)

Períodos: [24, 168] h. Força = max(0, 1 − Var(R)/Var(C+R)) (Wang, Smith & Hyndman, 2006).

| Componente | Força |
|---|---:|
| Tendência | 0.471 |
| Sazonal 24 h | 0.662 |
| Sazonal 168 h | 0.460 |

Periodograma — maiores picos:

| Período (h) | Potência relativa |
|---:|---:|
| 24.0 | 0.323 |
| 12.0 | 0.051 |
| 7008.0 | 0.033 |
| 83.9 | 0.024 |
| 4672.0 | 0.020 |
| 168.9 | 0.017 |
