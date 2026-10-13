# ETAPA 01 — Inspeção da base

Gerado em: 2026-10-05T15:07:24

## Arquivo

- Origem (somente leitura): `<PASTA_TCC>\SANEPAR_B_ALTO_2018_20.xlsx`
- Cópia de trabalho: `<PROJETO>\data\raw\SANEPAR_B_ALTO_2018_20.xlsx` (reutilizada)
- SHA-256 origem: `41ef8a4ca7c58cfd364293eb37be154f0f535a5243f07c5dfb7bbfc240116da6`
- SHA-256 cópia:  `41ef8a4ca7c58cfd364293eb37be154f0f535a5243f07c5dfb7bbfc240116da6`
- Hashes idênticos: OK

## Estrutura

- Faixa usada da aba no Excel (inclui cabeçalho): 26481 linhas × 12 colunas — o excedente são linhas/colunas vazias (resíduo de formatação), ignoradas pelo pandas
- Lido pelo pandas: 26280 linhas × 10 colunas
- Linhas/colunas totalmente vazias removidas adicionalmente: 0 / 0
- **Shape efetivo: 26280 linhas × 10 colunas**
- Colunas: `Timestamp`, `B1`, `B2`, `B3`, `LT01`, `FT01`, `FT02`, `FT03`, `PT01`, `PT02`
- Valores ausentes (total): 0
- Linhas inteiramente duplicadas: 0

## Coluna temporal

- Período: 2018-01-01 00:00:00 a 2020-12-31 23:00:00
- Ordenada crescente: True (passos ≤ 0: 1)
- **Frequência real (passo dominante): 0 days 01:00:00**
- Distribuição dos passos: {'0 days 00:00:00': 1, '0 days 01:00:00': 26276, '0 days 02:00:00': 1, '1 days 01:00:00': 1}
- Registros esperados no passo dominante: 26304
- Timestamps ausentes: 25, em 2 intervalo(s): 2018-11-04 00:00:00 a 2018-11-04 00:00:00 (1 h); 2020-02-29 00:00:00 a 2020-02-29 23:00:00 (24 h)
- Timestamps duplicados: 1 → ['2018-12-31 21:00:00'] (1 linha(s) excedente(s))

### Registros por ano

| ano | registros | esperado | diferenca |
|---|---|---|---|
| 2018 | 8760 | 8760 | 0 |
| 2019 | 8760 | 8760 | 0 |
| 2020 | 8760 | 8784 | -24 |

### Vizinhança das anomalias temporais

**2018-11-04 00:00:00 (ausente, 1 h)**

| Timestamp | B1 | B2 | B3 | LT01 | FT01 | FT02 | FT03 | PT01 | PT02 |
|---|---|---|---|---|---|---|---|---|---|
| 2018-11-03 22:00:00 | 51.9154 | 0.0000 | 0.0000 | 3.4098 | 262.5116 | 90.8992 | 91.5746 | 4.4536 | 17.9241 |
| 2018-11-03 23:00:00 | 51.5765 | 0.0000 | 0.0000 | 3.5353 | 261.9213 | 90.8341 | 90.4077 | 4.5885 | 17.9905 |
| 2018-11-04 01:00:00 | 46.2912 | 0.0000 | 0.0000 | 3.6797 | 260.3206 | 72.9134 | 78.5409 | 4.7907 | 15.9867 |
| 2018-11-04 02:00:00 | 45.4769 | 0.0000 | 0.0000 | 3.8678 | 260.0810 | 68.6055 | 74.9647 | 4.9876 | 15.9867 |

**2018-12-31 21:00:00 (duplicado)**

| Timestamp | B1 | B2 | B3 | LT01 | FT01 | FT02 | FT03 | PT01 | PT02 |
|---|---|---|---|---|---|---|---|---|---|
| 2018-12-31 19:00:00 | 57.9888 | 37.0161 | 0.0000 | 3.8110 | 0.0000 | 147.4111 | 127.0420 | 4.5907 | 23.0035 |
| 2018-12-31 20:00:00 | 0.0000 | 41.4027 | 57.9888 | 3.3620 | 0.0000 | 141.0779 | 126.3328 | 4.1611 | 22.6695 |
| 2018-12-31 21:00:00 | 0.0000 | 37.4956 | 57.9888 | 2.9400 | 0.0000 | 122.8611 | 118.5721 | 3.7979 | 22.0486 |
| 2018-12-31 21:00:00 | 0.0000 | 28.0294 | 57.9888 | 2.5808 | 0.0000 | 102.6522 | 100.0798 | 3.5452 | 18.9959 |
| 2018-12-31 22:00:00 | 0.0000 | 0.0000 | 57.3863 | 2.6685 | 270.8348 | 85.4773 | 94.7391 | 3.6618 | 18.9959 |
| 2018-12-31 23:00:00 | 50.0335 | 0.0000 | 0.0000 | 2.9191 | 270.4438 | 75.8599 | 82.3348 | 3.9784 | 18.0700 |

**2020-02-29 00:00:00 a 2020-02-29 23:00:00 (ausente, 24 h)**

| Timestamp | B1 | B2 | B3 | LT01 | FT01 | FT02 | FT03 | PT01 | PT02 |
|---|---|---|---|---|---|---|---|---|---|
| 2020-02-28 22:00:00 | 57.9888 | 36.1318 | 0.0000 | 2.1426 | 276.0934 | 127.5127 | 113.3098 | 2.9812 | 22.0052 |
| 2020-02-28 23:00:00 | 56.7090 | 0.0000 | 0.0000 | 2.2143 | 275.3326 | 120.3044 | 100.9955 | 3.1031 | 19.1197 |
| 2020-03-01 00:00:00 | 52.2256 | 0.0000 | 0.0000 | 3.0382 | 262.0333 | 98.8688 | 89.5650 | 4.0245 | 17.9832 |
| 2020-03-01 01:00:00 | 47.4916 | 0.0000 | 0.0000 | 3.1787 | 263.6675 | 86.8952 | 79.6782 | 4.2119 | 15.9884 |

### Transições do horário de verão (verificação)

- 2018-02-17 23:00 — fim do horário de verão 2017/18 (hora repetida esperada): ocorrências do timestamp = 1
- 2018-11-04 00:00 — início do horário de verão 2018/19 (hora ausente esperada): ocorrências do timestamp = 0
- 2019-02-16 23:00 — fim do horário de verão 2018/19 (hora repetida esperada): ocorrências do timestamp = 1

## Perfil das variáveis

| coluna | dtype | nulos | unicos | zeros | pct_zeros | negativos | inteiro | maior_seq_constante | valor_seq_constante |
|---|---|---|---|---|---|---|---|---|---|
| B1 | float64 | 0 | 13473 | 2419 | 9.2000 | 0 | False | 91 | 0.0000 |
| B2 | float64 | 0 | 11970 | 13799 | 52.5100 | 0 | False | 640 | 0.0000 |
| B3 | float64 | 0 | 2831 | 23100 | 87.9000 | 0 | False | 565 | 0.0000 |
| LT01 | float64 | 0 | 26096 | 0 | 0.0000 | 0 | False | 19 | 3.0000 |
| FT01 | float64 | 0 | 19764 | 1309 | 4.9800 | 0 | False | 27 | 0.1157 |
| FT02 | float64 | 0 | 24719 | 1530 | 5.8200 | 0 | False | 30 | 0.0000 |
| FT03 | float64 | 0 | 24523 | 208 | 0.7900 | 0 | False | 28 | 0.0289 |
| PT01 | float64 | 0 | 26057 | 0 | 0.0000 | 0 | False | 16 | 5.5411 |
| PT02 | float64 | 0 | 13150 | 475 | 1.8100 | 0 | False | 26 | 0.0000 |

| coluna | media | desvio | min | p01 | p25 | mediana | p75 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 48.8468 | 17.0234 | 0.0000 | 0.0000 | 47.1237 | 57.7663 | 57.9888 | 59.9150 | 59.9883 |
| B2 | 17.5459 | 19.9816 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 36.8784 | 59.8318 | 59.9919 |
| B3 | 4.2913 | 14.1111 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 57.9888 | 59.9883 |
| LT01 | 3.5447 | 0.6683 | 0.2588 | 1.6338 | 3.1218 | 3.7402 | 4.0713 | 4.3776 | 4.4574 |
| FT01 | 211.7839 | 114.0387 | 0.0000 | 0.0000 | 254.3020 | 265.5722 | 273.7386 | 363.4499 | 390.6829 |
| FT02 | 113.8291 | 52.9522 | 0.0000 | 0.0000 | 72.3510 | 122.3407 | 150.4183 | 225.2633 | 442.8578 |
| FT03 | 100.1616 | 35.5071 | 0.0000 | 0.0289 | 77.6512 | 109.0524 | 124.6360 | 159.2854 | 199.5949 |
| PT01 | 4.4854 | 0.7215 | 0.6290 | 2.4862 | 4.0332 | 4.6233 | 5.0309 | 5.6051 | 5.7230 |
| PT02 | 19.4271 | 6.0030 | 0.0000 | 0.0000 | 17.9688 | 20.9925 | 23.0354 | 27.3983 | 29.0075 |

## Correlação de Pearson (base completa, apenas descritiva)

|  | B1 | B2 | B3 | LT01 | FT01 | FT02 | FT03 | PT01 | PT02 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 1.0000 | 0.2730 | -0.1290 | -0.1340 | 0.2710 | 0.4650 | 0.7170 | -0.2490 | 0.8090 |
| B2 | 0.2730 | 1.0000 | -0.1600 | -0.0330 | 0.0470 | 0.5940 | 0.6550 | -0.1920 | 0.5520 |
| B3 | -0.1290 | -0.1600 | 1.0000 | -0.0820 | 0.0330 | 0.2010 | 0.2090 | -0.1340 | 0.1500 |
| LT01 | -0.1340 | -0.0330 | -0.0820 | 1.0000 | -0.0740 | -0.1210 | -0.1380 | 0.9610 | -0.0910 |
| FT01 | 0.2710 | 0.0470 | 0.0330 | -0.0740 | 1.0000 | 0.2510 | 0.2760 | -0.1140 | 0.2930 |
| FT02 | 0.4650 | 0.5940 | 0.2010 | -0.1210 | 0.2510 | 1.0000 | 0.7860 | -0.2990 | 0.6160 |
| FT03 | 0.7170 | 0.6550 | 0.2090 | -0.1380 | 0.2760 | 0.7860 | 1.0000 | -0.3380 | 0.8970 |
| PT01 | -0.2490 | -0.1920 | -0.1340 | 0.9610 | -0.1140 | -0.2990 | -0.3380 | 1.0000 | -0.2520 |
| PT02 | 0.8090 | 0.5520 | 0.1500 | -0.0910 | 0.2930 | 0.6160 | 0.8970 | -0.2520 | 1.0000 |

## Variável-alvo `LT01`

- Presente na base: OK; tipo `float64`; nulos: 0
- Faixa: 0.2588 a 4.4574; média 3.5447; desvio 0.6683
- Zeros: 0; negativos: 0
- Maior sequência de valor constante: 19 registros (valor 3.0000)
- Unidade e significado: **a confirmar pelo usuário**.

## Observações e hipóteses (a validar; nenhuma alteração foi feita nos dados)

1. `B1`–`B3` são contínuas entre 0 e ~60, não binárias. Hipótese: frequência do inversor das bombas (Hz, rede de 60 Hz), com 0 = bomba desligada.
2. `LT01` varia de 0,26 a 4,46, compatível com nível em metros. `PT01` tem correlação 0,961 com `LT01` e `PT01 − LT01` ≈ 0,93 de mediana: provavelmente a pressão no fundo do reservatório (m.c.a.), quase redundante com o alvo. Usá-la como exógena no mesmo instante equivale a vazar o alvo.
3. O timestamp duplicado de 2018-12-31 21:00 tem valores **diferentes** nas duas linhas (não é cópia), formando uma sequência contínua. Junto com a hora ausente em 2018-11-04 00:00 (início do horário de verão), sugere que o registrador aplicou o horário de verão em 04/11/2018 e foi reajustado em 31/12/2018. Porém, nos fins do horário de verão (17/02/2018 e 16/02/2019) **não** há hora repetida. Hipótese não confirmada: tratar na ETAPA 04.
4. O dia 29/02/2020 está ausente por inteiro (24 h); 2020 será excluído na ETAPA 02.
