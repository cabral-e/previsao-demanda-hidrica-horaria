# ETAPA 13 — Custo computacional

Gerado em: 2026-10-05T18:46:19

## Como foi medido

- Tempo: `time.perf_counter`; 3 execuções independentes; mediana [mín–máx].
- Threads: n_jobs = 1 em todos os modelos (libsvm e statsmodels são monothread).
- Tempo de previsão: todas as 3.504 previsões de teste do horizonte (inclui montar atributos/filtrar).
- Memória: pico de RSS do processo acima do nível inicial (psutil), em execução separada da cronometragem.
- Tuning registrado à parte; não entra no tempo de treino.
- Remedição controlada em lote único (aplicativos pesados fechados); medição anterior preservada no registro.
- Estado da máquina no início do lote: RAM livre 1.78 GB de 7.86 GB; CPU 16.3% (1 s); na tomada: True; 237 processos. Maiores: claude.exe (481 MB), MemCompression (408 MB), MsMpEng.exe (349 MB), Dropbox.exe (267 MB), claude.exe (244 MB).

## Custo por modelo e horizonte

| Modelo | h | t_treino (s) | t_previsão (s) | t_total (s) | ms/previsão | Memória (MB) |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (direto) | 1 | 2.443 [2.427–2.523] | 0.4038 [0.4024–0.4074] | 2.851 | 0.1152 | 9.7 |
| LightGBM (direto) | 6 | 0.7695 [0.7541–0.7789] | 0.09974 [0.09784–0.1006] | 0.8701 | 0.0285 | 5.8 |
| LightGBM (direto) | 12 | 0.6861 [0.6548–0.7381] | 0.09653 [0.0919–0.101] | 0.7826 | 0.0275 | 9.3 |
| LightGBM (direto) | 24 | 0.4351 [0.421–0.4708] | 0.05916 [0.05727–0.05966] | 0.4948 | 0.0169 | 7.1 |
| LightGBM (recursivo) | 1 | 2.354 [2.346–2.384] | 0.4075 [0.4009–0.4241] | 2.755 | 0.1163 | 13.4 |
| LightGBM (recursivo) | 6 | 2.354 [2.346–2.384] | 2.426 [2.366–2.452] | 4.781 | 0.6925 | 14.1 |
| LightGBM (recursivo) | 12 | 2.354 [2.346–2.384] | 4.956 [4.794–5.099] | 7.34 | 1.4145 | 14.1 |
| LightGBM (recursivo) | 24 | 2.354 [2.346–2.384] | 9.562 [9.497–9.588] | 11.91 | 2.7288 | 14.1 |
| Persistência | 1 | 1.87e-05 [1.83e-05–2.31e-05] | 0.00129 [0.001288–0.002713] | 0.001308 | 0.0004 | 0.0 |
| Persistência | 6 | 1.99e-05 [1.82e-05–2.83e-05] | 0.001452 [0.001314–0.00229] | 0.001472 | 0.0004 | 0.0 |
| Persistência | 12 | 1.47e-05 [1.38e-05–1.53e-05] | 0.001276 [0.001217–0.00131] | 0.001291 | 0.0004 | 0.0 |
| Persistência | 24 | 1.55e-05 [1.55e-05–3.07e-05] | 0.001224 [0.001215–0.001274] | 0.001245 | 0.0003 | 0.0 |
| SARIMA | 1 | 368.7 [367.4–370.2] | 0.4028 [0.4026–0.4365] | 369.2 | 0.1149 | 20.6 |
| SARIMA | 6 | 368.7 [367.4–370.2] | 0.4001 [0.4–0.462] | 369.2 | 0.1142 | 20.6 |
| SARIMA | 12 | 368.7 [367.4–370.2] | 0.4077 [0.4014–0.4245] | 369.1 | 0.1164 | 20.6 |
| SARIMA | 24 | 368.7 [367.4–370.2] | 0.4005 [0.4001–0.4388] | 369.2 | 0.1143 | 20.6 |
| Sazonal diária | 1 | 1.62e-05 [1.53e-05–1.96e-05] | 0.00127 [0.001216–0.001761] | 0.001285 | 0.0004 | 0.0 |
| Sazonal diária | 6 | 1.62e-05 [1.44e-05–1.88e-05] | 0.001218 [0.001187–0.001415] | 0.001232 | 0.0003 | 0.0 |
| Sazonal diária | 12 | 1.46e-05 [1.38e-05–2.08e-05] | 0.001243 [0.001215–0.001652] | 0.001257 | 0.0004 | 0.0 |
| Sazonal diária | 24 | 1.49e-05 [1.44e-05–1.52e-05] | 0.001266 [0.001206–0.001352] | 0.001281 | 0.0004 | 0.0 |
| Sazonal semanal | 1 | 1.5e-05 [1.42e-05–2.69e-05] | 0.001223 [0.001175–0.001523] | 0.001237 | 0.0003 | 0.0 |
| Sazonal semanal | 6 | 1.87e-05 [1.54e-05–2.93e-05] | 0.001291 [0.001188–0.001291] | 0.001306 | 0.0004 | 0.0 |
| Sazonal semanal | 12 | 1.51e-05 [1.46e-05–1.51e-05] | 0.001208 [0.001198–0.001416] | 0.001224 | 0.0003 | 0.0 |
| Sazonal semanal | 24 | 1.51e-05 [1.34e-05–1.57e-05] | 0.001215 [0.00118–0.001338] | 0.001231 | 0.0003 | 0.0 |
| SVR | 1 | 28.81 [28.72–28.85] | 4.306 [4.295–4.333] | 33.11 | 1.2290 | 207.9 |
| SVR | 6 | 18.59 [18.57–18.71] | 4.833 [4.756–4.991] | 23.55 | 1.3794 | 212.6 |
| SVR | 12 | 13.05 [13.05–13.22] | 5.425 [5.385–5.438] | 18.49 | 1.5483 | 143.6 |
| SVR | 24 | 16.72 [16.59–16.82] | 6.991 [6.935–7.118] | 23.71 | 1.9951 | 164.1 |

SARIMA e LightGBM recursivo: um único ajuste serve aos 4 horizontes; o t_treino de cada h é o desse ajuste (convenção da ETAPA 13).

## Custo para servir os 4 horizontes

| Modelo | Modelos ajustados | Treino (s) | Previsão, soma dos 4 h (s) | Total (s) | Memória máx. (MB) | Tuning (s, à parte) |
|---|---:|---:|---:|---:|---:|---:|
| Sazonal semanal | 0 | 6.39e-05 | 0.004938 | 0.005001 | 0.0 | 0 |
| Sazonal diária | 0 | 6.19e-05 | 0.004995 | 0.005057 | 0.0 | 0 |
| Persistência | 0 | 6.88e-05 | 0.005242 | 0.00531 | 0.0 | 0 |
| LightGBM (direto) | 4 | 4.334 | 0.6592 | 4.993 | 9.7 | 74.68 |
| LightGBM (recursivo) | 1 | 2.354 | 17.35 | 19.71 | 14.1 | 74.68 |
| SVR | 4 | 77.17 | 21.56 | 98.73 | 212.6 | 1095 |
| SARIMA | 1 | 368.7 | 1.611 | 370.3 | 20.6 | 890.3 |

Outros custos de preparação: seleção de defasagens (ETAPA 06) 5.1 s (comum a SVR e LightGBM); sondagem do SVR 44.3 s. LightGBM recursivo reutiliza o tuning do LightGBM (h = 1).

## Estabilidade entre medições (anterior × remedição controlada)

Métricas de erro idênticas em todas as 28 combinações: **sim** (modelos determinísticos).

| Modelo | h | t_treino ant. → atual (s) | t_previsão ant. → atual (s) | Memória ant. → atual (MB) |
|---|---:|---|---|---|
| LightGBM (direto) | 1 | 2.368 → 2.443 | 0.4018 → 0.4038 | 11.0 → 9.7 |
| LightGBM (direto) | 6 | 0.7589 → 0.7695 | 0.0974 → 0.09974 | 5.8 → 5.8 |
| LightGBM (direto) | 12 | 0.5914 → 0.6861 | 0.08684 → 0.09653 | 6.2 → 9.3 |
| LightGBM (direto) | 24 | 0.3881 → 0.4351 | 0.05117 → 0.05916 | 6.2 → 7.1 |
| LightGBM (recursivo) | 1 | 2.368 → 2.354 | 0.4056 → 0.4075 | 13.4 → 13.4 |
| LightGBM (recursivo) | 6 | 2.368 → 2.354 | 2.365 → 2.426 | 17.3 → 14.1 |
| LightGBM (recursivo) | 12 | 2.368 → 2.354 | 4.795 → 4.956 | 17.4 → 14.1 |
| LightGBM (recursivo) | 24 | 2.368 → 2.354 | 9.608 → 9.562 | 17.4 → 14.1 |
| Persistência | 1 | 1.86e-05 → 1.87e-05 | 0.001471 → 0.00129 | 0.1 → 0.0 |
| Persistência | 6 | 1.94e-05 → 1.99e-05 | 0.003396 → 0.001452 | 0.0 → 0.0 |
| Persistência | 12 | 1.49e-05 → 1.47e-05 | 0.001223 → 0.001276 | 0.0 → 0.0 |
| Persistência | 24 | 1.36e-05 → 1.55e-05 | 0.001246 → 0.001224 | 0.0 → 0.0 |
| SARIMA | 1 | 377.7 → 368.7 | 0.4209 → 0.4028 | 25.3 → 20.6 |
| SARIMA | 6 | 377.7 → 368.7 | 0.4519 → 0.4001 | 25.3 → 20.6 |
| SARIMA | 12 | 377.7 → 368.7 | 0.4172 → 0.4077 | 25.3 → 20.6 |
| SARIMA | 24 | 377.7 → 368.7 | 0.4106 → 0.4005 | 25.3 → 20.6 |
| Sazonal diária | 1 | 1.9e-05 → 1.62e-05 | 0.001717 → 0.00127 | 0.0 → 0.0 |
| Sazonal diária | 6 | 1.82e-05 → 1.62e-05 | 0.001272 → 0.001218 | 0.0 → 0.0 |
| Sazonal diária | 12 | 1.34e-05 → 1.46e-05 | 0.001274 → 0.001243 | 0.0 → 0.0 |
| Sazonal diária | 24 | 4.01e-05 → 1.49e-05 | 0.002681 → 0.001266 | 0.0 → 0.0 |
| Sazonal semanal | 1 | 1.63e-05 → 1.5e-05 | 0.001772 → 0.001223 | 0.0 → 0.0 |
| Sazonal semanal | 6 | 2.06e-05 → 1.87e-05 | 0.001349 → 0.001291 | 0.0 → 0.0 |
| Sazonal semanal | 12 | 1.42e-05 → 1.51e-05 | 0.001464 → 0.001208 | 0.0 → 0.0 |
| Sazonal semanal | 24 | 1.49e-05 → 1.51e-05 | 0.001288 → 0.001215 | 0.0 → 0.0 |
| SVR | 1 | 33.37 → 28.81 | 5.572 → 4.306 | 212.2 → 207.9 |
| SVR | 6 | 20.7 → 18.59 | 6.101 → 4.833 | 205.8 → 212.6 |
| SVR | 12 | 16.16 → 13.05 | 6.56 → 5.425 | 156.1 → 143.6 |
| SVR | 24 | 18.38 → 16.72 | 7.876 → 6.991 | 157.6 → 164.1 |
