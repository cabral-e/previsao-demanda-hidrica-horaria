# Registro de decisões metodológicas

Decisões tomadas pelo usuário, em ordem cronológica. Cada item aponta onde está aplicado.

| Data | Etapa | Decisão | Onde |
|---|---|---|---|
| 2026-10-05 | 00 | Ambiente conda em `<USUARIO>\envs\tcc-hidrico`, Python 3.11 | `00_ambiente.md` |
| 2026-10-05 | 00 | Versões fixadas (`requirements.txt` + `requirements-lock.txt`) | raiz do projeto |
| 2026-10-05 | 01 | `LT01` modelado e reportado em metros; equivalente em m³ exibido (× 555,2); série não convertida | `config.yaml` → `data.target_unit`, `data.volume_equivalent` |
| 2026-10-05 | 01 | Horário de verão não confirmável com a Sanepar; tratar com base nos dados | `config.yaml` → `data.dst_confirmation_available` |
| 2026-10-05 | 03 | `train_fraction` = 0,80 (corte em 2019-08-08 00:00) | `config.yaml` → `split` |
| 2026-10-05 | 03 | EDA em duas camadas (descritiva no total; inferencial só no treino) | `config.yaml` → `eda` |
| 2026-10-05 | 04 | Manter os 37 episódios de nível baixo (operação real, não remover) | `config.yaml` → `treatment.keep_low_level_episodes` |
| 2026-10-05 | 04 | Horário de verão: realinhar −1 h os 1.389 rótulos de 2018-11-04 01:00 até a 1ª ocorrência de 2018-12-31 21:00 (sequência contínua; nenhum valor criado ou descartado) | `config.yaml` → `treatment.dst_strategy`; `04_diagnostico.md` |
| 2026-10-05 | 04 | Interpolar LT01 em 2019-01-30 06–08 h (salto incompatível com o balanço de massa); manter 2019-04-28 12 h | `config.yaml` → `treatment.jump_*`; `04_diagnostico.md` |
| 2026-10-05 | 04 | EDA não é refeita sobre a série tratada (diferença desprezível: 3 valores + rótulos de 57 dias) | registrar como limitação |
| 2026-10-05 | 06 | Defasagens base 1, 2, 3, 6, 12, 24, 48, 168; avaliar 23 e 25 | `config.yaml` → `features` |
| 2026-10-05 | 06 | Médias/desvios móveis só com valores passados (janelas 6/24/168 h e 24/168 h); calendário seno/cosseno de hora e dia da semana; **mês não entra** | `config.yaml` → `features` |
| 2026-10-05 | 06 | 23 e 25 h avaliadas só no treino (validação cronológica interna); regra ≥ 1% de ganho médio de RMSE → resultado +0,66% → **não incluídas** | `06_atributos.md`, `06_selecao_atributos.json` |
| 2026-10-05 | 08 | SARIMA com s = 24, d = 0, D = 1 e calendário semanal exógeno (senos/cossenos ou indicadores) | `config.yaml` → `sarima` |
| 2026-10-05 | 07 | Protocolo de teste: todas as horas do teste como alvos, origem = alvo − h, ajuste único no treino, observações reais até cada origem | `config.yaml` → `evaluation` |
| 2026-10-05 | 08 | Ordem no limite da grade mantida (sem ampliar a busca); registrar como limitação | `08_sarima.md` |
| 2026-10-05 | 09 | SVR RBF, estratégia direta; padronização de X e y só no ajuste; tuning em 2 fases (12 + 2 ajustes por h) na validação interna do treino; escolha por RMSE de validação | `config.yaml` → `svr`; `09_svr.md` |
| 2026-10-05 | 10 | LightGBM, estratégia direta; lr 0,05 fixo; árvores por parada antecipada na validação interna; grade 3×2×2 por h; ajuste final no treino inteiro com o nº de árvores encontrado | `config.yaml` → `lightgbm`; `10_lightgbm.md` |
| 2026-10-05 | 11 | Estratégias: baselines por indexação; SARIMA recursivo (Kalman); SVR e LightGBM diretos; variante LightGBM recursiva (modelo de 1 passo iterado) para isolar o efeito da estratégia | `11_horizontes_estrategias.md` |
| 2026-10-05 | 12 | Incluir teste de Diebold-Mariano na ETAPA 12 | ETAPA 12 |
| 2026-10-05 | 13 | Remedição controlada de todos os custos em lote único (aplicativos fechados, na tomada); versão mais recente (v002) é a oficial; v001 preservada | `13_custo.md`, `13_estado_maquina.json` |
| 2026-10-05 | 14 | Score composto: normalização B principal (A, C robustez); RMSE principal (MAE verificação); α de 0,1 a 0,9; variante com tuning | `config.yaml` → `composite_score` |
| 2026-10-05 | 14 | Conjunto principal do score = somente os 4 treináveis; com baseline = análise de sensibilidade (efeito estrutural: score do baseline = α, pois E = 1 e T = 0) | `config.yaml` → `composite_score.main_set`; `14_score.md` |
| 2026-10-05 | 17 | Figuras: previsão por modelo (melhor baseline por h numa figura), eixo x a cada 500 (teste completo) e 100 (1.300 amostras), curva de eficiência = RMSE × horizonte com custo na legenda; vírgula decimal (ABNT) | `config.yaml` → `plots`; `17_figuras.md` |
| 2026-10-05 | 13 | Convenção de custo: modelo de ajuste único (SARIMA) recebe o tempo do ajuste completo em cada horizonte; explicitar que serve aos 4 h | ETAPA 13 |
| 2026-10-05 | 08 | Seleção controlada: 8 ordens + 3 exógenas nas últimas 26 semanas do treino, menor BIC → SARIMA(2,0,1)(1,1,1)24 + Fourier semanal K = 3; ajuste final no treino inteiro | `08_sarima_selecao.json`, `08_sarima.md` |
