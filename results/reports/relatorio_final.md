# Relatório final — Trade-off entre desempenho preditivo e custo computacional na previsão de demanda hídrica

Gerado em: 2026-10-05. Todos os números provêm dos artefatos do projeto (`results/reports/00` a `17`,
`results/metrics/`, `results/experiments/registry.csv`); nenhum valor foi estimado ou ajustado manualmente.
Decisões metodológicas: `results/reports/decisoes.md`.

---

## 1. Contexto

A previsão de curto prazo do nível de reservatórios apoia a operação de sistemas de abastecimento (acionamento de
bombas, manobras, garantia de reservação). Modelos mais sofisticados tendem a reduzir o erro, mas exigem mais
tempo de desenvolvimento, treino, previsão e memória. Este estudo avalia esse equilíbrio de forma empírica, com
dados horários de um reservatório da Sanepar (bairro Alto, Curitiba), comparando um modelo estatístico clássico
(SARIMA), dois modelos de aprendizado de máquina (SVR e LightGBM) e baselines ingênuos, em quatro horizontes.

## 2. Questão de pesquisa

Qual é o equilíbrio ideal entre acurácia preditiva e custo computacional na previsão de demanda de água,
considerando diferentes modelos e horizontes de previsão?

## 3. Hipótese

Modelos mais complexos não necessariamente apresentam o melhor custo-benefício. A hipótese foi tratada como a ser
testada, sem presumir o resultado.

## 4. Dados

- **Fonte**: Sanepar, reservatório do bairro Alto (RAP 01, Câmara 1), arquivo `SANEPAR_B_ALTO_2018_20.xlsx`;
  cópia de trabalho com SHA-256 idêntico ao original (`41ef8a4c…6da6`).
- **Estrutura**: 26.280 registros horários (01/01/2018 a 31/12/2020), 10 colunas, sem valores ausentes.
- **Variável-alvo**: `LT01`, **nível da câmara em metros**. O equivalente em volume é V ≈ 555,2 × LT01 (m³),
  fator obtido da tela do supervisório da Sanepar (4,29 m ↔ 2.382 m³) e confirmado de forma independente pelo
  balanço de massa (ΔV × vazão líquida: r = 0,944, inclinação 0,88).
- **Demais variáveis** (não usadas como entrada; modelagem univariada): frequência das bombas de sucção B1–B3 (Hz),
  vazões de entrada, gravidade e recalque FT01–FT03 (m³/h), pressões de sucção e recalque PT01–PT02 (mca). A pressão
  de sucção PT01 tem correlação de 0,961 com o alvo e seria quase uma cópia dele como variável explicativa.
- **Período usado**: somente 2018 e 2019 (17.520 h); 2020 excluído (8.760 h), conforme o desenho do estudo.

## 5. Metodologia

Pipeline incremental, reprodutível e auditável, com checkpoint e validação pelo usuário a cada etapa:

1. Inspeção, filtro temporal e tratamento (ETAPAS 01–04).
2. Divisão **cronológica** 80/20 com corte em **2019-08-08 00:00**: treino de 01/01/2018 a 07/08/2019
   (14.016 h) e teste de 08/08/2019 a 31/12/2019 (3.504 h).
3. Análise exploratória em duas camadas: descritiva no período completo; inferencial (estacionariedade,
   autocorrelação, decomposição) **somente no treino**.
4. Engenharia de atributos sem informação futura; escolhas de atributos e hiperparâmetros feitas **só com o treino**
   (validação cronológica interna: últimos 20% do treino).
5. Modelos: 3 baselines, SARIMA, SVR, LightGBM direto e LightGBM recursivo.
6. Avaliação: MAE e RMSE (treino e teste), teste de Diebold-Mariano, custo computacional medido, score composto
   e fronteira de Pareto.

**Controle de vazamento de informação**: testes automatizados verificam que alterar qualquer valor posterior à
origem não muda nenhuma previsão (todos os modelos, todos os horizontes). Escalonadores ajustados só nos dados de
ajuste. Total de 79 testes automatizados aprovados.

**Ambiente**: Windows 11, Intel Core i5-10210U (4 núcleos / 8 threads), 7,86 GB de RAM, Python 3.11.17;
versões fixadas em `requirements.txt` e `requirements-lock.txt`; semente 42; **1 thread** em todos os modelos.

## 6. Análise exploratória (EDA)

- **Estacionariedade** (treino): ADF rejeita raiz unitária (p < 10⁻²¹) e KPSS não rejeita estacionariedade
  (p > 0,10) — série estacionária em nível.
- **Sazonalidade diária dominante**: ACF de 0,63 em 24 h; o ciclo de 24 h concentra 32% da potência espectral;
  força sazonal diária (MSTL) de 0,66. Perfil médio: mínimo às 22 h (2,70 m), máximo às 9 h (3,99 m).
- **Efeito semanal relevante**: força 0,46; ACF de 0,58 em 168 h. Nos dias úteis o nível cai a ~2,5 m entre 21 e
  23 h; nos fins de semana permanece em ~3,3 m (diferença de até 0,85 m às 22 h). Sábado tem a menor média
  (3,30 m) e domingo a maior (3,77 m).
- **Memória curta forte**: ACF de 0,96 em 1 h.
- **Diferença entre os períodos**: o teste (agosto a dezembro de 2019) é mais regular que o treino
  (média 3,62 m × 3,49 m; desvio 0,60 × 0,68; mínimo 1,38 × 0,29 m).

Implicações: SARIMA com período de 24 h e sem diferenciação simples; calendário semanal como informação
explicativa; defasagens de 1–3, 24 e 168 h como atributos.

## 7. Tratamento dos dados

| Problema | Evidência | Tratamento (decisão do usuário) |
|---|---|---|
| Hora ausente em 2018-11-04 00:00 e hora duplicada em 2018-12-31 21:00 | As variações entre linhas vizinhas são típicas de 1 h (percentil 74 em 1 h × 5 em 2 h): nenhuma medição foi perdida nem repetida; só os rótulos entre as duas anomalias estão adiantados 1 h (provável horário de verão aplicado pelo registrador) | **Realinhamento −1 h** de 1.389 rótulos; nenhum valor criado ou descartado |
| Salto de 2019-01-30, 06–08 h | Queda de 1,44 m em 1 h e retorno às 9 h; o balanço de massa prevê +79 m³ contra −800 m³ observados (25 desvios-padrão) | **Interpolação linear** de 3 valores de `LT01` |
| Salto de 2019-04-28, 12 h | Também incompatível com as vazões (19 desvios-padrão), mas o novo patamar persiste | Mantido |
| 37 episódios de nível baixo (167 h) | Concentrados entre 20 e 22 h, no pico de consumo | Mantidos (operação real) |
| Parada em 2018-02-27, 11–22 h | Entrada, saída e bombas zeradas; nível constante fisicamente coerente | Mantida |

Resultado: 17.520 h contínuas, sem lacunas nem duplicatas. Todas as alterações estão no período de treino.

## 8. Modelos

| Modelo | Configuração final | Seleção (somente treino) | Custo de *tuning* |
|---|---|---|---:|
| Persistência | ŷ(T) = y(T − h) | — | — |
| Sazonal diária | ŷ(T) = y(T − 24) | — | — |
| Sazonal semanal | ŷ(T) = y(T − 168) | — | — |
| SARIMA | (2,0,1)(1,1,1)₂₄ + senos/cossenos semanais (K = 3) | 8 ordens + 3 formas de calendário, menor BIC, nas últimas 26 semanas do treino | 890 s |
| SVR (RBF) | Um modelo por horizonte; C ∈ {1, 10}, γ ∈ {0,018; 0,059}, ε ∈ {0,05; 0,1; 0,2} | Grade em 2 fases (56 ajustes); X e y padronizados no ajuste | 1.095 s (+ 44 s de sondagem) |
| LightGBM direto | Um modelo por horizonte; 15–31 folhas; 83–819 árvores (parada antecipada interna) | Grade 3 × 2 × 2 por horizonte (48 ajustes) | 75 s |
| LightGBM recursivo | Modelo de 1 h aplicado h vezes | Reaproveita os hiperparâmetros de h = 1 | (o mesmo) |

**Atributos dos modelos de aprendizado de máquina** (17): defasagens 1, 2, 3, 6, 12, 24, 48 e 168 h; médias
móveis de 6, 24 e 168 h e desvios móveis de 24 e 168 h (somente valores passados); seno e cosseno da hora e do
dia da semana do instante-alvo (sem mês). As defasagens 23 e 25 h foram testadas e excluídas por regra
pré-registrada (ganho médio de +0,66% < 1%).

Achados: a ordem do SARIMA e parte dos hiperparâmetros do SVR ficaram no limite das grades (não ampliadas, por
desenho); o coeficiente sazonal de média móvel do SARIMA ficou em −0,968, indicando padrão diário muito estável.

## 9. Estratégia experimental

- **Horizontes**: 1, 6, 12 e 24 h (= passos, pela frequência real de 1 h).
- **Estratégias**: baselines por indexação; SARIMA recursivo (filtro de Kalman com parâmetros fixos); SVR e
  LightGBM diretos; LightGBM recursivo como variante para isolar o efeito da estratégia.
- **Protocolo comum**: cada modelo é ajustado uma vez no treino; no teste, todas as 3.504 horas são alvos em cada
  horizonte (origem = alvo − h), usando só observações reais até a origem. **Alvos e origens são idênticos entre
  todos os modelos** (verificado nos arquivos de previsão: 56 combinações).
- Métricas de treino calculadas sobre os mesmos 13.824 alvos para todos (após 192 h de aquecimento).

## 10. Métricas de erro

RMSE e MAE de teste, em metros (equivalente em volume: × 555,2 m³):

| Modelo | RMSE 1 h | RMSE 6 h | RMSE 12 h | RMSE 24 h | MAE 1 h | MAE 6 h | MAE 12 h | MAE 24 h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Melhor baseline | 0,197 | 0,521 | 0,521 | 0,521 | 0,149 | 0,371 | 0,371 | 0,371 |
| SARIMA | 0,116 | 0,375 | 0,404 | 0,400 | 0,080 | 0,278 | 0,300 | 0,295 |
| SVR | 0,092 | 0,299 | 0,365 | 0,397 | 0,059 | 0,207 | 0,251 | 0,277 |
| LightGBM direto | **0,080** | 0,280 | 0,362 | **0,390** | **0,052** | 0,195 | 0,255 | 0,279 |
| LightGBM recursivo | **0,080** | **0,272** | **0,352** | 0,395 | **0,052** | **0,184** | **0,238** | **0,276** |

(Melhor baseline: persistência em 1 h; sazonal semanal em 6–24 h. Em 1 h, os dois LightGBM são o mesmo modelo.)

**Significância (Diebold-Mariano com variância HAC, correção HLN e Holm, 5%)**:
- Todos os modelos treináveis superam o melhor baseline em todos os horizontes.
- 1 e 6 h: os LightGBM são significativamente melhores que SVR e SARIMA.
- 12 h: LightGBM e SVR equivalentes, ambos melhores que o SARIMA.
- **24 h: os quatro treináveis são estatisticamente equivalentes.**
- LightGBM direto × recursivo: sem diferença significativa na perda quadrática; o recursivo é melhor na perda
  absoluta em 6 e 12 h.

**Generalização** (RMSE teste/treino): SARIMA 0,88–1,08; SVR 1,09–1,16; LightGBM 1,03–1,33. Os modelos de
aprendizado de máquina ajustam-se mais ao treino, mas têm menor erro absoluto no teste. Razões < 1 nos baselines
refletem o teste mais regular.

## 11. Custo computacional

Medido (nunca estimado) em remedição controlada, lote único, máquina na tomada e aplicativos pesados fechados:
`time.perf_counter`, 3 repetições (mediana; dispersão < 3% nos treináveis), memória como pico de RSS em execução
separada. As métricas de erro foram idênticas entre medições (modelos determinísticos).

| Modelo | Modelos ajustados | Treino (s) | Previsão, soma dos 4 h (s) | Total, 4 h (s) | Memória (MB) | *Tuning* (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baselines | 0 | ~0 | 0,005 | 0,005 | ~0 | 0 |
| LightGBM direto | 4 | 4,3 | 0,66 | **5,0** | 6–10 | 75 |
| LightGBM recursivo | 1 | 2,4 | 17,4 | 19,7 | 13–14 | 75 |
| SVR | 4 | 77,2 | 21,6 | 98,7 | 144–213 | 1.095 |
| SARIMA | 1 | 368,7 | 1,6 | 370,3 | 21 | 890 |

- Entre os treináveis, os custos totais variam ~75×; com os baselines, 5 ordens de grandeza.
- O *tuning* custa mais que o treino final em todos os treináveis: SARIMA 2,4×, SVR 14×, LightGBM 17×.
- O SVR usa ~10× mais memória (vetores de suporte).
- No LightGBM recursivo, a previsão domina o custo e cresce linearmente com h (0,4 s em 1 h; 9,6 s em 24 h).

## 12. Resultados

- **Menor erro**: LightGBM em todos os horizontes (direto ou recursivo); média nos 4 horizontes: LightGBM
  recursivo (RMSE 0,275 m; MAE 0,188 m), seguido do direto (0,278 m; 0,195 m).
- **Ganho sobre o melhor baseline**: redução de RMSE de 22% a 59% entre os treináveis; o maior ganho ocorre em
  1 h (LightGBM: −59%).
- **Menor custo entre os treináveis**: LightGBM direto (5 s para os 4 horizontes).
- **Score composto principal** (normalização B, somente treináveis, α = 0,5): LightGBM direto em 6, 12 e 24 h
  (0,269; 0,348; 0,375); empate dos dois LightGBM em 1 h (~0,20). SVR (0,49–0,67) e SARIMA (0,80–0,89) por
  último. **O vencedor é o mesmo para todo α de 0,1 a 0,9** e se mantém nas normalizações A e C, com MAE e com o
  custo de *tuning* somado (alternando apenas entre os dois LightGBM).

## 13. Trade-off

- **Fronteira de Pareto** (erro × tempo): apenas os LightGBM. **SVR e SARIMA são dominados em todos os
  horizontes** — há um modelo mais rápido e mais preciso que cada um deles.
- O único trade-off real ocorre dentro da família LightGBM, em 6 e 12 h: o recursivo custa 5,5× e 9,4× mais
  tempo para reduzir o RMSE em 2,5% e 2,6% (~0,01 m por década de tempo), ganho não significativo na perda
  quadrática. Em 24 h o direto domina o recursivo.
- **Análise de sensibilidade com o baseline no score**: por construção, o baseline tem score igual a α
  (E = 1, T = 0); como leva milissegundos, vence para α ≤ 0,5–0,6, e o LightGBM só passa à frente com α ≥ 0,6–0,7.
  Isso decorre da escala logarítmica de tempo dominada pelos baselines, não de evidência de que não modelar seja
  preferível — todos os treináveis o superam com significância.

## 14. Discussão

- **A hipótese é sustentada.** O modelo de maior custo computacional (SARIMA, 370 s) e o segundo mais caro (SVR,
  99 s) não compraram acurácia: são dominados pelo LightGBM (5 s), que tem o menor erro. A ordem de custo entre os
  treináveis é praticamente a inversa da ordem de erro.
- **"Complexidade" precisa ser definida.** Em custo computacional, o SARIMA é o mais complexo; em capacidade de
  modelagem, SVR e LightGBM são os mais flexíveis. O melhor custo-benefício veio de um modelo flexível **e** barato,
  não do mais simples nem do mais caro.
- **O horizonte muda a magnitude, não o líder.** A vantagem do aprendizado de máquina diminui com h: em 24 h os
  quatro treináveis são estatisticamente equivalentes, e a escolha passa a depender quase só do custo — o que
  favorece ainda mais o LightGBM direto. A recursão ajuda em 6–12 h, mas com custo de previsão crescente.
- **Custo de desenvolvimento**: o *tuning* domina o custo total para quem desenvolve; o LightGBM também é o mais
  barato nesse quesito (75 s, contra 890 s do SARIMA e 1.095 s do SVR).
- **Implicação prática**: para previsão horária do nível deste reservatório, o LightGBM direto oferece o melhor
  equilíbrio em todos os horizontes; o recursivo é uma alternativa quando 6–12 h são prioritários e o custo de
  previsão (segundos) é irrelevante para a operação.
- **Sobre scores compostos**: o resultado do score depende do conjunto de modelos e da escala de tempo. Incluir um
  baseline de custo quase nulo em escala logarítmica favorece "não modelar" por construção; por isso a fronteira de
  Pareto e os testes de significância foram usados como complemento.

## 15. Limitações

- **`LT01` é um indicador indireto da demanda**: o nível resulta do balanço entre entrada, saídas e a operação das
  bombas, não só do consumo.
- **Uma única unidade** (uma câmara de um reservatório): a generalização para outros sistemas não é garantida.
- **Tempos dependentes da máquina local** e de execução com 1 thread; a ordem de custo pode mudar com paralelismo ou
  outro hardware. Havia processos de fundo durante a remedição (registrados).
- **Horário de verão não confirmado** com a Sanepar: o realinhamento se apoia na continuidade da amostragem; a hora
  do dia de 57 dias de 2018 pode conter erro de 1 h.
- **Causa dos saltos abruptos** (artefato de medição ou manobra entre câmaras) não confirmada; um deles permanece no
  treino.
- **Grades de busca pequenas por desenho**; ordem do SARIMA e parte dos hiperparâmetros do SVR no limite da grade.
- **Um único corte treino/teste**; o teste (agosto a dezembro) é mais regular que o treino e cobre só parte do ano.
- **Dois anos de dados** não permitem modelar a sazonalidade anual.
- **Pareto em estimativas pontuais**: diferenças de erro não significativas (por exemplo, em 24 h) devem ser lidas
  com cautela.
- A EDA foi feita sobre a série antes do tratamento (diferença desprezível: 3 valores e um deslocamento de rótulos).

## 16. Conclusão

Nos dados do reservatório do bairro Alto (2018–2019), **mais custo computacional não resultou em mais acurácia**.
O LightGBM, com treino de segundos, obteve o menor erro em todos os horizontes, reduziu o RMSE em até 59% em
relação ao melhor baseline e dominou, em erro e tempo, o SVR e o SARIMA — este último cerca de 75× mais caro. O
equilíbrio ideal é o LightGBM direto, com o recursivo como alternativa marginal em 6–12 h. A vantagem dos modelos
de aprendizado de máquina diminui com o horizonte e, em 24 h, os modelos treináveis tornam-se estatisticamente
equivalentes, de modo que o custo passa a ser o critério decisivo. A hipótese de que modelos mais complexos não
necessariamente apresentam o melhor custo-benefício é sustentada pelas três evidências usadas — métricas com
significância estatística, score composto robusto a α e fronteira de Pareto.

## 17. Trabalhos futuros

- Avaliação em múltiplas origens (*rolling origin*) e em períodos de teste de estações diferentes.
- Variáveis exógenas operacionais (vazões, estado das bombas) com defasagem adequada, e previsão direta da
  vazão de saída como medida mais próxima da demanda.
- Outras unidades e reservatórios da rede para testar a generalização.
- Medição de custo com paralelismo, em outro hardware e incluindo energia consumida.
- Grades de hiperparâmetros mais amplas ou busca bayesiana, com o custo de *tuning* explicitamente no trade-off.
- Modelos adicionais (por exemplo, redes recorrentes) e previsões probabilísticas (intervalos).
- Séries mais longas, para incorporar a sazonalidade anual.

---

### Rastreabilidade

| Seção | Fonte |
|---|---|
| Dados, inspeção | `01_inspecao.md`, `01b_confirmacoes_usuario.md`, `02_filtro_temporal.md` |
| EDA | `03_eda.md`, `results/figures/eda/` |
| Tratamento | `04_diagnostico.md`, `04_tratamento.md` |
| Divisão, atributos | `05_divisao.md`, `06_atributos.md` |
| Modelos | `07_baseline.md`, `08_sarima.md`, `09_svr.md`, `10_lightgbm.md`, `11_horizontes_estrategias.md` |
| Métricas, DM | `12_metricas.md`, `results/metrics/12_*.csv` |
| Custo | `13_custo.md`, `13_estado_maquina.json`, `results/metrics/13_*.csv` |
| Score, Pareto | `14_score.md`, `15_pareto.md` |
| Tabela final | `16_consolidacao.md`, `results/metrics/resultados_finais.csv` / `.parquet` |
| Figuras | `17_figuras.md`, `results/figures/resultados/` |
| Decisões | `decisoes.md` |
| Experimentos | `results/experiments/registry.csv` (versão oficial dos tempos: v002) |
