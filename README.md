# Previsão horária de demanda hídrica: desempenho preditivo × custo computacional

Código, resultados e relatórios do Trabalho de Conclusão de Curso (pós-graduação em Análise de Dados e Big Data)
*Análise do trade-off entre desempenho preditivo e custo computacional em modelos de previsão de demanda hídrica*.

O estudo compara, em horizontes de **1, 6, 12 e 24 horas**, três baselines ingênuos, **SARIMA**, **SVR** e
**LightGBM** (estratégias direta e recursiva) na previsão do nível horário de um reservatório da Sanepar
(bairro Alto, Curitiba, 2018–2019), medindo de forma conjunta **erro** (MAE, RMSE, teste de Diebold-Mariano) e
**custo computacional** (tempo de treino, de previsão e memória), com score composto e fronteira de Pareto.

## Principais resultados

| Modelo | RMSE 1 h | RMSE 6 h | RMSE 12 h | RMSE 24 h | Custo para os 4 horizontes |
|---|---:|---:|---:|---:|---:|
| Melhor baseline | 0,197 | 0,521 | 0,521 | 0,521 | ~0,005 s |
| SARIMA | 0,116 | 0,375 | 0,404 | 0,400 | 370 s |
| SVR | 0,092 | 0,299 | 0,365 | 0,397 | 99 s |
| LightGBM direto | **0,080** | 0,280 | 0,362 | **0,390** | **5 s** |
| LightGBM recursivo | **0,080** | **0,272** | **0,352** | 0,395 | 20 s |

RMSE de teste em metros (nível da câmara; volume ≈ 555,2 m³ por metro).

- O LightGBM obteve o menor erro em todos os horizontes e **dominou, em erro e tempo, o SVR e o SARIMA**.
- A vantagem dos modelos de aprendizado de máquina diminui com o horizonte; em 24 h, os quatro modelos treináveis
  são estatisticamente equivalentes (Diebold-Mariano com correção de Holm).
- Mais custo computacional não resultou em mais acurácia, o que sustenta a hipótese do trabalho.

Relatório completo: [`results/reports/relatorio_final.md`](results/reports/relatorio_final.md) ·
decisões metodológicas: [`results/reports/decisoes.md`](results/reports/decisoes.md) ·
tabela final: [`results/metrics/resultados_finais.csv`](results/metrics/resultados_finais.csv) ·
figuras: [`results/figures/`](results/figures/).

## Estrutura

```
config/config.yaml        parâmetros e decisões do experimento (fonte única de configuração)
src/data/                 carga, inspeção, filtro, EDA, tratamento e divisão temporal
src/features/             atributos (sem informação futura) e seleção de defasagens
src/models/               baselines, SARIMA, SVR, LightGBM (direto e recursivo)
src/evaluation/           protocolo, métricas, Diebold-Mariano, custo, score composto, Pareto, consolidação
src/visualization/        estilo e figuras
src/utils/                caminhos, logging, persistência, ambiente e preparação para publicação
tests/                    79 testes automatizados (pytest), incluindo verificação de vazamento de informação
results/reports/          relatório de cada etapa (00–17), decisões e relatório final
results/metrics/          métricas, custos, scores e tabela final (CSV e Parquet)
results/figures/          figuras da EDA e dos resultados (PNG 300 dpi e PDF)
results/experiments/      registro de experimentos (registry.csv)
```

## Dados

A base original da Sanepar (`SANEPAR_B_ALTO_2018_20.xlsx`) **não é distribuída** neste repositório, assim como os
dados intermediários e os arquivos de previsão com a série completa. Os resultados, métricas e figuras derivados
são publicados com autorização da Sanepar. Para reproduzir, coloque a planilha na pasta acima do projeto
(caminho definido em `config/config.yaml → paths.raw_source`).

## Reprodução

Ambiente: Python 3.11, versões fixadas em `requirements.txt` (diretas) e `requirements-lock.txt` (completas).

```
conda create -p <pasta-de-ambientes>/tcc-hidrico python=3.11 pip -y
<pasta-de-ambientes>/tcc-hidrico/python -m pip install -r requirements.txt
```

Na raiz do projeto, cada etapa reutiliza os resultados persistidos quando a entrada e a configuração não mudaram
(controle por hash); use `--force` para recalcular.

```
python -m src.utils.env_info                     # 00 ambiente
python -m src.data.inspection                    # 01 inspeção
python -m src.data.temporal_filter               # 02 filtro 2018–2019
python -m src.data.eda                           # 03 análise exploratória
python -m src.data.treatment --diagnose          # 04 evidências para o tratamento
python -m src.data.treatment                     # 04 tratamento
python -m src.data.split                         # 05 divisão temporal
python -m src.features.lag_evaluation            # 06 atributos
python -m src.models.baseline                    # 07 baselines
python -m src.models.sarima --select             # 08 seleção do SARIMA (~15 min)
python -m src.models.sarima                      # 08 ajuste final (~25 min)
python -m src.models.svr --probe                 # 09 sondagem do SVR
python -m src.models.svr --tune                  # 09 tuning (~18 min)
python -m src.models.svr                         # 09 ajuste final
python -m src.models.lightgbm_model --tune       # 10 tuning do LightGBM
python -m src.models.lightgbm_model              # 10 ajuste final
python -m src.models.lightgbm_model --recursive  # 11 variante recursiva
python -m src.evaluation.strategy_check          # 11 verificação dos pontos de avaliação
python -m src.evaluation.metrics_report          # 12 métricas e Diebold-Mariano
python -m src.evaluation.cost_batch              # 13 remedição controlada de custos (~40 min)
python -m src.evaluation.cost_report             # 13 consolidação de custos
python -m src.evaluation.composite               # 14 score composto
python -m src.evaluation.pareto                  # 15 fronteira de Pareto
python -m src.evaluation.consolidate             # 16 tabela final
python -m src.visualization.make_figures         # 17 figuras
python -m pytest                                 # testes
python -m src.utils.publish                      # remove caminhos locais antes de publicar
```

Os tempos medidos dependem do hardware (referência: Intel Core i5-10210U, 1 thread, Windows 11).

## Licença

- Código: [MIT](LICENSE).
- Textos, relatórios e figuras: [CC BY 4.0](LICENSE-CONTENT.md).
