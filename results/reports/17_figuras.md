# ETAPA 17 — Visualizações

Gerado em: 2026-10-05T19:43:26

Estilo da referência (`docs/referencia_grafico_previsao.png`): fundo branco, borda preta, grade clara, fonte serifada, observado em preto contínuo e previsto em vermelho tracejado, legenda abaixo. PNG 300 dpi + PDF.

## Resultados — `results/figures/resultados/` (18 figuras)

- `17_curva_eficiencia.png`
- `17_mae_modelo_horizonte.png`
- `17_previsao_baseline_primeiras_1300.png`
- `17_previsao_baseline_teste_completo.png`
- `17_previsao_lightgbm_primeiras_1300.png`
- `17_previsao_lightgbm_rec_primeiras_1300.png`
- `17_previsao_lightgbm_rec_teste_completo.png`
- `17_previsao_lightgbm_teste_completo.png`
- `17_previsao_sarima_primeiras_1300.png`
- `17_previsao_sarima_teste_completo.png`
- `17_previsao_svr_primeiras_1300.png`
- `17_previsao_svr_teste_completo.png`
- `17_rmse_modelo_horizonte.png`
- `17_score_principal.png`
- `17_score_sensibilidade_alpha.png`
- `17_score_sensibilidade_com_baseline.png`
- `17_tempo_computacional.png`
- `17_tradeoff_pareto.png`

## EDA (ETAPA 03) — `results/figures/eda/`

- `03_acf_pacf_diferenca.png`
- `03_acf_pacf_nivel.png`
- `03_boxplot_dia_semana.png`
- `03_boxplot_hora.png`
- `03_boxplot_mes.png`
- `03_decomposicao_mstl.png`
- `03_decomposicao_mstl_3_semanas.png`
- `03_distribuicao_lt01.png`
- `03_heatmap_hora_dia.png`
- `03_janela_2_semanas.png`
- `03_perfil_horario.png`
- `03_serie_lt01.png`

Notas: previsão × observado em duas janelas (teste completo, marcações a cada 500; primeiras 1.300 amostras, a cada 100). A figura "baseline" mostra o melhor baseline de cada horizonte (indicado na faixa). Curva de eficiência = RMSE de teste × horizonte; custo para servir os 4 horizontes na legenda. ACF e PACF estão na mesma figura (painéis empilhados).
