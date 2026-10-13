# ETAPA 05 — Divisão temporal

Gerado em: 2026-10-05T15:54:05

Divisão **cronológica**, sem embaralhar. Corte calculado sobre a grade horária do período
(`train_fraction` = 0.8); o mesmo corte usado na EDA (ETAPA 03).

| Conjunto | Início | Fim | Registros | Fração | Dias |
|---|---|---|---:|---:|---:|
| Treino | 2018-01-01 00:00:00 | 2019-08-07 23:00:00 | 14016 | 0.8000 | 584 |
| Teste | 2019-08-08 00:00:00 | 2019-12-31 23:00:00 | 3504 | 0.2000 | 146 |

**Corte (primeiro instante do teste): 2019-08-08 00:00:00**

## LT01 por conjunto

| Conjunto | média | desvio | mín | mediana | máx |
|---|---:|---:|---:|---:|---:|
| Treino | 3.4870 | 0.6835 | 0.2941 | 3.6807 | 4.4451 |
| Teste | 3.6163 | 0.6022 | 1.3752 | 3.8334 | 4.4043 |

## Validações

- tamanhos_somam: OK
- sem_sobreposicao: OK
- treino_antes_do_corte: OK
- teste_a_partir_do_corte: OK
- teste_comeca_no_corte: OK
- contiguos: OK
- ordem_preservada: OK
- nao_vazios: OK

## Notas para as etapas seguintes

- Atributos defasados no início do teste usam as últimas observações do treino (até 168 h).
  Isso é legítimo: na origem de cada previsão, esse passado já é conhecido. Nenhum valor do
  teste entra no ajuste de modelos, escalonadores ou na seleção de hiperparâmetros.
- Qualquer validação para ajuste de hiperparâmetros será feita **dentro do treino**, também
  de forma cronológica (ETAPAS 09–10).

## Rastreabilidade

- Entrada: `data/processed/04_tratado.parquet` — SHA-256 `14722d7a19c40a48e444d925ddeadbf4515cb362ebd31d1c1d5b2dde88da9d9b`
- Hash da configuração (data.timestamp_column, data.start, data.end, split): `8ad573026668de7a`
