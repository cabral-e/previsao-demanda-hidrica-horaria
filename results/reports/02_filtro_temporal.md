# ETAPA 02 — Filtro temporal

Gerado em: 2026-10-05T15:22:13 (artefato reutilizado)

## Antes × depois

| | Início | Fim | Registros |
|---|---|---|---:|
| Base original | 2018-01-01 00:00:00 | 2020-12-31 23:00:00 | 26280 |
| Selecionado (2018–2019) | 2018-01-01 00:00:00 | 2019-12-31 23:00:00 | 17520 |
| Excluído | 2020-01-01 00:00:00 | 2020-12-31 23:00:00 | 8760 |

### Registros por ano no período selecionado

| ano | registros | esperado | diferenca |
|---|---|---|---|
| 2018 | 8760 | 8760 | 0 |
| 2019 | 8760 | 8760 | 0 |

## Anomalias temporais que permanecem (tratamento na ETAPA 04)

- Ordenada crescente: True
- Lacunas: 2018-11-04 00:00:00 a 2018-11-04 00:00:00 (1 h)
- Timestamps duplicados: 2018-12-31 21:00:00
- Grade horária completa esperada: 17520 registros; presentes 17520 (inclui 1 duplicado)

## Rastreabilidade

- Entrada: `data/interim/01_raw_loaded.parquet` — SHA-256 `b0938030607041253ddf14168949f9a1c63b4a4fd875501143ba94bca86a84b3`
- Saída: `data/interim/02_filtrado_2018_2019.parquet` — SHA-256 `075d201bc572d3631f128a61882414bb5c05ab6cf467e615de1528aedf5092cc`
- Hash da configuração (data.timestamp_column, data.start, data.end): `603aa2736e2a8375`
- Critério: `data.start` ≤ Timestamp ≤ `data.end` (inclusivo); nenhuma linha alterada.
