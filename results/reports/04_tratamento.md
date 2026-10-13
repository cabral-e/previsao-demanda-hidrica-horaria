# ETAPA 04 — Tratamento

Gerado em: 2026-10-05T15:51:10

Configuração: `{"keep_low_level_episodes": true, "dst_strategy": "realign", "duplicate_strategy": null, "jump_strategy": "interpolate_segments", "jump_segments": [["2019-01-30 06:00", "2019-01-30 08:00"]]}`

## Transformações (em ordem)

1. Realinhamento −1 h de 1389 registros (2018-11-04 01:00:00 até a 1ª ocorrência de 2018-12-31 21:00:00).
2. Grade horária regular; pontos interpolados: 0 [].
3. LT01 interpolado linearmente em 3 registro(s), trechos [['2019-01-30 06:00', '2019-01-30 08:00']].
4. Episódios de nível baixo mantidos (decisão do usuário).

## Validações

- ordenado: OK
- sem_duplicados: OK
- sem_lacunas: OK
- frequencia_1h: OK
- sem_nan: OK
- sem_inf: OK

Registros: 17520 → 17520 (2018-01-01 00:00:00 a 2019-12-31 23:00:00).

Saída: `data/processed/04_tratado.parquet` — SHA-256 `14722d7a19c40a48e444d925ddeadbf4515cb362ebd31d1c1d5b2dde88da9d9b`; hash da configuração `4e28c432dc960ca2`.
