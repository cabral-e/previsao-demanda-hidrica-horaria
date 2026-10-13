# ETAPA 01 — Confirmações do usuário

Registrado em: 2026-10-05

## Significado das variáveis (informado pelo usuário)

| Coluna | Descrição | Unidade |
|---|---|---|
| B1, B2, B3 | Bombas de sucção (frequência; máximo 60 Hz) | Hz |
| LT01 | Nível do reservatório (Câmara 1) — **variável-alvo** | m³ (informado) |
| FT01 | Vazão de entrada | m³/h |
| FT02 | Vazão de gravidade | m³/h |
| FT03 | Vazão de recalque | m³/h |
| PT01 (PT01SU) | Pressão de sucção | mca |
| PT02 (PT02RBAL) | Pressão de recalque | mca |

Variáveis consideradas importantes pelo usuário: vazão, pressão e nível.

## Confronto com os dados

- **B1–B3:** máximo observado 59,99 Hz, valores contínuos de 0 a 60. Consistente.
- **PT01 × LT01:** correlação 0,961; `PT01 − LT01` com mediana ≈ 0,93. A pressão de sucção em mca
  equivale à altura da lâmina d'água sobre a tomada da bomba, mais um desnível fixo de ≈ 0,93 m.
  Isso só é fisicamente coerente se `LT01` estiver em **metros**. Implicação: PT01 é quase
  redundante com o alvo. Usá-la no mesmo instante como exógena vazaria o alvo.
- **LT01 em m³ — inconsistência a resolver:** a faixa observada é de 0,26 a 4,46. Como volume de
  reservatório urbano, 4,46 m³ é implausível (vazões de entrada de ~265 m³/h encheriam esse
  volume em ~1 minuto). Os valores são compatíveis com **nível em metros** ou, menos provável,
  com volume em **milhares de m³**. Os gráficos usarão o rótulo definido após a confirmação.

### Evidência da tela supervisória (SCADA Sanepar, Bairro Alto, 20/08/2021)

O usuário enviou capturas da tela do supervisório. O mostrador de cada câmara exibe **dois valores**:
nível em **m** (caixa laranja) e volume em **m³** (abaixo).

| Tag | Nível (m) | Volume (m³) | m³ por m |
|---|---:|---:|---:|
| LT01 (RAP 01, Câmara 1) | 4,29 | 2382 | 555,2 |
| LT02 (RAP 01, Câmara 2) | 4,24 | 2353 | 555,0 |
| LT03 (RAP 02, Câmara 1) | 4,31 | 2393 | 555,2 |
| LT04 (RAP 02, Câmara 2) | 4,13 | 2293 | 555,2 |

- A relação é linear e constante, ≈ 555 m³ por metro (área da câmara ≈ 555 m²).
- RAP 01 = 5000 m³ em 2 câmaras, ou seja, ≈ 2500 m³ por câmara. 2500 / 555 ≈ 4,5 m, que coincide
  com o máximo observado na série (4,46).
- A faixa da série (0,26 a 4,46) corresponde ao valor **em metros** do mostrador. O equivalente em
  volume é V ≈ 555 × LT01 (≈ 144 a 2476 m³).
- Por ser uma transformação linear, o ranking dos modelos não muda com a unidade. MAE e RMSE
  apenas são multiplicados por ≈ 555.
- Na tela, PT01 (tanque de sucção) = 5,1 mca e LT01 = 4,29 m. A diferença de ≈ 0,8 é coerente
  com o desnível de ≈ 0,93 estimado a partir dos dados.

### Decisão do usuário (2026-10-05)

1. Modelar e reportar `LT01` em **metros**.
2. Exibir também o equivalente em **m³** (fator 555,2 m³/m, origem: tela do supervisório) em
   tabelas e gráficos finais.
3. **Não converter** a série. A conversão é feita apenas na apresentação.

## Horário de verão

O usuário **não consegue confirmar** com a Sanepar a hipótese de horário de verão (hora ausente em
2018-11-04 00:00 e hora duplicada em 2018-12-31 21:00). O tratamento será decidido na ETAPA 04
com base apenas nos dados, documentando a incerteza como limitação.
