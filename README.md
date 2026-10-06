# Detección de fraude en verificación de identidad

Sistema que estima el riesgo de fraude de cada intento de verificación de identidad y decide
automáticamente **aprobar**, **verificar** (OTP, nueva selfie o analista) o **rechazar**, casi en
tiempo real, explicando cada decisión.

## Resultados

| | Valor |
|---|---|
| Modelo | LightGBM monótono (22 variables) |
| PR-AUC en test | **0,544** (IC 95 %: 0,43–0,64), frente a 0,163 de un modelo al azar |
| ROC-AUC en test | 0,811 |
| Recall con 5 % de falsos positivos | 41,6 % |
| Calibración (Brier) | 0,102: el score se puede leer como probabilidad de fraude |

La validación cruzada en train (PR-AUC 0,549 ± 0,014) coincide con test: el modelo no está sobreajustado.

**Política de decisión** (rechazar a un cliente bueno cuesta 2 veces lo que cuesta un fraude, supuesto S6):

| Decisión | Regla | % del tráfico en test |
|---|---|---|
| Aprobar | score < 0,28 | 81,0 % |
| Verificar | 0,28 ≤ score < 0,83 | 17,4 % |
| Rechazar | score ≥ 0,83 | 1,5 % |

Detiene el **43 %** del fraude, pierde el **1,3 %** de los clientes buenos (solo el 0,17 % por rechazo
directo) y cuesta **24,5 % menos** que no tener modelo.

## Estructura

```
config.yaml                  Todas las decisiones: variables, exclusiones y su motivo, costos
notebooks/01_eda.ipynb       Análisis exploratorio: pregunta → evidencia → conclusión
src/
├── config.py                Lee config.yaml
├── data.py                  Carga, valida, limpia, excluye periodos sesgados y parte por cuenta
├── features.py              Variables calculadas solo con información pasada
├── models.py                El Pipeline (preprocesamiento + modelo) y las métricas
├── train.py                 Validación cruzada, comparación de modelos y modelo final
├── decision.py              Umbrales por costo esperado: aprobar / verificar / rechazar
├── evaluate.py              Evaluación final en test y gráficos
├── inference.py             Puntúa procesos nuevos y explica cada decisión
├── api.py                   API REST (FastAPI)
├── monitor.py               Deriva de datos (PSI), de decisiones y desempeño con etiquetas
└── pipeline.py              Ejecuta datos → entrenamiento → umbrales → evaluación
tests/                       42 tests: datos, fuga temporal, decisión, explicabilidad, inferencia y API
docs/
├── analisis_extendido.md    El análisis completo, con las respuestas a los 4 escenarios
├── supuestos.md             Supuestos con su evidencia e impacto
├── slides.pdf               Slides de la sustentación
└── figuras/                 Gráficos del análisis
```

## Uso

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.11+. Copia el dataset en
`data/raw/data_anonymized.csv` (los datos no se incluyen en el repositorio).

```bash
uv sync
```

**Entrenar todo** (datos → modelo → umbrales → evaluación):

```bash
uv run python -m src.pipeline
```

**Puntuar datos nuevos** (un CSV con el formato original; la etiqueta no hace falta). El pipeline deja
ejemplos listos en `data/sample/`:

```bash
uv run python -m src.inference --input data/sample/nuevos.csv
```

Cada proceso sale en `reports/decisiones.csv` con su score, su decisión y, si se verifica o se
rechaza, sus motivos (por ejemplo, "Rechazos previos por señales de fraude: 4").

**API** (documentación interactiva con ejemplos en http://127.0.0.1:8000/docs):

```bash
uv run uvicorn src.api:app
```

**Monitorear** las predicciones registradas, con o sin etiquetas:

```bash
uv run python -m src.monitor --etiquetas data/sample/etiquetas.csv
```

Para ver una alerta, puntúa el lote con deriva simulada y monitoréalo:

```bash
uv run python -m src.inference --input data/sample/nuevos_deriva.csv --output reports/decisiones_deriva.csv --log logs/deriva.jsonl
uv run python -m src.monitor --log logs/deriva.jsonl
```

**Tests:**

```bash
uv run pytest -q
```

## Decisiones principales

- **Exclusión con evidencia.** Las 25 columnas de dispositivo, red y geografía fueron aleatorizadas
  por la anonimización: un modelo con todas ellas obtiene AUC 0,49, igual que con la etiqueta barajada.
- **Sin fuga temporal.** Cada variable solo usa filas anteriores; un test lo verifica recalculando
  con fechas de corte.
- **Validación honesta.** Partición agrupada por cuenta (la etiqueta es casi por cuenta) y sin los
  periodos con sesgo de selección, que inflaban el PR-AUC de 0,52 a 0,62.
- **Sin el resultado del proceso actual** (`STATUS`, `DECLINED_REASON`): con él el PR-AUC sube a 0,61,
  pero no se conoce al decidir.
- **Probabilidades calibradas**, sin SMOTE ni ponderación de clases: la decisión usa el score como
  probabilidad.
- **Decisión por costo esperado.** Los umbrales salen de los costos; cambiarlos no requiere reentrenar.
- **Explicable por diseño.** Restricciones monótonas (más señal de riesgo nunca baja el score) y
  motivos que son valores SHAP exactos.
- **Mismo código en entrenamiento y producción**, verificado por un test.

## Comparación (validación cruzada, 5 folds agrupados por cuenta)

| Experimento | PR-AUC | ROC-AUC | Recall @ 5 % FPR |
|---|---|---|---|
| **Principal (LightGBM monótono)** | **0,549 ± 0,014** | 0,818 | 0,394 |
| Línea base (regresión logística) | 0,511 ± 0,021 | 0,806 | 0,350 |
| Sin las 7 variables nuevas | 0,524 ± 0,018 | 0,810 | 0,372 |
| Sin `DISTRUST` | 0,532 ± 0,015 | 0,807 | 0,366 |
| Con fuga (`STATUS`, `DECLINED_REASON` del intento actual) | 0,613 ± 0,011 | 0,846 | 0,457 |

