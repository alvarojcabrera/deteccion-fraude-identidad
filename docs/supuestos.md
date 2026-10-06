# Supuestos

Cada supuesto indica en qué se basa, qué decisión depende de él y qué cambiaría si fuera falso.
Las referencias "EDA §N" apuntan a las secciones de `notebooks/01_eda.ipynb`.

## Verificados con datos

| # | Supuesto | Evidencia | Impacto | Si es falso |
|---|---|---|---|---|
| S1 | Cada fila es un intento de verificación de identidad previo a aprobar una compra o registro en XX | `PROCESS_ID` único en las 17.781 filas; columnas de documento, rostro y dispositivo (EDA §1, §4) | El modelo puntúa intentos, no compras | Faltarían variables de monto y producto |
| S2 | `IS_FRAUD` es la etiqueta final y es casi por cuenta | 1.217 fraudes con `STATUS = success`; solo 159 de 4.343 cuentas con varios intentos mezclan etiquetas (EDA §7) | Partición agrupada por `ACCOUNT_ID` | Una partición aleatoria bastaría |
| S4 | Las variables de dispositivo, red y geografía fueron aleatorizadas por la anonimización | AUC ≈ 0,5 individual (EDA §8) y combinado (0,490, EDA §9.5); país, ciudad y zona horaria combinados al azar; velocidades de IP imposibles; `FIRST_SEEN_*` posteriores a todas las transacciones (EDA §9) | Se excluyen (`excluded.ruido_sintetico`) | Con datos reales serían las primeras señales a evaluar |
| S5 | Las señales de riesgo vienen de dos esquemas según el cliente | Parejas `DISTRUST`/`Distrust`, `ANOMALY`/`Anomaly`, `LIFETIME`/`Lifetime` nunca llenas a la vez; cada versión pertenece a un cliente (EDA §4, §5) | Se unifican (`unify`) y se agregan indicadores de faltante | Habría que tratarlas como variables distintas |
| S9 | `PROCESS_DATE` y `FirstTime` son redundantes | `PROCESS_DATE` = `DATE` en todas las filas (EDA §5); `FirstTime` equivale a `Lifetime` (correlación 0,99998) | Se excluyen | — |
| S10 | La distribución temporal no representa el tráfico real | Meses del cliente A con 100 % de fraude; los clientes no se solapan en el tiempo (EDA §6). Esas 295 filas inflaban el PR-AUC de validación de 0,52 a 0,62, y entrenar sin ellas no cambia el desempeño en tráfico representativo (Fase 3) | Validación agrupada por cuenta, no temporal. Las 295 filas sesgadas se excluyen de entrenamiento y evaluación, pero cuentan como historial para las variables (`biased_periods`) | Se usaría una partición temporal y se conservarían esas filas |
| S13 | El modelo debe servir para clientes nuevos | Agregar `CLIENTE` como variable no mejora el PR-AUC de forma relevante (0,549 con y sin `CLIENTE`) | `CLIENTE` no es variable del modelo; se conserva para estratificar, evaluar y monitorear | Se incluiría `CLIENTE` y un cliente nuevo necesitaría reentrenar |

## De negocio (no verificables con estos datos)

| # | Supuesto | Razón | Impacto | Si es falso |
|---|---|---|---|---|
| S3 | `STATUS` y `DECLINED_REASON` del intento actual no están disponibles al decidir | Son el resultado de la verificación; algunos motivos casi revelan la etiqueta (EDA §10) | Modelo principal sin ellos; variante con ellos como comparación | El modelo funcionaría como segunda capa después de la verificación |
| S6 | Rechazar a un cliente bueno cuesta **2 veces** lo que cuesta aprobar un fraude | En onboarding, un rechazo injusto pierde a un cliente nuevo: su LTV y el costo de adquirirlo; rara vez vuelve a intentarlo. Para XX (productos agrícolas, clientes recurrentes y estacionales) se asume que eso pesa más que la mercancía y el flete de un fraude. No hay montos ni márgenes en los datos para verificarlo | Fija los umbrales por costo esperado mínimo: con 2×, verificar si score ≥ 0,28 y rechazar si ≥ 0,83 | Es un parámetro (`decision.costs`). `reports/sensibilidad_costos.csv` muestra la política para 0,1× a 5×: con 0,5× se rechaza el 6,1 % y se detiene el 58 % del fraude; con 5×, el 0,8 % y el 27 % |
| S7 | Existe una verificación adicional (OTP, nueva selfie o analista) | Práctica estándar de fricción dinámica (step-up) | Banda "verificar" entre aprobar y rechazar (17,4 % del tráfico en test) | La decisión sería binaria y habría que elegir entre más fraude o más clientes perdidos |
| S8 | Las etiquetas de fraude llegan con retraso | Lo plantea el enunciado; es típico en fraude | No se usan fraudes previos como variable; la evaluación espera a que las etiquetas maduren | Se podría evaluar y reentrenar en continuo |
| S11 | `DISTRUST`, `ANOMALY` y `LIFETIME` se obtienen antes de decidir | Son puntajes de proveedores externos consultados durante el proceso | Son las variables más fuertes del modelo | Serían fuga. El riesgo está acotado: sin `DISTRUST` el PR-AUC baja solo de 0,549 a 0,532 (Fase 3) |
| S12 | El resultado (`STATUS` y `DECLINED_REASON`) de intentos **anteriores** sí se conoce al decidir el actual | El proceso anterior ya terminó | Se usan `fallos_previos`, `tasa_fallos_previos`, `rechazos_riesgo_previos` y los fallos previos por documento y email | Se eliminarían esas variables y bajaría el desempeño |
| S14 | La verificación adicional cuesta 0,05 por caso, detiene el 70 % de los fraudes y el 10 % de los clientes buenos la abandona | Valores ilustrativos: un OTP o una nueva selfie no son infalibles y generan algo de abandono | Definen la amplitud de la banda "verificar" y el costo estimado | Son parámetros (`decision.step_up`); con datos reales de conversión se recalibran sin reentrenar |
| S15 | La lista de motivos de rechazo del proveedor de verificación es estable | Es la taxonomía observada en los datos (`risk_reasons`) | `rechazos_riesgo_previos` depende de ella | Si el proveedor renombra motivos, la variable deja de contar sin dar error: hay que monitorear su distribución |
