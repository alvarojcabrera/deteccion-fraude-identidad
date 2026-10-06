"""Monitoreo básico de un lote de predicciones registradas, en tres niveles:

1. Deriva de datos (al instante, sin etiquetas): PSI de cada variable y del score frente a train.
2. Deriva de decisiones (al instante, sin etiquetas): % de aprobar / verificar / rechazar
   frente a lo esperado.
3. Desempeño (con retraso): solo cuando hay suficientes etiquetas maduras (S8).

Uso:
    python -m src.monitor --log logs/predicciones.jsonl --etiquetas data/sample/etiquetas.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.models import classification_metrics


def psi(referencia, actual, bins=10):
    """Índice de estabilidad poblacional: < 0,10 estable; 0,10–0,25 cambio moderado; > 0,25 deriva.

    1. Corta la referencia en 10 tramos con la misma cantidad de casos (cuantiles).
    2. Calcula qué proporción cae en cada tramo, en la referencia y en el lote actual.
       Los vacíos son un tramo más: que una señal deje de llegar también es deriva.
    3. Suma (actual − referencia) × ln(actual / referencia) en todos los tramos.
    """
    ref = pd.to_numeric(referencia, errors="coerce")
    act = pd.to_numeric(actual, errors="coerce")
    cortes = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf  # los extremos abiertos atrapan valores nunca vistos

    def proporciones(x):
        conteo = pd.cut(x.dropna(), cortes, include_lowest=True).value_counts(sort=False).to_numpy()
        return np.append(conteo, x.isna().sum()) / len(x)

    # Un mínimo de 0,0001 evita dividir por cero o calcular ln(0) en tramos vacíos
    p_ref = np.clip(proporciones(ref), 1e-4, None)
    p_act = np.clip(proporciones(act), 1e-4, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def monitor(log, cfg, etiquetas=None):
    mcfg, cols = cfg["monitoring"], cfg["columns"]
    modelos_dir = Path(cfg["paths"]["models"])
    referencia = pd.read_parquet(modelos_dir / "referencia.parquet")
    esperado = json.loads((modelos_dir / "umbrales.json").read_text(encoding="utf-8"))[
        "esperado_en_train"]["por_decision"]
    alertas = []

    # 1. Deriva de datos
    deriva = {c: psi(referencia[c], log[c], mcfg["psi_bins"])
              for c in ["score"] + cfg["features"]["numeric"]}
    for variable, valor in sorted(deriva.items(), key=lambda kv: -kv[1]):
        if valor > mcfg["psi_alert"]:
            alertas.append(f"Deriva en {variable} (PSI {valor:.2f})")

    # 2. Deriva de decisiones
    tasas = {d: float((log["decision"] == d).mean()) for d in esperado}
    for d, actual in tasas.items():
        if abs(actual - esperado[d]["pct_trafico"]) > mcfg["decision_rate_tolerance"]:
            alertas.append(f"Tasa de '{d}' {actual:.1%} vs {esperado[d]['pct_trafico']:.1%} esperado")

    # 3. Desempeño, solo con suficientes etiquetas maduras
    desempeno = None
    if etiquetas is not None:
        con_etiqueta = log.merge(etiquetas, on=cols["id"])
        if len(con_etiqueta) >= mcfg["min_labels"]:
            desempeno = classification_metrics(con_etiqueta[cols["target"]], con_etiqueta["score"],
                                               cfg["training"]["target_fpr"])

    return {"procesos": len(log), "estado": "ALERTA" if alertas else "OK", "alertas": alertas,
            "psi": deriva, "tasas_decision": tasas, "desempeno": desempeno}


def main():
    parser = argparse.ArgumentParser(description="Monitorea un lote de predicciones registradas.")
    parser.add_argument("--log", default="logs/predicciones.jsonl")
    parser.add_argument("--etiquetas", default=None, help="CSV con PROCESS_ID e IS_FRAUD, si ya llegaron")
    args = parser.parse_args()

    cfg = load_config()
    log = pd.read_json(args.log, lines=True, dtype={cfg["columns"]["id"]: str})
    etiquetas = pd.read_csv(args.etiquetas, dtype={cfg["columns"]["id"]: str}) if args.etiquetas else None
    reporte = monitor(log, cfg, etiquetas)

    salida = Path(cfg["paths"]["reports"]) / f"monitoreo_{Path(args.log).stem}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(reporte, indent=2), encoding="utf-8")

    print(f"Estado: {reporte['estado']}  ({reporte['procesos']} procesos)")
    for alerta in reporte["alertas"]:
        print(f"  - {alerta}")
    if reporte["desempeno"]:
        print(f"  Desempeño con etiquetas: PR-AUC {reporte['desempeno']['pr_auc']:.3f}, "
              f"ROC-AUC {reporte['desempeno']['roc_auc']:.3f}")
    print(f"Reporte: {salida}")


if __name__ == "__main__":
    main()
