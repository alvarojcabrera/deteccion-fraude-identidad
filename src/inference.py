"""Inferencia: puntúa procesos nuevos (en el formato del CSV original) y explica cada decisión.

Las variables de historial necesitan el pasado de cada cuenta. Aquí ese pasado es el dataset
histórico; en producción sería un almacén de variables (feature store). Se reutilizan
exactamente las mismas funciones del entrenamiento (clean, build_features y el Pipeline
guardado), así el modelo recibe lo mismo offline y en producción (tests/test_inference.py).

Uso:
    python -m src.inference --input data/sample/nuevos.csv
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import clean, load_raw, prepare_raw
from src.decision import decide
from src.features import build_features


def contributions(pipeline, X):
    """Valores SHAP: cuánto empujó cada variable el score de cada proceso.

    Positivo = hacia fraude. Están en log-odds y son exactos: valor base + suma = score del
    modelo (tests/test_explicabilidad.py). LightGBM los calcula de forma nativa.
    """
    Xt = pipeline.named_steps["pre"].transform(X)  # el mismo preprocesamiento del modelo
    valores = pipeline.named_steps["clf"].booster_.predict(Xt, pred_contrib=True)
    # La última columna es el valor base; el resto, una contribución por variable
    contrib = pd.DataFrame(valores[:, :-1], columns=Xt.columns, index=X.index)
    return contrib, valores[:, -1]


def formatear(valor):
    """Un valor legible: enteros sin decimales, decimales con dos, vacíos como "sin dato"."""
    if pd.isna(valor):
        return "sin dato"
    if isinstance(valor, (int, float, np.number)):
        return f"{int(valor)}" if float(valor).is_integer() else f"{valor:.2f}"
    return str(valor)


def reasons(contrib, X, decisiones, ecfg):
    """Motivos de cada decisión: las variables que más empujaron hacia fraude, con su valor.

    Solo para verificaciones y rechazos (una aprobación no necesita justificarse) y solo
    aportes relevantes, para no presentar ruido como motivo.
    """
    motivos = []
    for i, decision in enumerate(decisiones):
        if decision not in ecfg["decisiones"]:
            motivos.append([])
            continue
        aportes = contrib.iloc[i].sort_values(ascending=False).head(ecfg["top_k"])
        motivos.append([
            {"variable": variable,
             "valor": formatear(X.iloc[i][variable]),
             "contribucion": round(float(aporte), 3),
             "descripcion": f"{ecfg['nombres'].get(variable, variable)}: {formatear(X.iloc[i][variable])}"}
            for variable, aporte in aportes.items() if aporte >= ecfg["min_contribucion"]
        ])
    return motivos


class Scorer:
    """Carga una sola vez el modelo, los umbrales y el historial, y puntúa procesos nuevos."""

    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.cols = self.cfg["columns"]
        modelos_dir = Path(self.cfg["paths"]["models"])

        artefacto = joblib.load(modelos_dir / "modelo.joblib")
        self.pipeline = artefacto["pipeline"]
        self.version = artefacto["version"]
        self.variables = artefacto["numeric"] + artefacto["categorical"]
        self.umbrales = json.loads((modelos_dir / "umbrales.json").read_text(encoding="utf-8"))["umbrales"]

        # Historial sin la etiqueta: en producción, al decidir, todavía no se conoce (S8)
        self.historial = load_raw(self.cfg["paths"]["raw"], self.cols).drop(columns=self.cols["target"])

    def score(self, nuevos):
        """Score, decisión y motivos para un DataFrame de procesos nuevos."""
        id_col = self.cols["id"]
        nuevos = prepare_raw(nuevos, self.cols, require_target=False)
        nuevos = nuevos.drop(columns=self.cols["target"], errors="ignore")

        # Mismos tipos que en entrenamiento: en un JSON, una columna numérica toda vacía llega
        # como texto y rompería el modelo (un test encontró este error)
        for col in self.historial.select_dtypes("number").columns.intersection(nuevos.columns):
            nuevos[col] = pd.to_numeric(nuevos[col], errors="coerce")

        # El historial (sin los procesos que se están puntuando, para no contarlos dos veces)
        # más los procesos nuevos; las columnas que falten (p. ej. STATUS) quedan vacías
        pasado = self.historial[~self.historial[id_col].isin(nuevos[id_col])]
        combinado = clean(pd.concat([pasado, nuevos], ignore_index=True), self.cfg)
        variables = build_features(combinado, self.cfg)
        # Solo las filas nuevas, en el mismo orden en que llegaron
        variables = variables.set_index(id_col).loc[nuevos[id_col]].reset_index()

        X = variables[self.variables]
        score = self.pipeline.predict_proba(X)[:, 1]
        decisiones = decide(score, self.umbrales)
        contrib, _ = contributions(self.pipeline, X)

        resultado = variables[[id_col, self.cols["account"], self.cols["date"], "CLIENTE"]].copy()
        resultado["score"] = score.round(4)
        resultado["decision"] = decisiones
        resultado["motivos"] = reasons(contrib, X, decisiones, self.cfg["explanations"])
        resultado["version_modelo"] = self.version
        return resultado, X

    def log(self, resultado, X, ruta):
        """Registra cada predicción con sus variables: es lo que lee el monitoreo."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        registro = pd.concat([resultado.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        registro[self.cols["date"]] = registro[self.cols["date"]].astype(str)
        registro["scored_at"] = datetime.now(timezone.utc).isoformat()
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(registro.to_json(orient="records", lines=True, force_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Puntúa procesos nuevos y decide.")
    parser.add_argument("--input", required=True, help="CSV con procesos en el formato original")
    parser.add_argument("--output", default="reports/decisiones.csv")
    parser.add_argument("--log", default="logs/predicciones.jsonl")
    args = parser.parse_args()

    scorer = Scorer()
    resultado, X = scorer.score(pd.read_csv(args.input, low_memory=False))

    salida = resultado.copy()
    salida["motivos"] = [" | ".join(m["descripcion"] for m in ms) for ms in salida["motivos"]]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.output, index=False)
    scorer.log(resultado, X, args.log)

    print(f"{len(resultado)} procesos puntuados con {scorer.version}")
    print(resultado["decision"].value_counts().to_string())
    print(f"Resultados: {args.output}  |  Registro: {args.log}")


if __name__ == "__main__":
    main()
