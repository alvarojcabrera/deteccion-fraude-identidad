"""API de scoring casi en tiempo real.

Uso:
    uvicorn src.api:app --reload
    Documentación interactiva en http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict

from src.config import load_config
from src.inference import Scorer

estado = {}

# Id de cada cliente a partir de su nombre legible (A, B), tal como está en config.yaml
CLIENTE = {nombre: cid for cid, nombre in load_config()["clients"].items()}

# Ejemplos que aparecen precargados en /docs, uno por banda de decisión
EJEMPLOS = {
    "aprobar": {
        "summary": "Cliente nuevo con señales limpias (aprobar)",
        "value": [{
            "PROCESS_ID": "demo-001", "ACCOUNT_ID": "cuenta-nueva-001",
            "DATE": "2024-08-31T10:00:00-05:00", "CLIENT_ID": CLIENTE["B"],
            "DOCUMENT_TYPE": "national-id", "DOCUMENT_NUMBER": "doc-demo-001",
            "EMAIL": "cliente001@demo", "PHONE_NUMBER": "tel-demo-001",
            "DISTRUST": 0.0, "ANOMALY": 0.17, "LIFETIME": 2400,
        }],
    },
    "verificar": {
        "summary": "Cliente nuevo con DISTRUST alto y teléfono reciente (verificar)",
        "value": [{
            "PROCESS_ID": "demo-002", "ACCOUNT_ID": "cuenta-nueva-002",
            "DATE": "2024-08-31T10:05:00-05:00", "CLIENT_ID": CLIENTE["B"],
            "DOCUMENT_TYPE": "national-id", "DOCUMENT_NUMBER": "doc-demo-002",
            "EMAIL": "cliente002@demo", "PHONE_NUMBER": "tel-demo-002",
            "DISTRUST": 0.95, "ANOMALY": 0.46, "LIFETIME": 30,
        }],
    },
    "rechazar": {
        "summary": "Cuenta del historial con 21 rechazos por fraude que vuelve a intentar (rechazar)",
        "value": [{
            "PROCESS_ID": "demo-003",
            "ACCOUNT_ID": "8c5cc5fa71a5b9af95fac95772a994896005b34f268743f6b755649d02e8ec4b",
            "DATE": "2024-03-12T09:00:00-05:00", "CLIENT_ID": CLIENTE["A"],
            "DOCUMENT_NUMBER": "816967f0756a32f5ecbdfbb12379eca24d29781133090bf69394afdcf2d3f883",
            "EMAIL": "d3e98395ca7451029ff4c@f19e7063a",
            "PHONE_NUMBER": "9bee8969e0a3dd1db4e031827ba66cc16e3c0c005eafafaa265db0453e0108d0",
            "Distrust": 0.95,
        }],
    },
}


@asynccontextmanager
async def lifespan(app):
    # El modelo y el historial se cargan una sola vez, no en cada solicitud
    estado["scorer"] = Scorer()
    yield
    estado.clear()


app = FastAPI(title="Scoring de fraude en verificación de identidad", lifespan=lifespan)


class Proceso(BaseModel):
    """Un proceso en el formato crudo. Se aceptan columnas adicionales del CSV original;
    las señales que falten se tratan como faltantes, igual que en entrenamiento."""
    model_config = ConfigDict(extra="allow")

    PROCESS_ID: str
    ACCOUNT_ID: str
    DATE: AwareDatetime  # fecha ISO 8601 con zona horaria, p. ej. 2024-08-31T10:00:00-05:00
    CLIENT_ID: str | None = None


class Motivo(BaseModel):
    """Una variable que empujó la decisión hacia fraude."""
    variable: str
    valor: str
    contribucion: float  # aporte al score en log-odds (valor SHAP)
    descripcion: str


class Resultado(BaseModel):
    PROCESS_ID: str
    score: float
    decision: str
    motivos: list[Motivo]  # vacío en las aprobaciones
    version_modelo: str


@app.get("/health")
def health():
    scorer = estado["scorer"]
    return {"estado": "ok", "version_modelo": scorer.version, "umbrales": scorer.umbrales}


@app.post("/score", response_model=list[Resultado])
def score(procesos: Annotated[list[Proceso], Body(openapi_examples=EJEMPLOS)]):
    if not procesos:
        raise HTTPException(status_code=422, detail="La lista de procesos está vacía")
    scorer = estado["scorer"]
    # mode="json" devuelve la fecha como texto ISO, el mismo formato que el CSV original
    nuevos = pd.DataFrame([p.model_dump(mode="json") for p in procesos])
    try:
        resultado, X = scorer.score(nuevos)
    except (ValueError, KeyError, TypeError) as e:
        # Datos que pasan el esquema pero no se pueden procesar: error del cliente, no del servidor
        raise HTTPException(status_code=422, detail=f"No se pudo procesar la solicitud: {e}")
    scorer.log(resultado, X, Path(scorer.cfg["paths"]["logs"]) / "predicciones_api.jsonl")

    # FastAPI valida cada registro contra `Resultado`
    return resultado[list(Resultado.model_fields)].to_dict(orient="records")
