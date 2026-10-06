"""Tests de la inferencia y de la API.

El más importante es el de consistencia entre entrenamiento y producción: el mismo proceso debe
recibir el mismo score por el camino de producción (CSV crudo -> Scorer) que por el de
entrenamiento (test.parquet -> modelo). Si difieren, el modelo funcionaría bien offline y mal
en producción (escenario 2 de la prueba).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import load_config

CFG = load_config()
COLS = CFG["columns"]
MODELOS = Path(CFG["paths"]["models"])
MUESTRA = Path(CFG["paths"]["sample"]) / "nuevos.csv"

pytestmark = pytest.mark.skipif(
    not (MODELOS / "umbrales.json").exists() or not MUESTRA.exists(),
    reason="Faltan el modelo, los umbrales o la muestra: ejecutar el pipeline completo",
)


@pytest.fixture(scope="module")
def scorer():
    from src.inference import Scorer
    return Scorer()


@pytest.fixture(scope="module")
def nuevos():
    return pd.read_csv(MUESTRA, low_memory=False)


def test_puntua_todos_los_procesos(scorer, nuevos):
    resultado, _ = scorer.score(nuevos)
    assert len(resultado) == len(nuevos)
    assert set(resultado[COLS["id"]]) == set(nuevos[COLS["id"]])
    assert resultado["score"].between(0, 1).all()
    assert set(resultado["decision"]) <= {"aprobar", "verificar", "rechazar"}


def test_mismo_score_en_entrenamiento_y_produccion(scorer, nuevos):
    resultado, _ = scorer.score(nuevos)

    artefacto = joblib.load(MODELOS / "modelo.joblib")
    test = pd.read_parquet(Path(CFG["paths"]["processed"]) / "test.parquet")
    test = test[test[COLS["id"]].isin(nuevos[COLS["id"]])]
    offline = artefacto["pipeline"].predict_proba(test[artefacto["numeric"] + artefacto["categorical"]])[:, 1]

    comparado = test[[COLS["id"]]].assign(offline=offline).merge(resultado, on=COLS["id"])
    assert len(comparado) == len(nuevos)
    assert np.allclose(comparado["offline"], comparado["score"], atol=1e-4)


def test_respeta_el_orden_recibido(scorer, nuevos):
    # Las variables se calculan en orden cronológico, pero la respuesta debe seguir la solicitud
    desordenado = nuevos.sample(frac=1, random_state=1).head(30)
    resultado, _ = scorer.score(desordenado)
    assert resultado[COLS["id"]].tolist() == desordenado[COLS["id"]].tolist()


def test_no_necesita_columnas_que_no_existen_al_decidir(scorer, nuevos):
    # Al decidir no se conoce el resultado del proceso ni su motivo de rechazo
    sin_resultado = nuevos.drop(columns=["STATUS", "DECLINED_REASON"]).head(20)
    resultado, _ = scorer.score(sin_resultado)
    assert len(resultado) == 20


def test_api_score(nuevos):
    from fastapi.testclient import TestClient

    from src.api import app

    lote = nuevos.head(3)
    cuerpo = lote.astype(object).where(lote.notna(), None).to_dict(orient="records")
    with TestClient(app) as cliente:
        assert cliente.get("/health").json()["estado"] == "ok"
        respuesta = cliente.post("/score", json=cuerpo)

    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert [d["PROCESS_ID"] for d in datos] == lote[COLS["id"]].tolist()
    assert all(d["decision"] in {"aprobar", "verificar", "rechazar"} for d in datos)


def test_api_ejemplos_de_la_documentacion():
    """Los ejemplos precargados en /docs deben caer en la banda que anuncian."""
    from fastapi.testclient import TestClient

    from src.api import EJEMPLOS, app

    with TestClient(app) as cliente:
        for banda, ejemplo in EJEMPLOS.items():
            respuesta = cliente.post("/score", json=ejemplo["value"])
            assert respuesta.status_code == 200
            assert respuesta.json()[0]["decision"] == banda


@pytest.mark.parametrize("cuerpo", [
    [],                                                                         # lista vacía
    [{"PROCESS_ID": "x", "ACCOUNT_ID": "y", "DATE": "string"}],                 # fecha inválida
    [{"PROCESS_ID": "x", "ACCOUNT_ID": "y", "DATE": "2024-08-31T10:00:00"}],    # sin zona horaria
    [{"PROCESS_ID": "x", "DATE": "2024-08-31T10:00:00-05:00"}],                 # falta ACCOUNT_ID
    [{"PROCESS_ID": "x", "ACCOUNT_ID": "y", "DATE": "2024-08-31T10:00:00-05:00"},
     {"PROCESS_ID": "x", "ACCOUNT_ID": "y", "DATE": "2024-08-31T11:00:00-05:00"}],  # id repetido
])
def test_api_datos_invalidos_responden_422(cuerpo):
    from fastapi.testclient import TestClient

    from src.api import app

    with TestClient(app) as cliente:
        respuesta = cliente.post("/score", json=cuerpo)
    assert respuesta.status_code == 422
