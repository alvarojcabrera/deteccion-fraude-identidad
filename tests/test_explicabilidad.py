"""Tests de explicabilidad: las explicaciones deben ser exactas, coherentes y legibles."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.decision import decide
from src.inference import contributions, reasons

CFG = load_config()
MODELOS = Path(CFG["paths"]["models"])

pytestmark = pytest.mark.skipif(not (MODELOS / "modelo.joblib").exists(),
                                reason="Falta entrenar el modelo")


@pytest.fixture(scope="module")
def modelo():
    return joblib.load(MODELOS / "modelo.joblib")


@pytest.fixture(scope="module")
def muestra(modelo):
    test = pd.read_parquet(Path(CFG["paths"]["processed"]) / "test.parquet")
    return test[modelo["numeric"] + modelo["categorical"]].sample(400, random_state=0)


def test_contribuciones_suman_el_score(modelo, muestra):
    """Valor base + suma de contribuciones = log-odds del modelo: la explicación es exacta."""
    contrib, base = contributions(modelo["pipeline"], muestra)
    logit = base + contrib.sum(axis=1).to_numpy()
    proba = modelo["pipeline"].predict_proba(muestra)[:, 1]
    assert np.allclose(1 / (1 + np.exp(-logit)), proba, atol=1e-6)


def test_una_explicacion_por_variable_original(modelo, muestra):
    """Las categóricas no se parten en columnas binarias: se explica 'tipo de documento'."""
    contrib, _ = contributions(modelo["pipeline"], muestra)
    assert list(contrib.columns) == modelo["numeric"] + modelo["categorical"]


@pytest.mark.parametrize("variable", list(CFG["training"]["monotone"]))
def test_relaciones_monotonas(modelo, muestra, variable):
    """Más señal de riesgo nunca puede bajar el score (o subirlo, si la relación es inversa)."""
    sentido = CFG["training"]["monotone"][variable]
    valores = muestra[variable].dropna()
    grilla = np.unique(np.quantile(valores, np.linspace(0, 1, 12))) if len(valores) else []
    anterior = None
    for v in grilla:
        variado = muestra.copy()
        variado[variable] = v
        score = modelo["pipeline"].predict_proba(variado)[:, 1]
        if anterior is not None:
            assert (sentido * (score - anterior) >= -1e-9).all()
        anterior = score


def test_motivos_legibles_y_solo_cuando_hacen_falta(modelo, muestra):
    ecfg = CFG["explanations"]
    score = modelo["pipeline"].predict_proba(muestra)[:, 1]
    umbrales = {"verificar": 0.28, "rechazar": 0.83}
    decisiones = decide(score, umbrales)
    contrib, _ = contributions(modelo["pipeline"], muestra)
    motivos = reasons(contrib, muestra, decisiones, ecfg)

    for d, ms in zip(decisiones, motivos):
        if d not in ecfg["decisiones"]:
            assert ms == []                       # una aprobación no lleva motivos
            continue
        assert len(ms) <= ecfg["top_k"]
        aportes = [m["contribucion"] for m in ms]
        assert aportes == sorted(aportes, reverse=True)
        assert all(a >= ecfg["min_contribucion"] for a in aportes)
        # Cada motivo tiene un nombre legible configurado, no el nombre técnico de la columna
        assert all(m["variable"] in ecfg["nombres"] for m in ms)
    assert any(motivos), "Ningún caso de la muestra recibió motivos"
