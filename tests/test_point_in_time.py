"""Tests de las variables: solo pueden usar información del pasado.

Test de viaje en el tiempo: si una variable solo usa información pasada, calcularla con TODO
el dataset o solo con los datos hasta una fecha de corte debe dar el mismo valor para las filas
anteriores al corte. Si cambia, la variable está mirando el futuro.
"""
import pandas as pd
import pytest

from src.config import load_config
from src.data import clean, load_raw
from src.features import build_features

CFG = load_config()
COLS = CFG["columns"]
VARIABLES = CFG["features"]["numeric"] + CFG["features"]["categorical"]


@pytest.fixture(scope="module")
def limpio():
    return clean(load_raw(CFG["paths"]["raw"], COLS), CFG)


@pytest.fixture(scope="module")
def completo(limpio):
    return build_features(limpio, CFG).set_index(COLS["id"])


# Varios cortes: dentro del cliente A, entre clientes y dentro del cliente B
@pytest.mark.parametrize("corte", ["2024-02-15", "2024-06-01", "2024-08-15", "2024-08-25"])
def test_variables_no_cambian_al_quitar_el_futuro(limpio, completo, corte):
    fecha = pd.Timestamp(corte, tz="UTC")
    pasado = limpio[limpio[COLS["date"]] < fecha]
    truncado = build_features(pasado, CFG).set_index(COLS["id"])

    assert len(truncado) > 0, "El corte no deja filas"
    esperado = completo.loc[truncado.index, VARIABLES]
    pd.testing.assert_frame_equal(truncado[VARIABLES], esperado, check_dtype=False)


def test_primer_intento_no_tiene_historial(completo):
    df = completo.reset_index().sort_values([COLS["date"], COLS["id"]])
    primeros = df.groupby(COLS["account"]).head(1)
    assert (primeros["intentos_previos"] == 0).all()
    assert (primeros["fallos_previos"] == 0).all()
    assert primeros["horas_desde_intento_anterior"].isna().all()


def test_historial_sube_de_uno_en_uno(completo):
    df = completo.reset_index().sort_values([COLS["date"], COLS["id"]])
    pasos = df.groupby(COLS["account"])["intentos_previos"].diff().dropna()
    assert (pasos == 1).all()
