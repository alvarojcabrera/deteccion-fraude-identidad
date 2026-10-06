"""Tests de los datos procesados (salida de `python -m src.data`)."""
import pandas as pd
import pytest

from src.config import load_config

CFG = load_config()
COLS = CFG["columns"]


@pytest.fixture(scope="module")
def datos():
    ruta = CFG["paths"]["processed"]
    return pd.read_parquet(f"{ruta}/train.parquet"), pd.read_parquet(f"{ruta}/test.parquet")


def test_ninguna_cuenta_en_ambos_conjuntos(datos):
    train, test = datos
    comunes = set(train[COLS["account"]]) & set(test[COLS["account"]])
    assert not comunes


def test_no_hay_procesos_repetidos(datos):
    train, test = datos
    total = len(train) + len(test)
    assert total == pd.concat([train, test])[COLS["id"]].nunique()


def test_tasa_de_fraude_similar(datos):
    train, test = datos
    assert abs(train[COLS["target"]].mean() - test[COLS["target"]].mean()) < 0.02


def test_ninguna_columna_excluida_es_variable():
    excluidas = {c for grupo in CFG["excluded"].values() for c in grupo}
    variables = set(CFG["features"]["numeric"] + CFG["features"]["categorical"])
    assert not excluidas & variables


def test_sin_filas_de_periodos_sesgados(datos):
    df = pd.concat(datos)
    mes = df[COLS["date"]].dt.tz_convert(None).dt.to_period("M").astype(str)
    for cliente, meses in CFG["biased_periods"].items():
        assert not ((df["CLIENTE"] == cliente) & mes.isin(meses)).any()
