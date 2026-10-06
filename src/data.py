"""Datos: carga, validación, limpieza, partición y orquestación.

Uso:
    python -m src.data
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import load_config
from src.features import build_features


class DataValidationError(ValueError):
    """Los datos no cumplen lo que el pipeline espera. La API la traduce a un error 422."""


def load_raw(path, cols, require_target=True):
    """Lee el CSV crudo y aplica los tipos básicos."""
    return prepare_raw(pd.read_csv(path, low_memory=False), cols, require_target)


def prepare_raw(df, cols, require_target=True):
    """Valida y tipa. La usan el entrenamiento (con etiqueta) y la inferencia (sin ella).

    Falla rápido con una excepción (no con `assert`, que Python elimina al ejecutar con `-O`).
    """
    requeridas = [c for k, c in cols.items() if require_target or k != "target"]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise DataValidationError(f"Faltan columnas: {faltantes}")
    if not df[cols["id"]].is_unique:
        raise DataValidationError("Hay procesos duplicados")
    if df[cols["date"]].isna().any():
        raise DataValidationError("Hay filas sin fecha")
    if require_target and df[cols["target"]].isna().any():
        raise DataValidationError("Hay filas sin etiqueta")

    df = df.copy()
    try:
        df[cols["date"]] = pd.to_datetime(df[cols["date"]], utc=True, format="ISO8601")
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Fecha inválida en {cols['date']}: {e}") from e
    if require_target:
        df[cols["target"]] = df[cols["target"]].astype(int)
    return df


def clean(df, cfg):
    """Unifica los esquemas de ambos clientes y normaliza categorías."""
    df = df.copy()

    # Hallazgo 1: las parejas nunca están llenas a la vez, se combinan
    for nueva, (a, b) in cfg["unify"].items():
        df[nueva] = df[a].fillna(df[b])

    # Nombre legible y estable del cliente; uno nuevo no rompe el pipeline
    df["CLIENTE"] = df["CLIENT_ID"].map(cfg["clients"]).fillna("otro")

    # El tipo de documento solo existe para un cliente: el faltante es una categoría
    df["DOCUMENT_TYPE"] = df["DOCUMENT_TYPE"].fillna("desconocido")

    return df


def biased_mask(df, cfg):
    """True en las filas de periodos con sesgo de selección (S10)."""
    mes = df[cfg["columns"]["date"]].dt.tz_convert(None).dt.to_period("M").astype(str)
    mascara = pd.Series(False, index=df.index)
    for cliente, meses in cfg["biased_periods"].items():
        mascara |= (df["CLIENTE"] == cliente) & mes.isin(meses)
    return mascara


def group_folds(df, cfg, n_splits):
    """Folds agrupados por cuenta y estratificados por cliente y etiqueta."""
    cols = cfg["columns"]
    estrato = df["CLIENTE"] + "_" + df[cols["target"]].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    return list(sgkf.split(df, estrato, groups=df[cols["account"]]))


def split_train_test(df, cfg):
    """Partición train/test: se reserva uno de los folds como prueba."""
    train_idx, test_idx = group_folds(df, cfg, cfg["split"]["n_folds"])[0]
    return df.iloc[train_idx], df.iloc[test_idx]


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    cols, feats = cfg["columns"], cfg["features"]

    raw = load_raw(cfg["paths"]["raw"], cols)
    df = clean(raw, cfg)
    # Las variables se calculan con TODO el historial, incluidos los periodos sesgados:
    # esos intentos sí ocurrieron y son pasado legítimo para las filas posteriores
    df = build_features(df, cfg)

    sesgo = biased_mask(df, cfg)
    df = df[~sesgo]
    train, test = split_train_test(df, cfg)

    # Lista blanca: solo sale lo que está declarado en config.yaml
    meta = [cols["id"], cols["account"], cols["date"], cols["target"], "CLIENTE"]
    keep = meta + feats["numeric"] + feats["categorical"] + feats["leakage"]

    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    train[keep].to_parquet(out / "train.parquet", index=False)
    test[keep].to_parquet(out / "test.parquet", index=False)

    write_sample(raw, test, cfg)

    reporte = {
        "filas_excluidas_por_sesgo": int(sesgo.sum()),
        **{
            nombre: {
                "filas": len(parte),
                "cuentas": int(parte[cols["account"]].nunique()),
                "tasa_fraude": round(parte[cols["target"]].mean(), 4),
                "tasa_fraude_por_cliente": parte.groupby("CLIENTE")[cols["target"]].mean().round(4).to_dict(),
            }
            for nombre, parte in [("train", train), ("test", test)]
        },
    }
    (out / "data_report.json").write_text(json.dumps(reporte, indent=2), encoding="utf-8")
    print(json.dumps(reporte, indent=2))


def write_sample(raw, test, cfg, n=500):
    """Datos "nuevos" para probar la inferencia: filas crudas de test, sin la etiqueta.

    - nuevos.csv: lote normal, con el mismo formato que el CSV original.
    - nuevos_deriva.csv: el mismo lote con DISTRUST desplazado, para probar las alertas de monitoreo.
    - etiquetas.csv: las etiquetas reales, como si llegaran después (S8).
    """
    cols = cfg["columns"]
    ids = test[cols["id"]].sample(n=min(n, len(test)), random_state=cfg["seed"])
    lote = raw[raw[cols["id"]].isin(ids)].sort_values(cols["date"])

    out = Path(cfg["paths"]["sample"])
    out.mkdir(parents=True, exist_ok=True)
    lote.drop(columns=cols["target"]).to_csv(out / "nuevos.csv", index=False)
    lote[[cols["id"], cols["target"]]].to_csv(out / "etiquetas.csv", index=False)

    deriva = lote.drop(columns=cols["target"]).copy()
    for col in cfg["unify"]["DISTRUST_U"]:
        deriva[col] = np.clip(deriva[col] + 0.3, 0, 1)
    deriva.to_csv(out / "nuevos_deriva.csv", index=False)


if __name__ == "__main__":
    main()
