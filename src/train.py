"""Entrenamiento: validación cruzada, comparación de modelos y modelo final.

Solo usa train.parquet; el conjunto de prueba queda intacto para la evaluación final.

Uso:
    python -m src.train
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import group_folds
from src.models import build_pipeline, classification_metrics


def experiment_features(experiment, features):
    """Variables de un experimento: las del modelo principal, más `add` y menos `drop`."""
    numeric = [c for c in features["numeric"] if c not in experiment.get("drop", [])]
    categorical = [c for c in features["categorical"] if c not in experiment.get("drop", [])]
    categorical += experiment.get("add", [])
    return numeric, categorical


def cross_validate(df, folds, cfg, model, numeric, categorical):
    """Entrena en 4 partes y predice la quinta, para cada fold.

    Devuelve los scores "fuera de fold" (cada fila los recibe de un modelo que no la vio) y las
    métricas de cada fold.
    """
    tcfg = cfg["training"]
    y = df[cfg["columns"]["target"]]
    X = df[numeric + categorical]
    oof = np.zeros(len(df))
    por_fold = []
    for idx_train, idx_val in folds:
        pipe = build_pipeline(model, numeric, categorical, tcfg["models"][model], cfg["seed"],
                              tcfg["monotone"])
        pipe.fit(X.iloc[idx_train], y.iloc[idx_train])
        oof[idx_val] = pipe.predict_proba(X.iloc[idx_val])[:, 1]
        por_fold.append(classification_metrics(y.iloc[idx_val], oof[idx_val], tcfg["target_fpr"]))
    return oof, pd.DataFrame(por_fold)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    cols, tcfg = cfg["columns"], cfg["training"]
    modelos_dir, reportes_dir = Path(cfg["paths"]["models"]), Path(cfg["paths"]["reports"])
    modelos_dir.mkdir(exist_ok=True)
    reportes_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(Path(cfg["paths"]["processed"]) / "train.parquet")
    # Los mismos folds para todos los experimentos: así las comparaciones son justas
    folds = group_folds(df, cfg, tcfg["cv_folds"])

    # 1. Comparación de modelos y variantes (config.yaml → training.experiments)
    filas = []
    for nombre, exp in tcfg["experiments"].items():
        numeric, categorical = experiment_features(exp, cfg["features"])
        oof, por_fold = cross_validate(df, folds, cfg, exp["model"], numeric, categorical)
        metricas = por_fold.drop(columns=["n", "tasa_base"])
        filas.append({"experimento": nombre,
                      **{f"{k}_media": v for k, v in metricas.mean().items()},
                      **{f"{k}_desv": v for k, v in metricas.std().items()}})
        if nombre == "principal":
            oof_principal = oof
    experimentos = pd.DataFrame(filas)
    experimentos.to_csv(reportes_dir / "cv_experimentos.csv", index=False)
    print(experimentos[["experimento", "pr_auc_media", "pr_auc_desv", "roc_auc_media",
                        "recall_at_fpr_media", "brier_media"]].round(3).to_string(index=False))

    # 2. Modelo final: el experimento principal, entrenado con todo train
    numeric, categorical = experiment_features(tcfg["experiments"]["principal"], cfg["features"])
    modelo = build_pipeline("lightgbm", numeric, categorical, tcfg["models"]["lightgbm"],
                            cfg["seed"], tcfg["monotone"])
    modelo.fit(df[numeric + categorical], df[cols["target"]])

    version = f"lightgbm-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    metricas_cv = experimentos.set_index("experimento").loc["principal"].to_dict()
    joblib.dump({"pipeline": modelo, "version": version, "numeric": numeric,
                 "categorical": categorical, "metricas_cv": metricas_cv},
                modelos_dir / "modelo.joblib")

    # 3. Referencia: variables de train con su score fuera de fold. La usan la decisión (tasas
    #    esperadas de cada banda) y el monitoreo (contra qué se compara la deriva)
    referencia = df[[cols["id"], "CLIENTE", cols["target"]] + numeric + categorical].copy()
    referencia["score"] = oof_principal
    referencia.to_parquet(modelos_dir / "referencia.parquet", index=False)

    (reportes_dir / "entrenamiento.json").write_text(
        json.dumps({"version": version, "metricas_cv": metricas_cv}, indent=2), encoding="utf-8")
    print(f"\nModelo guardado: {modelos_dir / 'modelo.joblib'} (versión {version})")


if __name__ == "__main__":
    main()
