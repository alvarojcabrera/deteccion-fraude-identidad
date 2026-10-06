"""Modelos y métricas: el Pipeline de preprocesamiento + modelo, y las métricas compartidas."""
import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def classification_metrics(y, score, target_fpr=0.05):
    """Métricas de ranking y calibración. Las usan el entrenamiento, la evaluación y el monitoreo.

    - pr_auc: métrica principal; resume precisión y recall sin elegir un umbral.
      Un modelo al azar obtiene la tasa de fraude (≈ 0,16).
    - roc_auc: qué tan bien ordena los casos.
    - recall_at_fpr: % del fraude detectado si se acepta molestar al `target_fpr` de los legítimos.
    - brier: error de la probabilidad; mide si el score se puede leer como probabilidad.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "n": int(len(y)),
        "tasa_base": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_fpr": float(np.interp(target_fpr, fpr, tpr)),
        "brier": float(brier_score_loss(y, score)),
    }


def _a_categoria(X):
    # Función con nombre (no lambda) porque el modelo se guarda en disco y una lambda no se puede guardar
    return X.astype("category")


def build_pipeline(model, numeric, categorical, params, seed, monotone=None):
    """Preprocesamiento + modelo en un solo objeto, que se aplica igual al entrenar y al predecir.

    `monotone` ({variable: +1 | -1}) obliga a LightGBM a que el riesgo nunca baje (+1) o nunca
    suba (−1) cuando la variable aumenta.
    """
    # Una categoría vacía pasa a ser la categoría "ninguno": que falte también es información
    imputar_categoria = SimpleImputer(strategy="constant", fill_value="ninguno")

    if model == "logistic":
        # Línea base. No acepta vacíos: se imputa la mediana y se agrega una columna "faltaba".
        # Necesita escalas comparables y las categorías como columnas 1/0 (one-hot).
        numericas = Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            ("escalar", StandardScaler()),
        ])
        categoricas = Pipeline([
            ("imputar", imputar_categoria),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        clasificador = LogisticRegression(random_state=seed, **params)
    elif model == "lightgbm":
        # Modelo principal. Maneja los vacíos de forma nativa, así que las numéricas pasan tal cual.
        # Las categóricas entran como tipo `category`: cada variable es UNA columna del modelo, y
        # la explicación habla de "tipo de documento" y no de columnas binarias sueltas.
        numericas = "passthrough"
        categoricas = Pipeline([
            ("imputar", imputar_categoria),
            ("categoria", FunctionTransformer(_a_categoria, feature_names_out="one-to-one")),
        ])
        # Una restricción por columna, en el mismo orden en que entran al modelo (0 = libre)
        restricciones = [(monotone or {}).get(c, 0) for c in numeric + categorical]
        clasificador = lgb.LGBMClassifier(
            random_state=seed, monotone_constraints=restricciones,
            monotone_constraints_method="advanced", **params)
    else:
        raise ValueError(f"Modelo desconocido: {model}")

    # Aplica a cada grupo de columnas su preprocesamiento y conserva los nombres de columna
    preprocesamiento = ColumnTransformer(
        [("num", numericas, numeric), ("cat", categoricas, categorical)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline([("pre", preprocesamiento), ("clf", clasificador)])
