"""Evaluación final en test. Es la única etapa que mira test: ni el modelo ni los umbrales se
ajustaron con él.

Uso:
    python -m src.evaluate
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.config import load_config
from src.decision import decide, outcomes, sensitivity
from src.inference import contributions
from src.models import classification_metrics


def bootstrap_ci(df, target, account, n=500, seed=42):
    """Intervalo de confianza del 95 % del PR-AUC y del ROC-AUC.

    500 veces se toma una muestra con reposición de CUENTAS (no de filas, porque la etiqueta es
    casi por cuenta) y se recalcula la métrica; los percentiles 2,5 y 97,5 son el intervalo.
    """
    rng = np.random.default_rng(seed)
    por_cuenta = df.groupby(account).indices  # cuenta -> posiciones de sus filas
    cuentas = df[account].unique()
    pr, roc = [], []
    for _ in range(n):
        muestra = rng.choice(cuentas, size=len(cuentas), replace=True)
        idx = np.concatenate([por_cuenta[c] for c in muestra])
        y, s = df[target].to_numpy()[idx], df["score"].to_numpy()[idx]
        pr.append(average_precision_score(y, s))
        roc.append(roc_auc_score(y, s))
    return {"pr_auc": np.percentile(pr, [2.5, 97.5]).tolist(),
            "roc_auc": np.percentile(roc, [2.5, 97.5]).tolist()}


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def figures(test, target, umbrales, contrib, escenarios, ratio_supuesto, nombres, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    y, s = test[target], test["score"]

    # Curva precisión-recall: cuánto fraude se detecta (recall) y con qué precisión
    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Modelo (PR-AUC {average_precision_score(y, s):.3f})")
    ax.axhline(y.mean(), ls="--", c="gray", label=f"Azar (tasa de fraude {y.mean():.3f})")
    ax.set(xlabel="Recall (fraude detectado)", ylabel="Precisión", title="Curva precisión-recall (test)")
    ax.legend()
    guardar(fig, carpeta / "curva_pr.png")

    # Calibración: ¿entre los procesos con score ≈ 0,6 hay ≈ 60 % de fraude?
    real, predicho = calibration_curve(y, s, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", label="Calibración perfecta")
    ax.plot(predicho, real, marker="o", label="Modelo")
    ax.set(xlabel="Score medio", ylabel="Tasa de fraude real", title="Calibración (test)")
    ax.legend()
    guardar(fig, carpeta / "calibracion.png")

    # Distribución de scores de legítimos y fraudes, con los dos umbrales
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, density=True, label="Legítimo")
    ax.hist(s[y == 1], bins=bins, alpha=0.6, density=True, label="Fraude")
    ax.axvline(umbrales["verificar"], c="orange", ls="--", label=f"Verificar ≥ {umbrales['verificar']:.2f}")
    ax.axvline(umbrales["rechazar"], c="red", ls="--", label=f"Rechazar ≥ {umbrales['rechazar']:.2f}")
    ax.set(xlabel="Score", ylabel="Densidad", title="Distribución de scores y bandas de decisión (test)")
    ax.legend()
    guardar(fig, carpeta / "distribucion_scores.png")

    # Importancia: cuánto mueve el score cada variable, en promedio (valores SHAP)
    importancia = contrib.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([nombres.get(c, c) for c in importancia.index], importancia, color="tab:red")
    ax.set(xlabel="Contribución media absoluta al score (log-odds)", title="Importancia de variables (test)")
    guardar(fig, carpeta / "importancia.png")

    # Dependencia: aporte de cada variable según su valor. Con restricciones monótonas, siempre
    # avanza en una sola dirección
    variables = ["DISTRUST_U", "LIFETIME_U", "rechazos_riesgo_previos", "fallos_previos"]
    fig, ax = plt.subplots(1, len(variables), figsize=(16, 3.6))
    for a, col in zip(ax, variables):
        a.scatter(test[col], contrib[col], s=6, alpha=0.4, color="tab:red")
        a.axhline(0, c="gray", lw=1)
        a.set(xlabel=nombres.get(col, col), ylabel="Contribución (log-odds)")
    fig.suptitle("Efecto de cada valor: monótono por diseño")
    guardar(fig, carpeta / "shap_dependencia.png")

    # Sensibilidad: cómo cambia la política según cuánto cuesta rechazar a un cliente bueno
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = escenarios["costo_rechazo_vs_fraude"]
    for col, etiqueta in [("pct_aprobar", "Aprobar"), ("pct_verificar", "Verificar"), ("pct_rechazar", "Rechazar")]:
        ax[0].plot(x, escenarios[col], marker="o", label=etiqueta)
    ax[0].set(ylabel="% del tráfico", title="Decisiones según los costos")
    ax[1].plot(x, escenarios["fraude_detenido_pct"], marker="o", label="Fraude detenido")
    ax[1].plot(x, escenarios["buenos_perdidos_pct"], marker="o", label="Clientes buenos perdidos")
    ax[1].set(ylabel="%", title="Impacto según los costos (test)")
    for a in ax:
        a.set(xscale="log", xlabel="Costo de rechazar a un cliente bueno / costo de un fraude")
        a.axvline(ratio_supuesto, c="gray", ls="--", label=f"Supuesto S6 ({ratio_supuesto}×)")
        a.legend()
    guardar(fig, carpeta / "sensibilidad_costos.png")


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    cols, dcfg = cfg["columns"], cfg["decision"]
    target = cols["target"]
    modelos_dir, reportes_dir = Path(cfg["paths"]["models"]), Path(cfg["paths"]["reports"])

    artefacto = joblib.load(modelos_dir / "modelo.joblib")
    umbrales = json.loads((modelos_dir / "umbrales.json").read_text(encoding="utf-8"))["umbrales"]
    variables = artefacto["numeric"] + artefacto["categorical"]

    test = pd.read_parquet(Path(cfg["paths"]["processed"]) / "test.parquet")
    test["score"] = artefacto["pipeline"].predict_proba(test[variables])[:, 1]
    test["decision"] = decide(test["score"], umbrales)

    # Métricas del modelo, globales y por cliente, con su intervalo de confianza
    tfpr = cfg["training"]["target_fpr"]
    global_ = classification_metrics(test[target], test["score"], tfpr)
    por_cliente = {c: classification_metrics(g[target], g["score"], tfpr)
                   for c, g in test.groupby("CLIENTE")}
    ic = bootstrap_ci(test, target, cols["account"], seed=cfg["seed"])

    # Métricas de negocio: la política elegida y los escenarios de costos
    politica = outcomes(test[target], test["decision"], dcfg)
    escenarios = sensitivity(test[target], test["score"], dcfg)
    escenarios.to_csv(reportes_dir / "sensibilidad_costos.csv", index=False)

    contrib, _ = contributions(artefacto["pipeline"], test[variables])
    figures(test, target, umbrales, contrib, escenarios, dcfg["costs"]["legitimo_rechazado"],
            cfg["explanations"]["nombres"], reportes_dir / "figuras")

    reporte = {
        "version_modelo": artefacto["version"],
        "test_global": global_,
        "test_ic95_por_cuenta": ic,
        "test_por_cliente": por_cliente,
        "cv_train": artefacto["metricas_cv"],
        "umbrales": umbrales,
        "politica_en_test": politica,
        "importancia": contrib.abs().mean().sort_values(ascending=False).round(4).to_dict(),
    }
    (reportes_dir / "evaluacion.json").write_text(json.dumps(reporte, indent=2), encoding="utf-8")

    print(f"Test  PR-AUC {global_['pr_auc']:.3f} (IC95 {ic['pr_auc'][0]:.3f}–{ic['pr_auc'][1]:.3f})"
          f"  ROC-AUC {global_['roc_auc']:.3f}  Recall@FPR{tfpr:.0%} {global_['recall_at_fpr']:.3f}"
          f"  Brier {global_['brier']:.3f}  (tasa de fraude {global_['tasa_base']:.3f})")
    for c, m in por_cliente.items():
        print(f"  Cliente {c}: PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  n={m['n']}")
    print(f"Política: detiene {politica['fraude_detenido_pct']:.1%} del fraude; pierde "
          f"{politica['buenos_perdidos_pct']:.2%} de clientes buenos; ahorro "
          f"{politica['ahorro_vs_sin_modelo_pct']:.1%}")
    print("\nSensibilidad a los costos (test):")
    print(escenarios.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
