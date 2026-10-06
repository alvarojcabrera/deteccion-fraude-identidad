"""Lógica de decisión: de la probabilidad de fraude a aprobar, verificar o rechazar.

El modelo entrega una probabilidad calibrada p. Para cada proceso se elige la acción con menor
costo esperado (unidad: lo que cuesta aprobar un fraude = 1):

    aprobar    ->  p · C_fraude
    rechazar   ->  (1 − p) · C_rechazo
    verificar  ->  C_verificación + (1 − p) · abandono · C_rechazo + p · (1 − captura) · C_fraude

"Verificar" es una prueba adicional (OTP, nueva selfie o un analista) para la zona gris.
Como p está calibrada, los puntos donde una acción pasa a ser más barata tienen solución exacta:
los umbrales salen de los costos y cambiarlos no requiere reentrenar.

Uso:
    python -m src.decision
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

APROBAR, VERIFICAR, RECHAZAR = "aprobar", "verificar", "rechazar"


def cost_thresholds(dcfg):
    """Umbrales (verificar, rechazar) donde se cruzan los costos esperados de las acciones."""
    cf, cr = dcfg["costs"]["fraude_aprobado"], dcfg["costs"]["legitimo_rechazado"]
    su = dcfg["step_up"]
    cv, a, c = su["costo"], su["abandono_legitimos"], su["captura_fraude"]

    # Aprobar = verificar  ->  p · cf = cv + (1−p)·a·cr + p·(1−c)·cf   (despejando p)
    p_verificar = (cv + a * cr) / (c * cf + a * cr)
    # Verificar = rechazar  ->  cv + (1−p)·a·cr + p·(1−c)·cf = (1−p)·cr   (despejando p)
    p_rechazar = (cr * (1 - a) - cv) / ((1 - c) * cf + cr * (1 - a))

    if p_verificar >= p_rechazar:
        # Verificar nunca es lo más barato: queda una decisión binaria aprobar / rechazar
        p = cr / (cf + cr)
        return {"verificar": p, "rechazar": p}
    return {"verificar": p_verificar, "rechazar": p_rechazar}


def decide(score, umbrales):
    """Decisión para cada score."""
    score = np.asarray(score)
    return np.select([score >= umbrales["rechazar"], score >= umbrales["verificar"]],
                     [RECHAZAR, VERIFICAR], default=APROBAR)


def outcomes(y, decisiones, dcfg):
    """Evalúa una política ya aplicada: cuenta cada tipo de ERROR y lo multiplica por lo que
    cuesta. No decide nada. La verificación no atrapa todo el fraude y parte de los clientes
    buenos la abandona."""
    y = np.asarray(y).astype(bool)
    d = np.asarray(decisiones)
    cf, cr = dcfg["costs"]["fraude_aprobado"], dcfg["costs"]["legitimo_rechazado"]
    su = dcfg["step_up"]
    apr, ver, rec = d == APROBAR, d == VERIFICAR, d == RECHAZAR
    fraudes, buenos = y.sum(), (~y).sum()

    # Cada término = cuántos casos de ese error hubo × lo que cuesta cada uno
    costo = ((apr & y).sum() * cf                                   # error: fraude aprobado
             + (ver & y).sum() * (1 - su["captura_fraude"]) * cf   # error: fraude que pasó la verificación
             + (rec & ~y).sum() * cr                                # error: cliente bueno rechazado
             + (ver & ~y).sum() * su["abandono_legitimos"] * cr     # error: cliente bueno que abandonó
             + ver.sum() * su["costo"])                             # costo operativo de verificar
    costo_sin_modelo = fraudes * cf                                 # aprobar todo

    return {
        "por_decision": {
            nombre: {"pct_trafico": float(sel.mean()),
                     "tasa_fraude": float(y[sel].mean()) if sel.any() else None}
            for nombre, sel in [(APROBAR, apr), (VERIFICAR, ver), (RECHAZAR, rec)]
        },
        "fraude_detenido_pct": float(((rec & y).sum() + su["captura_fraude"] * (ver & y).sum()) / fraudes),
        "buenos_rechazados_pct": float((rec & ~y).sum() / buenos),
        "buenos_perdidos_pct": float(((rec & ~y).sum() + su["abandono_legitimos"] * (ver & ~y).sum()) / buenos),
        "costo_por_1000": float(1000 * costo / len(y)),
        "costo_por_1000_sin_modelo": float(1000 * costo_sin_modelo / len(y)),
        "ahorro_vs_sin_modelo_pct": float(1 - costo / costo_sin_modelo),
    }


def sensitivity(y, score, dcfg):
    """Cómo cambian umbrales e impacto según cuánto cuesta rechazar a un cliente bueno."""
    filas = []
    for ratio in dcfg["sensitivity"]:
        escenario = {**dcfg, "costs": {**dcfg["costs"], "legitimo_rechazado": ratio}}
        umbrales = cost_thresholds(escenario)
        r = outcomes(y, decide(score, umbrales), escenario)
        filas.append({
            "costo_rechazo_vs_fraude": ratio,
            "umbral_verificar": umbrales["verificar"],
            "umbral_rechazar": umbrales["rechazar"],
            **{f"pct_{k}": v["pct_trafico"] for k, v in r["por_decision"].items()},
            "fraude_detenido_pct": r["fraude_detenido_pct"],
            "buenos_perdidos_pct": r["buenos_perdidos_pct"],
            "costo_por_1000": r["costo_por_1000"],
        })
    return pd.DataFrame(filas)


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    dcfg, target = cfg["decision"], cfg["columns"]["target"]
    modelos_dir = Path(cfg["paths"]["models"])

    umbrales = cost_thresholds(dcfg)
    # Impacto esperado con los scores fuera de fold de train (nunca con test). El monitoreo
    # compara contra estas tasas de aprobar / verificar / rechazar.
    ref = pd.read_parquet(modelos_dir / "referencia.parquet")
    esperado = outcomes(ref[target], decide(ref["score"], umbrales), dcfg)

    (modelos_dir / "umbrales.json").write_text(json.dumps(
        {"umbrales": umbrales, "esperado_en_train": esperado}, indent=2), encoding="utf-8")

    print(f"Umbrales: verificar si score >= {umbrales['verificar']:.3f}, "
          f"rechazar si score >= {umbrales['rechazar']:.3f}")
    print(f"Esperado en train: detiene {esperado['fraude_detenido_pct']:.1%} del fraude, "
          f"pierde {esperado['buenos_perdidos_pct']:.2%} de clientes buenos")


if __name__ == "__main__":
    main()
