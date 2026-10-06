"""Tests de la lógica de decisión por costo esperado."""
import numpy as np
import pytest

from src.decision import APROBAR, RECHAZAR, VERIFICAR, cost_thresholds, decide, outcomes

STEP_UP = {"costo": 0.05, "abandono_legitimos": 0.10, "captura_fraude": 0.70}


def escenario(costo_rechazo):
    return {"costs": {"fraude_aprobado": 1.0, "legitimo_rechazado": costo_rechazo}, "step_up": STEP_UP}


def costo_esperado(p, accion, dcfg):
    """Las mismas fórmulas del docstring de src/decision.py, escritas aparte para comprobarlas."""
    cf, cr = dcfg["costs"]["fraude_aprobado"], dcfg["costs"]["legitimo_rechazado"]
    su = dcfg["step_up"]
    return {
        APROBAR: p * cf,
        RECHAZAR: (1 - p) * cr,
        VERIFICAR: su["costo"] + (1 - p) * su["abandono_legitimos"] * cr + p * (1 - su["captura_fraude"]) * cf,
    }[accion]


def test_bandas_en_orden():
    umbrales = {"verificar": 0.4, "rechazar": 0.7}
    decisiones = decide([0.1, 0.39, 0.4, 0.69, 0.7, 0.99], umbrales)
    assert list(decisiones) == [APROBAR, APROBAR, VERIFICAR, VERIFICAR, RECHAZAR, RECHAZAR]


@pytest.mark.parametrize("costo_rechazo", [0.1, 0.5, 1, 2, 5])
def test_umbrales_minimizan_el_costo_esperado(costo_rechazo):
    """Para cualquier score, la decisión tomada es la acción de menor costo esperado."""
    dcfg = escenario(costo_rechazo)
    umbrales = cost_thresholds(dcfg)
    for p in np.linspace(0.001, 0.999, 500):
        elegida = decide([p], umbrales)[0]
        minimo = min(costo_esperado(p, a, dcfg) for a in (APROBAR, VERIFICAR, RECHAZAR))
        assert costo_esperado(p, elegida, dcfg) <= minimo + 1e-9


def test_rechazo_injusto_mas_caro_rechaza_menos():
    assert cost_thresholds(escenario(5))["rechazar"] > cost_thresholds(escenario(0.5))["rechazar"]


def test_impacto_con_verificacion_parcial():
    y = np.array([1, 1, 0, 0])
    d = np.array([VERIFICAR, APROBAR, VERIFICAR, RECHAZAR])
    r = outcomes(y, d, escenario(2))
    assert r["fraude_detenido_pct"] == pytest.approx(0.70 / 2)        # la verificación atrapa el 70 %
    assert r["buenos_perdidos_pct"] == pytest.approx((1 + 0.10) / 2)  # 1 rechazado + 10 % que abandona
