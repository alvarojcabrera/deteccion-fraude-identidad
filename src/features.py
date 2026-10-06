import numpy as np
import pandas as pd


def build_features(df, cfg):
    """Variables calculadas solo con información anterior a cada fila."""
    cols = cfg["columns"]
    # Orden cronológico: todo lo acumulado solo puede mirar hacia atrás
    df = df.sort_values([cols["date"], cols["id"]]).reset_index(drop=True)
    # El resultado de intentos ANTERIORES sí se conoce al decidir el actual (S12).
    # En un proceso nuevo STATUS aún no existe (vacío), así que no cuenta como fallo.
    fallo = (df["STATUS"] == "failure").astype(int)

    df = add_history_features(df, fallo, cols["account"], cols["date"], cfg["risk_reasons"])
    df = add_velocity_features(df, cols["account"], cols["date"])
    df = add_shared_identity_features(df, cfg["identity"], cols["account"])
    df = add_identity_history_features(df, fallo, cfg["identity_history"])
    df = add_missing_flags(df, cfg["identity"])
    return df


def add_history_features(df, fallo, account, date, risk_reasons):
    """Historial de la propia cuenta: intentos, fallos y rechazos por motivos de riesgo previos."""
    cuenta = df.groupby(account)

    df["intentos_previos"] = cuenta.cumcount()
    # Suma acumulada menos el valor actual = solo lo anterior
    df["fallos_previos"] = fallo.groupby(df[account]).cumsum() - fallo
    df["tasa_fallos_previos"] = df["fallos_previos"] / df["intentos_previos"].replace(0, np.nan)

    # Rechazos anteriores por motivos asociados a fraude (foto falsa, fotocopia, rostro...)
    riesgo = df["DECLINED_REASON"].isin(risk_reasons).astype(int)
    df["rechazos_riesgo_previos"] = riesgo.groupby(df[account]).cumsum() - riesgo

    df["horas_desde_intento_anterior"] = cuenta[date].diff().dt.total_seconds() / 3600
    return df


def add_velocity_features(df, account, date, ventana_horas=24):
    """Intentos de la misma cuenta en las `ventana_horas` anteriores."""
    segundos = (df[date] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    ventana = ventana_horas * 3600

    def contar(t):
        # t: tiempos de UNA cuenta en orden cronológico. Para cada intento i, searchsorted
        # da la posición del primer intento dentro de la ventana; i menos esa posición es
        # cuántos intentos anteriores caen en ella.
        t = t.to_numpy()
        return np.arange(len(t)) - np.searchsorted(t, t - ventana, side="left")

    df[f"intentos_{ventana_horas}h"] = segundos.groupby(df[account]).transform(contar).astype(int)
    return df


def add_shared_identity_features(df, identity, account):
    """Cuántas OTRAS cuentas usaron antes el mismo documento, email o teléfono."""
    for col, nombre in identity.items():
        # True la primera vez que aparece cada pareja (valor, cuenta)
        pareja_nueva = (~df.duplicated([col, account])).astype(int)
        cuentas_vistas = pareja_nueva.groupby(df[col]).cumsum()
        # Se resta la cuenta actual; sin valor de identidad -> 0
        df[f"cuentas_previas_{nombre}"] = (cuentas_vistas - 1).fillna(0).astype(int)
    return df


def add_identity_history_features(df, fallo, identity):
    """Intentos y fallos previos con el mismo documento o email, en cualquier cuenta."""
    for col, nombre in identity.items():
        clave = df[col]
        df[f"intentos_previos_{nombre}"] = df.groupby(col).cumcount().where(clave.notna(), 0)
        df[f"fallos_previos_{nombre}"] = ((fallo.groupby(clave).cumsum() - fallo)
                                          .fillna(0).astype(int))
    return df


def add_missing_flags(df, identity):
    """Indicadores de faltante: que falte un dato también es información (EDA §11)."""
    for col, nombre in identity.items():
        df[f"falta_{nombre}"] = df[col].isna().astype(int)
    df["falta_distrust"] = df["DISTRUST_U"].isna().astype(int)
    df["falta_lifetime"] = df["LIFETIME_U"].isna().astype(int)
    return df
