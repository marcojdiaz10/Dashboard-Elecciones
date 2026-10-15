# -*- coding: utf-8 -*-
"""
Servicio del modelo de predicción de votantes por puesto.

Reproduce EXACTAMENTE la construcción de variables del notebook
`prediccion_votantes_por_puesto_mlflow_indice_ideologico.ipynb` (secciones 2 y 3)
para que el modelo reciba las mismas 17 variables con las que se entrenó:

    log_censo, log_n_mesas, pct_mujeres, log_electores_mesa, peso_puesto,
    censo_rank_pct, censo_rel_mun, log_n_puestos_mun, log_censo_mun,
    n_candidatos_mun, n_partidos_mun, pct_tradicional_mun, pct_coalicion_mun,
    incl_media_mun, incl_std_mun, incl_rango_mun, dep

El modelo predice la TASA de participación; votantes = tasa × censo.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DIR_APP = Path(__file__).resolve().parent
RUTA_MODELO = DIR_APP / ".." / "Notebooks" / "mejor_modelo_votaciones.pkl"
# Predicciones de test que guarda el notebook (sección 6): sirven para un intervalo honesto
RUTA_PRED_TEST = DIR_APP / ".." / "Notebooks" / "votantes_predichos_test.csv"

CENSO_MIN = 10   # mismo valor que en el notebook
NUM = ["log_censo", "log_n_mesas", "pct_mujeres", "log_electores_mesa", "peso_puesto", "censo_rank_pct",
       "censo_rel_mun", "log_n_puestos_mun", "log_censo_mun",
       "n_candidatos_mun", "n_partidos_mun", "pct_tradicional_mun", "pct_coalicion_mun",
       "incl_media_mun", "incl_std_mun", "incl_rango_mun"]
CAT = ["dep"]
FEATURES = NUM + CAT
TRADICIONAL = r"LIBERAL COLOMBIANO|CONSERVADOR COLOMBIANO|PARTIDO DE LA U\b|UNI[OÓ]N POR LA GENTE|CAMBIO RADICAL"
COLUMNAS = ["dep", "mun", "puesto_nom", "partido", "candidato", "votos_totales", "inclinacion_politica",
            "n_total_psto", "n_mesas_psto", "n_hombres_psto", "n_mujeres_psto"]

ERROR_CARGA: str | None = None   # último error al cargar el modelo (la app lo muestra)


# ---------------------------------------------------------------------------
# Tabla de puestos con las variables del modelo
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def tabla_puestos(ruta: str) -> pd.DataFrame:
    """Una fila por puesto válido con las 17 variables del modelo, más dep, mun,
    puesto_nom, censo, votos_puesto y participacion (0-1)."""
    df = pd.read_parquet(ruta, columns=COLUMNAS)
    for c in ["dep", "mun", "puesto_nom", "partido", "candidato"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["votos_totales", "inclinacion_politica", "n_total_psto", "n_mesas_psto",
              "n_hombres_psto", "n_mujeres_psto"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["mun_key"] = df["dep"] + " | " + df["mun"]
    df["puesto_key"] = df["mun_key"] + " | " + df["puesto_nom"]

    # Limpieza (sección 2 del notebook)
    df = df.drop_duplicates(["puesto_key", "partido", "candidato"], keep="first")
    df = df[df["votos_totales"] <= df["n_total_psto"]].reset_index(drop=True)

    puestos = (df.groupby("puesto_key").agg(
        mun_key=("mun_key", "first"), dep=("dep", "first"), mun=("mun", "first"),
        puesto_nom=("puesto_nom", "first"), votos_puesto=("votos_totales", "sum"),
        censo=("n_total_psto", "first"), n_mesas=("n_mesas_psto", "first"),
        n_hombres=("n_hombres_psto", "first"), n_mujeres=("n_mujeres_psto", "first")).reset_index())
    puestos["participacion"] = puestos["votos_puesto"] / puestos["censo"]
    invalido = (puestos["participacion"] > 1) | (puestos["participacion"] == 0) | (puestos["censo"] < CENSO_MIN)
    puestos = puestos[~invalido].reset_index(drop=True)

    # Candidatos inscritos por municipio
    cm = df.drop_duplicates(["mun_key", "candidato"])[["mun_key", "candidato", "partido", "inclinacion_politica"]].copy()
    cm["trad"] = cm["partido"].str.upper().str.contains(TRADICIONAL, regex=True).astype(int)
    cm["coal"] = cm["partido"].str.upper().str.contains("COALICI").astype(int)
    mun_c = cm.groupby("mun_key").agg(
        n_candidatos_mun=("candidato", "nunique"), n_partidos_mun=("partido", "nunique"),
        incl_media_mun=("inclinacion_politica", "mean"), incl_std_mun=("inclinacion_politica", "std"),
        incl_min=("inclinacion_politica", "min"), incl_max=("inclinacion_politica", "max"),
        pct_tradicional_mun=("trad", "mean"), pct_coalicion_mun=("coal", "mean"))
    mun_c["incl_std_mun"] = mun_c["incl_std_mun"].fillna(0.0)
    mun_c["incl_rango_mun"] = mun_c["incl_max"] - mun_c["incl_min"]
    mun_c = mun_c.drop(columns=["incl_min", "incl_max"])

    # Variables (sección 3 del notebook)
    d = puestos.merge(mun_c, left_on="mun_key", right_index=True, how="left")
    g = d.groupby("mun_key")
    d["log_censo"] = np.log(d["censo"])
    d["log_n_mesas"] = np.log1p(d["n_mesas"])
    d["pct_mujeres"] = d["n_mujeres"] / d["censo"]
    d["log_electores_mesa"] = np.log1p(d["censo"] / d["n_mesas"])
    d["n_puestos_mun"] = g["puesto_key"].transform("size")
    d["censo_mun"] = g["censo"].transform("sum")
    d["log_n_puestos_mun"] = np.log1p(d["n_puestos_mun"])
    d["log_censo_mun"] = np.log1p(d["censo_mun"])
    d["peso_puesto"] = d["censo"] / d["censo_mun"]
    d["censo_rank_pct"] = g["censo"].rank(pct=True)
    d["censo_rel_mun"] = d["censo"] / g["censo"].transform("mean")
    return d


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def modelo_disponible() -> bool:
    return RUTA_MODELO.exists()


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el Pipeline guardado con joblib. Devuelve None si falla y deja el motivo
    en ERROR_CARGA (p. ej. versión de scikit-learn distinta a la del entrenamiento)."""
    global ERROR_CARGA
    try:
        import joblib
        modelo = joblib.load(RUTA_MODELO)
    except Exception as e:  # noqa: BLE001 - se informa en la app
        ERROR_CARGA = f"{type(e).__name__}: {e}"
        return None
    faltan = set(getattr(modelo, "feature_names_in_", FEATURES)) - set(FEATURES)
    if faltan:
        ERROR_CARGA = f"El modelo pide variables que este servicio no construye: {sorted(faltan)}"
        return None
    ERROR_CARGA = None
    return modelo


def predecir(modelo, filas: pd.DataFrame, censos) -> pd.DataFrame:
    """Participación predicha (0-1) y votantes esperados = tasa × censo."""
    columnas = list(getattr(modelo, "feature_names_in_", FEATURES))
    faltan = [c for c in columnas if c not in filas.columns]
    if faltan:
        raise ValueError(f"Faltan variables para el modelo: {faltan}")
    tasa = np.clip(modelo.predict(filas[columnas]), 0, 1)
    censos = np.asarray(censos, dtype=float)
    return pd.DataFrame({"participacion_predicha": tasa,
                         "votantes_predichos": np.round(tasa * censos).astype(int)}, index=filas.index)


@lru_cache(maxsize=2)
def residuos_participacion(ruta: str) -> np.ndarray:
    """Residuos (real − predicho) de la tasa de participación para el intervalo empírico.
    Usa las predicciones de TEST del notebook si existen (municipios no vistos: honesto).
    Si no, usa todos los puestos (incluye entrenamiento: el intervalo sale más estrecho)."""
    if RUTA_PRED_TEST.exists():
        t = pd.read_csv(RUTA_PRED_TEST)
        t = t[t["censo"] > 0]
        return (t["votantes_reales"] / t["censo"] - t["participacion_predicha"]).to_numpy()
    modelo = cargar_modelo()
    if modelo is None:
        return np.array([])
    tabla = tabla_puestos(ruta)
    pred = predecir(modelo, tabla, tabla["censo"])
    return (tabla["participacion"] - pred["participacion_predicha"]).to_numpy()


def origen_residuos() -> str:
    return ("predicciones de test del notebook (municipios no vistos)" if RUTA_PRED_TEST.exists()
            else "todos los puestos, incluidos los de entrenamiento (intervalo optimista)")
