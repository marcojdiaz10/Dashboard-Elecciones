# -*- coding: utf-8 -*-
"""
Simulación de Montecarlo de las alcaldías 2023 (lógica de Simulacio_MonteCarlo_Elecciones.ipynb).

En cada escenario se perturban tres variables:
  1. Participación de cada puesto           ~ Normal(participación observada, sigma_part)
  2. Ideología de cada candidato (medición)  ~ Normal(clasificacion, sigma_cand),
                                              sigma_cand = 0.08 + 0.30·|clasificacion − inclinación del partido|
  3. Desplazamiento ideológico municipal     ~ Normal(0, sigma_desplaz)
Los votos de cada puesto se reparten según la afinidad base del candidato (suavizada) y
la distancia entre el electorado desplazado y el candidato: peso = share · exp(−λ·Δdistancia).

Cambio frente al notebook: solo se usan puestos con censo > 0 y participación < 100 %
(antes se subía el censo cuando los votos lo superaban). El ganador de referencia se
calcula con esos mismos puestos, para que la comparación sea consistente.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

COLS = ["dep", "mun", "puesto_nom", "partido", "candidato", "votos_totales",
        "clasificacion", "inclinacion_politica", "n_total_psto"]
CORTE_BLOQUE = 0.15   # bloques del notebook: izquierda < −0,15 ≤ centro ≤ 0,15 < derecha
BLOQUES = ["Izquierda", "Centro", "Derecha"]


@lru_cache(maxsize=2)
def preparar(ruta: str) -> dict:
    """Vectores base de la simulación (se calcula una vez por dataset)."""
    df = pd.read_parquet(ruta, columns=COLS)
    for c in ["dep", "mun", "puesto_nom", "partido", "candidato"]:
        df[c] = df[c].astype(str)
    for c in ["votos_totales", "clasificacion", "inclinacion_politica", "n_total_psto"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["votos_totales"] = df["votos_totales"].fillna(0)

    # Ideología del candidato: clasificación; si falta, la del partido; si falta, centro (0)
    ideo_faltante = int(df["clasificacion"].isna().sum())
    df["clasificacion"] = df["clasificacion"].fillna(df["inclinacion_politica"]).fillna(0.0)
    df["inclinacion_politica"] = df["inclinacion_politica"].fillna(df["clasificacion"])

    llave = ["dep", "mun", "puesto_nom"]
    puesto = df.groupby(llave).agg(votos_obs=("votos_totales", "sum"),
                                   censo_psto=("n_total_psto", "first")).reset_index()
    n_puestos_total = len(puesto)
    valido = (puesto["censo_psto"] > 0) & (puesto["votos_obs"] < puesto["censo_psto"])
    excluidos = puesto.loc[~valido, llave + ["votos_obs", "censo_psto"]]
    puesto = puesto[valido].copy()
    puesto["part_base"] = np.clip(puesto["votos_obs"] / puesto["censo_psto"], 0.05, 0.98)

    df = df.merge(puesto, on=llave, how="inner")     # solo filas de puestos válidos
    df["votos_x_ideo"] = df["votos_totales"] * df["clasificacion"]
    ideo = df.groupby(llave).agg(sum_vx=("votos_x_ideo", "sum"), sum_v=("votos_totales", "sum"),
                                 mean_ideo=("clasificacion", "mean")).reset_index()
    ideo["ideo_base_psto"] = np.where(ideo["sum_v"] > 0, ideo["sum_vx"] / ideo["sum_v"].replace(0, np.nan),
                                      ideo["mean_ideo"])
    df = df.merge(ideo[llave + ["ideo_base_psto"]], on=llave)

    # Afinidad base suavizada (Laplace) de cada candidato en cada puesto
    df["share_base"] = np.where(df["votos_obs"] > 0, (df["votos_totales"] + 0.5) / (df["votos_obs"] + 2.5), 0.20)
    df["puesto_id"] = df.groupby(llave).ngroup()
    df["mun_id"] = df.groupby(["dep", "mun"]).ngroup()
    df["cand_id"] = df.groupby(["dep", "mun", "candidato"]).ngroup()

    pu = df.drop_duplicates("puesto_id").sort_values("puesto_id")
    ca = df.drop_duplicates("cand_id").sort_values("cand_id")[
        ["cand_id", "mun_id", "dep", "mun", "candidato", "partido", "clasificacion", "inclinacion_politica"]
    ].reset_index(drop=True)
    ca["votos_reales"] = df.groupby("cand_id")["votos_totales"].sum().to_numpy()
    muns = df.drop_duplicates("mun_id").sort_values("mun_id")[["mun_id", "dep", "mun"]].reset_index(drop=True)
    muns["censo"] = pu.groupby("mun_id")["censo_psto"].sum().to_numpy()

    p_idx, c_idx = df["puesto_id"].to_numpy(), df["cand_id"].to_numpy()
    ideo_psto = pu["ideo_base_psto"].to_numpy()
    ideo_cand = ca["clasificacion"].to_numpy()
    base = {
        "n_puestos": len(pu), "n_muns": len(muns), "n_cands": len(ca),
        "censo": pu["censo_psto"].to_numpy(), "part_base": pu["part_base"].to_numpy(), "ideo_psto": ideo_psto,
        "ideo_cand": ideo_cand, "sigma_cand": 0.08 + 0.30 * np.abs(ideo_cand - ca["inclinacion_politica"].to_numpy()),
        "mun_de_cand": ca["mun_id"].to_numpy(),
        "p_idx": p_idx, "m_idx": df["mun_id"].to_numpy(), "c_idx": c_idx,
        "share_base": df["share_base"].to_numpy(),
        "dist_base": np.abs(ideo_psto[p_idx] - ideo_cand[c_idx]),
        "cands": ca, "muns": muns, "excluidos": excluidos,
        "n_puestos_total": n_puestos_total, "ideo_faltante": ideo_faltante,
    }
    base["ganador_real"] = _ganadores(base["cands"]["votos_reales"].to_numpy(), base["mun_de_cand"])
    return base


def _ganadores(votos_cand: np.ndarray, mun_de_cand: np.ndarray) -> np.ndarray:
    """Índice del candidato con más votos en cada municipio (mismo método del notebook)."""
    orden = np.lexsort((votos_cand, mun_de_cand))
    _, primero = np.unique(mun_de_cand[orden], return_index=True)
    ultimo = np.append(primero[1:] - 1, len(orden) - 1)
    return orden[ultimo]


def _bloque(ideo: np.ndarray) -> np.ndarray:
    return np.where(ideo < -CORTE_BLOQUE, 0, np.where(ideo > CORTE_BLOQUE, 2, 1))


def ensayo(b: dict, rng: np.random.Generator, sigma_part: float, sigma_desplaz: float,
           lambda_espacial: float, error_candidato: bool = True):
    """Un escenario. Devuelve (participación, ideología candidatos, desplazamiento, votos candidatos, ganadores)."""
    part = np.clip(rng.normal(b["part_base"], sigma_part), 0.05, 0.98) if sigma_part > 0 else b["part_base"]
    votos_puesto = np.round(b["censo"] * part)
    ideo_c = (np.clip(rng.normal(b["ideo_cand"], b["sigma_cand"]), -1.0, 1.0)
              if error_candidato else b["ideo_cand"])
    desplaz = rng.normal(0.0, sigma_desplaz, size=b["n_muns"]) if sigma_desplaz > 0 else np.zeros(b["n_muns"])

    p, m, c = b["p_idx"], b["m_idx"], b["c_idx"]
    ideo_elect = np.clip(b["ideo_psto"][p] + desplaz[m], -1.0, 1.0)
    dist = np.abs(ideo_elect - ideo_c[c])
    peso = b["share_base"] * np.exp(-lambda_espacial * (dist - b["dist_base"]))
    suma = np.bincount(p, weights=peso, minlength=b["n_puestos"])
    votos_fila = np.round(votos_puesto[p] * peso / np.maximum(suma[p], 1e-12))
    votos_cand = np.bincount(c, weights=votos_fila, minlength=b["n_cands"])
    return part, ideo_c, desplaz, votos_cand, _ganadores(votos_cand, b["mun_de_cand"])


def simular(ruta: str, n_sim: int = 150, sigma_part: float = 0.06, sigma_desplaz: float = 0.15,
            lambda_espacial: float = 2.0, semilla: int = 42) -> dict:
    """Corre n_sim escenarios y resume: probabilidad de ganar por candidato, rangos de votos,
    probabilidad de cambio de alcalde por municipio y registro por iteración."""
    b = preparar(ruta)
    rng = np.random.default_rng(semilla)
    victorias = np.zeros(b["n_cands"], dtype=np.int32)
    cambios = np.zeros(b["n_muns"], dtype=np.int32)
    votos = np.empty((n_sim, b["n_cands"]), dtype=np.float32)
    registro = []
    for i in range(n_sim):
        part, ideo_c, desplaz, votos_cand, gan = ensayo(b, rng, sigma_part, sigma_desplaz, lambda_espacial)
        votos[i] = votos_cand
        victorias[gan] += 1
        cambio = gan != b["ganador_real"]
        cambios += cambio
        bloque_c = _bloque(ideo_c)
        alcaldias = np.bincount(bloque_c[gan], minlength=3)
        registro.append({
            "iteracion": i + 1, "total_votos": float(votos_cand.sum()), "part_media": float(part.mean()),
            "desplaz_medio": float(desplaz.mean()), "alcaldias_cambian": int(cambio.sum()),
            **{f"votos_{n.lower()}": float(votos_cand[bloque_c == k].sum()) for k, n in enumerate(BLOQUES)},
            **{f"alcaldias_{n.lower()}": int(alcaldias[k]) for k, n in enumerate(BLOQUES)},
        })

    cands = b["cands"].copy()
    cands["prob_ganar"] = victorias / n_sim
    p05, p50, p95 = np.percentile(votos, [5, 50, 95], axis=0)
    cands["votos_p05"], cands["votos_mediana"], cands["votos_p95"] = p05, p50, p95
    cands["gano_2023"] = False
    cands.loc[b["ganador_real"], "gano_2023"] = True

    muns = b["muns"].copy()
    muns["prob_cambio"] = cambios / n_sim
    gan_real = cands.loc[b["ganador_real"]].reset_index(drop=True)
    muns["ganador_2023"] = gan_real["candidato"].to_numpy()
    muns["ideo_ganador_2023"] = gan_real["clasificacion"].to_numpy()
    # Margen 2023 entre primero y segundo (en % de los votos del municipio)
    v = cands.sort_values(["mun_id", "votos_reales"], ascending=[True, False])
    top2 = v.groupby("mun_id")["votos_reales"].apply(lambda s: (s.iloc[0] - (s.iloc[1] if len(s) > 1 else 0)) / max(s.sum(), 1))
    muns["margen_2023"] = top2.reindex(muns["mun_id"]).to_numpy()
    muns["n_candidatos"] = cands.groupby("mun_id").size().reindex(muns["mun_id"]).to_numpy()
    return {"cands": cands, "muns": muns, "registro": pd.DataFrame(registro),
            "parametros": dict(n_sim=n_sim, sigma_part=sigma_part, sigma_desplaz=sigma_desplaz,
                               lambda_espacial=lambda_espacial, semilla=semilla),
            "n_puestos": b["n_puestos"], "n_puestos_total": b["n_puestos_total"],
            "excluidos": len(b["excluidos"]), "ideo_faltante": b["ideo_faltante"]}


@lru_cache(maxsize=2)
def calibracion(ruta: str) -> dict:
    """Escenario SIN incertidumbre (las tres sigmas en cero). Debería reproducir los ganadores
    2023; los municipios que cambian aquí lo hacen por el suavizado, el redondeo o el recorte
    de la participación, no por la incertidumbre."""
    b = preparar(ruta)
    _, _, _, votos_cand, gan = ensayo(b, np.random.default_rng(0), 0.0, 0.0, 0.0, error_candidato=False)
    cambia = gan != b["ganador_real"]
    muns = b["muns"].loc[cambia, ["dep", "mun"]].copy()
    ca = b["cands"]
    muns["ganador_2023"] = ca.loc[b["ganador_real"][cambia], "candidato"].to_numpy()
    muns["ganador_sin_ruido"] = ca.loc[gan[cambia], "candidato"].to_numpy()
    return {"cambian": int(cambia.sum()), "n_muns": b["n_muns"], "detalle": muns.reset_index(drop=True),
            "votos_reales": float(ca["votos_reales"].sum()), "votos_sin_ruido": float(votos_cand.sum())}
