# -*- coding: utf-8 -*-
"""
Dashboard EDA - Resultados electorales Colombia (votos totales).

Ejecutar:  streamlit run app.py
Requiere:  ../Dataset_completo.parquet en la misma carpeta que este archivo.

Diseñado para ~13 millones de filas:
- Las columnas de texto se leen como categorías (diccionario de Arrow), no como
  texto repetido, y los números se reducen a 32 bits.
- El discurso se lee aparte y se guarda una sola vez por candidato.
- Cada combinación de filtros se resume UNA vez en tablas pequeñas (candidatos,
  puestos, territorio) que se guardan en caché; los gráficos usan esas tablas.
"""

from __future__ import annotations

import re
import unicodedata
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
from plotly.colors import sample_colorscale
import pyarrow.parquet as pq
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

import re
import unicodedata

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EDA Electoral Colombia", page_icon="🗳️", layout="wide")

RUTA_DATOS = Path(os.getenv("DASHBOARD_DATA_PATH", str(Path(__file__).parent / "../Dataset_completo.parquet"))).expanduser()
TAM_MUESTRA_SCATTER = 200_000      # máximo de puntos en los gráficos de dispersión
LIMITE_FILAS_EXCEL = 1_048_576     # Excel no abre más filas que esto
LIMITE_FILAS_CSV = 3_000_000       # por encima de esto no se ofrece la descarga

COLS_TEXTO = ["dep", "mun", "puesto_nom", "partido", "candidato"]
COLS_INDICE = ["clasificacion", "inclinacion_politica"]
COLS_PUESTO = ["n_mesas_psto", "n_hombres_psto", "n_mujeres_psto", "n_total_psto"]
COLS_REQUERIDAS = COLS_TEXTO + ["votos_totales"] + COLS_INDICE + COLS_PUESTO

# Grupos ideológicos a partir de los índices (-1 izquierda ... 1 derecha).
# Ajusta los cortes si tu índice tiene otra escala o distribución.
CORTES_IDEOLOGIA = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
ETIQUETAS_IDEOLOGIA = ["Izquierda", "Centro-izquierda", "Centro", "Centro-derecha", "Derecha"]
COLORES_IDEOLOGIA = dict(zip(
    ETIQUETAS_IDEOLOGIA, ["#b2182b", "#ef8a62", "#9e9e9e", "#67a9cf", "#2166ac"]
))
ESCALA_IDEOLOGIA = [[0.0, "#b2182b"], [0.5, "#9e9e9e"], [1.0, "#2166ac"]]
RANGO_COMPLETO = (-1.0, 1.0)

INDICES = {  # etiqueta en pantalla -> (columna numérica, columna de grupo, prefijo)
    "Inclinación del partido": ("inclinacion", "incl_cat", "inc"),
    "Clasificación del candidato": ("clasificacion", "clas_cat", "clas"),
}

# Stopwords básicas en español (sin tildes) para el análisis de discurso
STOPWORDS_ES = {
    "a","al","algo","algun","alguna","algunas","alguno","algunos","ante","antes",
    "aqui","asi","aun","cada","como","con","contra","cual","cuando","de","del",
    "desde","donde","dos","el","ella","ellas","ellos","en","entre","era","eran",
    "eres","es","esa","esas","ese","eso","esos","esta","estan","estar","estas",
    "este","esto","estos","fue","fueron","ha","han","hasta","hay","la","las",
    "le","les","lo","los","mas","me","mi","mis","mucho","muy","nada","ni","no",
    "nos","nosotros","nuestra","nuestras","nuestro","nuestros","o","para","pero",
    "por","porque","que","quien","quienes","se","sea","ser","si","sido","sin",
    "sobre","solo","son","soy","su","sus","tambien","te","tiene","tienen","toda",
    "todas","todo","todos","tu","tus","un","una","unas","uno","unos","y","ya","yo",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def fmt_num(n: float) -> str:
    """Número con punto de miles (formato colombiano)."""
    return "—" if pd.isna(n) else f"{n:,.0f}".replace(",", ".")


def fmt_pct(x: float) -> str:
    """Proporción (0-1) como porcentaje con coma decimal."""
    return "—" if pd.isna(x) else f"{x * 100:.1f} %".replace(".", ",")


def fmt_idx(x: float) -> str:
    """Índice ideológico con signo y coma decimal."""
    return "—" if pd.isna(x) else f"{x:+.2f}".replace(".", ",")


def texto(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas categóricas a texto antes de graficar
    (evita categorías vacías en los ejes y respeta el orden que les demos)."""
    cats = df.select_dtypes("category").columns
    return df.astype({c: object for c in cats}) if len(cats) else df


def categorizar(serie: pd.Series) -> pd.Series:
    """Agrupa un índice continuo (-1 a 1) en grupos ideológicos."""
    return pd.cut(serie, bins=CORTES_IDEOLOGIA, labels=ETIQUETAS_IDEOLOGIA,
                  include_lowest=True).astype(object)


def media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    m = valores.notna() & pesos.notna()
    return float(np.average(valores[m], weights=pesos[m])) if pesos[m].sum() > 0 else np.nan


def describir(x) -> pd.DataFrame:
    """Estadísticas descriptivas de un vector de votos."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"votos": []})
    p25, p50, p75, p90, p99 = np.percentile(x, [25, 50, 75, 90, 99])
    filas = {
        "n": len(x), "media": x.mean(), "desv. est.": x.std(ddof=1) if len(x) > 1 else np.nan,
        "mín": x.min(), "p25": p25, "mediana": p50, "p75": p75, "p90": p90, "p99": p99,
        "máx": x.max(), "% en cero": (x == 0).mean() * 100,
    }
    return pd.DataFrame({"votos": filas}).round(2)


def histograma(conteos: np.ndarray, bordes: np.ndarray, etiqueta_x: str,
               etiqueta_y: str = "Frecuencia"):
    """Histograma a partir de conteos ya calculados (no se envían millones de
    puntos al navegador, solo las barras)."""
    centros = (bordes[:-1] + bordes[1:]) / 2
    fig = px.bar(x=centros, y=conteos, labels={"x": etiqueta_x, "y": etiqueta_y})
    fig.update_traces(width=np.diff(bordes), marker_line_width=0)
    fig.update_layout(bargap=0)
    return fig


def _columnas_texto_arrow(ruta: str, columnas: list[str]) -> list[str]:
    """Columnas guardadas como texto en el Parquet (se leerán como diccionario)."""
    esquema = pq.read_schema(ruta)
    return [
        c for c in columnas
        if pa.types.is_string(esquema.field(c).type) or pa.types.is_large_string(esquema.field(c).type)
    ]


# ---------------------------------------------------------------------------
# Gráficos del EDA (Notebooks/EDA_indice_ideologico.ipynb)
# Cada función recibe datos ya filtrados y devuelve la figura, así responden a
# los filtros de la barra lateral.
# ---------------------------------------------------------------------------
GRIS_CLARO, GRIS_MEDIO, GRIS_OSCURO, ACENTO = "#cad2c5", "#84a98c", "#2f3e46", "#e76f51"
MIN_PUESTOS_DECILES = 10


def color_ideologia(valores) -> list[str]:
    """Color de la escala ideológica (-1 rojo · 0 gris · 1 azul) para cada valor."""
    v = np.clip((np.asarray(valores, dtype=float) + 1) / 2, 0, 1)
    return sample_colorscale(ESCALA_IDEOLOGIA, list(v))


def rgba(hex_color: str, alfa: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})"


def fmt_corto(n: float) -> str:
    """5,9 M · 850 mil · 120."""
    if n >= 1e6:
        return f"{n / 1e6:.1f} M".replace(".", ",")
    return f"{n / 1e3:.0f} mil" if n >= 1e3 else fmt_num(n)


def puestos_consistentes(pu: pd.DataFrame) -> pd.DataFrame:
    """Puestos con censo > 0 y participación <= 100 % (criterio del notebook)."""
    return pu[(pu["n_total_psto"] > 0) & (pu["participacion"] <= 100)]


def fig_distribucion_indice(serie: pd.Series):
    """G1 — Histograma del índice por puesto, coloreado por posición ideológica."""
    x = serie.dropna()
    conteos, bordes = np.histogram(x, bins=40, range=(-1, 1))
    centros = (bordes[:-1] + bordes[1:]) / 2
    pct_cat = categorizar(x).value_counts(normalize=True).reindex(ETIQUETAS_IDEOLOGIA).fillna(0) * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centros, y=conteos, width=(bordes[1] - bordes[0]) * 0.94, showlegend=False,
        marker=dict(color=color_ideologia(centros), line=dict(width=0)),
        customdata=np.c_[bordes[:-1], bordes[1:]],
        hovertemplate="Índice %{customdata[0]:+.2f} a %{customdata[1]:+.2f}<br>"
                      "<b>%{y:,} puestos</b><extra></extra>"))
    y_max = max(conteos.max(), 1) * 1.08
    for valor, nombre, estilo in [(x.mean(), "Media", "dash"), (x.median(), "Mediana", "dot")]:
        fig.add_trace(go.Scatter(x=[valor, valor], y=[0, y_max], mode="lines",
                                 line=dict(color="#111111", dash=estilo, width=2),
                                 name=f"{nombre} = {valor:+.3f}", hoverinfo="skip"))
    for c in CORTES_IDEOLOGIA[1:-1]:
        fig.add_vline(x=c, line=dict(color="#d0d0d0", width=1, dash="dot"))
    for cat, a, b in zip(ETIQUETAS_IDEOLOGIA, CORTES_IDEOLOGIA[:-1], CORTES_IDEOLOGIA[1:]):
        fig.add_annotation(x=(a + b) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"<b>{cat}</b><br>{pct_cat[cat]:.1f}% de puestos",
                           font=dict(color=COLORES_IDEOLOGIA[cat], size=12))
    fig.add_annotation(x=0.99, y=0.86, xref="paper", yref="paper", xanchor="right", showarrow=False,
                       align="right", bgcolor="rgba(255,255,255,0.9)", bordercolor="#e3e3e3",
                       font=dict(color="#2b2b2b"),
                       text=f"n = {len(x):,} puestos<br>sd = {x.std():.3f}<br>"
                            f"min {x.min():+.2f} · max {x.max():+.2f}")
    fig.update_layout(
        xaxis=dict(title="Índice ideológico del puesto (−1 izquierda · +1 derecha)", range=[-1, 1],
                   tickformat=".1f", dtick=0.2),
        yaxis=dict(title="Número de puestos", range=[0, y_max]),
        legend=dict(x=0.01, y=0.86, xanchor="left"), bargap=0, margin=dict(t=70), height=480)
    return fig


def fig_concentracion_deciles(votos: np.ndarray):
    """G1.1 — % del total de votos que reúne cada decil de tamaño del puesto."""
    serie_v = pd.Series(votos[votos > 0])
    decil = pd.qcut(serie_v.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    por_decil = serie_v.groupby(decil, observed=True).agg(["sum", "median"])
    por_decil["pct"] = por_decil["sum"] / por_decil["sum"].sum() * 100

    fig = go.Figure(go.Bar(
        x=por_decil.index.astype(str), y=por_decil["pct"],
        marker_color=[GRIS_CLARO] * 9 + [ACENTO], showlegend=False,
        text=por_decil["pct"], texttemplate="%{text:.1f}%", textposition="outside",
        customdata=por_decil["median"],
        hovertemplate="%{x}<br>%{y:.1f}% de los votos<br>"
                      "Votos medianos por puesto: %{customdata:,.0f}<extra></extra>"))
    fig.add_hline(y=10, line=dict(color="#9a9a9a", dash="dash", width=1),
                  annotation_text="10% si todos pesaran igual", annotation_position="top left",
                  annotation_font=dict(size=11, color="#6b6b6b"))
    fig.update_xaxes(title="Decil (D1 = puestos más pequeños · D10 = más grandes)")
    fig.update_yaxes(title="% del total de votos", ticksuffix="%",
                     range=[0, por_decil["pct"].max() * 1.18])
    fig.update_layout(height=450, margin=dict(t=30))
    return fig, por_decil


def fig_top_departamentos(top: pd.DataFrame, indice_nacional: float):
    """G2 — Índice agregado de los departamentos con más electores (top ya ordenado)."""
    etiquetas_y = [
        f"<b>{d}</b><br><span style='font-size:11px;color:#6b6b6b'>"
        f"{fmt_corto(top.loc[d, 'censo'])} electores · {fmt_num(top.loc[d, 'n_puestos'])} puestos</span>"
        for d in top.index
    ]
    fig = go.Figure(go.Bar(
        y=etiquetas_y, x=top["indice"], orientation="h", showlegend=False,
        marker=dict(color=color_ideologia(top["indice"])),
        text=[f"{v:+.3f}" for v in top["indice"]], textposition="outside",
        textfont=dict(size=12), cliponaxis=False,
        hovertemplate="%{y}<br>Índice: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#9a9a9a", width=1))
    fig.add_vline(x=indice_nacional, line=dict(color="#111111", width=1.5, dash="dash"),
                  annotation_text=f"Total filtrado: {indice_nacional:+.3f}", annotation_position="top")
    lim = max(top["indice"].abs().max(), abs(indice_nacional), 0.1) * 1.9
    fig.update_xaxes(range=[-lim, lim], tickformat=".2f", title="−1 izquierda · 0 centro · +1 derecha")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(margin=dict(l=210, t=40, b=60), height=max(360, 62 * len(top) + 100))
    return fig


def fig_indice_participacion(val: pd.DataFrame, col_ind: str, r_s: float, r_p: float):
    """G3 — Dispersión índice del puesto vs. participación (val ya muestreado)."""
    fig = go.Figure(go.Scattergl(
        x=val[col_ind], y=val["participacion"], mode="markers", name="Puesto",
        marker=dict(size=5, color=val[col_ind], colorscale=ESCALA_IDEOLOGIA, cmin=-1, cmax=1,
                    opacity=0.45, line=dict(width=0)),
        customdata=np.c_[val["dep"].astype(str), val["mun"].astype(str), val["puesto_nom"].astype(str)],
        hovertemplate="<b>%{customdata[2]}</b><br>%{customdata[1]} · %{customdata[0]}<br>"
                      "Índice %{x:+.2f} · Participación %{y:.1f}%<extra></extra>"))
    fig.add_annotation(x=0.01, y=0.04, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                       showarrow=False, align="left", bgcolor="rgba(255,255,255,0.9)",
                       bordercolor="#e3e3e3", font=dict(color="#2b2b2b"),
                       text=f"Spearman = {r_s:+.3f}<br>Pearson = {r_p:+.3f}")
    fig.update_layout(
        xaxis=dict(title="Índice ideológico del puesto", range=[-1, 1], tickformat=".1f"),
        yaxis=dict(title="Participación", range=[0, 100], ticksuffix="%"),
        height=480, margin=dict(t=30))
    return fig


def fig_participacion_categoria(val: pd.DataFrame):
    """G4 — Caja de la participación por categoría ideológica del puesto."""
    fig = go.Figure()
    for cat in ETIQUETAS_IDEOLOGIA:
        serie = val.loc[val["categoria"] == cat, "participacion"]
        fig.add_trace(go.Box(
            y=serie, name=f"<b>{cat}</b><br><span style='font-size:11px'>n={len(serie):,}</span>",
            marker=dict(color=COLORES_IDEOLOGIA[cat], size=3, opacity=0.4),
            line=dict(color=COLORES_IDEOLOGIA[cat], width=2.5),
            fillcolor=rgba(COLORES_IDEOLOGIA[cat], 0.35), boxpoints="outliers", showlegend=False,
            hovertemplate=f"{cat}<br>" + "%{y:.1f}%<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Participación", range=[0, 100], ticksuffix="%"),
                      xaxis=dict(title="Categoría ideológica del puesto", showgrid=False),
                      height=460, margin=dict(t=30))
    return fig


def fig_peso_electoral(peso_pct: pd.DataFrame):
    """G5 — % de puestos, censo y votos por categoría ideológica."""
    ticks = [f"<span style='color:{COLORES_IDEOLOGIA[c]}'><b>{c}</b></span>" for c in ETIQUETAS_IDEOLOGIA]
    fig = go.Figure()
    for medida, tono in zip(peso_pct.columns, [GRIS_CLARO, GRIS_MEDIO, GRIS_OSCURO]):
        fig.add_trace(go.Bar(
            x=ticks, y=peso_pct[medida], name=medida, marker_color=tono,
            text=peso_pct[medida], texttemplate="%{text:.1f}", textposition="outside",
            textfont=dict(size=11), hovertemplate=f"{medida}: " + "%{y:.1f}%<extra></extra>"))
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.06, height=460, margin=dict(t=40),
        yaxis=dict(title="% del total", ticksuffix="%", range=[0, max(peso_pct.to_numpy().max(), 1) * 1.18]),
        xaxis=dict(title="Categoría ideológica del puesto", showgrid=False),
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.02, yanchor="bottom"))
    return fig


# ---- G6: mapa estratégico ---------------------------------------------------
ORDEN_ESTRATEGIA = ["Movilizar", "Persuadir", "Consolidar", "Baja prioridad"]
COLOR_ESTRATEGIA = {"Movilizar": "#1f7a6f", "Persuadir": "#b5482b",
                    "Consolidar": "#7a7a7a", "Baja prioridad": "#7a7a7a"}


def clasificar_estrategia(m: pd.DataFrame, ideologia_candidato: float) -> tuple[pd.DataFrame, float, float]:
    """Distancia ideológica = |índice del puesto − candidato| / 2 (0 idéntico · 1 máxima distancia).
    Cortes = medianas del municipio. Cercano = distancia por debajo de la mediana."""
    m = m.copy()
    m["distancia"] = (m["indice"] - ideologia_candidato).abs() / 2
    m["abst_pct"] = (1 - m["participacion"]) * 100
    m["abstencionistas"] = (m["censo"] * (1 - m["participacion"])).round()
    corte_d, corte_ab = m["distancia"].median(), m["abst_pct"].median()
    cercano = m["distancia"] <= corte_d
    alta_ab = m["abst_pct"] >= corte_ab
    m["estrategia"] = np.select(
        [cercano & alta_ab, ~cercano & ~alta_ab, cercano & ~alta_ab],
        ["Movilizar", "Persuadir", "Consolidar"], default="Baja prioridad")
    return m, corte_d, corte_ab


def fig_mapa_estrategico(m: pd.DataFrame, corte_d: float, corte_ab: float, n_etiquetas: int = 3):
    """G6 — Distancia ideológica al candidato vs. abstención, por puesto del municipio."""
    x0, x1 = m["distancia"].min(), m["distancia"].max()
    y0, y1 = m["abst_pct"].min(), m["abst_pct"].max()
    px_, py_ = (x1 - x0) * 0.06 + 1e-3, (y1 - y0) * 0.08 + 1e-3
    x0, x1, y0, y1 = x0 - px_, x1 + px_, y0 - py_, y1 + py_

    cuadrantes = {
        "Movilizar": dict(caja=(x0, corte_d, corte_ab, y1), fondo="rgba(42,157,143,0.10)",
                          texto="<b>MOVILIZAR</b><br>cercanos que no votan", pos=(x0, y1, "left", "top")),
        "Persuadir": dict(caja=(corte_d, x1, y0, corte_ab), fondo="rgba(231,111,81,0.10)",
                          texto="<b>PERSUADIR</b><br>distantes que sí votan", pos=(x1, y0, "right", "bottom")),
        "Consolidar": dict(caja=(x0, corte_d, y0, corte_ab), fondo="rgba(0,0,0,0)",
                           texto="<b>CONSOLIDAR</b><br>base cercana que ya vota", pos=(x0, y0, "left", "bottom")),
        "Baja prioridad": dict(caja=(corte_d, x1, corte_ab, y1), fondo="rgba(0,0,0,0)",
                               texto="<b>BAJA PRIORIDAD</b><br>distantes y abstencionistas",
                               pos=(x1, y1, "right", "top")),
    }
    fig = go.Figure()
    for nombre, e in cuadrantes.items():
        a, b, c, d = e["caja"]
        fig.add_shape(type="rect", x0=a, x1=b, y0=c, y1=d, fillcolor=e["fondo"],
                      line=dict(width=0), layer="below")
        xa, ya, xan, yan = e["pos"]
        fig.add_annotation(x=xa, y=ya, xanchor=xan, yanchor=yan, showarrow=False, align=xan,
                           text=e["texto"], font=dict(size=12, color=COLOR_ESTRATEGIA[nombre]))
    fig.add_vline(x=corte_d, line=dict(color="#8a8a8a", dash="dash", width=1))
    fig.add_hline(y=corte_ab, line=dict(color="#8a8a8a", dash="dash", width=1))

    tam_ref = 2.0 * max(m["censo"].max(), 1) / (38 ** 2)
    fig.add_trace(go.Scatter(
        x=m["distancia"], y=m["abst_pct"], mode="markers", showlegend=False,
        marker=dict(size=m["censo"].clip(lower=1), sizemode="area", sizeref=tam_ref, sizemin=4,
                    color=m["indice"], colorscale=ESCALA_IDEOLOGIA, cmin=-1, cmax=1,
                    line=dict(color="white", width=1), opacity=0.9,
                    colorbar=dict(title=dict(text="Índice del<br>puesto"), tickformat=".1f",
                                  len=0.6, thickness=14)),
        customdata=np.c_[m["puesto_nom"].astype(str), m["estrategia"], m["censo"], m["votos"],
                         m["indice"], m["abstencionistas"]],
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                      "Índice %{customdata[4]:+.2f} · distancia %{x:.2f}<br>"
                      "Abstención %{y:.1f}% · %{customdata[5]:,.0f} abstencionistas<br>"
                      "Censo %{customdata[2]:,.0f} · votos %{customdata[3]:,.0f}<extra></extra>"))

    top_mov = m[m["estrategia"] == "Movilizar"].nlargest(n_etiquetas, "abstencionistas")
    top_per = m[m["estrategia"] == "Persuadir"].nlargest(n_etiquetas, "votos")
    for sub, color, dx in [(top_mov, COLOR_ESTRATEGIA["Movilizar"], 40),
                           (top_per, COLOR_ESTRATEGIA["Persuadir"], -40)]:
        for _, r in sub.iterrows():
            nombre = str(r["puesto_nom"])
            fig.add_annotation(x=r["distancia"], y=r["abst_pct"], ax=dx, ay=-28, showarrow=True,
                               arrowhead=0, arrowcolor=color, font=dict(size=10, color=color),
                               bgcolor="rgba(255,255,255,0.85)",
                               text=nombre if len(nombre) <= 24 else nombre[:23] + "…")
    fig.update_layout(
        xaxis=dict(title="Distancia ideológica respecto al candidato (0 = idéntica · 1 = máxima)",
                   range=[x0, x1], tickformat=".2f"),
        yaxis=dict(title="Abstención del puesto", range=[y0, y1], ticksuffix="%"),
        height=620, margin=dict(t=30, r=40))
    return fig


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
# Se usa cache_resource (y no cache_data) porque cache_data devuelve una COPIA
# del DataFrame en cada interacción: con 13 millones de filas eso duplica la
# memoria y hace lenta la app. Por eso el DataFrame nunca se modifica después.
@st.cache_resource(show_spinner="Cargando dataset (la primera vez tarda un poco)…")
def cargar_datos(ruta: str) -> pd.DataFrame:
    disponibles = pq.read_schema(ruta).names
    faltantes = [c for c in COLS_REQUERIDAS if c not in disponibles]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")

    tabla = pq.read_table(
        ruta, columns=COLS_REQUERIDAS,
        read_dictionary=_columnas_texto_arrow(ruta, COLS_TEXTO),
    )
    df = tabla.to_pandas()
    del tabla

    for c in COLS_TEXTO:
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")

    df["votos_totales"] = (
        pd.to_numeric(df["votos_totales"], errors="coerce").fillna(0).astype("int32")
    )
    for c in COLS_INDICE + COLS_PUESTO:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # Solo puestos con participación < 100 % (ver puestos_validos)
    valido = puestos_validos(df)
    if not valido.all():
        df = df.loc[valido].reset_index(drop=True)

    # Identificador numérico denso de puesto (0..N-1): acelera las agregaciones
    df["puesto_id"] = (
        df.groupby(LLAVE_PUESTO, observed=True, sort=False, dropna=False)
        .ngroup()
        .astype("int32")
    )
    return df


LLAVE_PUESTO = ["dep", "mun", "puesto_nom"]


def puestos_validos(df: pd.DataFrame) -> np.ndarray:
    """Máscara por fila: True si el puesto tiene censo > 0 y menos votos que personas habilitadas
    (participación < 100 %). Se excluyen los puestos con censo 0, sin censo o con participación ≥ 100 %."""
    grupo = df.groupby(LLAVE_PUESTO, observed=True, sort=False, dropna=False)
    votos = grupo["votos_totales"].transform("sum").to_numpy(dtype=np.float64)
    censo = grupo["n_total_psto"].transform("first").to_numpy(dtype=np.float64)
    return (censo > 0) & (votos < censo)


@st.cache_data(show_spinner=False)
def puestos_excluidos(ruta: str) -> pd.DataFrame:
    """Puestos que cargar_datos deja por fuera (para mostrarlos en la barra lateral)."""
    df = pd.read_parquet(ruta, columns=LLAVE_PUESTO + ["votos_totales", "n_total_psto"])
    df["votos_totales"] = pd.to_numeric(df["votos_totales"], errors="coerce").fillna(0)
    df["n_total_psto"] = pd.to_numeric(df["n_total_psto"], errors="coerce")
    fuera = df.loc[~puestos_validos(df)]
    t = (fuera.groupby(LLAVE_PUESTO, observed=True, dropna=False)
              .agg(votos=("votos_totales", "sum"), censo=("n_total_psto", "first")).reset_index())
    t["participacion"] = t["votos"] / t["censo"].where(t["censo"] > 0)
    return texto(t)


@st.cache_resource(show_spinner="Cargando discursos…")
def cargar_discursos(ruta: str) -> pd.DataFrame:
    """Un discurso por (candidato, partido). Se lee como diccionario para no
    cargar 13 millones de copias del mismo texto."""
    columnas = ["candidato", "partido", "discurso"]
    if "discurso" not in pq.read_schema(ruta).names:
        return pd.DataFrame(columns=columnas)

    tabla = pq.read_table(ruta, columns=columnas,
                          read_dictionary=_columnas_texto_arrow(ruta, columnas))
    d = tabla.to_pandas().drop_duplicates(subset=["candidato", "partido"])
    d = d[d["discurso"].notna()].astype({c: str for c in columnas})
    d = d[d["discurso"].str.strip() != ""]
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def opciones(ruta: str, col: str, deps: tuple = (), muns: tuple = ()) -> list:
    """Valores disponibles de una columna, dados los filtros territoriales previos."""
    df = cargar_datos(ruta)
    serie = df[col]
    if deps or muns:
        mask = np.ones(len(df), dtype=bool)
        if deps:
            mask &= df["dep"].isin(deps).to_numpy()
        if muns:
            mask &= df["mun"].isin(muns).to_numpy()
        serie = serie[mask]
    return sorted(serie.dropna().unique().tolist(), key=str)


def barra_lateral(ruta: str) -> tuple:
    """Dibuja los filtros y devuelve una tupla (hashable) con la selección."""
    st.sidebar.header("Filtros")
    st.sidebar.caption("Deja un filtro vacío para incluir todos los valores.")
    excluidos = puestos_excluidos(ruta)

    deps = tuple(st.sidebar.multiselect("Departamento", opciones(ruta, "dep")))
    muns = tuple(st.sidebar.multiselect("Municipio", opciones(ruta, "mun", deps)))
    partidos = tuple(st.sidebar.multiselect("Partido", opciones(ruta, "partido", deps, muns)))

    r_clas = st.sidebar.slider(
        "Índice del candidato", -1.0, 1.0, RANGO_COMPLETO, step=0.05,
        help="-1 = izquierda, 1 = derecha.",
    )
    redondear = lambda r: (round(r[0], 2), round(r[1], 2))
    # El filtro de inclinación del partido se retiró: se devuelve el rango completo (sin filtro)
    return deps, muns, partidos, RANGO_COMPLETO, redondear(r_clas)


@st.cache_resource(max_entries=2, show_spinner="Aplicando filtros…")
def filtrar(ruta: str, filtros: tuple) -> pd.DataFrame:
    """Filas que cumplen los filtros. Sin filtros devuelve el mismo DataFrame
    (sin copiarlo)."""
    df = cargar_datos(ruta)
    deps, muns, partidos, r_inc, r_clas = filtros
    mask = np.ones(len(df), dtype=bool)
    activo = False
    for col, sel in [("dep", deps), ("mun", muns), ("partido", partidos)]:
        if sel:
            mask &= df[col].isin(sel).to_numpy()
            activo = True
    # Si el rango es el completo no se filtra (así se conservan los índices vacíos)
    for col, rango in [("inclinacion_politica", r_inc), ("clasificacion", r_clas)]:
        if tuple(rango) != RANGO_COMPLETO:
            mask &= df[col].between(*rango).to_numpy()
            activo = True
    return df.loc[mask] if activo else df


# ---------------------------------------------------------------------------
# Resúmenes (se calculan una vez por combinación de filtros)
# ---------------------------------------------------------------------------
@st.cache_resource(max_entries=8, show_spinner="Calculando resúmenes…")
def resumenes(ruta: str, filtros: tuple) -> dict | None:
    d = filtrar(ruta, filtros)
    if d.empty:
        return None
    votos = d["votos_totales"].to_numpy().astype(np.float64)

    # --- Candidatos: una fila por (candidato, partido) ---------------------
    cand = (
        d.groupby(["candidato", "partido"], observed=True)
        .agg(
            votos=("votos_totales", "sum"),
            puestos=("puesto_id", "size"),
            clasificacion=("clasificacion", "first"),
            inclinacion=("inclinacion_politica", "first"),
        )
        .reset_index()
    )
    cand["incl_cat"] = categorizar(cand["inclinacion"])
    cand["clas_cat"] = categorizar(cand["clasificacion"])

    # --- Territorio: votos por municipio y grupo ideológico ----------------
    cubo = (
        d.groupby(["dep", "mun", "candidato", "partido"], observed=True)["votos_totales"]
        .sum().reset_index(name="votos")
        .merge(cand[["candidato", "partido", "inclinacion", "clasificacion",
                     "incl_cat", "clas_cat"]],
               on=["candidato", "partido"], how="left")
    )
    for col, pref in [("inclinacion", "inc"), ("clasificacion", "clas")]:
        tiene = cubo[col].notna()
        cubo[f"v_{pref}"] = np.where(tiene, cubo["votos"] * cubo[col], 0.0)  # votos × índice
        cubo[f"n_{pref}"] = np.where(tiene, cubo["votos"], 0)                # votos con índice
    cubo = (
        cubo.groupby(["dep", "mun", "incl_cat", "clas_cat"], observed=True, dropna=False)
        [["votos", "v_inc", "n_inc", "v_clas", "n_clas"]].sum().reset_index()
    )
    n_municipios = cubo.groupby(["dep", "mun"], observed=True).ngroups

    # --- Puestos: una fila por puesto (n_*_psto sin inflar) -----------------
    pu = (
        d[["puesto_id", "dep", "mun", "puesto_nom"] + COLS_PUESTO]
        .drop_duplicates("puesto_id").set_index("puesto_id")
    )
    pu = texto(pu).astype({c: "float64" for c in COLS_PUESTO})
    gid = d["puesto_id"].to_numpy()
    n_ids = int(gid.max()) + 1
    pu["votos"] = np.bincount(gid, weights=votos, minlength=n_ids)[pu.index]
    for col, nombre in [("inclinacion_politica", "ind_inc"), ("clasificacion", "ind_clas")]:
        x = d[col].to_numpy().astype(np.float64)
        tiene = ~np.isnan(x)
        suma = np.bincount(gid, weights=np.where(tiene, votos * x, 0.0), minlength=n_ids)
        peso = np.bincount(gid, weights=np.where(tiene, votos, 0.0), minlength=n_ids)
        with np.errstate(invalid="ignore", divide="ignore"):
            pu[nombre] = (suma / peso)[pu.index]
    habil = pu["n_total_psto"].replace(0, np.nan)
    pu["participacion"] = pu["votos"] / habil * 100
    pu["pct_mujeres"] = pu["n_mujeres_psto"] / habil * 100

    # --- Distribución por fila (candidato en un puesto) ---------------------
    hist_lin = np.histogram(votos, bins=60)
    hist_log = np.histogram(np.log10(votos + 1), bins=60)

    cand = texto(cand)
    cand[["candidato", "partido"]] = cand[["candidato", "partido"]].astype(str)

    return {
        "cand": cand, "cubo": cubo, "puestos": pu,
        "hist_lin": hist_lin, "hist_log": hist_log, "desc_filas": describir(votos),
        "n_filas": len(d), "n_municipios": n_municipios,
    }


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
def mostrar_kpis(R: dict) -> None:
    cand, pu = R["cand"], R["puestos"]
    votos = cand["votos"].sum()
    habilitados = pu["n_total_psto"].sum()

    fila1, fila2 = st.columns(4), st.columns(4)
    fila1[0].metric("Votos totales", fmt_num(votos))
    fila1[1].metric("Candidatos", fmt_num(cand["candidato"].nunique()))
    fila1[2].metric("Partidos", fmt_num(cand["partido"].nunique()))
    fila1[3].metric("Puestos", fmt_num(len(pu)))
    fila2[0].metric("Municipios", fmt_num(R["n_municipios"]))
    fila2[1].metric(
        "Participación", fmt_pct(votos / habilitados if habilitados else np.nan),
        help="Votos filtrados / personas habilitadas en los puestos donde aparecen. "
             "Si filtras por partido o índice, se lee como % del electorado que "
             "votó por esa opción.",
    )
    fila2[2].metric(
        "Índice del voto (partido)", fmt_idx(media_ponderada(cand["inclinacion"], cand["votos"])),
        help="Promedio de inclinacion_politica ponderado por votos (-1 izquierda, 1 derecha).",
    )
    fila2[3].metric(
        "Índice del voto (candidato)", fmt_idx(media_ponderada(cand["clasificacion"], cand["votos"])),
        help="Promedio de clasificacion ponderado por votos (-1 izquierda, 1 derecha).",
    )


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
def tab_distribucion(R: dict) -> None:
    st.subheader("Distribución de votos")
    nivel = st.radio(
        "Unidad de análisis",
        ["Candidato en un puesto (fila)", "Candidato (total)", "Puesto (total)"],
        horizontal=True,
    )
    log = st.checkbox("Escala logarítmica: log10(votos + 1)", value=True)
    etiqueta = "log10(votos + 1)" if log else "Votos"

    if nivel == "Candidato en un puesto (fila)":
        conteos, bordes = R["hist_log"] if log else R["hist_lin"]
        resumen = R["desc_filas"]
    else:
        datos = (R["cand"] if nivel == "Candidato (total)" else R["puestos"])["votos"].to_numpy()
        conteos, bordes = np.histogram(np.log10(datos + 1) if log else datos, bins=60)
        resumen = describir(datos)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(histograma(conteos, bordes, etiqueta))
    with col2:
        st.dataframe(resumen)

    st.subheader("Concentración de la votación en los puestos según su tamaño")
    votos_puesto = R["puestos"]["votos"].to_numpy(dtype=np.float64)
    if (votos_puesto > 0).sum() < MIN_PUESTOS_DECILES:
        st.info(f"Se necesitan al menos {MIN_PUESTOS_DECILES} puestos con votos para formar deciles.")
    else:
        fig, por_decil = fig_concentracion_deciles(votos_puesto)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.metric("Votos en el 10 % de puestos más grandes", f"{por_decil['pct'].iloc[-1]:.1f} %".replace(".", ","))
            st.metric("Votos en el 50 % de puestos más pequeños",
                      f"{por_decil['pct'].iloc[:5].sum():.1f} %".replace(".", ","))
        st.caption("Puestos agrupados en deciles según sus votos (D1 = 10 % más pequeño, D10 = 10 % más grande). "
                   "Si D10 supera ampliamente la línea del 10 %, unos pocos puestos concentran buena parte del voto.")

    st.subheader("Votos por candidato según grupo ideológico")
    eje = st.radio("Agrupar por", list(INDICES), horizontal=True, key="eje_box")
    _, cat, _ = INDICES[eje]
    datos = R["cand"].dropna(subset=[cat])
    fig = px.box(
        datos, x=cat, y="votos", color=cat, log_y=log, points="outliers",
        category_orders={cat: ETIQUETAS_IDEOLOGIA}, color_discrete_map=COLORES_IDEOLOGIA,
        hover_data=["candidato", "partido"],
        labels={cat: eje, "votos": "Votos del candidato"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


def tab_rankings(R: dict) -> None:
    top_n = st.slider("Número de elementos a mostrar", 5, 50, 15)
    cand = R["cand"]
    total = cand["votos"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {top_n} candidatos")
        top = cand.nlargest(top_n, "votos").copy()
        top["etiqueta"] = top["candidato"] + " (" + top["partido"] + ")"
        fig = px.bar(
            top.sort_values("votos"), x="votos", y="etiqueta", orientation="h",
            color="clasificacion", color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
            hover_data=["partido", "inclinacion"],
            labels={"votos": "Votos", "etiqueta": "", "clasificacion": "Clasificación"},
        )
        fig.update_layout(height=max(400, 28 * top_n))
        st.plotly_chart(fig)

    with col2:
        st.subheader(f"Top {top_n} partidos")
        part = (
            cand.groupby("partido")
            .agg(votos=("votos", "sum"), inclinacion=("inclinacion", "first"))
            .nlargest(top_n, "votos").reset_index().sort_values("votos")
        )
        part["pct"] = part["votos"] / total * 100 if total else 0
        fig = px.bar(
            part, x="votos", y="partido", orientation="h",
            text=part["pct"].map(lambda v: f"{v:.1f}%"),
            color="inclinacion", color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
            labels={"votos": "Votos", "partido": "", "inclinacion": "Inclinación"},
        )
        fig.update_layout(height=max(400, 28 * top_n))
        st.plotly_chart(fig)

    st.caption("Color: índice ideológico (rojo = izquierda, gris = centro, azul = derecha).")


def tab_territorio(R: dict) -> None:
    cubo, pu = R["cubo"], R["puestos"]
    cols = ["votos", "v_inc", "n_inc"]

    dep = texto(cubo.groupby("dep", observed=True)[cols].sum().reset_index()).set_index("dep")
    dep["habilitados"] = pu.groupby("dep")["n_total_psto"].sum()
    dep["participacion"] = dep["votos"] / dep["habilitados"].replace(0, np.nan) * 100
    dep["indice"] = dep["v_inc"] / dep["n_inc"].replace(0, np.nan)
    dep = dep.reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Votos por departamento")
        fig = px.bar(
            dep.sort_values("votos"), x="votos", y="dep", orientation="h",
            color="indice", color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
            labels={"votos": "Votos", "dep": "", "indice": "Índice partido"},
        )
        fig.update_layout(height=max(450, 22 * len(dep)))
        st.plotly_chart(fig)
    with col2:
        st.subheader("Participación por departamento (%)")
        fig = px.bar(
            dep.sort_values("participacion"), x="participacion", y="dep", orientation="h",
            labels={"participacion": "Votos / habilitados (%)", "dep": ""},
        )
        fig.update_layout(height=max(450, 22 * len(dep)))
        st.plotly_chart(fig)

    st.subheader("Índice ideológico de los departamentos con más electores")
    top_dep = st.slider("Número de departamentos", 3, 33, 10, key="top_dep_indice")
    base = dep.set_index("dep")
    base["censo"] = base["habilitados"]
    base["n_puestos"] = pu.groupby("dep").size()
    top = base.dropna(subset=["indice", "censo"]).nlargest(top_dep, "censo").sort_values("indice")
    if top.empty:
        st.info("No hay departamentos con índice para los filtros seleccionados.")
    else:
        indice_total = cubo["v_inc"].sum() / cubo["n_inc"].sum() if cubo["n_inc"].sum() else np.nan
        st.plotly_chart(fig_top_departamentos(top, indice_total), width="stretch")
        st.caption("Índice del partido ponderado por los votos del departamento; se eligen los departamentos "
                   "con más personas habilitadas. Línea punteada: índice del total filtrado.")

    st.subheader("Municipios con más votos")
    top_n = st.slider("Número de municipios", 10, 60, 25, key="top_mun")
    mun = texto(cubo.groupby(["dep", "mun"], observed=True)[cols].sum().reset_index())
    mun = mun.nlargest(top_n, "votos")
    mun["indice"] = mun["v_inc"] / mun["n_inc"].replace(0, np.nan)
    mun["etiqueta"] = mun["mun"] + " (" + mun["dep"] + ")"
    fig = px.bar(
        mun.sort_values("votos"), x="votos", y="etiqueta", orientation="h",
        color="indice", color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
        labels={"votos": "Votos", "etiqueta": "", "indice": "Índice partido"},
    )
    fig.update_layout(height=max(450, 24 * top_n))
    st.plotly_chart(fig)

    st.caption(
        "Color: índice ideológico del partido ponderado por votos. "
        "Para un mapa se necesitan los códigos DANE de departamento y municipio y un "
        "GeoJSON; con solo los nombres el cruce no es confiable."
    )


def tab_ideologia(R: dict, parcial: bool = False) -> None:
    cand, cubo, pu = R["cand"], R["cubo"], R["puestos"]
    eje = st.radio("Índice a analizar", list(INDICES), horizontal=True, key="eje_ideo")
    col_idx, cat, pref = INDICES[eje]
    col_pu = "ind_inc" if pref == "inc" else "ind_clas"

    st.subheader("Distribución del índice ideológico por puesto")
    if parcial:
        st.caption("⚠️ Hay filtros de partido o de índice activos: el índice de cada puesto solo "
                   "considera los votos que cumplen esos filtros.")
    if pu[col_pu].notna().sum() == 0:
        st.info("No hay puestos con índice para los filtros seleccionados.")
    else:
        st.plotly_chart(fig_distribucion_indice(pu[col_pu]), width="stretch")
        st.caption(f"Índice de cada puesto = {eje.lower()} ponderada por los votos del puesto. "
                   "Las líneas punteadas verticales marcan los cortes entre categorías.")

    st.subheader("Peso de cada categoría ideológica en puestos, censo y votos")
    base = pu.assign(categoria=categorizar(pu[col_pu])).dropna(subset=["categoria"])
    if base.empty:
        st.info("No hay puestos con índice para los filtros seleccionados.")
    else:
        consistentes = puestos_consistentes(base)
        peso = pd.DataFrame({
            "% de puestos": base.groupby("categoria").size(),
            "% del censo": base[base["n_total_psto"] > 0].groupby("categoria")["n_total_psto"].sum(),
            "% de votos": consistentes.groupby("categoria")["votos"].sum(),
        }).reindex(ETIQUETAS_IDEOLOGIA).fillna(0)
        peso_pct = peso.div(peso.sum(axis=0).replace(0, np.nan), axis=1).fillna(0) * 100
        st.plotly_chart(fig_peso_electoral(peso_pct), width="stretch")
        st.caption("Si la barra de censo o de votos de una categoría supera a la de puestos, esa categoría "
                   "reúne puestos más grandes y pesa más de lo que sugiere el conteo de puestos.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Votos por grupo ideológico")
        g = cand.dropna(subset=[cat]).groupby(cat)["votos"].sum().reset_index()
        g["pct"] = g["votos"] / g["votos"].sum() * 100
        fig = px.bar(
            g, x=cat, y="votos", color=cat, text=g["pct"].map(lambda v: f"{v:.1f}%"),
            category_orders={cat: ETIQUETAS_IDEOLOGIA}, color_discrete_map=COLORES_IDEOLOGIA,
            labels={cat: "", "votos": "Votos"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

    with col2:
        st.subheader("¿Dónde se concentra el voto en el espectro?")
        base = cand.dropna(subset=[col_idx])
        bordes = np.linspace(-1, 1, 21)
        pv, _ = np.histogram(base[col_idx], bins=bordes, weights=base["votos"])
        pc, _ = np.histogram(base[col_idx], bins=bordes)
        centros = np.round((bordes[:-1] + bordes[1:]) / 2, 2)
        largo = pd.DataFrame({
            "índice": np.tile(centros, 2),
            "pct": np.concatenate([pv / max(pv.sum(), 1), pc / max(pc.sum(), 1)]) * 100,
            "serie": ["% de votos"] * len(centros) + ["% de candidatos"] * len(centros),
        })
        fig = px.bar(largo, x="índice", y="pct", color="serie", barmode="group",
                     labels={"pct": "%", "serie": ""})
        st.plotly_chart(fig)

    st.subheader("Candidato vs. partido: ¿el discurso se aparta de la línea del partido?")
    sc = cand.dropna(subset=["inclinacion", "clasificacion"])
    fig = px.scatter(
        sc, x="inclinacion", y="clasificacion", size="votos", size_max=35, opacity=0.6,
        color="partido" if sc["partido"].nunique() <= 12 else None,
        hover_data=["candidato", "partido", "votos"],
        range_x=[-1.05, 1.05], range_y=[-1.05, 1.05],
        labels={"inclinacion": "Inclinación del partido", "clasificacion": "Clasificación del candidato"},
    )
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig)
    st.caption("Sobre la diagonal: el candidato es más de derecha que su partido; debajo, más de izquierda. "
               "El tamaño del punto es proporcional a los votos.")

    col3, col4 = st.columns(2)
    dep = texto(cubo.groupby("dep", observed=True)[["votos", f"v_{pref}", f"n_{pref}"]].sum().reset_index())
    dep["indice"] = dep[f"v_{pref}"] / dep[f"n_{pref}"].replace(0, np.nan)
    with col3:
        st.subheader("Índice ideológico del voto por departamento")
        fig = px.bar(
            dep.sort_values("indice"), x="indice", y="dep", orientation="h",
            color="indice", color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
            range_x=[-1, 1], labels={"indice": "Índice ponderado por votos", "dep": ""},
        )
        fig.update_layout(height=max(450, 22 * len(dep)), coloraxis_showscale=False)
        st.plotly_chart(fig)

    with col4:
        st.subheader("Composición del voto por departamento (%)")
        comp = cubo.groupby(["dep", cat], observed=True)["votos"].sum().reset_index()
        comp["pct"] = comp["votos"] / comp.groupby("dep", observed=True)["votos"].transform("sum") * 100
        comp = texto(comp)
        orden_dep = dep.sort_values("indice")["dep"].tolist()
        fig = px.bar(
            comp, x="pct", y="dep", color=cat, orientation="h",
            category_orders={cat: ETIQUETAS_IDEOLOGIA, "dep": orden_dep[::-1]},
            color_discrete_map=COLORES_IDEOLOGIA,
            labels={"pct": "% de votos", "dep": "", cat: ""},
        )
        fig.update_layout(barmode="stack", height=max(450, 22 * len(dep)))
        st.plotly_chart(fig)

    st.subheader("Cruce: grupo del partido × grupo del candidato (votos)")
    cruce = (
        pd.pivot_table(cand, index="incl_cat", columns="clas_cat", values="votos",
                       aggfunc="sum", fill_value=0)
        .reindex(index=ETIQUETAS_IDEOLOGIA, columns=ETIQUETAS_IDEOLOGIA, fill_value=0)
        .rename_axis(index="Partido ↓ / Candidato →", columns=None)
    )
    st.dataframe(cruce.style.format(fmt_num).background_gradient(cmap="Blues", axis=None))


def tab_puestos(R: dict, parcial: bool = False) -> None:
    p = R["puestos"]

    n_raros = int((p["participacion"] > 100).sum())
    if n_raros:
        st.warning(
            f"{fmt_num(n_raros)} puestos tienen más votos que personas habilitadas "
            "(participación > 100 %). Puede indicar problemas en el cruce de datos."
        )

    muestra = p.sample(TAM_MUESTRA_SCATTER, random_state=42) if len(p) > TAM_MUESTRA_SCATTER else p
    if len(p) > TAM_MUESTRA_SCATTER:
        st.caption(f"Mostrando una muestra aleatoria de {fmt_num(TAM_MUESTRA_SCATTER)} puestos.")
    muestra = muestra.reset_index()
    hover = ["dep", "mun", "puesto_nom"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Votos vs. habilitados del puesto")
        log = st.checkbox("Ejes en escala logarítmica", value=True, key="log_puestos")
        fig = px.scatter(
            muestra, x="n_total_psto", y="votos", hover_data=hover, opacity=0.4,
            log_x=log, log_y=log,
            labels={"n_total_psto": "Personas habilitadas", "votos": "Votos"},
        )
        st.plotly_chart(fig)
    with col2:
        st.subheader("Distribución de la participación")
        x = p["participacion"].dropna().to_numpy()
        conteos, bordes = np.histogram(x, bins=60)
        st.plotly_chart(histograma(conteos, bordes, "Participación (%)", "Puestos"))

    st.subheader("Índice ideológico del puesto y participación")
    if parcial:
        st.caption("⚠️ Hay filtros de partido o de índice activos: la participación se lee como % del "
                   "electorado que votó por las opciones filtradas.")
    eje_p = st.radio("Índice", list(INDICES), horizontal=True, key="eje_part")
    col_p = "ind_inc" if INDICES[eje_p][2] == "inc" else "ind_clas"
    val = puestos_consistentes(p).dropna(subset=[col_p, "participacion"]).reset_index()
    if len(val) < 3:
        st.info("No hay suficientes puestos con índice y participación válidos para los filtros seleccionados.")
    else:
        r_s = val[[col_p, "participacion"]].corr(method="spearman").iloc[0, 1]
        r_p = val[[col_p, "participacion"]].corr(method="pearson").iloc[0, 1]
        muestra_v = val.sample(TAM_MUESTRA_SCATTER, random_state=42) if len(val) > TAM_MUESTRA_SCATTER else val
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Índice vs. participación**")
            st.plotly_chart(fig_indice_participacion(muestra_v, col_p, r_s, r_p), width="stretch")
        with col6:
            st.markdown("**Participación por categoría ideológica**")
            val["categoria"] = categorizar(val[col_p])
            st.plotly_chart(fig_participacion_categoria(val), width="stretch")
        excluidos = len(p) - len(puestos_consistentes(p))
        st.caption(f"{fmt_num(len(val))} puestos válidos (se excluyen {fmt_num(excluidos)} con censo 0 o "
                   "participación > 100 %). La participación es cota inferior: no incluye votos en blanco ni nulos. "
                   "La asociación es descriptiva, no causal.")

    st.subheader("Correlaciones entre variables del puesto (Spearman)")
    cols = ["votos", "participacion", "n_total_psto", "n_mesas_psto", "pct_mujeres", "ind_inc", "ind_clas"]
    nombres = {"ind_inc": "índice partido", "ind_clas": "índice candidato"}
    st.dataframe(p[cols].corr(method="spearman").round(2).rename(index=nombres, columns=nombres))


def tokenizar(texto_: str) -> list[str]:
    """Normaliza texto y devuelve palabras de 3+ letras sin stopwords."""
    if not isinstance(texto_, str):
        return []

    t = texto_.lower().strip()

    # Eliminar URLs
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)

    # Eliminar menciones y hashtags conservando el texto
    t = re.sub(r"[@#]", "", t)

    # Eliminar tildes
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()

    # Extraer palabras, filtrar palabras cortas y stopwords
    return [
        w for w in re.findall(r"[a-z]+", t)
        if len(w) > 2 and w not in STOPWORDS_ES
    ]

@st.cache_data(show_spinner="Contando palabras…")
def palabras_frecuentes(ruta: str, filtros: tuple, cat: str, grupo: str, n: int = 25) -> pd.DataFrame:
    d = discursos_filtrados(ruta, filtros)
    if grupo != "Todos":
        d = d[d[cat] == grupo]
    contador = Counter()
    for t in d["discurso"]:
        contador.update(tokenizar(t))
    return pd.DataFrame(contador.most_common(n), columns=["palabra", "frecuencia"])


def discursos_filtrados(ruta: str, filtros: tuple) -> pd.DataFrame:
    R = resumenes(ruta, filtros)
    return cargar_discursos(ruta).merge(R["cand"], on=["candidato", "partido"], how="inner")


def tab_discurso(ruta: str, filtros: tuple) -> None:
    d = discursos_filtrados(ruta, filtros)
    if d.empty:
        st.info("No hay discursos para los filtros seleccionados (o el dataset no tiene la columna 'discurso').")
        return
    d["n_palabras"] = d["discurso"].str.split().str.len()

    eje = st.radio("Agrupar por", list(INDICES), horizontal=True, key="eje_disc")
    col_idx, cat, _ = INDICES[eje]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Longitud del discurso por grupo ideológico")
        fig = px.box(
            d.dropna(subset=[cat]), x=cat, y="n_palabras", color=cat,
            category_orders={cat: ETIQUETAS_IDEOLOGIA}, color_discrete_map=COLORES_IDEOLOGIA,
            hover_data=["candidato", "partido"], labels={cat: "", "n_palabras": "Palabras"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)

        st.subheader("Longitud vs. índice ideológico")
        fig = px.scatter(
            d.dropna(subset=[col_idx]), x=col_idx, y="n_palabras", size="votos", size_max=30,
            opacity=0.6, hover_data=["candidato", "partido"],
            color=col_idx, color_continuous_scale=ESCALA_IDEOLOGIA, range_color=[-1, 1],
            labels={col_idx: eje, "n_palabras": "Palabras"},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig)

    with col2:
        st.subheader("Palabras más frecuentes")
        presentes = set(d[cat].dropna())
        grupos = ["Todos"] + [g for g in ETIQUETAS_IDEOLOGIA if g in presentes]
        grupo = st.selectbox("Grupo ideológico", grupos)
        top = palabras_frecuentes(ruta, filtros, cat, grupo)
        fig = px.bar(
            top.sort_values("frecuencia"), x="frecuencia", y="palabra", orientation="h",
            labels={"frecuencia": "Frecuencia", "palabra": ""},
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig)

    st.caption(f"{fmt_num(len(d))} discursos únicos (uno por candidato y partido).")

# ---------------------------------------------------------------------------
# Nuevas secciones
# ---------------------------------------------------------------------------
import streamlit.components.v1 as components  # noqa: E402

DIR_IMAGENES = Path(__file__).parent / "Imagenes"


def _incrustar_html(nombre: str, alto: int = 500) -> None:
    """Reutiliza un gráfico HTML ya generado si existe."""
    ruta = DIR_IMAGENES / nombre
    if ruta.exists():
        components.html(ruta.read_text(encoding="utf-8"), height=alto, scrolling=True)
    else:
        st.info(f"Gráfico no encontrado: {nombre}")


@st.cache_data(show_spinner="Resumiendo el esquema del dataset…")
def resumen_dataset(ruta: str) -> dict:
    df = cargar_datos(ruta)
    cols = [c for c in df.columns if c != "puesto_id"]
    info = pd.DataFrame({
        "columna": cols,
        "tipo": [str(df[c].dtype) for c in cols],
        "faltantes": [int(df[c].isna().sum()) for c in cols],
    })
    info["% faltantes"] = (info["faltantes"] / len(df) * 100).round(2)
    num = df[[c for c in COLS_INDICE + COLS_PUESTO + ["votos_totales"]]].describe().T
    return {"filas": len(df), "columnas": len(cols), "info": info, "describe": num.round(2)}

@st.cache_resource(show_spinner="Preparando puestos para el mapa estratégico…")
def puestos_estrategia(ruta: str) -> pd.DataFrame:
    """Una fila por puesto con TODOS sus votos (sin filtros de partido ni de índice):
    el índice del puesto debe reflejar a todo su electorado. Solo puestos consistentes
    (censo > 0, participación <= 100 %) y con índice."""
    df = cargar_datos(ruta)
    gid = df["puesto_id"].to_numpy()
    n_ids = int(gid.max()) + 1
    votos = df["votos_totales"].to_numpy().astype(np.float64)
    x = df["inclinacion_politica"].to_numpy().astype(np.float64)
    tiene = ~np.isnan(x)
    total = np.bincount(gid, weights=votos, minlength=n_ids)
    suma = np.bincount(gid, weights=np.where(tiene, votos * x, 0.0), minlength=n_ids)
    peso = np.bincount(gid, weights=np.where(tiene, votos, 0.0), minlength=n_ids)

    pu = texto(df[["puesto_id", "dep", "mun", "puesto_nom", "n_total_psto"]]
               .drop_duplicates("puesto_id").set_index("puesto_id"))
    pu["votos"] = total[pu.index]
    with np.errstate(invalid="ignore", divide="ignore"):
        pu["indice"] = (suma / peso)[pu.index]
    pu["censo"] = pu["n_total_psto"].astype(np.float64)
    pu["participacion"] = pu["votos"] / pu["censo"].where(pu["censo"] > 0)
    pu = pu[(pu["censo"] > 0) & (pu["participacion"] <= 1) & pu["indice"].notna()]
    return pu.drop(columns="n_total_psto").reset_index(drop=True)


def tab_estrategia(ruta: str, filtros: tuple) -> None:
    st.subheader("Mapa estratégico de un municipio: ¿movilizar o persuadir?")
    st.caption("Ubica cada puesto del municipio según su distancia ideológica a un candidato hipotético "
               "y su abstención. Los cortes son las medianas del municipio.")
    base = puestos_estrategia(ruta)
    if base.empty:
        st.warning("No hay puestos con índice y participación válidos.")
        return

    # Valores iniciales: la selección de la barra lateral si es única; si no, el municipio con más puestos.
    tam = base.groupby(["dep", "mun"]).size()
    dep_ini, mun_ini = tam.idxmax()
    deps_lat, muns_lat = filtros[0], filtros[1]
    if len(deps_lat) == 1:
        dep_ini = str(deps_lat[0])
        mun_ini = tam.loc[dep_ini].idxmax() if dep_ini in tam.index.get_level_values(0) else mun_ini
    if len(muns_lat) == 1:
        coinc = tam[tam.index.get_level_values(1) == str(muns_lat[0])]
        if len(deps_lat) == 1:
            coinc = coinc[coinc.index.get_level_values(0) == dep_ini]
        if len(coinc):
            dep_ini, mun_ini = coinc.idxmax()

    lista_dep = sorted(base["dep"].unique())
    c1, c2, c3 = st.columns(3)
    dep = c1.selectbox("Departamento", lista_dep,
                       index=lista_dep.index(dep_ini) if dep_ini in lista_dep else 0, key="est_dep")
    lista_mun = sorted(base.loc[base["dep"] == dep, "mun"].unique())
    mun = c2.selectbox("Municipio", lista_mun,
                       index=lista_mun.index(mun_ini) if mun_ini in lista_mun else 0, key="est_mun")
    ideologia = c3.slider("Índice ideológico del candidato", -1.0, 1.0, 0.0, step=0.05, key="est_cand",
                          help="-1 = izquierda, 0 = centro, 1 = derecha.")

    m = base[(base["dep"] == dep) & (base["mun"] == mun)]
    if len(m) < 2:
        st.warning(f"{mun} tiene {len(m)} puesto válido: no se pueden formar cuadrantes.")
        return
    if len(m) < 10:
        st.info(f"{mun} tiene solo {len(m)} puestos válidos: los cuadrantes por mediana son poco informativos.")
    m, corte_d, corte_ab = clasificar_estrategia(m, ideologia)

    st.plotly_chart(fig_mapa_estrategico(m, corte_d, corte_ab), width="stretch")
    coma = lambda v, d: f"{v:.{d}f}".replace(".", ",")
    st.caption(f"Candidato en {fmt_idx(ideologia)} · cortes: distancia {coma(corte_d, 2)}, "
               f"abstención {coma(corte_ab, 1)} % · tamaño = censo del puesto · color = índice del puesto. "
               "La abstención es cota superior "
                 "(el dataset no incluye votos en blanco ni nulos). El índice usa todos los votos del puesto, "
                 "sin los filtros de partido o índice de la barra lateral.")

    resumen = (m.groupby("estrategia")
               .agg(puestos=("puesto_nom", "size"), censo=("censo", "sum"),
                    abstencionistas=("abstencionistas", "sum"), votos=("votos", "sum"))
               .reindex(ORDEN_ESTRATEGIA).fillna(0))
    k = st.columns(4)
    k[0].metric("Puestos para movilizar", fmt_num(resumen.loc["Movilizar", "puestos"]))
    k[1].metric("Abstencionistas cercanos", fmt_num(resumen.loc["Movilizar", "abstencionistas"]))
    k[2].metric("Puestos para persuadir", fmt_num(resumen.loc["Persuadir", "puestos"]))
    k[3].metric("Votantes distantes", fmt_num(resumen.loc["Persuadir", "votos"]))

    def principales(estrategia: str, orden: str) -> list:
        sub = m[m["estrategia"] == estrategia].nlargest(3, orden)
        return [{"puesto": r["puesto_nom"], "censo": int(r["censo"]), "votos": int(r["votos"]),
                 "indice_puesto": round(float(r["indice"]), 2), "distancia": round(float(r["distancia"]), 3),
                 "abstencion_pct": round(float(r["abst_pct"]), 1), "abstencionistas": int(r["abstencionistas"])}
                for _, r in sub.iterrows()]

    boton_explicar("Estrategia", {
        "municipio": mun, "departamento": dep, "indice_candidato": ideologia,
        "puestos_validos": len(m), "indice_medio_municipio": round(float(m["indice"].mean()), 3),
        "corte_distancia_mediana": round(float(corte_d), 3), "corte_abstencion_mediana_pct": round(float(corte_ab), 1),
        "cuadrantes": {e: {"puestos": int(r["puestos"]), "censo": int(r["censo"]),
                           "abstencionistas": int(r["abstencionistas"]), "votos": int(r["votos"]),
                           "pct_censo": round(float(r["censo"] / max(resumen["censo"].sum(), 1) * 100), 1)}
                       for e, r in resumen.iterrows()},
        "top_movilizar_por_abstencionistas": principales("Movilizar", "abstencionistas"),
        "top_persuadir_por_votos": principales("Persuadir", "votos"),
    }, key="estrategia")

    with st.expander("¿Cómo leer el mapa?"):
        st.markdown(
            "| Cuadrante | Distancia | Abstención | Enfoque |\n|---|---|---|---|\n"
            "| **Movilizar** | baja | alta | Convencer de **salir a votar** a electores afines |\n"
            "| **Persuadir** | alta | baja | Convencer de **cambiar de postura** a quienes ya votan |\n"
            "| Consolidar | baja | baja | Base que ya vota: mantener |\n"
            "| Baja prioridad | alta | alta | Lejanos y poco participativos |\n\n"
            "Distancia = |índice del puesto − índice del candidato| / 2 (0 = idéntica, 1 = máxima)."
        )

    resumen_tab = resumen.assign(**{"% censo": resumen["censo"] / resumen["censo"].sum()})
    st.dataframe(
        resumen_tab.style.format({"puestos": fmt_num, "censo": fmt_num, "abstencionistas": fmt_num,
                                  "votos": fmt_num, "% censo": fmt_pct}),
        width="stretch")

    cols = ["puesto_nom", "censo", "votos", "indice", "distancia", "abst_pct", "abstencionistas"]
    nombres = {"puesto_nom": "Puesto", "censo": "Censo", "votos": "Votos", "indice": "Índice",
               "distancia": "Distancia", "abst_pct": "Abstención (%)", "abstencionistas": "Abstencionistas"}
    fmt = {"Censo": fmt_num, "Votos": fmt_num, "Índice": fmt_idx, "Distancia": lambda v: coma(v, 2),
           "Abstención (%)": lambda v: coma(v, 1), "Abstencionistas": fmt_num}
    c4, c5 = st.columns(2)
    with c4:
        st.markdown("**Movilizar** — 10 puestos con más abstencionistas cercanos")
        t = m[m["estrategia"] == "Movilizar"].nlargest(10, "abstencionistas")[cols].rename(columns=nombres)
        st.dataframe(t.style.format(fmt), hide_index=True, width="stretch")
    with c5:
        st.markdown("**Persuadir** — 10 puestos con más votantes distantes")
        t = m[m["estrategia"] == "Persuadir"].nlargest(10, "votos")[cols].rename(columns=nombres)
        st.dataframe(t.style.format(fmt), hide_index=True, width="stretch")

    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(mun)).strip("_").lower()
    csv = (m.sort_values(["estrategia", "censo"], ascending=[True, False])[["dep", "mun", "estrategia"] + cols]
           .to_csv(index=False).encode("utf-8-sig"))
    st.download_button("Descargar clasificación de puestos (CSV)", csv,
                       file_name=f"estrategia_{slug}.csv", mime="text/csv")


@st.cache_data(show_spinner=False, max_entries=200)
def explicacion_ia(pestana: str, contexto_json: str, modelo: str) -> str:
    """Una llamada al LLM por combinación de pestaña + datos + modelo (se reutiliza si no cambian)."""
    import explicador_ia as ex
    return ex.explicar(pestana, contexto_json, modelo)


def boton_explicar(pestana: str, contexto: dict, key: str) -> None:
    """Botón '✨ Explícame esto': envía al agente las cifras que hay en pantalla y muestra
    su interpretación. Si los datos cambian, la explicación anterior se oculta."""
    try:
        import explicador_ia as ex
    except ImportError:
        return
    ok, motivo = ex.disponible()
    if not ok:
        st.caption(f"✨ La explicación con IA no está disponible: {motivo}.")
        return
    datos = ex.a_json(contexto)
    clave = f"explicacion_{key}"
    if st.button("✨ Explícame esto", key=f"boton_{key}",
                 help="La IA interpreta en palabras simples las cifras que estás viendo."):
        with st.spinner("Interpretando los resultados…"):
            try:
                st.session_state[clave] = (datos, explicacion_ia(pestana, datos, ex.MODELO_LLM))
            except Exception as e:  # noqa: BLE001 - clave inválida, modelo inexistente, sin conexión…
                st.session_state[clave] = (datos, f"⚠️ No pude generar la explicación: `{type(e).__name__}: "
                                                  f"{str(e)[:300]}`")
    guardada = st.session_state.get(clave)
    if guardada and guardada[0] == datos:
        with st.container(border=True):
            st.markdown(guardada[1])
            st.caption(f"Generado por IA ({ex.MODELO_LLM}) a partir de las cifras de esta pantalla. "
                       "Verifica las cifras antes de usarlas.")
    elif guardada:
        st.caption("Cambiaste la selección: vuelve a pedir la explicación.")


def contexto_chat_dashboard(R: dict, filtros: tuple) -> dict:
    """Reduce el estado filtrado del dashboard a evidencia compacta para el agente."""
    cand = R["cand"].copy()
    puestos = R["puestos"].copy()
    cubo = R["cubo"].copy()
    votos = float(cand["votos"].sum())
    censo = float(puestos["n_total_psto"].sum())
    top_cand = cand.nlargest(8, "votos")[["candidato", "partido", "votos", "clasificacion", "inclinacion"]].to_dict("records")
    top_mun = (cubo.groupby(["dep", "mun"], observed=True)["votos"].sum()
               .sort_values(ascending=False).head(8).reset_index().to_dict("records"))
    dep, mun, partidos, r_inc, r_clas = filtros
    return {
        "descripcion": "Dashboard EDA electoral Colombia; datos observados de votos por candidatos.",
        "filtros_activos": {"departamentos": list(dep), "municipios": list(mun), "partidos": list(partidos),
                            "rango_inclinacion": list(r_inc), "rango_clasificacion": list(r_clas)},
        "indicadores_filtrados": {"votos_candidatos": int(votos), "censo_habilitado_en_puestos": int(censo),
                                  "participacion_aparente_pct": round(100*votos/censo, 2) if censo else None,
                                  "candidatos": int(cand["candidato"].nunique()), "partidos": int(cand["partido"].nunique()),
                                  "puestos": int(len(puestos)), "municipios": int(R["n_municipios"]),
                                  "indice_voto_partido": round(media_ponderada(cand["inclinacion"], cand["votos"]), 4),
                                  "indice_voto_candidato": round(media_ponderada(cand["clasificacion"], cand["votos"]), 4)},
        "principales_candidatos_por_votos": top_cand,
        "municipios_con_mas_votos_en_filtros": top_mun,
        "limitaciones": ["Los votos son por candidatos; pueden excluir votos en blanco y nulos.",
                         "La participación aparente puede cambiar de interpretación si hay filtros de partido o índice."]
    }


def tab_asistente(ruta: str, R: dict, filtros: tuple) -> None:
    """Chat contextual persistente durante la sesión de Streamlit."""
    import agente_adk as adk
    adk.configurar_ruta_datos(ruta)
    st.subheader("Asistente analítico · Google ADK + OpenAI")
    st.caption("Pregunta por los indicadores filtrados o solicita una comparación territorial. Las cifras deben provenir del dashboard o de las herramientas de datos.")
    ok, motivo = adk.adk_disponible()
    if not ok:
        st.warning(f"El asistente no está disponible: {motivo}")
        st.info("Configura OPENAI_API_KEY y DASHBOARD_DATA_PATH (o utiliza la ruta de datos del dashboard), e instala requirements.txt.")
        return

    contexto = contexto_chat_dashboard(R, filtros)
    mensajes = st.session_state.setdefault("chat_analitica_mensajes", [])
    for msg in mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    pregunta = st.chat_input("Ej.: compara la participación de Tunja con el promedio de Boyacá", key="chat_analitico_input")
    if pregunta:
        mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("Consultando el contexto y las herramientas analíticas…"):
                try:
                    # Una sesión ADK por turno mantiene el contexto visible y evita mezclar usuarios.
                    import uuid
                    historial = "\n".join(f"{m['role']}: {m['content']}" for m in mensajes[-8:])
                    pregunta_con_historial = (f"Historial reciente de esta conversación:\n{historial}\n\nNueva pregunta: {pregunta}" if historial else pregunta)
                    respuesta = adk.consultar(pregunta_con_historial, contexto, session_id="chat-" + uuid.uuid4().hex)
                except Exception as exc:
                    respuesta = (f"No pude completar la consulta ({type(exc).__name__}). "
                                 "Verifica OPENAI_API_KEY, el nombre del modelo y la conectividad.")
            st.markdown(respuesta)
        mensajes.append({"role": "assistant", "content": respuesta})
    with st.expander("Contexto analítico enviado al agente"):
        st.json(contexto)
    if st.button("Limpiar conversación", key="limpiar_chat_analitico"):
        st.session_state["chat_analitica_mensajes"] = []
        st.rerun()


COLORES_BLOQUE = {"Izquierda": "#b2182b", "Centro": "#9e9e9e", "Derecha": "#2166ac"}


@st.cache_data(show_spinner=False, max_entries=4)
def correr_simulacion(ruta: str, n_sim: int, sigma_part: float, sigma_desplaz: float,
                      lambda_espacial: float, semilla: int) -> dict:
    import simulacion_mc as smc
    return smc.simular(ruta, n_sim, sigma_part, sigma_desplaz, lambda_espacial, semilla)


def tab_simulacion(ruta: str, filtros: tuple) -> None:
    import simulacion_mc as smc

    st.subheader("Simulación de Montecarlo: ¿qué tan seguros son los resultados?")
    st.caption(
        "Cada escenario perturba tres variables: la participación de cada puesto, la ideología medida de cada "
        "candidato y un desplazamiento ideológico de cada municipio. Los votos se reparten según la afinidad "
        "observada en 2023 y la cercanía ideológica. Repetir muchos escenarios muestra qué municipios son "
        "reñidos y la probabilidad de ganar de cada candidato."
    )

    with st.form("form_simulacion"):
        c1, c2, c3, c4, c5 = st.columns(5)
        n_sim = c1.slider("Escenarios", 50, 1000, 150, step=50)
        sigma_part = c2.slider("Incertidumbre de participación (σ)", 0.0, 0.20, 0.06, step=0.01,
                               help="Desviación estándar de la participación de cada puesto (0,06 = ±6 puntos).")
        sigma_desplaz = c3.slider("Desplazamiento ideológico (σ)", 0.0, 0.40, 0.15, step=0.01,
                                  help="Cuánto puede moverse el electorado de cada municipio en el eje −1 a 1.")
        lambda_esp = c4.slider("Sensibilidad a la distancia (λ)", 0.0, 5.0, 2.0, step=0.1,
                               help="Cuánto castiga el votante la distancia ideológica con el candidato.")
        semilla = c5.number_input("Semilla", 0, 9999, 42, help="Misma semilla y parámetros = mismos resultados.")
        if st.form_submit_button("Correr simulación", type="primary"):
            st.session_state["sim_params"] = (int(n_sim), float(sigma_part), float(sigma_desplaz),
                                              float(lambda_esp), int(semilla))

    cal = smc.calibracion(ruta)
    with st.expander(f"Prueba de calibración: sin incertidumbre cambian {cal['cambian']} de "
                     f"{fmt_num(cal['n_muns'])} alcaldías", expanded=cal["cambian"] > 0):
        st.markdown(
            "Con las tres incertidumbres en cero la simulación debería reproducir los ganadores de 2023. "
            "Los municipios que cambian aquí lo hacen por el suavizado de la afinidad, el redondeo de votos o el "
            "recorte de la participación a 5–98 %, no por la incertidumbre: su volatilidad en la simulación está "
            "inflada y conviene leerla con cuidado."
        )
        if cal["cambian"]:
            st.dataframe(cal["detalle"], hide_index=True, width="stretch")

    if "sim_params" not in st.session_state:
        st.info("Ajusta los parámetros y presiona **Correr simulación**.")
        return
    with st.spinner("Simulando escenarios…"):
        S = correr_simulacion(ruta, *st.session_state["sim_params"])
    reg, muns, cands = S["registro"], S["muns"], S["cands"]
    st.caption(f"{fmt_num(S['n_puestos'])} de {fmt_num(S['n_puestos_total'])} puestos usados: se excluyen "
               f"{fmt_num(S['excluidos'])} con censo 0 o participación ≥ 100 %. Parámetros: "
               f"{S['parametros']['n_sim']} escenarios, σ participación {S['parametros']['sigma_part']}, "
               f"σ desplazamiento {S['parametros']['sigma_desplaz']}, λ {S['parametros']['lambda_espacial']}, "
               f"semilla {S['parametros']['semilla']}.")

    # --- Indicadores nacionales -------------------------------------------------
    k = st.columns(4)
    p05, p50, p95 = np.percentile(reg["alcaldias_cambian"], [5, 50, 95])
    k[0].metric("Alcaldías que cambian de ganador", f"{fmt_num(p50)}",
                help=f"Mediana por escenario; 90 % de los escenarios entre {fmt_num(p05)} y {fmt_num(p95)}.")
    k[1].metric("Municipios reñidos (cambio ≥ 50 %)", fmt_num((muns["prob_cambio"] >= 0.5).sum()))
    k[2].metric("Municipios estables (cambio < 5 %)", fmt_num((muns["prob_cambio"] < 0.05).sum()))
    k[3].metric("Participación media simulada", fmt_pct(reg["part_media"].mean()))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Convergencia del % de votos por bloque ideológico**")
        fig = go.Figure()
        for b in smc.BLOQUES:
            serie = (reg[f"votos_{b.lower()}"] / reg["total_votos"] * 100).expanding().mean()
            fig.add_trace(go.Scatter(x=reg["iteracion"], y=serie, mode="lines", name=b,
                                     line=dict(color=COLORES_BLOQUE[b], width=2.5),
                                     hovertemplate=b + ": %{y:.2f}%<extra>escenario %{x}</extra>"))
        fig.update_layout(height=380, margin=dict(t=20), xaxis_title="Escenario",
                          yaxis=dict(title="% acumulado de votos", ticksuffix="%"),
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Bloques: izquierda < −{smc.CORTE_BLOQUE}, centro entre ±{smc.CORTE_BLOQUE}, "
                   f"derecha > {smc.CORTE_BLOQUE} (ideología del candidato en cada escenario). "
                   "Si las curvas se aplanan, hay suficientes escenarios.")
    with col2:
        st.markdown("**Alcaldías ganadas por bloque en los escenarios**")
        real = np.bincount(smc._bloque(cands.loc[cands["gano_2023"], "clasificacion"].to_numpy()), minlength=3)
        fig = go.Figure()
        for i, b in enumerate(smc.BLOQUES):
            fig.add_trace(go.Box(y=reg[f"alcaldias_{b.lower()}"], name=b, marker_color=COLORES_BLOQUE[b],
                                 fillcolor=rgba(COLORES_BLOQUE[b], 0.35), line=dict(color=COLORES_BLOQUE[b]),
                                 boxpoints=False, showlegend=False))
            fig.add_trace(go.Scatter(x=[b], y=[real[i]], mode="markers", showlegend=i == 0, name="Resultado 2023",
                                     marker=dict(symbol="diamond", size=12, color="#111111"),
                                     hovertemplate=f"2023: {real[i]}<extra></extra>"))
        fig.update_layout(height=380, margin=dict(t=20), yaxis_title="Alcaldías ganadas",
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        top_n = st.slider("Municipios más reñidos a mostrar", 10, 40, 20, key="sim_top")
        top = muns.nlargest(top_n, "prob_cambio").sort_values("prob_cambio")
        fig = go.Figure(go.Bar(
            x=top["prob_cambio"] * 100, y=top["mun"] + " (" + top["dep"] + ")", orientation="h",
            marker_color=color_ideologia(top["ideo_ganador_2023"]),
            customdata=np.c_[top["ganador_2023"], top["margen_2023"] * 100, top["n_candidatos"]],
            hovertemplate="%{y}<br>Cambia de ganador en %{x:.0f}% de los escenarios<br>"
                          "Ganador 2023: %{customdata[0]}<br>Margen 2023: %{customdata[1]:.1f} puntos · "
                          "%{customdata[2]} candidatos<extra></extra>"))
        fig.update_layout(height=max(380, 24 * top_n), margin=dict(t=20, l=10),
                          xaxis=dict(title="% de escenarios en que cambia el ganador", ticksuffix="%"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Color: ideología del ganador de 2023 (rojo izquierda, azul derecha).")
    with col4:
        st.markdown("**Margen de 2023 vs. probabilidad de cambio**")
        fig = go.Figure(go.Scattergl(
            x=muns["margen_2023"] * 100, y=muns["prob_cambio"] * 100, mode="markers",
            marker=dict(size=6, opacity=0.5, color=muns["ideo_ganador_2023"], colorscale=ESCALA_IDEOLOGIA,
                        cmin=-1, cmax=1),
            customdata=np.c_[muns["mun"], muns["dep"]],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Margen %{x:.1f} pts · cambio %{y:.0f}%<extra></extra>"))
        fig.update_layout(height=380, margin=dict(t=20), xaxis=dict(title="Margen entre 1.º y 2.º en 2023 (puntos)"),
                          yaxis=dict(title="% de escenarios con cambio", ticksuffix="%", range=[-2, 102]))
        st.plotly_chart(fig, width="stretch")
        st.caption("Lo esperable: a menor margen en 2023, mayor probabilidad de cambio. Puntos alejados de esa "
                   "tendencia merecen revisión.")

    # --- Un municipio -------------------------------------------------------------
    st.subheader("Probabilidad de ganar por candidato en un municipio")
    lista_dep = sorted(muns["dep"].unique())
    deps_lat, muns_lat = filtros[0], filtros[1]
    dep_ini = str(deps_lat[0]) if len(deps_lat) == 1 and str(deps_lat[0]) in lista_dep else \
        muns.loc[muns["prob_cambio"].idxmax(), "dep"]
    c6, c7 = st.columns(2)
    dep = c6.selectbox("Departamento", lista_dep, index=lista_dep.index(dep_ini), key="sim_dep")
    lista_mun = sorted(muns.loc[muns["dep"] == dep, "mun"].unique())
    mun_ini = str(muns_lat[0]) if len(muns_lat) == 1 and str(muns_lat[0]) in lista_mun else \
        muns[muns["dep"] == dep].nlargest(1, "prob_cambio")["mun"].iloc[0]
    mun = c7.selectbox("Municipio", lista_mun, index=lista_mun.index(mun_ini), key="sim_mun")

    fila_m = muns[(muns["dep"] == dep) & (muns["mun"] == mun)].iloc[0]
    cm = cands[cands["mun_id"] == fila_m["mun_id"]].sort_values("prob_ganar", ascending=False)
    margen_txt = f"{fila_m['margen_2023'] * 100:.1f}".replace(".", ",")
    st.caption(f"Ganador 2023: **{fila_m['ganador_2023']}** con un margen de {margen_txt} puntos · "
               f"cambia de ganador en {fila_m['prob_cambio'] * 100:.0f} % de los escenarios.")
    col5, col6 = st.columns(2)
    with col5:
        orden = cm.sort_values("prob_ganar")
        fig = go.Figure(go.Bar(
            x=orden["prob_ganar"] * 100, y=orden["candidato"], orientation="h",
            marker_color=color_ideologia(orden["clasificacion"]),
            text=[f"{v * 100:.0f}%" for v in orden["prob_ganar"]], textposition="outside", cliponaxis=False,
            customdata=np.c_[orden["partido"], orden["clasificacion"]],
            hovertemplate="%{y}<br>%{customdata[0]}<br>Ideología %{customdata[1]:.2f}<br>"
                          "Gana en %{x:.0f}% de los escenarios<extra></extra>"))
        fig.update_layout(height=max(280, 50 * len(cm) + 80), margin=dict(t=20, l=10, r=40),
                          xaxis=dict(title="Probabilidad de ganar", ticksuffix="%", range=[0, 110]))
        st.plotly_chart(fig, width="stretch")
    with col6:
        orden = cm.sort_values("votos_mediana")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orden["votos_mediana"], y=orden["candidato"], mode="markers", name="Simulación (mediana y 90 %)",
            marker=dict(size=11, color=color_ideologia(orden["clasificacion"])),
            error_x=dict(type="data", symmetric=False, array=orden["votos_p95"] - orden["votos_mediana"],
                         arrayminus=orden["votos_mediana"] - orden["votos_p05"], color="#888888"),
            hovertemplate="%{y}<br>Mediana %{x:,.0f} votos<extra></extra>"))
        fig.add_trace(go.Scatter(x=orden["votos_reales"], y=orden["candidato"], mode="markers", name="2023",
                                 marker=dict(symbol="line-ns-open", size=16, color="#111111", line=dict(width=2)),
                                 hovertemplate="%{y}<br>2023: %{x:,.0f} votos<extra></extra>"))
        fig.update_layout(height=max(280, 50 * len(cm) + 80), margin=dict(t=20, l=10),
                          xaxis_title="Votos", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

    cal_mun = cal["detalle"]
    boton_explicar("Simulación", {
        "parametros": S["parametros"],
        "calibracion": {"alcaldias_que_cambian_sin_incertidumbre": cal["cambian"], "total_alcaldias": cal["n_muns"],
                        "este_municipio_cambia_sin_incertidumbre": bool(
                            ((cal_mun["dep"] == dep) & (cal_mun["mun"] == mun)).any()) if len(cal_mun) else False},
        "nacional": {"alcaldias_que_cambian_mediana": int(p50), "rango_90": [int(p05), int(p95)],
                     "total_alcaldias": len(muns), "municipios_reñidos_cambio_50pct_o_mas": int((muns["prob_cambio"] >= 0.5).sum()),
                     "municipios_estables_cambio_menor_5pct": int((muns["prob_cambio"] < 0.05).sum())},
        "municipio": {"nombre": mun, "departamento": dep, "ganador_2023": fila_m["ganador_2023"],
                      "margen_2023_puntos": round(float(fila_m["margen_2023"] * 100), 1),
                      "prob_cambio_de_ganador_pct": round(float(fila_m["prob_cambio"] * 100), 1)},
        "candidatos": [{"candidato": r["candidato"], "partido": r["partido"],
                        "ideologia": round(float(r["clasificacion"]), 2), "gano_2023": bool(r["gano_2023"]),
                        "prob_ganar_pct": round(float(r["prob_ganar"] * 100), 1), "votos_2023": int(r["votos_reales"]),
                        "votos_mediana_simulada": int(r["votos_mediana"]),
                        "votos_rango_90": [int(r["votos_p05"]), int(r["votos_p95"])]}
                       for _, r in cm.head(8).iterrows()],
    }, key="simulacion")

    tabla = cm[["candidato", "partido", "clasificacion", "votos_reales", "votos_p05", "votos_mediana",
                "votos_p95", "prob_ganar", "gano_2023"]].rename(columns={
        "candidato": "Candidato", "partido": "Partido", "clasificacion": "Ideología", "votos_reales": "Votos 2023",
        "votos_p05": "Votos p5", "votos_mediana": "Votos mediana", "votos_p95": "Votos p95",
        "prob_ganar": "Prob. ganar", "gano_2023": "Ganó 2023"})
    st.dataframe(tabla.style.format({"Ideología": fmt_idx, "Votos 2023": fmt_num, "Votos p5": fmt_num,
                                     "Votos mediana": fmt_num, "Votos p95": fmt_num, "Prob. ganar": fmt_pct}),
                 hide_index=True, width="stretch")

    csv = (muns.sort_values("prob_cambio", ascending=False)
           [["dep", "mun", "ganador_2023", "margen_2023", "n_candidatos", "prob_cambio"]]
           .to_csv(index=False).encode("utf-8-sig"))
    st.download_button("Descargar volatilidad por municipio (CSV)", csv, file_name="simulacion_volatilidad.csv",
                       mime="text/csv")
    st.caption("Limitaciones: una sola elección como base; la incertidumbre la fijan los parámetros elegidos, "
               "no se estima de los datos; no incluye votos en blanco ni nulos. Es un análisis de sensibilidad, "
               "no un pronóstico.")

def tab_prediccion(ruta: str) -> None:
    import servicio_modelo as sm
    from servicio_modelo import (cargar_modelo, modelo_disponible, predecir,
                                 residuos_participacion, tabla_puestos)

    st.subheader("Predicción de votantes por puesto")
    if not modelo_disponible():
        st.error("No se encontró ../Notebooks/mejor_modelo_votaciones.pkl. "
                 "La interfaz está lista pero no puede predecir sin el modelo.")
        return
    modelo = cargar_modelo()
    if modelo is None:
        st.error("El modelo existe pero no pudo cargarse.")
        motivo = getattr(sm, "ERROR_CARGA", None)
        if motivo is None:  # módulo viejo en memoria: se intenta cargar aquí para ver el error real
            try:
                import joblib
                joblib.load(sm.RUTA_MODELO if hasattr(sm, "RUTA_MODELO")
                            else Path(__file__).parent / ".." / "Notebooks" / "mejor_modelo_votaciones.pkl")
                motivo = "El archivo carga bien; reinicia Streamlit para usar la versión nueva de servicio_modelo.py."
            except Exception as e:  # noqa: BLE001
                motivo = f"{type(e).__name__}: {e}"
        st.code(motivo, language="text")
        st.caption("Si el error menciona scikit-learn o un módulo que no se encuentra, instala en el "
                   "entorno de la app la misma versión de scikit-learn con la que entrenaste el modelo.")
        return
    tabla = tabla_puestos(ruta)
    dep = st.selectbox("Departamento", sorted(tabla["dep"].unique()), key="pred_dep")
    sub = tabla[tabla["dep"] == dep]
    mun = st.selectbox("Municipio", sorted(sub["mun"].unique()), key="pred_mun")
    sub = sub[sub["mun"] == mun]
    pnom = st.selectbox("Puesto", sorted(sub["puesto_nom"].unique()), key="pred_psto")
    fila = sub[sub["puesto_nom"] == pnom].head(1)
    if fila.empty:
        st.warning("Puesto sin datos válidos.")
        return
    censo = float(fila["censo"].iloc[0])
    st.write(f"Censo (habilitados) del puesto: **{fmt_num(censo)}**")
    seleccion = (dep, mun, pnom)
    if st.button("Predecir votantes", type="primary"):
        try:
            res = predecir(modelo, fila, [censo])
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            return
        # Se guarda para que el resultado siga visible en las siguientes interacciones
        st.session_state["prediccion"] = (seleccion, float(res["participacion_predicha"].iloc[0]),
                                          int(res["votantes_predichos"].iloc[0]))
    guardada = st.session_state.get("prediccion")
    if guardada and guardada[0] == seleccion:
        _, tasa, votos = guardada
        reales = float(fila["votos_puesto"].iloc[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Participación estimada", fmt_pct(tasa))
        c2.metric("Votantes estimados", fmt_num(votos))
        c3.metric("Votantes reales 2023", fmt_num(reales),
                  help="Votos por candidatos en 2023 (sin blancos ni nulos), como referencia.")
        resid = residuos_participacion(ruta)
        origen = sm.origen_residuos() if hasattr(sm, "origen_residuos") else "el modelo"
        intervalo = None
        if len(resid) > 100:
            lo, hi = np.percentile(resid, [5, 95])
            t_lo, t_hi = max(0.0, tasa + lo), min(1.0, tasa + hi)
            intervalo = (t_lo, t_hi)
            st.caption(
                f"Intervalo empírico 90 % (residuos de {origen}): participación "
                f"{fmt_pct(t_lo)} – {fmt_pct(t_hi)}  →  votantes "
                f"{fmt_num(round(t_lo * censo))} – {fmt_num(round(t_hi * censo))}."
            )
        st.caption("Qué significa: personas esperadas que votarían en el puesto. "
                   "Qué NO significa: no indica por quién votan ni quién gana.")

        mun_tab = tabla[(tabla["dep"] == dep) & (tabla["mun"] == mun)]
        boton_explicar("Predicción", {
            "puesto": pnom, "municipio": mun, "departamento": dep, "censo": int(censo),
            "participacion_predicha_pct": round(tasa * 100, 1), "votantes_estimados": votos,
            "votantes_reales_2023": int(reales),
            "diferencia_estimado_menos_real": int(votos - reales),
            "rango_90_votantes": [round(intervalo[0] * censo), round(intervalo[1] * censo)] if intervalo else None,
            "origen_del_rango": origen if intervalo else None,
            "contexto": {
                "participacion_real_2023_pct": round(reales / censo * 100, 1) if censo else None,
                "participacion_media_puestos_del_municipio_pct": round(float(mun_tab["participacion"].mean() * 100), 1),
                "participacion_media_nacional_pct": round(float(tabla["participacion"].mean() * 100), 1),
                "puestos_en_el_municipio": len(mun_tab),
            },
        }, key="prediccion")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🗳️ EDA electoral Colombia: votos totales")

    if not RUTA_DATOS.exists():
        if RUTA_DATOS.with_suffix(".xlsx").exists():
            st.error(
                "Encontré Dataset_completo.xlsx, pero la app lee Parquet (Excel es demasiado "
                "lento para este tamaño). Conviértelo una vez a ../Dataset_completo.parquet."
            )
        else:
            st.error(f"No encuentro {RUTA_DATOS.name}. Ponlo en la misma carpeta que app.py.")
        st.stop()

    ruta = str(RUTA_DATOS.resolve())
    try:
        import agente_adk as adk
        adk.configurar_ruta_datos(ruta)
    except Exception:
        pass
    try:
        cargar_datos(ruta)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    filtros = barra_lateral(ruta)
    R = resumenes(ruta, filtros)
    if R is None:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()

    mostrar_kpis(R)
    st.divider()

    # Filtros que cambian el significado de índice y participación por puesto
    _, _, partidos, r_inc, r_clas = filtros
    parcial = bool(partidos) or tuple(r_inc) != RANGO_COMPLETO or tuple(r_clas) != RANGO_COMPLETO

    tabs = st.tabs([
        "Distribución", "Rankings", "Territorio",
        "Ideología", "Puestos", "Discurso",
        "Estrategia", "Simulación", "Predicción", "Asistente IA",
    ])
    with tabs[0]:
        tab_distribucion(R)
    with tabs[1]:
        tab_rankings(R)
    with tabs[2]:
        tab_territorio(R)
    with tabs[3]:
        tab_ideologia(R, parcial)
    with tabs[4]:
        tab_puestos(R, parcial)
    with tabs[5]:
        tab_discurso(ruta, filtros)
    with tabs[6]:
        tab_estrategia(ruta, filtros)
    with tabs[7]:
        tab_simulacion(ruta, filtros)
    with tabs[8]:
        tab_prediccion(ruta)
    with tabs[9]:
        tab_asistente(ruta, R, filtros)


if __name__ == "__main__":
    main()
