# -*- coding: utf-8 -*-
"""Agente analítico Google ADK + OpenAI vía LiteLLM para el dashboard electoral.

Las herramientas calculan o recuperan cifras en Python; el LLM solo interpreta resultados.
No estima probabilidad de victoria a partir del modelo de participación.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

APP_NAME = "dashboard_electoral"
MODELO_LLM = os.getenv("MODELO_LLM", "openai/gpt-4o-mini")
RUTA_DATOS: str | None = os.getenv("DASHBOARD_DATA_PATH")

INSTRUCCION = """Eres un analista de datos que acompaña un dashboard electoral de Colombia.
Responde en español claro y preciso. Usa primero el contexto de pantalla y las herramientas
analíticas cuando se necesiten cifras territoriales. No inventes cifras, fuentes ni resultados.
Distingue datos observados, estimaciones del modelo y escenarios de simulación. El modelo de
predicción estima participación/votantes, no probabilidad de victoria. La simulación es análisis
de sensibilidad, no pronóstico. Si una herramienta no tiene datos suficientes, dilo claramente.
Menciona el departamento/municipio y las unidades cuando corresponda. Responde con una conclusión,
la evidencia numérica y una limitación relevante. No excedas 250 palabras salvo que te pidan detalle.
"""


def configurar_ruta_datos(ruta: str | Path) -> None:
    """Configura la ruta de datos del dashboard para las herramientas territoriales."""
    global RUTA_DATOS
    RUTA_DATOS = str(Path(ruta).expanduser().resolve())


def adk_disponible() -> tuple[bool, str]:
    try:
        import google.adk  # noqa: F401
        import litellm  # noqa: F401
    except Exception:
        return False, "Faltan dependencias: instala los paquetes de requirements.txt."
    if not os.getenv("OPENAI_API_KEY"):
        return False, "No se encontró OPENAI_API_KEY en las variables de entorno o secretos."
    if not RUTA_DATOS or not Path(RUTA_DATOS).exists():
        return False, "No se encontró el Parquet. Configura DASHBOARD_DATA_PATH o la ruta de datos."
    return True, "Agente ADK listo."


def _json_seguro(obj: Any) -> str:
    def conv(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.bool_,)): return bool(v)
        if pd.isna(v): return None
        return str(v)
    return json.dumps(obj, ensure_ascii=False, default=conv)


def herramienta_contexto_dashboard(contexto_json: str) -> dict:
    """Devuelve el contexto de indicadores y filtros actualmente visible en el dashboard."""
    try:
        contexto = json.loads(contexto_json)
    except (TypeError, json.JSONDecodeError):
        return {"error": "El contexto del dashboard no es JSON válido."}
    return {"fuente": "contexto_visible_dashboard", "contexto": contexto}


def herramienta_resumen_municipio(departamento: str, municipio: str) -> dict:
    """Calcula votos por candidatos, censo y participación observada en un municipio."""
    if not RUTA_DATOS or not Path(RUTA_DATOS).exists():
        return {"error": "Ruta de datos no configurada o inexistente."}
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(RUTA_DATOS, format="parquet")
        required = ["dep", "mun", "puesto_nom", "votos_totales", "n_total_psto"]
        faltantes = [c for c in required if c not in dataset.schema.names]
        if faltantes:
            return {"error": f"Faltan columnas requeridas: {faltantes}"}
        filtro = (ds.field("dep") == str(departamento)) & (ds.field("mun") == str(municipio))
        tabla = dataset.to_table(columns=required, filter=filtro).to_pandas()
        if tabla.empty:
            return {"error": "No se encontraron filas para ese departamento y municipio. Verifica los nombres exactos."}
        tabla["votos_totales"] = pd.to_numeric(tabla["votos_totales"], errors="coerce").fillna(0)
        tabla["n_total_psto"] = pd.to_numeric(tabla["n_total_psto"], errors="coerce")
        puestos = tabla.groupby(["dep", "mun", "puesto_nom"], dropna=False).agg(
            votos=("votos_totales", "sum"), censo=("n_total_psto", "first")).reset_index()
        puestos = puestos[(puestos.censo > 0) & (puestos.votos < puestos.censo)]
        if puestos.empty:
            return {"error": "El municipio no tiene puestos válidos con el criterio del dashboard (censo > 0 y votos < censo)."}
        votos = float(puestos.votos.sum()); censo = float(puestos.censo.sum())
        return {"departamento": str(departamento), "municipio": str(municipio),
                "puestos_validos": int(len(puestos)), "votos_candidatos_observados": int(votos),
                "censo_habilitado": int(censo), "participacion_observada_pct": round(100*votos/censo, 2) if censo else None,
                "fuente": "Dataset Parquet del dashboard", "nota": "Votos por candidatos; puede excluir votos en blanco y nulos."}
    except Exception as exc:  # devuelve un error seguro al modelo y al usuario
        return {"error": f"No fue posible consultar el municipio ({type(exc).__name__}). Revisa el dataset y sus columnas."}


def herramienta_comparar_municipio_departamento(departamento: str, municipio: str) -> dict:
    """Compara la participación observada del municipio con otros municipios del departamento."""
    if not RUTA_DATOS or not Path(RUTA_DATOS).exists():
        return {"error": "Ruta de datos no configurada o inexistente."}
    try:
        import pyarrow.dataset as ds
        dataset = ds.dataset(RUTA_DATOS, format="parquet")
        required = ["dep", "mun", "puesto_nom", "votos_totales", "n_total_psto"]
        filtro = ds.field("dep") == str(departamento)
        t = dataset.to_table(columns=required, filter=filtro).to_pandas()
        if t.empty:
            return {"error": "No se encontraron datos para el departamento indicado."}
        t["votos_totales"] = pd.to_numeric(t["votos_totales"], errors="coerce").fillna(0)
        t["n_total_psto"] = pd.to_numeric(t["n_total_psto"], errors="coerce")
        p = t.groupby(["dep", "mun", "puesto_nom"], dropna=False).agg(votos=("votos_totales", "sum"), censo=("n_total_psto", "first")).reset_index()
        p = p[(p.censo > 0) & (p.votos < p.censo)]
        m = p.groupby("mun", dropna=False).agg(votos=("votos", "sum"), censo=("censo", "sum"), puestos=("puesto_nom", "nunique")).reset_index()
        m["participacion_pct"] = 100*m["votos"]/m["censo"]
        objetivo = m[m["mun"].astype(str).str.casefold() == str(municipio).casefold()]
        if objetivo.empty:
            return {"error": "El municipio no aparece con ese nombre en el departamento."}
        x = float(objetivo["participacion_pct"].iloc[0]); media = float(m["participacion_pct"].mean()); mediana = float(m["participacion_pct"].median())
        return {"departamento": str(departamento), "municipio": str(municipio),
                "participacion_municipio_pct": round(x, 2), "media_municipios_departamento_pct": round(media, 2),
                "mediana_municipios_departamento_pct": round(mediana, 2),
                "diferencia_vs_media_pp": round(x-media, 2), "municipios_comparados": int(len(m)),
                "nota": "Media simple de tasas municipales, no ponderada por censo; votos por candidatos."}
    except Exception as exc:
        return {"error": f"No fue posible comparar territorios ({type(exc).__name__}). Revisa el dataset."}


@lru_cache(maxsize=4)
def _runner(modelo: str):
    from google.adk.agents import Agent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    agent = Agent(name="analista_dashboard", model=LiteLlm(model=modelo), description="Analista del dashboard electoral",
                  instruction=INSTRUCCION,
                  tools=[herramienta_contexto_dashboard, herramienta_resumen_municipio,
                         herramienta_comparar_municipio_departamento])
    sesiones = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=sesiones)
    return runner, sesiones


def consultar(pregunta: str, contexto: dict | str, modelo: str | None = None, session_id: str = "dashboard-chat") -> str:
    """Responde una pregunta usando ADK, herramientas Python y contexto de filtros/indicadores."""
    from google.genai import types
    modelo = modelo or MODELO_LLM
    runner, sesiones = _runner(modelo)
    user_id = "dashboard_user"
    try:
        sesiones.create_session_sync(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    except Exception:
        # La sesión puede existir en conversaciones persistentes de Streamlit.
        pass
    contexto_json = contexto if isinstance(contexto, str) else _json_seguro(contexto)
    mensaje = (f"Contexto actual del dashboard (JSON; úsalo como evidencia y filtros activos):\n{contexto_json}\n\n"
               f"Pregunta del usuario: {pregunta}\n\nPuedes usar herramienta_contexto_dashboard para revisar el contexto; "
               "si la pregunta requiere datos territoriales, usa las herramientas de municipio.")
    partes = []
    for evento in runner.run(user_id=user_id, session_id=session_id,
                             new_message=types.Content(role="user", parts=[types.Part(text=mensaje)])):
        if evento.is_final_response() and evento.content and evento.content.parts:
            partes.extend(p.text for p in evento.content.parts if getattr(p, "text", None))
    return "\n".join(partes).strip() or "No se recibió una respuesta del modelo."


def explicar(pestana: str, contexto_json: str, modelo: str | None = None) -> str:
    pregunta = (f"Explica la pestaña {pestana} con la estructura: **En una frase**, **Lo que muestran los datos** "
                "(2-3 viñetas con cifras), **Qué significa** (1-2 acciones) y **Ojo con** (una limitación). Máximo 180 palabras.")
    from explicador_ia import GUIAS
    return consultar(pregunta, {"pestana": pestana, "guia_de_lectura": GUIAS.get(pestana, ""), "datos": json.loads(contexto_json)}, modelo,
                     session_id="explicacion-" + str(abs(hash((pestana, contexto_json))))[:16])


def construir_payload(accion: str, contenido: dict) -> dict:
    return {"accion": accion, "contenido": contenido}
