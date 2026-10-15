# -*- coding: utf-8 -*-
"""Compatibilidad con la interfaz existente del explicador IA.
La implementación real del agente y sus herramientas está en agente_adk.py.
"""
from __future__ import annotations
import json
import os
import numpy as np

MODELO_LLM = os.getenv("MODELO_LLM", "openai/gpt-4o-mini")
GUIAS = {
    "Estrategia": "Interpreta los cuadrantes territoriales; la abstención puede sobreestimarse porque no incluye votos en blanco ni nulos.",
    "Simulación": "Es análisis de sensibilidad basado en parámetros del usuario, no un pronóstico. La probabilidad de ganar solo aplica al escenario simulado.",
    "Predicción": "El modelo estima participación y número de votantes; no predice intención de voto ni probabilidad de victoria.",
}

def disponible() -> tuple[bool, str]:
    import agente_adk as adk
    return adk.adk_disponible()

def a_json(contexto: dict) -> str:
    def convertir(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return None if np.isnan(o) else round(float(o), 4)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    return json.dumps(contexto, ensure_ascii=False, sort_keys=True, default=convertir)

def explicar(pestana: str, contexto_json: str, modelo: str | None = None) -> str:
    import agente_adk as adk
    return adk.explicar(pestana, contexto_json, modelo or MODELO_LLM)
