"""Chequeo local de configuración; no realiza llamadas al LLM."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

checks = {
    "OPENAI_API_KEY definida": bool(os.getenv("OPENAI_API_KEY")),
    "google-adk instalado": importlib.util.find_spec("google.adk") is not None,
    "litellm instalado": importlib.util.find_spec("litellm") is not None,
    "streamlit instalado": importlib.util.find_spec("streamlit") is not None,
    "pyarrow instalado": importlib.util.find_spec("pyarrow") is not None,
}
for name, default in [
    ("Parquet", ROOT.parent / "Dataset_completo.parquet"),
    ("Modelo de predicción", ROOT.parent / "Notebooks" / "mejor_modelo_votaciones.pkl"),
    ("CSV residuos test (opcional)", ROOT.parent / "Notebooks" / "votantes_predichos_test.csv"),
]:
    key = {"Parquet":"DASHBOARD_DATA_PATH", "Modelo de predicción":"DASHBOARD_MODEL_PATH", "CSV residuos test (opcional)":"DASHBOARD_PRED_TEST_PATH"}[name]
    value = Path(os.getenv(key, str(default))).expanduser()
    if name == "CSV residuos test (opcional)":
        checks[f"{name}: {value} (no bloqueante)"] = True
        if not value.exists():
            print(f"AVISO  {name}: no se encontró; la predicción puede usar residuos de todo el dataset.")
    else:
        checks[f"{name}: {value}"] = value.exists()

for label, ok in checks.items():
    print(f"{'OK' if ok else 'FALTA'}  {label}")
print("\nEste chequeo no valida compatibilidad interna del .pkl ni hace una petición a OpenAI.")
