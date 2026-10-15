# Integración de Google ADK + OpenAI (LiteLLM)

## Qué se agregó

- `agente_adk.py`: agente Google ADK con modelo OpenAI a través de LiteLLM y tres herramientas:
  - `herramienta_contexto_dashboard`: revisa el contexto resumido de filtros e indicadores.
  - `herramienta_resumen_municipio`: calcula votos por candidatos, censo y participación observada de un municipio desde el Parquet.
  - `herramienta_comparar_municipio_departamento`: compara participación municipal con media y mediana simples del departamento.
- `explicador_ia.py`: conserva la interfaz anterior de los botones “Explícame esto” y la delega al agente ADK.
- Nueva pestaña **Asistente IA** con chat contextual, historial de sesión, expander del contexto enviado y control para limpiar la conversación.
- Rutas configurables para el Parquet, el modelo serializado y el CSV de residuos mediante variables de entorno.
- `.env.example`, `.gitignore` y `Dockerfile` para facilitar ejecución local y despliegue en nube.

## Archivos externos requeridos

El ZIP original NO incluye el dataset Parquet ni los artefactos del modelo de predicción. Por tanto:

1. `Dataset_completo.parquet`: necesario para abrir el dashboard y ejecutar las herramientas territoriales. Debe contener las columnas usadas por `app.py`, incluidas `dep`, `mun`, `puesto_nom`, `partido`, `candidato`, `votos_totales`, `clasificacion`, `inclinacion_politica`, `n_mesas_psto`, `n_hombres_psto`, `n_mujeres_psto` y `n_total_psto`; también puede necesitar `discurso` para análisis de discurso.
2. `mejor_modelo_votaciones.pkl`: necesario solo para la pestaña Predicción.
3. `votantes_predichos_test.csv`: opcional, recomendado para el intervalo empírico de la predicción. Sin él el servicio puede construir residuos sobre todo el dataset, lo que puede producir un intervalo optimista.

Ubicación local sugerida:

```text
proyecto/
├── Dataset_completo.parquet
├── Notebooks/
│   ├── mejor_modelo_votaciones.pkl
│   └── votantes_predichos_test.csv  # opcional
└── Dashboard/
    ├── app.py
    ├── agente_adk.py
    ├── explicador_ia.py
    ├── servicio_modelo.py
    ├── simulacion_mc.py
    ├── requirements.txt
    └── .env
```

## Ejecución local (Windows PowerShell)

Desde la carpeta `Dashboard`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edita `.env` y define `OPENAI_API_KEY`. Ajusta `DASHBOARD_DATA_PATH`, `DASHBOARD_MODEL_PATH` y `DASHBOARD_PRED_TEST_PATH` a rutas reales. Las rutas relativas se interpretan desde el directorio de trabajo de Streamlit; para evitar ambigüedad se recomiendan rutas absolutas. En PowerShell puedes cargar las variables así:

```powershell
Get-Content .env | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' -and $_ -notmatch '^\s*#' } | ForEach-Object {
  $name, $value = $_ -split '=', 2
  [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
streamlit run app.py
```

Alternativamente, usa un gestor de variables de entorno y configura los valores antes de iniciar Streamlit. No subas `.env` al repositorio.

## Ejecutar en Linux/macOS

```bash
cd Dashboard
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env y exporta las variables (o usa python-dotenv desde tu entorno de ejecución)
set -a; source .env; set +a
streamlit run app.py
```

## Despliegue en nube

### Opción A — Streamlit Community Cloud

- Sube el código a un repositorio Git privado si el código o los datos son sensibles.
- Configura `OPENAI_API_KEY`, `MODELO_LLM`, `DASHBOARD_DATA_PATH`, `DASHBOARD_MODEL_PATH` y `DASHBOARD_PRED_TEST_PATH` en Secrets/variables del despliegue.
- No subas un Parquet grande al repositorio. El entorno debe poder acceder al archivo desde almacenamiento permitido o desde un artefacto seguro descargado durante el despliegue. Ajusta `DASHBOARD_DATA_PATH` a la ruta disponible.
- Streamlit Community Cloud no ofrece un disco persistente general para todos los usos; confirma límites de memoria, tamaño y persistencia antes de desplegar un dataset de millones de filas.

### Opción B — Docker / Cloud Run / servidor propio

Construye la imagen desde la carpeta `Dashboard`:

```bash
docker build -t dashboard-electoral .
docker run --rm -p 8501:8501 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e MODELO_LLM="openai/gpt-4o-mini" \
  -e DASHBOARD_DATA_PATH="/data/Dataset_completo.parquet" \
  -e DASHBOARD_MODEL_PATH="/models/mejor_modelo_votaciones.pkl" \
  -v /ruta/real/datos:/data:ro \
  -v /ruta/real/modelos:/models:ro \
  dashboard-electoral
```

Para Cloud Run, la imagen debe tener acceso a los datos durante la ejecución. Para archivos grandes, es preferible almacenamiento de objetos o un volumen soportado y descargar/montar los artefactos con identidad de servicio; no incluyas secretos ni datasets privados dentro de la imagen. Inyecta la clave de OpenAI desde un gestor de secretos. Configura memoria suficiente: el dashboard original carga un Parquet grande en memoria.

## Validaciones recomendadas

```bash
python -m py_compile app.py agente_adk.py explicador_ia.py servicio_modelo.py simulacion_mc.py
streamlit run app.py
```

En la aplicación:

1. Confirma que las pestañas tradicionales siguen abriendo.
2. Pulsa “Explícame esto” en Estrategia, Simulación o Predicción cuando haya datos disponibles.
3. En Asistente IA, pregunta por indicadores visibles y luego solicita una comparación territorial con nombres exactos del dataset.
4. Compara la salida de las herramientas con cálculos independientes. Las comparaciones departamentales usan una media simple de tasas municipales; no se ponderan por censo.
5. Comprueba consumo de memoria/tiempo con el dataset real.

## Limitaciones conocidas y siguientes pasos

- El archivo ZIP no contiene los datos ni el modelo serializado: no es posible validar predicciones o abrir el dashboard de punta a punta sin esos artefactos.
- El primer uso de una herramienta territorial puede leer muchas filas del Parquet; el rendimiento debe medirse con el archivo real. Para uso frecuente conviene materializar una tabla agregada municipal y departamental.
- La pestaña Predicción solo funcionará si el `.pkl` existe y es compatible con la versión de scikit-learn usada al entrenarlo.
- No hay un modelo probabilístico independiente de victoria en el proyecto; el agente debe mantener esa limitación explícita.
- El historial de chat vive en la sesión de Streamlit. No es una memoria persistente entre reinicios ni entre usuarios.
