import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import pandas as pd

from src.api.schemas import ChurnRequest, ChurnResponse
from src.api import model_loader

# Configurar logging para trazabilidad y MLOps (útil para el Hito 3)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("andeslink_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar el servidor para optimizar tiempos de respuesta."""
    try:
        model_loader.get_model()
        logger.info("Modelo cargado correctamente en startup.")
    except Exception as e:
        logger.critical(f"Error crítico al cargar el modelo durante el inicio: {e}")
    yield

app = FastAPI(
    title="AndesLink Churn Prediction API",
    description="API REST para predecir la probabilidad de abandono (churn) de clientes de AndesLink.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def index():
    """Redirige automáticamente a la documentación interactiva de Swagger."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Endpoint de estado de salud (healthcheck) de la API."""
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(data: ChurnRequest):
    """
    Predice la probabilidad de churn a partir de los datos comportementales del cliente.
    """
    try:
        start_time = time.time()
        pipeline = model_loader.get_model()
        
        # Convertir a diccionario de forma compatible con Pydantic v1 y v2
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        
        # Convertir a DataFrame (el modelo de scikit-learn espera un pandas DataFrame)
        df = pd.DataFrame([payload])
        
        # Obtener probabilidades e inferir etiqueta
        probs = pipeline.predict_proba(df)[0]
        prob = float(probs[1])
        label = int(prob >= 0.5)
        
        duration = time.time() - start_time
        
        # Log del payload y resultado para monitoreo futuro (Hito 3)
        logger.info(
            f"PREDICT: duration={duration:.4f}s | prob={prob:.4f} | label={label} | payload={payload}"
        )
        
        return ChurnResponse(churn_probability=round(prob, 4), churn_label=label)
        
    except Exception as e:
        logger.error(f"Error procesando la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del modelo: {str(e)}")
