from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import datetime

# Inicializa la app
app = FastAPI(
    title="Ensemble Model API",
    description="API para servir un modelo de ensemble en Render",
    version="1.0.0"
)

# Carga del modelo entrenado
model = joblib.load("model.pkl")

# Clase de entrada para predicción
class Features(BaseModel):
    features: list[float]


# ---------- ENDPOINTS ----------

@app.get("/health")
def health():
    """
    Verifica que la API esté viva.
    """
    return {"status": "ok", "timestamp": datetime.datetime.utcnow()}


@app.get("/info")
def info():
    """
    Devuelve metadatos del modelo y la API.
    """
    return {
        "model_name": "Ensemble Classifier",
        "author": "Vania Janet Raya Rios",
        "version": "1.0.0",
        "description": "Modelo de ensemble entrenado en scikit-learn y desplegado en Render",
        "framework": "FastAPI",
        "last_update": datetime.datetime.fromtimestamp(
            int(datetime.datetime.now().timestamp())
        ).isoformat(),
    }


@app.post("/predict")
def predict(input: Features):
    """
    Realiza una predicción con el modelo cargado.
    """
    X = np.array([input.features])
    y_pred = model.predict(X)
    return {"prediction": y_pred.tolist()}
