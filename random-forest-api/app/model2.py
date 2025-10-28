from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import datetime
from typing import List, Optional, Union
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


class SimpleRandomForest:
    """
    Random Forest 'desde cero' usando DecisionTreeClassifier como base.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Union[int, float, str, None] = "sqrt",
        max_depth: Optional[int] = None,
        criterion: str = "gini",
        random_state: Optional[int] = 42,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifier] = []
        self._rng = np.random.default_rng(random_state)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        idx = self._rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        seeds = self._rng.integers(0, 10_000_000, size=self.n_estimators)
        for s in seeds:
            Xi, yi = self._bootstrap_sample(X, y)
            clf = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(s),
            )
            clf.fit(Xi, yi)
            self.trees.append(clf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([t.predict(X) for t in self.trees])
        maj = []
        for row in preds:
            c = Counter(row)
            maj.append(c.most_common(1)[0][0])
        return np.array(maj)


# Inicializa la app
app = FastAPI(
    title="Ensemble Model API",
    description="API para servir un modelo de ensemble en Render",
    version="1.0.0"
)

# Cargar el bundle completo
bundle = joblib.load("./model/model.pkl")
model = bundle["model"]  # ⬅️ Extrae el modelo del diccionario
feature_names = bundle.get("feature_names", [])
class_names = bundle.get("class_names", ["setosa", "versicolor", "virginica"])

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
        "model_name": "SimpleRandomForest",
        "author": "Vania Janet Raya Rios",
        "version": "1.0.0",
        "description": "Modelo de ensemble entrenado en scikit-learn y desplegado en Render",
        "framework": "FastAPI",
        "n_estimators": model.n_estimators,
        "feature_names": feature_names,
        "class_names": class_names,
        "last_update": datetime.datetime.now().isoformat(),
    }


@app.post("/predict")
def predict(input: Features):
    """
    Realiza una predicción con el modelo cargado.
    """
    X = np.array([input.features])
    y_pred = model.predict(X)  # ⬅️ Ahora usa 'model' no 'model.rf'
    
    # Mapear predicción a nombre de clase
    predicted_class = class_names[int(y_pred[0])] if y_pred[0] < len(class_names) else str(y_pred[0])
    
    return {
        "prediction": int(y_pred[0]),
        "class_name": predicted_class
    }