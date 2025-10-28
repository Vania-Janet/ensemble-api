"""
Script para entrenar y guardar el modelo SimpleRandomForest
Ejecutar desde: /Users/vania/Documents/Practica8/random-forest-api
"""

import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.model_class import SimpleRandomForest


if __name__ == "__main__":
    print("🔄 Entrenando modelo SimpleRandomForest...")
    
    # 1) Cargar datos de Iris desde sklearn
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()
    
    # 2) Split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3) Entrenar
    print(f"   - Entrenando con {len(Xtr)} muestras...")
    rf = SimpleRandomForest(
        n_estimators=1000,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        criterion="gini",
    )
    rf.fit(Xtr, ytr)
    
    # 4) Evaluar
    yp = rf.predict(Xte)
    acc = accuracy_score(yte, yp)
    print(f"   - Accuracy en test: {acc:.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(yte, yp))
    print("\nReporte de clasificación:")
    print(classification_report(yte, yp, target_names=class_names))
    
    # 5) Guardar modelo
    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": rf,
        "feature_names": feature_names,
        "class_names": class_names,
    }
    joblib.dump(bundle, "model/model.pkl")
    print("\n✅ Modelo guardado en: model/model.pkl")
    print(f"   - Features: {feature_names}")
    print(f"   - Clases: {class_names}")
    print(f"   - Número de árboles: {rf.n_estimators}")
