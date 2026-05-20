"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Configuración de rutas
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0)

class IrisSample(BaseModel):
    sepal_length: float = Field(...)
    sepal_width: float  = Field(...)
    petal_length: float = Field(...)
    petal_width: float  = Field(...)

# --- NUEVO: PARÁMETROS DE POLÍTICA AÑADIDOS ---
class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(..., min_items=5)
    retrain_from_scratch: bool = Field(False)
    policy: str = Field(
        "any_improvement", 
        description="Políticas soportadas: any_improvement, min_delta, per_class_f1"
    )
    min_delta: float = Field(0.02, description="Umbral para la política min_delta")
    target_class: int = Field(0, description="Clase objetivo para per_class_f1 (0, 1 o 2)")

class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str

class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    policy_used: str
    message: str

class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}

def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []

def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    return history[-1] if history else None

def bootstrap_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)",
        "policy": "bootstrap",
        "activated": True
    }]
    save_history(history)

app = FastAPI(title="Iris Continuous Training API")

@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()

@app.get("/health")
def health():
    meta = get_active_model_meta()
    return {"status": "ok", "active_model_version": meta["version"] if meta else "none"}

@app.post("/predict", response_model=PredictResponse)
def predict(sample: IrisSample):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    clf = joblib.load(MODEL_PATH)
    X = np.array([[sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()
    return PredictResponse(prediction=pred, class_name=CLASS_NAMES[pred], model_version=meta["version"])

@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width] for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    history = load_history()
    previous_accuracy = history[-1]["accuracy"] if history else None
    
    # Cargar modelo viejo para comparar F1 si fuera necesario
    clf_old = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

    data_file = MODELS_DIR / "accumulated_data.joblib"
    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    if len(np.unique(y_train)) < 2:
        raise HTTPException(status_code=422, detail="El dataset de entrenamiento debe contener al menos 2 clases distintas.")

    clf_new = LogisticRegression(max_iter=300, random_state=42)

    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        clf_new.fit(X_tr, y_tr)
        eval_X, eval_y = X_val, y_val
        eval_note = f"validación con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        eval_X, eval_y = X_train, y_train
        eval_note = "evaluación en train"

    accuracy_new = round(float(accuracy_score(eval_y, clf_new.predict(eval_X))), 4)

    # --- NUEVO: EVALUACIÓN DE POLÍTICAS ---
    model_updated = False
    policy_reason = ""

    if request.policy == "any_improvement":
        model_updated = (previous_accuracy is None) or (accuracy_new >= previous_accuracy)
        policy_reason = f"Acc ({accuracy_new}) >= Prev ({previous_accuracy})" if model_updated else "Acc no mejoró"
    
    elif request.policy == "min_delta":
        if previous_accuracy is None:
            model_updated, policy_reason = True, "Primer modelo (Bootstrap)"
        else:
            delta_actual = accuracy_new - previous_accuracy
            model_updated = delta_actual >= request.min_delta
            policy_reason = f"Mejora ({delta_actual:.4f}) >= MinDelta ({request.min_delta})" if model_updated else f"Mejora menor a {request.min_delta}"
    
    elif request.policy == "per_class_f1":
        if clf_old is None:
             model_updated, policy_reason = True, "Primer modelo"
        else:
             f1_new = f1_score(eval_y, clf_new.predict(eval_X), labels=[request.target_class], average=None, zero_division=0)[0]
             f1_old = f1_score(eval_y, clf_old.predict(eval_X), labels=[request.target_class], average=None, zero_division=0)[0]
             model_updated = f1_new > f1_old
             policy_reason = f"F1 Clase {request.target_class} ({f1_new:.4f}) > Prev ({f1_old:.4f})" if model_updated else f"F1 no mejoró respecto a {f1_old:.4f}"
    else:
        raise HTTPException(status_code=400, detail="Política no reconocida")

    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"
    status = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)

    message = f"Modelo {status} usando política '{request.policy}'. Razón: {policy_reason}"

    history.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": accuracy_new,
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "status": status,
        "activated": model_updated,
        "policy_used": request.policy,
        "policy_reason": policy_reason
    })
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        policy_used=request.policy,
        message=message
    )

@app.get("/model/info", response_model=ModelInfo)
def model_info():
    history = load_history()
    if not history:
         raise HTTPException(status_code=404, detail="No model")
    active_entries = [h for h in history if h.get("activated", True)]
    active = active_entries[-1] if active_entries else history[-1]
    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active["algorithm"],
        history=history
    )

@app.delete("/model/history")
def reset_history():
    for p in [HISTORY_PATH, MODEL_PATH, MODELS_DIR / "accumulated_data.joblib"]:
        if p.exists(): p.unlink()
    bootstrap_model()
    return {"status": "ok"}