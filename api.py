from fastapi import FastAPI, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import logging
import numpy as np

from etl import run_etl
from ml_model import train_rf
from nn_model import train_nn

# Налаштування логування
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ML Платформа Titanic")
model = None

class Passenger(BaseModel):
    Sex: int = Field(..., ge=0, le=1, description="0=Female, 1=Male")
    Age: float = Field(..., ge=0, description="Age must be non-negative")
    Fare: float = Field(..., ge=0, description="Fare must be non-negative")
    FamilySize: int = Field(..., ge=1, description="At least 1 (the passenger)")
    IsAlone: int = Field(..., ge=0, le=1)
    Title: int
    AgeGroup: int = Field(..., ge=0, le=3)
    Embarked: int

@app.get("/")
def home():
    """Головна сторінка з HTML-інтерфейсом"""
    return FileResponse("web/index.html")

@app.post("/etl")
def etl_pipeline(
    input_file: str = Query("titanic.csv", description="Шлях до вхідного CSV"),
    output_file: str = Query("titanic_prepared.csv", description="Шлях до вихідного CSV")
):
    """Запуск ETL-процесу"""
    try:
        df = run_etl(input_file, output_file)
        logging.info(f"ETL завершено: {len(df)} рядків")
        return {
            "status": "ETL complete",
            "rows": len(df),
            "input_file": input_file,
            "output_file": output_file
        }
    except Exception as e:
        logging.error(f"ETL error: {e}")
        return {"error": str(e)}

@app.post("/train")
def train_model(model_type: str = Query("rf", description="Тип моделі: rf або nn")):
    """Навчання моделі"""
    try:
        df = pd.read_csv("titanic_prepared.csv")
        global model

        if model_type == "rf":
            model, metrics = train_rf(df)
        elif model_type == "nn":
            model, history = train_nn(df)
            metrics = {"accuracy": history.history["val_accuracy"][-1]}
        else:
            return {"error": f"Unknown model type: {model_type}"}

        joblib.dump(model, "model.pkl")
        logging.info(f"Model {model_type} trained")
        return {"status": "Model trained", "model_type": model_type, "metrics": metrics}
    except Exception as e:
        logging.error(f"Train error: {e}")
        return {"error": str(e)}

@app.post("/predict")
def predict(passenger: Passenger):
    """Прогноз для одного пасажира"""
    global model
    if model is None:
        try:
            model = joblib.load("model.pkl")
        except Exception:
            return {"error": "Model not trained yet"}

    data = [[
        passenger.Sex, passenger.Age, passenger.Fare, passenger.FamilySize,
        passenger.IsAlone, passenger.Title, passenger.AgeGroup, passenger.Embarked
    ]]

    # Перетворення у NumPy масив
    X = np.array(data).reshape(1, -1)

    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Приведення результату до int
    if isinstance(prediction, (np.ndarray, list)):
        prediction = prediction.item()
    return {"prediction": int(prediction)}

@app.post("/predict_batch_json")
def predict_batch_json(passengers: List[Passenger]):
    """Прогноз для групи пасажирів (JSON)"""
    global model
    if model is None:
        try:
            model = joblib.load("model.pkl")
        except Exception:
            return {"error": "Model not trained yet"}

    data = [[p.Sex, p.Age, p.Fare, p.FamilySize,
             p.IsAlone, p.Title, p.AgeGroup, p.Embarked] for p in passengers]

    predictions = model.predict(data).tolist()
    return {"predictions": predictions}

@app.post("/predict_batch_file")
def predict_batch_file(file: UploadFile = File(...)):
    """Прогноз для групи пасажирів (CSV-файл)"""
    global model
    if model is None:
        try:
            model = joblib.load("model.pkl")
        except Exception:
            return {"error": "Model not trained yet"}

    df = pd.read_csv(file.file)
    features = ["Sex","Age","Fare","FamilySize","IsAlone","Title","AgeGroup","Embarked"]
    data = df[features]

    predictions = model.predict(data).tolist()
    return {"predictions": predictions}

@app.get("/metrics")
def get_metrics():
    """Отримати confusion matrix"""
    return FileResponse("confusion_matrix.png")
