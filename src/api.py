from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("models") / "model.joblib"

app = FastAPI(title="Production MLOps Platform")

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "MLOps API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]

    return {
        "prediction": int(prediction)
    }