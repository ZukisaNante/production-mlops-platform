from pathlib import Path
import os

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# Path to the trained ML model
MODEL_PATH = Path("models") / "model.joblib"

# Application/model version information
API_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"
BUILD_SHA = os.getenv("BUILD_SHA", "local")


# Create FastAPI application
app = FastAPI(title="Production MLOps Platform")


# Load trained model
model = joblib.load(MODEL_PATH)


# Request schema for predictions
class PredictionRequest(BaseModel):
    features: list[float]


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "MLOps API is running"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Version endpoint
@app.get("/version")
def version():
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "build_sha": BUILD_SHA
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])[0]

    return {
        "prediction": int(prediction)
    }