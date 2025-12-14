from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

# Load the model
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


class InputData(BaseModel):
    feature1: float
    feature2: float


@app.get("/health")
def health_check():
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}


@app.post("/predict")
def predict(data: InputData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare data for prediction
    input_df = pd.DataFrame([data.dict()])

    # Predict
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}