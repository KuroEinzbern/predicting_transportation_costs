from fastapi import FastAPI
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from challenge_spotter.api.schema_validation import PredictionRequest, PredictionResponse
from challenge_spotter import config as cfg
import os
from dotenv import load_dotenv



pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    load_dotenv()   
    model_version = os.getenv("model_version", 1.0)
    pipeline = joblib.load(cfg.MODEL_DIR / f"model_{model_version}")
    yield 
    pipeline= None

app = FastAPI(title="Model API", lifespan=lifespan )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([request.model_dump()])
    X_to_predict= input_data.drop(columns=["load_id"])
    prediction= pipeline.predict(X=X_to_predict)
    return PredictionResponse(load_id=request.load_id,prediction=float(prediction[0]))

@app.get("/health")
def health():
    return {"status": "ok"}