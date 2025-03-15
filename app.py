from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
model = load("injury_prediction_model.joblib")

# Define FastAPI app
app = FastAPI(title="Football Injury Prediction API")

# Root endpoint to check if API is running
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def home():
    return {"message": "Football Injury Prediction API is running!"}

# Define input schema
class PlayerData(BaseModel):
    age: int
    previous_injuries: int
    training_hours_per_week: float
    sleep_hours_per_night: float
    hydration_level: int  # 1: Adequate, 2: Insufficient, 3: Optimal
    nutrition_habits: int  # 1: Balanced, 2: Varied, 3: High Protein
    fitness_level: int  # 1: Low, 2: Moderate, 3: High

# API endpoint for prediction
@app.post("/predict")
def predict_injury(player: PlayerData):
    player_df = pd.DataFrame([player.dict()])
    prediction = model.predict(player_df)

    return {
        "injury_likelihood": prediction[0][0],
        "preventive_techniques": prediction[0][1],
        "predicted_injury_type": prediction[0][2]
    }

