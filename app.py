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
@app.get("/")
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
    # Convert input data to a DataFrame
    player_df = pd.DataFrame([player.dict()])

    # Make prediction
    prediction = model.predict(player_df)

    # Return response as JSON
    return {
        "injury_likelihood": prediction[0][0],
        "preventive_techniques": prediction[0][1],
        "predicted_injury_type": prediction[0][2]
    }

