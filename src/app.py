from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import datetime
import numpy as np
import os
import mlflow

# Get the project root directory (parent of src)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, "models/best_model.pkl")

# Load the best model
model = joblib.load(model_path)

# FastAPI app with metadata (shows up in Swagger UI)
app = FastAPI(
    title="Ride ETA Prediction API",
    description="REST API to predict ETA for taxi rides using trained ML models",
    version="1.0.0"
)

# Input schema
class TripDetails(BaseModel):
    pickup_datetime: str
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    weather: str
    actual_duration: float | None = None   # optional field for monitoring drift

# Utility: Haversine distance
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@app.post("/predict", summary="Predict ETA", response_description="Predicted ETA in minutes")
def predict_eta(trip: TripDetails):
    # Parse datetime
    pickup_time = datetime.datetime.fromisoformat(trip.pickup_datetime)
    hour_of_day = pickup_time.hour
    day_of_week = pickup_time.weekday()
    is_weekend = 1 if day_of_week in [5,6] else 0
    rush_hour = 1 if hour_of_day in [7,8,9,17,18,19] else 0

    # Distance
    distance_km = haversine(trip.pickup_longitude, trip.pickup_latitude,
                            trip.dropoff_longitude, trip.dropoff_latitude)

    # Weather one-hot (simplified) - must match training encoding
    weather_map = {"sunny": [0,0], "rainy": [1,0], "cloudy": [0,1]}
    weather_features = weather_map.get(trip.weather.lower(), [0,0])

    # Feature vector
    features = [hour_of_day, day_of_week, is_weekend, distance_km, rush_hour] + weather_features
    features = np.array(features).reshape(1, -1)

    # Predict
    eta = model.predict(features)[0]

    # --- MLflow Logging ---
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'logs/mlflow.db')}")
    mlflow.set_experiment("Serving")

    with mlflow.start_run(run_name="API_Prediction", nested=True):
        mlflow.log_param("hour_of_day", hour_of_day)
        mlflow.log_param("day_of_week", day_of_week)
        mlflow.log_param("is_weekend", is_weekend)
        mlflow.log_param("rush_hour", rush_hour)
        mlflow.log_param("distance_km", distance_km)
        mlflow.log_param("weather", trip.weather)
        mlflow.log_metric("predicted_eta", float(eta))

        # If actual duration is provided, log error
        error = None
        if trip.actual_duration is not None:
            error = trip.actual_duration - float(eta)
            mlflow.log_metric("actual_duration", trip.actual_duration)
            mlflow.log_metric("prediction_error", error)

    return {
        "predicted_eta_minutes": round(float(eta), 2),
        "actual_duration": trip.actual_duration,
        "prediction_error": round(error, 2) if error is not None else None
    }
