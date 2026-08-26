"""Week 3: FastAPI endpoint for ride ETA prediction."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import BEST_MODEL_PATH, MODEL_METADATA_PATH, MODEL_VERSION, PREDICTION_LOG_PATH
from src.features import build_trip_features
from src.monitoring import log_prediction


Weather = Literal["clear", "cloudy", "foggy", "rainy", "snowy", "unknown"]


class TripDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pickup_datetime: datetime
    pickup_latitude: float = Field(ge=39, le=42)
    pickup_longitude: float = Field(ge=-75, le=-72)
    dropoff_latitude: float = Field(ge=39, le=42)
    dropoff_longitude: float = Field(ge=-75, le=-72)
    weather: Weather = "unknown"
    actual_duration: float | None = Field(default=None, gt=0, le=86_400)


def create_app(
    model_path: str | Path = BEST_MODEL_PATH,
    prediction_log_path: str | Path = PREDICTION_LOG_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> FastAPI:
    model_path = Path(model_path)
    try:
        model = joblib.load(model_path)
    except Exception:
        model = None
    metadata_path = Path(metadata_path)
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"model_version": MODEL_VERSION}
    )
    service = FastAPI(title="Ride ETA Prediction API", version=MODEL_VERSION)

    @service.post("/predict")
    def predict_eta(trip: TripDetails) -> dict:
        if model is None:
            raise HTTPException(status_code=503, detail="Model unavailable. Run the training script first.")
        features = build_trip_features(
            trip.pickup_datetime,
            trip.pickup_latitude,
            trip.pickup_longitude,
            trip.dropoff_latitude,
            trip.dropoff_longitude,
            trip.weather,
        )
        eta_seconds = max(float(np.asarray(model.predict(features))[0]), 1.0)
        error = trip.actual_duration - eta_seconds if trip.actual_duration is not None else None
        prediction_id = str(uuid.uuid4())
        log_prediction(
            {
                "prediction_id": prediction_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **features.iloc[0].to_dict(),
                "predicted_eta_seconds": eta_seconds,
                "actual_eta_seconds": trip.actual_duration,
                "prediction_error_seconds": error,
                "model_version": metadata["model_version"],
            },
            prediction_log_path,
        )
        return {
            "prediction_id": prediction_id,
            "predicted_eta_seconds": round(eta_seconds, 2),
            "predicted_eta_minutes": round(eta_seconds / 60, 2),
            "actual_duration": trip.actual_duration,
            "prediction_error": round(error, 2) if error is not None else None,
        }

    return service


app = create_app()
