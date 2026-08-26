from __future__ import annotations

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.app import create_app
from src.config import MODEL_FEATURES
from src.monitoring import monitoring_summary
from src.train_model import make_preprocessor


def fitted_model(path):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(120):
        distance = float(rng.uniform(0.5, 15))
        rows.append(
            {
                "hour_of_day": i % 24,
                "day_of_week": i % 7,
                "is_weekend": int(i % 7 >= 5),
                "trip_distance_km": distance,
                "rush_hour": int(i % 24 in (7, 8, 9, 16, 17, 18, 19)),
                "weather": "rainy" if i % 4 == 0 else "clear",
                "trip_duration": 180 + distance * 150,
            }
        )
    frame = pd.DataFrame(rows)
    pipeline = Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("regressor", HistGradientBoostingRegressor(max_iter=30, random_state=42)),
        ]
    )
    pipeline.fit(frame[MODEL_FEATURES], frame["trip_duration"])
    joblib.dump(pipeline, path)


def test_prediction_actual_and_monitoring_flow(tmp_path):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    fitted_model(model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "test",
                "target_unit": "seconds",
                "test_metrics": {"rmse_seconds": 100},
            }
        )
    )
    prediction_log = tmp_path / "predictions.jsonl"
    app = create_app(model_path, prediction_log, metadata_path)
    client = TestClient(app)

    payload = {
        "pickup_datetime": datetime.now().isoformat(),
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7308,
        "dropoff_longitude": -73.9973,
        "weather": "rainy",
        "actual_duration": 900,
    }
    prediction = client.post("/predict", json=payload)
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["predicted_eta_minutes"] == round(body["predicted_eta_seconds"] / 60, 2)

    assert body["actual_duration"] == 900
    assert body["prediction_error"] is not None
    summary = monitoring_summary(prediction_log, metadata_path)
    assert summary["prediction_count"] == 1
    assert summary["labeled_prediction_count"] == 1


def test_request_validation_rejects_impossible_coordinates(tmp_path):
    app = create_app(tmp_path / "missing.pkl", tmp_path / "predictions.jsonl")
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "pickup_datetime": datetime.now().isoformat(),
            "pickup_latitude": 95,
            "pickup_longitude": -73.9,
            "dropoff_latitude": 40.7,
            "dropoff_longitude": -73.9,
            "weather": "clear",
        },
    )
    assert response.status_code == 422
