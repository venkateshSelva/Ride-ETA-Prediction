"""Shared project configuration.

All paths are resolved from the repository root so commands work regardless of
the current working directory.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "nyc_taxi.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"
WEATHER_DATA_PATH = DATA_DIR / "external" / "nyc_weather_daily.csv"
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
LOG_DIR = PROJECT_ROOT / "logs"
MLFLOW_DB_PATH = LOG_DIR / "mlflow.db"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.jsonl"
DRIFT_SIMULATION_LOG_PATH = LOG_DIR / "drift_predictions.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_QUALITY_REPORT_PATH = REPORT_DIR / "data_quality_report.json"
DATASET_METADATA_PATH = DATA_DIR / "processed" / "dataset_metadata.json"

TARGET = "trip_duration"
NUMERIC_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "trip_distance_km",
    "rush_hour",
]
CATEGORICAL_FEATURES = ["weather"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
WEATHER_CATEGORIES = ("clear", "cloudy", "foggy", "rainy", "snowy", "unknown")
MODEL_VERSION = "2.0.0"
PREPROCESSING_VERSION = "2.0.0"


def ensure_project_directories() -> None:
    """Create directories used by generated project artifacts."""

    for path in (DATA_DIR / "external", DATA_DIR / "processed", MODEL_DIR, LOG_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)
