"""Week 2: train, compare, and track two ETA models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    DATASET_METADATA_PATH,
    MLFLOW_DB_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    REPORT_DIR,
    TARGET,
    ensure_project_directories,
)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "weather",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    rmse = float(mean_squared_error(actual, predicted) ** 0.5)
    mae = float(mean_absolute_error(actual, predicted))
    return {
        "rmse_seconds": rmse,
        "rmse_minutes": rmse / 60,
        "mae_seconds": mae,
        "mae_minutes": mae / 60,
        "r2": float(r2_score(actual, predicted)),
    }


def dataset_version() -> str:
    if not DATASET_METADATA_PATH.exists():
        return "unknown"
    return str(json.loads(DATASET_METADATA_PATH.read_text(encoding="utf-8"))["dataset_version"])


def train_models(
    input_path: str | Path = PROCESSED_DATA_PATH,
    model_output_path: str | Path = BEST_MODEL_PATH,
    *,
    max_rows: int | None = None,
    enable_mlflow: bool = True,
    metadata_output_path: str | Path = MODEL_METADATA_PATH,
    comparison_output_path: str | Path = REPORT_DIR / "model_comparison.json",
) -> dict:
    """Compare Linear Regression and Gradient Boosting on one train/test split."""

    ensure_project_directories()
    frame = pd.read_csv(input_path, low_memory=False)
    missing = sorted(set(MODEL_FEATURES + [TARGET]).difference(frame.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        frame[MODEL_FEATURES], frame[TARGET], test_size=0.2, random_state=42
    )
    candidates = {
        "linear_regression": (LinearRegression(), {}),
        "gradient_boosting": (
            HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42),
            {"max_iter": 200, "learning_rate": 0.08},
        ),
    }
    if enable_mlflow:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        mlflow.set_experiment("ETA_Prediction_Simple")

    results = []
    trained_models = {}
    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    for name, (regressor, parameters) in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("regressor", regressor)])
        pipeline.fit(X_train, y_train)
        predicted = np.maximum(pipeline.predict(X_test), 1.0)
        result = {"model": name, "parameters": parameters, **metrics(y_test, predicted)}
        results.append(result)
        trained_models[name] = pipeline

        candidate_path = model_output_path.parent / f"{name}.pkl"
        joblib.dump(pipeline, candidate_path)
        if enable_mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_name", name)
                mlflow.log_param("dataset_version", dataset_version())
                mlflow.log_params(parameters)
                mlflow.log_metrics({key: value for key, value in result.items() if isinstance(value, float)})

    results.sort(key=lambda item: item["rmse_seconds"])
    best_name = results[0]["model"]
    joblib.dump(trained_models[best_name], model_output_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": best_name,
        "target": TARGET,
        "target_unit": "seconds",
        "features": MODEL_FEATURES,
        "dataset_version": dataset_version(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_metrics": results[0],
    }
    metadata_output_path = Path(metadata_output_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = {"selected_model": best_name, "models": results}
    comparison_output_path = Path(comparison_output_path)
    comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--output", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            train_models(
                args.input,
                args.output,
                max_rows=args.max_rows,
                enable_mlflow=not args.no_mlflow,
            ),
            indent=2,
        )
    )
