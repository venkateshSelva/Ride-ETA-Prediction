"""Week 4: simulate a rush-hour surge and run the monitoring check."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    DRIFT_SIMULATION_LOG_PATH,
    MODEL_FEATURES,
    MODEL_METADATA_PATH,
    PROCESSED_DATA_PATH,
    REPORT_DIR,
)
from src.monitoring import log_prediction, monitoring_summary


def simulate_drift(
    input_path: str | Path = PROCESSED_DATA_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    log_path: str | Path = DRIFT_SIMULATION_LOG_PATH,
    samples: int = 100,
) -> dict:
    frame = pd.read_csv(input_path, nrows=max(2_000, samples * 5)).sample(samples, random_state=42)
    frame["hour_of_day"] = 18
    frame["rush_hour"] = 1
    model = joblib.load(model_path)
    predicted = np.maximum(model.predict(frame[MODEL_FEATURES]), 1.0)
    rng = np.random.default_rng(42)
    actual = predicted * rng.normal(1.5, 0.08, size=len(frame))

    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
    for (_, row), prediction, outcome in zip(frame.iterrows(), predicted, actual, strict=True):
        log_prediction(
            {
                "prediction_id": str(uuid.uuid4()),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                **{feature: row[feature] for feature in MODEL_FEATURES},
                "predicted_eta_seconds": float(prediction),
                "actual_eta_seconds": float(outcome),
                "prediction_error_seconds": float(outcome - prediction),
                "model_version": "simulation",
            },
            log_path,
        )

    summary = monitoring_summary(log_path, MODEL_METADATA_PATH)
    report = {"scenario": "rush-hour/festival surge", "samples": samples, **summary}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "drift_simulation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(simulate_drift(samples=args.samples), indent=2))
