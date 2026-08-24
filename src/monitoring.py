"""Week 4: simple prediction logging, monitoring, and retraining flag."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODEL_METADATA_PATH, PREDICTION_LOG_PATH, REPORT_DIR


def log_prediction(record: dict, path: str | Path = PREDICTION_LOG_PATH) -> None:
    """Append one prediction record to a JSON Lines file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_predictions(path: str | Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def monitoring_summary(
    path: str | Path = PREDICTION_LOG_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> dict:
    """Report prediction errors and flag a basic rush-hour distribution shift."""

    frame = load_predictions(path)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": len(frame),
        "labeled_prediction_count": 0,
        "rush_hour_share": None,
        "rmse_seconds": None,
        "mae_seconds": None,
        "drift_detected": False,
        "performance_degraded": False,
        "retrain_required": False,
    }
    if frame.empty:
        return summary

    summary["rush_hour_share"] = float(frame["rush_hour"].mean())
    summary["drift_detected"] = bool(summary["rush_hour_share"] > 0.6)
    labeled = frame.dropna(subset=["actual_eta_seconds"])
    summary["labeled_prediction_count"] = len(labeled)
    if not labeled.empty:
        errors = labeled["actual_eta_seconds"] - labeled["predicted_eta_seconds"]
        summary["rmse_seconds"] = float(np.sqrt(np.mean(errors**2)))
        summary["mae_seconds"] = float(np.mean(np.abs(errors)))
        metadata_path = Path(metadata_path)
        if metadata_path.exists():
            baseline_rmse = float(json.loads(metadata_path.read_text())["test_metrics"]["rmse_seconds"])
            summary["performance_degraded"] = bool(summary["rmse_seconds"] > baseline_rmse * 1.25)

    enough_actuals = summary["labeled_prediction_count"] >= 20
    summary["retrain_required"] = bool(
        enough_actuals and (summary["drift_detected"] or summary["performance_degraded"])
    )
    return summary


def save_report(
    prediction_path: str | Path = PREDICTION_LOG_PATH,
    output_path: str | Path = REPORT_DIR / "monitoring_summary.json",
) -> dict:
    summary = monitoring_summary(prediction_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTION_LOG_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "monitoring_summary.json")
    args = parser.parse_args()
    print(json.dumps(save_report(args.predictions, args.output), indent=2))
