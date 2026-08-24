from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES
from src.train_model import train_models


def test_model_comparison_and_packaging(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for i in range(360):
        distance = float(rng.uniform(0.2, 20))
        hour = i % 24
        rainy = i % 5 == 0
        rows.append(
            {
                "pickup_datetime": start + pd.to_timedelta(i, unit="m"),
                "hour_of_day": hour,
                "day_of_week": (i // 24) % 7,
                "is_weekend": int((i // 24) % 7 >= 5),
                "trip_distance_km": distance,
                "rush_hour": int(hour in (7, 8, 9, 16, 17, 18, 19)),
                "weather": "rainy" if rainy else "clear",
                "trip_duration": 120 + distance * 170 + rainy * 90 + rng.normal(0, 20),
            }
        )
    input_path = tmp_path / "processed.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    report = train_models(
        input_path,
        tmp_path / "best.pkl",
        enable_mlflow=False,
        metadata_output_path=tmp_path / "metadata.json",
        comparison_output_path=tmp_path / "comparison.json",
    )
    assert len(report["models"]) == 2
    assert report["models"][0]["r2"] > 0.8
    assert (tmp_path / "best.pkl").exists()
    assert set(MODEL_FEATURES).issubset(pd.DataFrame(rows).columns)
