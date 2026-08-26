"""Run the full four-week project pipeline."""

from __future__ import annotations

import argparse
import json

from src.config import WEATHER_DATA_PATH
from src.data_preprocessing import validate_and_engineer
from src.fetch_weather import fetch_weather
from src.simulate_drift import simulate_drift
from src.train_model import train_models


def main(*, refresh_weather: bool = False, max_rows: int | None = None, run_drift: bool = False) -> dict:
    if refresh_weather or not WEATHER_DATA_PATH.exists():
        weather = fetch_weather()
        weather_rows = len(weather)
    else:
        weather_rows = None
    quality = validate_and_engineer(require_weather=True)
    model_report = train_models(max_rows=max_rows)
    result = {
        "weather_rows_downloaded": weather_rows,
        "data_quality": quality,
        "model_training": model_report,
    }
    if run_drift:
        result["drift_simulation"] = simulate_drift()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-weather", action="store_true")
    parser.add_argument("--max-rows", type=int, help="Optional development-only training row cap")
    parser.add_argument("--simulate-drift", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            main(
                refresh_weather=arguments.refresh_weather,
                max_rows=arguments.max_rows,
                run_drift=arguments.simulate_drift,
            ),
            indent=2,
        )
    )
