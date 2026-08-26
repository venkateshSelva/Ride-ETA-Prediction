"""Download daily historical NYC weather used by Week 1 enrichment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.config import RAW_DATA_PATH, WEATHER_DATA_PATH, ensure_project_directories


API_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060


def weather_category(code: int | float | None, precipitation: float, snowfall: float) -> str:
    if snowfall > 0.05 or (code is not None and 71 <= int(code) <= 77):
        return "snowy"
    if precipitation > 0.05 or (code is not None and int(code) >= 51):
        return "rainy"
    if code is not None and 45 <= int(code) <= 48:
        return "foggy"
    if code is not None and int(code) >= 2:
        return "cloudy"
    return "clear"


def raw_date_range(raw_path: Path) -> tuple[str, str]:
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    for chunk in pd.read_csv(raw_path, usecols=["pickup_datetime"], chunksize=250_000):
        values = pd.to_datetime(chunk["pickup_datetime"], errors="coerce")
        chunk_min, chunk_max = values.min(), values.max()
        earliest = chunk_min if earliest is None else min(earliest, chunk_min)
        latest = chunk_max if latest is None else max(latest, chunk_max)
    if earliest is None or latest is None or pd.isna(earliest) or pd.isna(latest):
        raise ValueError("Could not determine pickup date range from raw data")
    return earliest.strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d")


def fetch_weather(raw_path: Path = RAW_DATA_PATH, output_path: Path = WEATHER_DATA_PATH) -> pd.DataFrame:
    ensure_project_directories()
    start_date, end_date = raw_date_range(raw_path)
    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code,temperature_2m_mean,precipitation_sum,snowfall_sum",
        "timezone": "America/New_York",
    }
    url = f"{API_ENDPOINT}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ride-eta-prediction/2.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload = json.load(response)
    if "daily" not in payload:
        raise RuntimeError(f"Weather API returned no daily data: {payload}")

    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    daily["weather"] = [
        weather_category(code, float(rain or 0), float(snow or 0))
        for code, rain, snow in zip(
            daily["weather_code"],
            daily["precipitation_sum"],
            daily["snowfall_sum"],
            strict=True,
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "source": "Open-Meteo Historical Weather API (ERA5 reanalysis)",
                "source_url": url,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "location": {"latitude": NYC_LATITUDE, "longitude": NYC_LONGITUDE},
                "date_range": {"start": start_date, "end": end_date},
                "rows": len(daily),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=WEATHER_DATA_PATH)
    arguments = parser.parse_args()
    result = fetch_weather(arguments.raw, arguments.output)
    print(f"Saved {len(result)} daily weather rows to {arguments.output}")
