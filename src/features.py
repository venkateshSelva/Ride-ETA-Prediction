"""Shared feature engineering for training and online inference."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, WEATHER_CATEGORIES


def haversine_km(
    pickup_longitude: Any,
    pickup_latitude: Any,
    dropoff_longitude: Any,
    dropoff_latitude: Any,
) -> Any:
    """Return great-circle distance in kilometres for scalars or arrays."""

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude],
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0088 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def normalize_weather(value: str | None) -> str:
    """Normalize a weather value to the model's stable category contract."""

    weather = (value or "unknown").strip().lower()
    return weather if weather in WEATHER_CATEGORIES else "unknown"


def add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time and distance features to a trip dataframe."""

    result = df.copy()
    pickup = pd.to_datetime(result["pickup_datetime"], errors="coerce")
    result["hour_of_day"] = pickup.dt.hour.astype("int8")
    result["day_of_week"] = pickup.dt.weekday.astype("int8")
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("int8")
    result["rush_hour"] = result["hour_of_day"].isin([7, 8, 9, 16, 17, 18, 19]).astype("int8")
    result["trip_distance_km"] = haversine_km(
        result["pickup_longitude"].to_numpy(),
        result["pickup_latitude"].to_numpy(),
        result["dropoff_longitude"].to_numpy(),
        result["dropoff_latitude"].to_numpy(),
    )
    if "weather" not in result:
        result["weather"] = "unknown"
    result["weather"] = result["weather"].map(normalize_weather)
    return result


def build_trip_features(
    pickup_datetime: datetime,
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    weather: str,
) -> pd.DataFrame:
    """Build a one-row feature frame using exactly the training contract."""

    hour = pickup_datetime.hour
    weekday = pickup_datetime.weekday()
    row = {
        "hour_of_day": hour,
        "day_of_week": weekday,
        "is_weekend": int(weekday in (5, 6)),
        "trip_distance_km": float(
            haversine_km(
                pickup_longitude,
                pickup_latitude,
                dropoff_longitude,
                dropoff_latitude,
            )
        ),
        "rush_hour": int(hour in (7, 8, 9, 16, 17, 18, 19)),
        "weather": normalize_weather(weather),
    }
    return pd.DataFrame([row], columns=MODEL_FEATURES)
