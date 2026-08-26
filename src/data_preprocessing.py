"""Week 1: validate, enrich, feature-engineer, and version trip data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATASET_METADATA_PATH,
    DATA_QUALITY_REPORT_PATH,
    PREPROCESSING_VERSION,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    WEATHER_DATA_PATH,
    ensure_project_directories,
)
from src.features import add_trip_features, normalize_weather


REQUIRED_COLUMNS = (
    "id",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "store_and_fwd_flag",
    "trip_duration",
)
NUMERIC_COLUMNS = (
    "vendor_id",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_duration",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_materialized_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline().strip()
    if first_line == "version https://git-lfs.github.com/spec/v1":
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not a dataset. Install Git LFS and run `git lfs pull`."
        )


def _load_weather(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    weather = pd.read_csv(path)
    missing = {"date", "weather"}.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather file is missing columns: {sorted(missing)}")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if weather["date"].isna().any() or weather["date"].duplicated().any():
        raise ValueError("Weather dates must be valid and unique")
    weather["weather"] = weather["weather"].map(normalize_weather)
    return weather


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def validate_and_engineer(
    input_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
    weather_path: str | Path = WEATHER_DATA_PATH,
    report_path: str | Path = DATA_QUALITY_REPORT_PATH,
    metadata_path: str | Path = DATASET_METADATA_PATH,
    require_weather: bool = False,
) -> dict[str, Any]:
    """Create the model-ready dataset and return its quality report.

    The expected grain is one completed taxi trip per unique ``id``. Invalid
    rows are rejected with auditable reason counts; the original input is never
    modified.
    """

    ensure_project_directories()
    input_path = Path(input_path)
    output_path = Path(output_path)
    weather_path = Path(weather_path)
    report_path = Path(report_path)
    metadata_path = Path(metadata_path)
    _assert_materialized_csv(input_path)

    raw = pd.read_csv(input_path, low_memory=False)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw dataset is missing required columns: {missing_columns}")

    input_rows = len(raw)
    exact_duplicate_count = int(raw.duplicated().sum())
    raw_null_counts = {column: int(value) for column, value in raw.isna().sum().items()}

    for column in NUMERIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["pickup_datetime"] = pd.to_datetime(raw["pickup_datetime"], errors="coerce")
    raw["dropoff_datetime"] = pd.to_datetime(raw["dropoff_datetime"], errors="coerce")

    required_missing = raw[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    duplicate_id = raw["id"].duplicated(keep="first")
    invalid_timestamp = raw[["pickup_datetime", "dropoff_datetime"]].isna().any(axis=1)
    timestamp_order = raw["dropoff_datetime"] <= raw["pickup_datetime"]
    timestamp_duration = (raw["dropoff_datetime"] - raw["pickup_datetime"]).dt.total_seconds()
    duration_mismatch = (timestamp_duration - raw["trip_duration"]).abs() > 1
    invalid_duration = ~raw["trip_duration"].between(60, 14_400, inclusive="both")
    invalid_passenger = ~raw["passenger_count"].between(1, 6, inclusive="both")
    invalid_coordinate = (
        ~raw["pickup_latitude"].between(-90, 90)
        | ~raw["dropoff_latitude"].between(-90, 90)
        | ~raw["pickup_longitude"].between(-180, 180)
        | ~raw["dropoff_longitude"].between(-180, 180)
    )
    outside_service_area = (
        ~raw["pickup_latitude"].between(39, 42)
        | ~raw["dropoff_latitude"].between(39, 42)
        | ~raw["pickup_longitude"].between(-75, -72)
        | ~raw["dropoff_longitude"].between(-75, -72)
    )

    base_reasons = {
        "missing_required_value": required_missing,
        "duplicate_trip_id": duplicate_id,
        "invalid_timestamp": invalid_timestamp,
        "dropoff_not_after_pickup": timestamp_order,
        "duration_timestamp_mismatch": duration_mismatch,
        "duration_outside_60_to_14400_seconds": invalid_duration,
        "passenger_count_outside_1_to_6": invalid_passenger,
        "coordinate_outside_global_range": invalid_coordinate,
        "coordinate_outside_service_area": outside_service_area,
    }
    base_invalid = np.logical_or.reduce([mask.fillna(True).to_numpy() for mask in base_reasons.values()])
    clean = raw.loc[~base_invalid].copy()

    clean = add_trip_features(clean)
    invalid_distance = ~clean["trip_distance_km"].between(0.05, 100, inclusive="both")
    distance_rejections = int(invalid_distance.sum())
    clean = clean.loc[~invalid_distance].copy()

    weather = _load_weather(weather_path)
    weather_warning = None
    clean["pickup_date"] = clean["pickup_datetime"].dt.strftime("%Y-%m-%d")
    if weather is None:
        if require_weather:
            raise FileNotFoundError(
                f"Historical weather is required but missing: {weather_path}. Run `python -m src.fetch_weather`."
            )
        clean["weather"] = "unknown"
        weather_warning = "Historical weather file missing; weather set to unknown."
    else:
        weather_columns = [column for column in weather.columns if column != "date"]
        clean = clean.drop(columns=[column for column in weather_columns if column in clean.columns])
        clean = clean.merge(weather, how="left", left_on="pickup_date", right_on="date", validate="many_to_one")
        clean = clean.drop(columns=["date"])
        clean["weather"] = clean["weather"].fillna("unknown").map(normalize_weather)

    clean = clean.sort_values(["pickup_datetime", "id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    clean.to_csv(output_path_tmp, index=False)
    output_path_tmp.replace(output_path)

    reason_counts = {name: int(mask.fillna(True).sum()) for name, mask in base_reasons.items()}
    reason_counts["distance_outside_0.05_to_100_km"] = distance_rejections
    weather_counts = {str(k): int(v) for k, v in clean["weather"].value_counts().items()}
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(input_path),
        "grain": "one completed taxi trip per unique id",
        "input_rows": input_rows,
        "output_rows": len(clean),
        "rejected_rows": input_rows - len(clean),
        "rejected_rate": round((input_rows - len(clean)) / input_rows, 6),
        "input_columns": list(raw.columns),
        "raw_null_counts": raw_null_counts,
        "exact_duplicate_rows": exact_duplicate_count,
        "rejection_reason_counts_overlapping": reason_counts,
        "output_duplicate_ids": int(clean["id"].duplicated().sum()),
        "output_null_values": int(clean.isna().sum().sum()),
        "pickup_time_min": clean["pickup_datetime"].min().isoformat(),
        "pickup_time_max": clean["pickup_datetime"].max().isoformat(),
        "weather_counts": weather_counts,
        "weather_coverage_rate": round(1 - weather_counts.get("unknown", 0) / len(clean), 6),
        "weather_warning": weather_warning,
        "status": "passed_with_rejections" if input_rows != len(clean) else "passed",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=_json_scalar) + "\n", encoding="utf-8")

    metadata = {
        "dataset_version": sha256_file(output_path)[:16],
        "preprocessing_version": PREPROCESSING_VERSION,
        "generated_at_utc": report["generated_at_utc"],
        "raw_sha256": sha256_file(input_path),
        "processed_sha256": sha256_file(output_path),
        "rows": len(clean),
        "columns": list(clean.columns),
        "weather_source": "Open-Meteo ERA5 historical archive" if weather is not None else "unavailable",
        "quality_report": str(report_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--weather", type=Path, default=WEATHER_DATA_PATH)
    parser.add_argument("--report", type=Path, default=DATA_QUALITY_REPORT_PATH)
    parser.add_argument("--metadata", type=Path, default=DATASET_METADATA_PATH)
    parser.add_argument("--require-weather", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate_and_engineer(
        args.input,
        args.output,
        args.weather,
        args.report,
        args.metadata,
        args.require_weather,
    )
    print(json.dumps(result, indent=2))
