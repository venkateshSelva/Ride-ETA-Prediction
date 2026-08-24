from __future__ import annotations

import json

import pandas as pd

from src.data_preprocessing import validate_and_engineer


def trip(trip_id: str, duration: int = 600, pickup_longitude: float = -73.99) -> dict:
    pickup = pd.Timestamp("2016-01-01 08:00:00")
    return {
        "id": trip_id,
        "vendor_id": 1,
        "pickup_datetime": pickup,
        "dropoff_datetime": pickup + pd.to_timedelta(duration, unit="s"),
        "passenger_count": 1,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": 40.75,
        "dropoff_longitude": -73.98,
        "dropoff_latitude": 40.76,
        "store_and_fwd_flag": "N",
        "trip_duration": duration,
    }


def test_validation_rejects_bad_rows_and_writes_version_metadata(tmp_path):
    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed.csv"
    report = tmp_path / "quality.json"
    metadata = tmp_path / "metadata.json"
    rows = [
        trip("valid-1"),
        trip("too-short", duration=20),
        trip("outside-nyc", pickup_longitude=-120),
        trip("valid-1"),
    ]
    pd.DataFrame(rows).to_csv(raw, index=False)

    result = validate_and_engineer(
        raw,
        processed,
        tmp_path / "missing_weather.csv",
        report,
        metadata,
    )

    output = pd.read_csv(processed)
    assert result["input_rows"] == 4
    assert result["output_rows"] == 1
    assert result["output_duplicate_ids"] == 0
    assert result["weather_coverage_rate"] == 0
    assert output.loc[0, "weather"] == "unknown"
    assert output.loc[0, "rush_hour"] == 1
    assert output.loc[0, "trip_distance_km"] > 0
    assert len(json.loads(metadata.read_text())["processed_sha256"]) == 64


def test_git_lfs_pointer_has_clear_error(tmp_path):
    pointer = tmp_path / "pointer.csv"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n"
    )
    try:
        validate_and_engineer(pointer, tmp_path / "out.csv")
    except RuntimeError as error:
        assert "git lfs pull" in str(error)
    else:
        raise AssertionError("Expected a Git LFS pointer error")
