from datetime import datetime

from src.features import build_trip_features, haversine_km, normalize_weather


def test_online_feature_contract_and_units():
    features = build_trip_features(
        datetime(2026, 8, 24, 18, 30),
        40.7580,
        -73.9855,
        40.7308,
        -73.9973,
        "RAINY",
    )
    assert features.loc[0, "hour_of_day"] == 18
    assert features.loc[0, "rush_hour"] == 1
    assert features.loc[0, "weather"] == "rainy"
    assert 3 < features.loc[0, "trip_distance_km"] < 4
    assert normalize_weather("storm") == "unknown"
    assert haversine_km(-73.9855, 40.7580, -73.9855, 40.7580) == 0
