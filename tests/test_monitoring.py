import json

from src.monitoring import log_prediction, monitoring_summary


def test_rush_hour_surge_sets_retraining_flag(tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"test_metrics": {"rmse_seconds": 100}}))
    for i in range(20):
        log_prediction(
            {
                "prediction_id": str(i),
                "rush_hour": 1,
                "predicted_eta_seconds": 600,
                "actual_eta_seconds": 900,
            },
            log_path,
        )
    summary = monitoring_summary(log_path, metadata_path)
    assert summary["drift_detected"] is True
    assert summary["performance_degraded"] is True
    assert summary["retrain_required"] is True
