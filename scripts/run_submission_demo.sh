#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Follow the README Setup section first."
  exit 1
fi

if [[ ! -f data/external/nyc_weather_daily.csv ]]; then
  echo "Missing historical weather. Run: .venv/bin/python -m src.fetch_weather"
  exit 1
fi

EVIDENCE_DIR="$PROJECT_DIR/submission_evidence"
mkdir -p "$EVIDENCE_DIR"

echo "[1/6] Running tests"
LOKY_MAX_CPU_COUNT=8 .venv/bin/pytest --disable-warnings 2>&1 \
  | tee "$EVIDENCE_DIR/01_tests.txt"

echo "[2/6] Running data validation and feature engineering"
.venv/bin/python -m src.data_preprocessing --require-weather 2>&1 \
  | tee "$EVIDENCE_DIR/02_data_preprocessing.txt"

echo "[3/6] Training and comparing two models"
LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m src.train_model 2>&1 \
  | tee "$EVIDENCE_DIR/03_model_training.txt"

echo "[4/6] Starting the API and sending a sample request"
LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m uvicorn src.app:app --port 8001 \
  >"$EVIDENCE_DIR/04_api_server.txt" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in {1..20}; do
  if curl --silent --fail http://127.0.0.1:8001/openapi.json >/dev/null; then
    break
  fi
  sleep 1
done

curl --silent --show-error --fail -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "pickup_datetime": "2016-06-15T18:30:00",
    "pickup_latitude": 40.7580,
    "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.7308,
    "dropoff_longitude": -73.9973,
    "weather": "rainy",
    "actual_duration": 900
  }' | .venv/bin/python -m json.tool | tee "$EVIDENCE_DIR/04_api_response.json"

cleanup
trap - EXIT

echo "[5/6] Simulating rush-hour drift"
LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m src.simulate_drift 2>&1 \
  | tee "$EVIDENCE_DIR/05_drift_simulation.txt"

echo "[6/6] Running monitoring and retraining check"
.venv/bin/python -m src.monitoring --predictions logs/drift_predictions.jsonl 2>&1 \
  | tee "$EVIDENCE_DIR/06_monitoring.txt"

cp reports/data_quality_report.json "$EVIDENCE_DIR/data_quality_report.json"
cp reports/model_comparison.json "$EVIDENCE_DIR/model_comparison.json"
cp reports/drift_simulation_report.json "$EVIDENCE_DIR/drift_simulation_report.json"

echo "Submission demo completed. Evidence saved in: $EVIDENCE_DIR"
