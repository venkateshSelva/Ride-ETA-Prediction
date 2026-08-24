import pandas as pd
import numpy as np
import mlflow
import os

# Get the project root directory (parent of src)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'logs/mlflow.db')}")
mlflow.set_experiment("DriftSimulation")

# Load processed dataset
df = pd.read_csv(os.path.join(project_root, "data/processed/processed_data.csv"))

# Simulate festival surge: multiply durations by 1.5
df['actual_duration'] = df['trip_duration'] * 1.5

# Create logs directory if it doesn't exist
logs_dir = os.path.join(project_root, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Log drifted dataset
with mlflow.start_run(run_name="Festival_Surge"):
    df.head(100).to_csv(os.path.join(project_root, "logs/drift_sample.csv"), index=False)
    mlflow.log_artifact(os.path.join(project_root, "logs/drift_sample.csv"), artifact_path="drift")
    mlflow.log_param("drift_type", "festival surge")
    mlflow.log_metric("avg_duration_increase", 1.5)