import mlflow
import os
import numpy as np
from subprocess import call

# Get the project root directory (parent of src)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'logs/mlflow.db')}")
client = mlflow.tracking.MlflowClient()

# Get Serving experiment
experiment = client.get_experiment_by_name("Serving")
runs = client.search_runs(experiment.experiment_id)

# Collect prediction errors
errors = []
for run in runs:
    if "prediction_error" in run.data.metrics:
        errors.append(run.data.metrics["prediction_error"])

if errors:
    avg_error = np.mean(errors)
    print("Average prediction error:", avg_error)

    # Trigger retraining if error > threshold
    if avg_error > 5.0:  # threshold in minutes
        print("⚠️ Drift detected! Triggering retraining...")
        call(["python", os.path.join(project_root, "src/train_model.py")])
else:
    print("ℹ️ No prediction errors logged yet.")