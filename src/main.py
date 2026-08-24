import subprocess
import os
from data_preprocessing import validate_and_engineer
from train_model import train_models

def simulate_drift(processed_data_path, project_root):
    print("\n🔹 Step 3: Simulating Drift (Festival Surge)")
    import pandas as pd
    import mlflow
    
    df = pd.read_csv(processed_data_path)
    df['actual_duration'] = df['trip_duration'] * 1.5  # simulate surge

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    drift_sample_path = os.path.join(logs_dir, "drift_sample.csv")
    df.head(100).to_csv(drift_sample_path, index=False)

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'logs/mlflow.db')}")
    mlflow.set_experiment("DriftSimulation")

    with mlflow.start_run(run_name="Festival_Surge"):
        mlflow.log_param("drift_type", "festival surge")
        mlflow.log_metric("avg_duration_increase", 1.5)
        mlflow.log_artifact(drift_sample_path, artifact_path="drift")

def monitor_and_retrain(project_root):
    print("\n🔹 Step 4: Monitoring & Retraining Trigger")
    import mlflow
    import numpy as np
    
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'logs/mlflow.db')}")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Serving")

    if experiment:
        runs = client.search_runs(experiment.experiment_id)
        errors = [r.data.metrics["prediction_error"] for r in runs if "prediction_error" in r.data.metrics]
        if errors:
            avg_error = np.mean(errors)
            print("📊 Average prediction error:", avg_error)
            if avg_error > 5.0:  # threshold in minutes
                print("⚠️ Drift detected! Triggering retraining...")
                subprocess.call(["python", os.path.join(project_root, "src/train_model.py")])
        else:
            print("ℹ️ No prediction errors logged yet.")
    else:
        print("ℹ️ No Serving experiment found.")

def main():
    print("🚀 Starting Ride ETA Prediction pipeline...")

    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths relative to project root
    raw_data_path = os.path.join(project_root, "data/raw/nyc_taxi.csv")
    processed_data_path = os.path.join(project_root, "data/processed/processed_data.csv")
    model_output_path = os.path.join(project_root, "models/best_model.pkl")

    # Step 1: Preprocessing
    print("\n🔹 Step 1: Data Preprocessing")
    validate_and_engineer(raw_data_path, processed_data_path)

    # Step 2: Model Training + Experiment Tracking
    print("\n🔹 Step 2: Model Training & MLflow Tracking")
    train_models(processed_data_path, model_output_path)

    print("\n✅ Pipeline complete. Best model saved at:", model_output_path)
    
    # Step 3: Drift Simulation
    simulate_drift(processed_data_path, project_root)

    # Step 4: Monitoring & Retraining
    monitor_and_retrain(project_root)
    
    # Path for MLflow database
    mlflow_db_path = os.path.join(project_root, "logs/mlflow.db")
    print("\n📊 To view MLflow experiments, run:")
    print(f"   mlflow ui --backend-store-uri sqlite:///{mlflow_db_path}")

if __name__ == "__main__":
    main()
