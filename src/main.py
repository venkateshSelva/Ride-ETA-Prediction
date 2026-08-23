import subprocess
import os
from data_preprocessing import validate_and_engineer
from train_model import train_models

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
    
    # Path for MLflow database
    mlflow_db_path = os.path.join(project_root, "logs/mlflow.db")
    print("\n📊 To view MLflow experiments, run:")
    print(f"   mlflow ui --backend-store-uri sqlite:///{mlflow_db_path}")

if __name__ == "__main__":
    main()
