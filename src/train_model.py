import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

# Set MLflow tracking directory to SQLite backend inside logs/
# Use absolute path to ensure it works from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
mlflow_db_path = os.path.join(project_root, "logs/mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
mlflow.set_experiment("ETA_Prediction")

def train_models(input_path, model_output_path):   
    print("🔹 Loading dataset...")
    df = pd.read_csv(input_path)
    df = pd.get_dummies(df, columns=['weather'], drop_first=True)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)

    X = df[['hour_of_day', 'day_of_week', 'is_weekend',
            'trip_distance_km', 'rush_hour'] + 
            [col for col in df.columns if col.startswith('weather_')]]
    y = df['trip_duration']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Linear Regression ---
    with mlflow.start_run(run_name="LinearRegression"):
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred_lr = lin_reg.predict(X_test)

        lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        lr_r2 = r2_score(y_test, y_pred_lr)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", lr_rmse)
        mlflow.log_metric("r2", lr_r2)
        mlflow.sklearn.log_model(lin_reg, "linear_regression_model")

        # Save model + plot as artifacts
        linear_reg_path = os.path.join(project_root, "models/linear_reg.pkl")
        joblib.dump(lin_reg, linear_reg_path)
        mlflow.log_artifact(linear_reg_path, artifact_path="models")

        plt.figure(figsize=(8,5))
        sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.3)
        plt.xlabel("Actual Duration")
        plt.ylabel("Predicted Duration")
        plt.title("Linear Regression Predictions")
        plot_path = os.path.join(project_root, "logs/lr_predictions.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    # --- XGBoost ---
    with mlflow.start_run(run_name="XGBoost"):
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)

        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_r2 = r2_score(y_test, y_pred_xgb)

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 6)
        mlflow.log_metric("rmse", xgb_rmse)
        mlflow.log_metric("r2", xgb_r2)
        mlflow.xgboost.log_model(xgb, "xgboost_model")

        # Save model + plot as artifacts
        xgb_model_path = os.path.join(project_root, "models/xgb_model.pkl")
        joblib.dump(xgb, xgb_model_path)
        mlflow.log_artifact(xgb_model_path, artifact_path="models")

        plt.figure(figsize=(8,5))
        sns.barplot(x=xgb.feature_importances_, y=X.columns)
        plt.title("XGBoost Feature Importance")
        plot_path = os.path.join(project_root, "logs/xgb_feature_importance.png")
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path, artifact_path="plots")

    # --- Save best model locally ---
    if xgb_rmse < lr_rmse:
        joblib.dump(xgb, model_output_path)
        print(f"✅ Best model (XGBoost) saved to {model_output_path}")
    else:
        joblib.dump(lin_reg, model_output_path)
        print(f"✅ Best model (Linear Regression) saved to {model_output_path}")

if __name__ == "__main__":
    print("🚀 Starting model training...")
    # Use the same project_root logic for standalone execution
    processed_data_path = os.path.join(project_root, "data/processed/processed_data.csv")
    model_output_path = os.path.join(project_root, "models/best_model.pkl")
    train_models(processed_data_path, model_output_path)
    print("🎯 Training completed!")
