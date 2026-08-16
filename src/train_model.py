import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def train_models(input_path, model_output_path):
    # Load processed dataset
    df = pd.read_csv(input_path)

    # Encode categorical features (weather)
    df = pd.get_dummies(df, columns=['weather'], drop_first=True)

    # Features and target
    X = df[['hour_of_day', 'day_of_week', 'is_weekend',
            'trip_distance_km', 'rush_hour'] + 
            [col for col in df.columns if col.startswith('weather_')]]
    y = df['trip_duration']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Linear Regression ---
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)

    print(f"Linear Regression RMSE: {lr_rmse:.2f}, R2: {lr_r2:.2f}")

    # --- Gradient Boosting (XGBoost) ---
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2 = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost RMSE: {xgb_rmse:.2f}, R2: {xgb_r2:.2f}")

    # --- Select Best Model ---
    if xgb_rmse < lr_rmse:
        best_model = xgb
        model_name = "XGBoost"
    else:
        best_model = lin_reg
        model_name = "LinearRegression"

    # Save best model
    joblib.dump(best_model, model_output_path)
    print(f"Best model ({model_name}) saved to {model_output_path}")

if __name__ == "__main__":
    train_models("data/processed/processed_data.csv", "models/best_model.pkl")
