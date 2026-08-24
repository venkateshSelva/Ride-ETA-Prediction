# Ride-ETA-Prediction
A logistics or ride-hailing platform wants to predict delivery time (or ride ETA) based on trip distance, time of day, weather conditions, traffic patterns, and pickup/drop-off location.

# Project Structure
```
Ride-ETA-Prediction/
├── data/
│   ├── raw/              # Original dataset (NYC taxi trips)
│   └── processed/        # Cleaned and engineered dataset
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── train_model.py      # Model training and evaluation with MLflow
│   ├── main.py            # Main pipeline orchestration script
│   ├── app.py             # FastAPI REST API for model serving
│   ├── monitoring.py      # Monitoring script for drift detection
│   └── simulate_drift.py  # Drift simulation script
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and visualization
│   ├── 02_feature_engineering.ipynb  # Feature engineering experiments
│   └── 03_model_experiments.ipynb    # Model training experiments
├── models/               # Trained model files
├── logs/                 # MLflow tracking logs and database
├── venv/                 # Python virtual environment
├── .gitignore
├── requirements.txt
└── README.md
```

# Pipeline Steps

## Step 1: Data Preprocessing

**Script:** `src/data_preprocessing.py`

**What It Does:**
- Loads the raw dataset from `data/raw/nyc_taxi.csv`
- Validates data by removing missing values and unrealistic trip durations
- Engineers features:
  - Time features (hour_of_day, day_of_week, is_weekend)
  - Distance (trip_distance_km using Haversine formula)
  - Traffic (rush_hour flag)
  - Weather (synthetic categorical feature)
- Saves the cleaned dataset to `data/processed/processed_data.csv`
- Logs progress for monitoring

**How to Run:**
```bash
python src/data_preprocessing.py
```

## Step 2: Model Training with MLflow

**Script:** `src/train_model.py`

**What It Does:**
- Loads the processed dataset from Step 1
- Encodes categorical features (weather)
- Splits data into train/test sets
- Trains two models:
  - Linear Regression (baseline)
  - XGBoost (advanced)
- Evaluates models using RMSE and R² metrics
- **Logs all experiments to MLflow** with:
  - Model parameters
  - Performance metrics
  - Trained models
  - Generated plots (predictions, feature importance)
- Saves the best model to `models/best_model.pkl`
- Generates visualization plots saved to `logs/`

**How to Run:**
```bash
python src/train_model.py
```

## Step 3: MLflow Experiment Tracking

**What It Does:**
- Launches the MLflow UI for experiment visualization
- Allows comparison of different model runs
- Provides detailed metrics, parameters, and artifacts
- Enables model versioning and management

**How to Run:**
```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

Then open http://127.0.0.1:5000 in your browser to view:
- Training runs and metrics comparison
- Model hyperparameters
- Performance plots and visualizations
- Model artifacts and downloads

## Step 4: Monitoring & Drift Detection

**Scripts:** `src/monitoring.py` and `src/simulate_drift.py`

**What It Does:**
- **Drift Simulation:** Simulates data drift scenarios (e.g., festival surge) to test model robustness
- **Monitoring:** Tracks prediction errors from API serving
- **Automatic Retraining:** Triggers model retraining when prediction error exceeds threshold (5 minutes)
- **MLflow Integration:** Logs drift metrics and monitoring results

**How to Run:**
```bash
# Run drift simulation
python src/simulate_drift.py

# Run monitoring check
python src/monitoring.py
```

**Integrated Pipeline:**
The complete pipeline includes automatic drift simulation and monitoring:
```bash
python src/main.py
```

This will:
1. Preprocess data
2. Train models
3. Simulate drift scenarios
4. Check for model drift and trigger retraining if needed

# Suggested Notebooks

## 1. `01_data_exploration.ipynb`

**Purpose:** Explore the raw dataset before preprocessing with MLflow integration.

**Contents:**   
- Load data/raw/nyc_taxi.csv
- Show head, shape, missing values
- Plot distributions (trip duration, passenger count, pickup times)
- **MLflow Integration:**
  - Logs EDA artifacts (summary stats, missing values reports)
  - Saves visualization plots to MLflow
  - Uses "EDA" experiment for tracking exploratory analysis

## 2. `02_feature_engineering.ipynb`

**Purpose:** Prototype new features interactively with MLflow tracking.

**Contents:**   
- Test Haversine distance calculation
- Validate raw data (remove invalid trips)
- Engineer features (hour_of_day, day_of_week, is_weekend, trip_distance_km, rush_hour, weather)
- Visualize feature relationships (distance vs duration, hourly patterns, weekend vs weekday)
- **MLflow Integration:**
  - Logs engineered dataset samples
  - Saves feature relationship plots to MLflow
  - Uses "FeatureEngineering" experiment for tracking feature development

## 3. `03_model_experiments.ipynb`

**Purpose:** Try models interactively before finalizing in train_model.py with MLflow experiment tracking.

**Contents:**   
- Load processed dataset from data/processed/processed_data.csv
- Train Linear Regression with MLflow parameter/metric logging
- Train XGBoost with hyperparameter tracking
- Plot predicted vs actual durations
- Compare RMSE, R² metrics
- Visualize feature importance (XGBoost)
- **MLflow Integration:**
  - Logs model parameters (n_estimators, learning_rate, max_depth)
  - Logs performance metrics (RMSE, R²)
  - Saves trained models to MLflow
  - Logs prediction plots and feature importance visualizations
  - Uses "ModelExperiments" experiment for tracking model development
  - Saves best model locally to models/best_model_notebook.pkl

# How to Use

## Setup
1. Clone the repository and navigate to the project directory
2. Create and activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required packages:
```bash
pip install -r requirements.txt
```

## Run the Complete Pipeline
The main pipeline script automates the entire process from data preprocessing to model training:

```bash
# From project root
source venv/bin/activate
python src/main.py
```

This will:
- Preprocess the raw data
- Train both Linear Regression and XGBoost models
- Log experiments to MLflow
- Save the best model to `models/best_model.pkl`

## Individual Steps

### Step 1: Data Preprocessing
```bash
python src/data_preprocessing.py
```
Cleans and engineers features from the raw dataset.

### Step 2: Model Training
```bash
python src/train_model.py
```
Trains and evaluates models, logs to MLflow.

### Step 3: View MLflow Experiments
```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```
Then open http://127.0.0.1:5000 in your browser to view:
- Training runs and metrics
- Model comparisons
- Hyperparameters
- Generated plots

## Run Jupyter Notebooks
```bash
source venv/bin/activate
jupyter notebook
```
Access the notebooks in the `notebooks/` directory for exploratory analysis and model experimentation.

## File Paths
- Raw data: `data/raw/nyc_taxi.csv`
- Processed data: `data/processed/processed_data.csv`
- Best model: `models/best_model.pkl`
- MLflow database: `logs/mlflow.db`

## API Testing

### Start the API Server
```bash
source venv/bin/activate
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Sample API Test Data

#### Example 1: Sunny weekday, rush hour
```json
{
  "pickup_datetime": "2026-08-24T08:30:00",
  "pickup_latitude": 40.7128,
  "pickup_longitude": -74.0060,
  "dropoff_latitude": 40.7306,
  "dropoff_longitude": -73.9352,
  "weather": "sunny",
  "actual_duration": 18.0
}
```

#### Example 2: Rainy evening, rush hour
```json
{
  "pickup_datetime": "2026-08-24T18:15:00",
  "pickup_latitude": 40.7580,
  "pickup_longitude": -73.9855,
  "dropoff_latitude": 40.7306,
  "dropoff_longitude": -73.9352,
  "weather": "rainy",
  "actual_duration": 25.0
}
```

#### Example 3: Cloudy weekend, off‑peak
```json
{
  "pickup_datetime": "2026-08-23T14:00:00",
  "pickup_latitude": 40.7306,
  "pickup_longitude": -73.9352,
  "dropoff_latitude": 40.6500,
  "dropoff_longitude": -73.9500,
  "weather": "cloudy",
  "actual_duration": 12.0
}
```

#### Example 4: Sunny late night (no rush hour)
```json
{
  "pickup_datetime": "2026-08-24T23:45:00",
  "pickup_latitude": 40.6400,
  "pickup_longitude": -73.7800,
  "dropoff_latitude": 40.7306,
  "dropoff_longitude": -73.9352,
  "weather": "sunny",
  "actual_duration": 30.0
}
```

### Using curl to Test
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2026-08-24T08:30:00",
    "pickup_latitude": 40.7128,
    "pickup_longitude": -74.0060,
    "dropoff_latitude": 40.7306,
    "dropoff_longitude": -73.9352,
    "weather": "sunny"
  }'
```

### Using Swagger UI
1. Open http://localhost:8000/docs
2. Click on the `/predict` endpoint
3. Click "Try it out"
4. Copy and paste any of the JSON examples above
5. Click "Execute"