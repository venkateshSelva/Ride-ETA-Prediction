# Ride-ETA-Prediction
A logistics or ride-hailing platform wants to predict delivery time (or ride ETA) based on trip distance, time of day, weather conditions, traffic patterns, and pickup/drop-off location.

# Project Structure
```
Ride-ETA-Prediction/
├── data/
│   ├── raw/              # Original dataset (e.g., NYC taxi trips)
│   └── processed/        # Cleaned and engineered dataset
├── src/
│   ├── data_preprocessing.py  # Data cleaning and feature engineering
│   ├── model_training.py      # Model training and evaluation
│   └── model_serving.py       # API for predictions
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA and visualization
├── models/               # Trained model files
├── logs/                 # MLflow tracking logs
├── .gitignore
├── requirements.txt
└── README.md
```

# Step 1

## What This Script Does
- Loads the raw dataset.
- Validates by removing missing values and unrealistic trip durations.
- Engineers features:
	- Time features (hour_of_day, day_of_week, is_weekend).
	- Distance (trip_distance_km using Haversine formula).
	- Traffic (rush_hour flag).
	- Weather (synthetic categorical feature).
- Saves the cleaned dataset into data/processed/processed_data.csv.
- Logs progress so you see exactly what’s happening.

### How to Run
```bash
python3 src/data_preprocessing.py
```

# Step 2

## What This Script Does
- Loads the processed dataset from Step 1.
- Encodes categorical features (weather).
- Splits into train/test sets.
- Trains Linear Regression and XGBoost.
- Evaluates with RMSE and R².
- Saves the best model into models/best_model.pkl.

### How to Run
```bash

python3 train_model.py

```
