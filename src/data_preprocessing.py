import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate distance between two lat/lon points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def validate_and_engineer(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Validation
    df = df.dropna()  # remove missing values
    df = df[df['trip_duration'] > 60]  # remove trips < 1 min
    df = df[df['trip_duration'] < 86400]  # remove trips > 24 hrs

    # Feature Engineering
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # Distance feature
    df['trip_distance_km'] = df.apply(
        lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'],
                              row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

    # Traffic feature (rush hour flag)
    df['rush_hour'] = df['hour_of_day'].isin([7,8,9,17,18,19]).astype(int)

    # Synthetic weather feature (for simplicity, random assignment)
    np.random.seed(42)
    df['weather'] = np.random.choice(['sunny', 'rainy', 'cloudy'], size=len(df))

    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    validate_and_engineer("../data/raw/nyc_taxi.csv", "../data/processed/processed_data.csv")
