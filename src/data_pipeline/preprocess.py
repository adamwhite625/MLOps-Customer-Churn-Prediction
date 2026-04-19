import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path):
    """
    Reads the raw CSV file from the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Handles missing values and removes non-predictive columns.
    """
    df = df.dropna()
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    return df

def encode_features(df):
    """
    Converts categorical string features into numerical format.
    """
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(df):
    """
    Normalizes numerical features using standard scaling.
    """
    scaler = StandardScaler()
    target_col = 'Churn'
    if target_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col)
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def save_processed_data(df, csv_path, parquet_path):
    """
    Saves the cleaned dataframe as both CSV and Parquet formats.
    Parquet is required by Feast Feature Store.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Feast requires an event_timestamp column
    df["event_timestamp"] = pd.Timestamp.now()
    df.to_parquet(parquet_path, index=False)

def run_preprocessing():
    """
    Orchestrates the end-to-end data preprocessing pipeline.
    """
    input_path = "data/raw/customer_churn_dataset-training-master.csv"
    csv_output = "data/processed/churn_cleaned.csv"
    parquet_output = "data/processed/churn_cleaned.parquet"

    try:
        data = load_data(input_path)
        data = clean_data(data)
        data = encode_features(data)
        data = scale_features(data)
        save_processed_data(data, csv_output, parquet_output)
        print(f"Successfully saved CSV to {csv_output}")
        print(f"Successfully saved Parquet to {parquet_output}")
    except Exception as e:
        print(f"Preprocessing failed: {str(e)}")

if __name__ == "__main__":
    run_preprocessing()
