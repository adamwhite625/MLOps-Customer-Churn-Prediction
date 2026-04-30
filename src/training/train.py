import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import os
import argparse

def train_model(model_obj, X_train, y_train, X_test, y_test, run_name):
    """
    Trains a model and logs metrics to MLflow.
    """
    with mlflow.start_run(run_name=run_name, nested=True):
        model_obj.fit(X_train, y_train)
        
        preds = model_obj.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model_obj, "model")
        
        return f1, model_obj

def preprocess(df):
    # Encode text columns to numeric values expected by the model
    if df['Gender'].dtype == object:
        df['Gender'] = df['Gender'].str.strip().map({'Female': 0, 'Male': 1}).fillna(0).astype(int)
    if df['Subscription Type'].dtype == object:
        df['Subscription Type'] = df['Subscription Type'].str.strip().map({'Basic': 0, 'Standard': 1, 'Premium': 2}).fillna(1).astype(int)
    if df['Contract Length'].dtype == object:
        df['Contract Length'] = df['Contract Length'].str.strip().map({'Monthly': 0, 'Quarterly': 1, 'Annual': 2}).fillna(2).astype(int)

    # Drop any non-feature columns that may be present
    drop_cols = [c for c in ['CustomerID', 'customer_id', 'event_timestamp'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna()
    return df

def run_experiment(data_path):
    df = pd.read_csv(data_path)
    df = preprocess(df)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start a Parent Run to group all trials
    with mlflow.start_run(run_name="Churn_Model_Comparison"):
        # 1. Trial with Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_f1, rf_model = train_model(rf, X_train, y_train, X_test, y_test, "RandomForest_Trial")
        
        # 2. Trial with XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        xgb_f1, xgb_model = train_model(xgb, X_train, y_train, X_test, y_test, "XGBoost_Trial")
        
        # 3. Model Selection & Registration
        best_f1 = max(rf_f1, xgb_f1)
        best_model = rf_model if rf_f1 > xgb_f1 else xgb_model
        best_name = "RandomForest" if rf_f1 > xgb_f1 else "XGBoost"
        
        print(f"Best Model Found: {best_name} with F1: {best_f1:.4f}")
        
        mlflow.log_param("best_model_type", best_name)
        mlflow.log_metric("final_best_f1", best_f1)
        
        # Register the best model specifically
        run_id = mlflow.active_run().info.run_id
        mlflow.sklearn.log_model(best_model, "best_churn_model")
        mlflow.register_model(f"runs:/{run_id}/best_churn_model", "CustomerChurnModel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default="data/processed/churn_cleaned.csv")
    args = parser.parse_args()
    run_experiment(args.input_data)
