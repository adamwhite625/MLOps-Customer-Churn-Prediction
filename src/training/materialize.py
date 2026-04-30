import os
import sys
import glob
import subprocess
import pandas as pd
from datetime import datetime, timedelta, timezone
from feast import FeatureStore

FEATURE_REPO_PATH = "feature_repo/feature_repo"

def prepare_real_data():
    # Find the CSV file pulled via DVC
    csv_files = glob.glob("data/raw/**/*.csv", recursive=True)
    if not csv_files:
        print("No data found. Skipping data preparation.")
        return False

    csv_path = csv_files[0]
    print(f"Processing data from: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna()

    df.rename(columns={"CustomerID": "customer_id"}, inplace=True)

    if df['Gender'].dtype == 'O':
        df['Gender'] = df['Gender'].str.strip().map({"Female": 0, "Male": 1}).fillna(0).astype("int64")
    if df['Subscription Type'].dtype == 'O':
        df['Subscription Type'] = df['Subscription Type'].str.strip().map({"Basic": 0, "Standard": 1, "Premium": 2}).fillna(1).astype("int64")
    if df['Contract Length'].dtype == 'O':
        df['Contract Length'] = df['Contract Length'].str.strip().map({"Monthly": 0, "Quarterly": 1, "Annual": 2}).fillna(2).astype("int64")

    float_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    for c in float_cols:
        df[c] = df[c].astype("float32")

    df["event_timestamp"] = datetime.utcnow()

    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/churn_cleaned.parquet", index=False)
    print("-> Parquet file created successfully!")
    return True

def main():
    prepare_real_data()

    redis_conn = os.environ.get("REDIS_CONNECTION_STRING")
    if not redis_conn:
        print("ERROR: REDIS_CONNECTION_STRING is empty. Check Github Repository Secrets.")
        sys.exit(1)

    print(f"Redis connection string found (length={len(redis_conn)}). Initializing Feast store...")

    # Run feast apply via subprocess (uses feature_store.yaml config)
    print("\n--- Running Feast Apply ---")
    subprocess.run(["feast", "apply"], cwd=FEATURE_REPO_PATH, check=True)

    # Load store and inject the connection string directly to bypass YAML env-var substitution
    print("\n--- Running Feast Materialize (Pushing to Redis) ---")
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    store.config.online_store.connection_string = redis_conn

    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    store.materialize(start_date=start, end_date=end)

    print("\nFeast Materialization Complete! Online Store (Redis) is ready.")

if __name__ == "__main__":
    main()
