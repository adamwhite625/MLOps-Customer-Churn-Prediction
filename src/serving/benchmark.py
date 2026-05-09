"""
Benchmark script to measure read/write latency and throughput
for the MLOps Customer Churn Prediction system.

Measures:
1. Feast Online Store (Redis) read latency
2. Azure ML Endpoint inference latency
3. Blob Storage upload latency
"""

import os
import time
import json
import statistics
import requests
from dotenv import load_dotenv

load_dotenv()

SCORING_URI = os.getenv(
    "AZURE_ML_SCORING_URI",
    "https://churn-endpoint.southeastasia.inference.ml.azure.com/score"
)
PRIMARY_KEY = os.getenv("AZURE_ML_PRIMARY_KEY", "")
REDIS_CONN = os.getenv("REDIS_CONNECTION_STRING", "")

# Sample data matching the model's expected input
SAMPLE_INPUT = {
    "Age": 30, "Gender": 0, "Tenure": 20,
    "Usage Frequency": 10, "Support Calls": 1,
    "Payment Delay": 2, "Subscription Type": 1,
    "Contract Length": 1, "Total Spend": 1000.0,
    "Last Interaction": 20
}


def benchmark_feast_read(num_requests=20):
    """Measures Feast Online Store (Redis) read latency."""
    try:
        import subprocess
        import sys

        # Build path from project root (current working directory)
        project_root = os.getcwd()
        feast_repo = os.path.join(project_root, "feature_repo", "feature_repo")
        feast_repo = os.path.normpath(feast_repo)

        from feast import FeatureStore
        store = FeatureStore(repo_path=feast_repo)
        if REDIS_CONN:
            store.config.online_store.connection_string = REDIS_CONN

        features = [
            "churn_features:Age", "churn_features:Gender",
            "churn_features:Tenure", "churn_features:Usage Frequency",
            "churn_features:Support Calls", "churn_features:Payment Delay",
            "churn_features:Subscription Type", "churn_features:Contract Length",
            "churn_features:Total Spend", "churn_features:Last Interaction"
        ]

        # Warm-up
        store.get_online_features(
            features=features, entity_rows=[{"customer_id": 1}]
        ).to_dict()

        latencies = []
        test_ids = list(range(1, num_requests + 1))

        for cid in test_ids:
            start = time.perf_counter()
            result = store.get_online_features(
                features=features, entity_rows=[{"customer_id": cid}]
            ).to_dict()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        print("=" * 60)
        print("FEAST ONLINE STORE (REDIS) READ LATENCY")
        print("=" * 60)
        print(f"  Number of requests : {num_requests}")
        print(f"  Average latency    : {statistics.mean(latencies):.2f} ms")
        print(f"  Median latency     : {statistics.median(latencies):.2f} ms")
        print(f"  Min latency        : {min(latencies):.2f} ms")
        print(f"  Max latency        : {max(latencies):.2f} ms")
        if len(latencies) > 1:
            print(f"  Std deviation      : {statistics.stdev(latencies):.2f} ms")
        print()
        return latencies

    except Exception as e:
        print(f"[SKIP] Feast benchmark failed: {e}")
        print()
        return []


def benchmark_endpoint(num_requests=20):
    """Measures Azure ML Endpoint inference latency."""
    if not PRIMARY_KEY:
        print("[SKIP] AZURE_ML_PRIMARY_KEY not set, skipping endpoint benchmark.")
        print()
        return []

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PRIMARY_KEY}"
    }
    payload = json.dumps({"data": [SAMPLE_INPUT]})

    # Warm-up request
    try:
        requests.post(SCORING_URI, data=payload, headers=headers, timeout=30)
    except Exception:
        pass

    latencies = []
    errors = 0

    for i in range(num_requests):
        start = time.perf_counter()
        try:
            resp = requests.post(
                SCORING_URI, data=payload, headers=headers, timeout=30
            )
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1
                print(f"  Request {i+1}: HTTP {resp.status_code}")
        except Exception as e:
            errors += 1
            print(f"  Request {i+1}: Error - {e}")

    print("=" * 60)
    print("AZURE ML ENDPOINT INFERENCE LATENCY")
    print("=" * 60)
    print(f"  Number of requests : {num_requests}")
    print(f"  Successful         : {len(latencies)}")
    print(f"  Errors             : {errors}")
    if latencies:
        print(f"  Average latency    : {statistics.mean(latencies):.2f} ms")
        print(f"  Median latency     : {statistics.median(latencies):.2f} ms")
        print(f"  Min latency        : {min(latencies):.2f} ms")
        print(f"  Max latency        : {max(latencies):.2f} ms")
        if len(latencies) > 1:
            print(f"  Std deviation      : {statistics.stdev(latencies):.2f} ms")
    print()
    return latencies


def benchmark_blob_upload():
    """Measures Azure Blob Storage upload latency with a sample CSV."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            print("[SKIP] AZURE_STORAGE_CONNECTION_STRING not set.")
            print()
            return None

        # Create a sample CSV content (5 rows)
        csv_content = (
            "Age,Gender,Tenure,Usage Frequency,Support Calls,"
            "Payment Delay,Subscription Type,Contract Length,"
            "Total Spend,Last Interaction,Churn\n"
        )
        for _ in range(5):
            csv_content += "30,Female,20,10,1,2,Standard,Monthly,1000,20,0\n"

        blob_service = BlobServiceClient.from_connection_string(conn_str)
        container_name = "churn-feedback"
        blob_name = "benchmark_test_upload.csv"

        container_client = blob_service.get_container_client(container_name)

        # Measure upload latency
        start = time.perf_counter()
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(csv_content, overwrite=True)
        upload_ms = (time.perf_counter() - start) * 1000

        # Measure download latency
        start = time.perf_counter()
        downloaded = blob_client.download_blob().readall()
        download_ms = (time.perf_counter() - start) * 1000

        # Clean up
        blob_client.delete_blob()

        size_bytes = len(csv_content.encode("utf-8"))

        print("=" * 60)
        print("AZURE BLOB STORAGE READ/WRITE LATENCY")
        print("=" * 60)
        print(f"  File size          : {size_bytes} bytes")
        print(f"  Upload latency     : {upload_ms:.2f} ms")
        print(f"  Download latency   : {download_ms:.2f} ms")
        print()
        return {"upload_ms": upload_ms, "download_ms": download_ms}

    except Exception as e:
        print(f"[SKIP] Blob Storage benchmark failed: {e}")
        print()
        return None


def main():
    """Runs all benchmarks and prints a summary."""
    print()
    print("*" * 60)
    print("  MLOPS CUSTOMER CHURN - PERFORMANCE BENCHMARK")
    print("*" * 60)
    print()

    feast_latencies = benchmark_feast_read(num_requests=20)
    endpoint_latencies = benchmark_endpoint(num_requests=20)
    blob_result = benchmark_blob_upload()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if feast_latencies:
        print(f"  Feast Redis Avg Read  : {statistics.mean(feast_latencies):.2f} ms")
    if endpoint_latencies:
        print(f"  ML Endpoint Avg Infer : {statistics.mean(endpoint_latencies):.2f} ms")
    if blob_result:
        print(f"  Blob Upload           : {blob_result['upload_ms']:.2f} ms")
        print(f"  Blob Download         : {blob_result['download_ms']:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
