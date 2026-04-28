import requests
import json
import os

# Load endpoint config from environment variables (never hardcode secrets)
SCORING_URI = os.getenv("AZURE_ML_SCORING_URI", "https://churn-endpoint.southeastasia.inference.ml.azure.com/score")
PRIMARY_KEY = os.getenv("AZURE_ML_PRIMARY_KEY")

def test_endpoint(data):
    """Sends a prediction request to the deployed endpoint and returns the response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PRIMARY_KEY}"
    }
    payload = json.dumps({"data": data})
    response = requests.post(SCORING_URI, data=payload, headers=headers)
    return response.status_code, response.json()

if __name__ == "__main__":
    # Sample customer data reflecting the required feature names
    sample_customers = [
        {
            "Age": 35,
            "Gender": 0,             # 0: Female, 1: Male (ví dụ)
            "Tenure": 24,
            "Usage Frequency": 15,
            "Support Calls": 1,
            "Payment Delay": 5,
            "Subscription Type": 1,  # 0: Basic, 1: Standard, 2: Premium
            "Contract Length": 2,    # 0: Monthly, 1: Quarterly, 2: Annual
            "Total Spend": 1500.0,
            "Last Interaction": 30
        },
        {
            "Age": 55,
            "Gender": 1,
            "Tenure": 2,
            "Usage Frequency": 3,
            "Support Calls": 8,
            "Payment Delay": 25,
            "Subscription Type": 0,
            "Contract Length": 0,
            "Total Spend": 200.0,
            "Last Interaction": 5
        }
    ]

    print("Sending prediction request to Azure ML Endpoint...")
    print(f"URL: {SCORING_URI}\n")

    status_code, result = test_endpoint(sample_customers)

    print(f"HTTP Status Code: {status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")

    if status_code == 200 and result.get("status") == "success":
        predictions = result.get("predictions", [])
        print("\nPrediction Results:")
        for i, pred in enumerate(predictions):
            label = "Churn (1)" if pred == 1 else "Loyal (0)"
            print(f"  Customer {i + 1}: {label}")
    else:
        print(f"\nError Details: {result.get('message')}")
