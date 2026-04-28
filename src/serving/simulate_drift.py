import requests
import json
import time
import os

SCORING_URI = os.getenv("AZURE_ML_SCORING_URI", "https://churn-endpoint.southeastasia.inference.ml.azure.com/score")
PRIMARY_KEY = os.getenv("AZURE_ML_PRIMARY_KEY")

def send_request(data):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {PRIMARY_KEY}"}
    response = requests.post(SCORING_URI, data=json.dumps({"data": data}), headers=headers)
    return response.status_code

if __name__ == "__main__":
    # Kịch bản 1: Dữ liệu bình thường (Khớp với tập train)
    normal_data = [{"Age": 30, "Gender": 0, "Tenure": 20, "Usage Frequency": 10, "Support Calls": 1, "Payment Delay": 2, "Subscription Type": 1, "Contract Length": 1, "Total Spend": 1000.0, "Last Interaction": 20}]
    
    # Kịch bản 2: Dữ liệu bị DRIFT (Hành vi thay đổi - Payment Delay tăng vọt)
    drifted_data = [{"Age": 30, "Gender": 0, "Tenure": 20, "Usage Frequency": 10, "Support Calls": 1, "Payment Delay": 100, "Subscription Type": 1, "Contract Length": 1, "Total Spend": 1000.0, "Last Interaction": 20}]

    print("--- Đang gửi dữ liệu bình thường ---")
    for _ in range(5):
        send_request(normal_data)
        print(".", end="", flush=True)
        time.sleep(1)

    print("\n--- Đang gửi dữ liệu bị DRIFT (Cảnh báo: Payment Delay tăng bất thường) ---")
    for _ in range(10):
        send_request(drifted_data)
        print("!", end="", flush=True)
        time.sleep(1)
    
    print("\n\nHoàn tất! Bây giờ bạn có thể vào Azure Portal -> Endpoint -> Monitoring để xem biểu đồ Drift.")
