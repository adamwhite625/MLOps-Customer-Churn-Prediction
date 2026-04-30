import pandas as pd
from datetime import datetime, timedelta
import subprocess
import os
import glob

def prepare_real_data():
    # Tìm file CSV đã được tải về từ Azure ML
    csv_files = glob.glob("data/raw/**/*.csv", recursive=True)
    if not csv_files:
        print("Không tìm thấy dữ liệu từ Azure ML. Khởi tạo mock data...")
        return False
        
    csv_path = csv_files[0]
    print(f"Đang xử lý dữ liệu thật từ: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Chuẩn hóa cột giống y hệt file Training
    df.rename(columns={"CustomerID": "customer_id"}, inplace=True)
    
    # Mapping các cột Text sang Số (Vì Feast schema và Model yêu cầu Int64)
    if df['Gender'].dtype == 'O':
        df['Gender'] = df['Gender'].map({"Female": 0, "Male": 1}).astype("int64")
    if df['Subscription Type'].dtype == 'O':
        df['Subscription Type'] = df['Subscription Type'].map({"Basic": 0, "Standard": 1, "Premium": 2}).astype("int64")
    if df['Contract Length'].dtype == 'O':
        df['Contract Length'] = df['Contract Length'].map({"Monthly": 0, "Quarterly": 1, "Annual": 2}).astype("int64")
        
    # Ép kiểu Float32 cho các cột số còn lại
    float_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    for c in float_cols:
        df[c] = df[c].astype("float32")
        
    # Thêm timestamp cho Feast
    df["event_timestamp"] = datetime.utcnow()
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/churn_cleaned.parquet", index=False)
    print("-> Đã tạo xong file Parquet từ dữ liệu gốc của bạn!")
    return True

def main():
    prepare_real_data()
    
    print("\n--- Running Feast Apply ---")
    subprocess.run(["feast", "apply"], cwd="feature_repo/feature_repo", check=True)
    
    print("\n--- Running Feast Materialize (Pushing to Redis) ---")
    end_date = datetime.utcnow() + timedelta(days=1)
    
    subprocess.run([
        "feast", "materialize", 
        "2020-01-01T00:00:00", 
        end_date.strftime("%Y-%m-%dT%H:%M:%S")
    ], cwd="feature_repo/feature_repo", check=True)
    
    print("\n✅ Feast Materialization Complete! Online Store (Redis) is ready.")

if __name__ == "__main__":
    main()
