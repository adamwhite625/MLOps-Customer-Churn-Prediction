import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Lấy Connection String từ biến môi trường (do Github Actions cung cấp)
CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "churn-feedback"
HISTORICAL_DATA_PATH = "data/raw/customer_churn_dataset-training-master.csv"

def main():
    if not CONNECTION_STRING:
        print("Lỗi: Không tìm thấy AZURE_STORAGE_CONNECTION_STRING.")
        return

    print("Kết nối tới Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Lấy danh sách các file batch_*.csv
    blobs = list(container_client.list_blobs(name_starts_with="batch_"))
    
    if not blobs:
        print("Không có dữ liệu feedback mới nào để gộp. Bỏ qua.")
        return

    print(f"Tìm thấy {len(blobs)} file feedback mới. Đang tải xuống...")
    
    new_data_frames = []
    blobs_to_delete = []

    for blob in blobs:
        print(f" - Đang đọc: {blob.name}")
        blob_client = container_client.get_blob_client(blob)
        download_stream = blob_client.download_blob()
        
        # Đọc trực tiếp từ memory stream
        df_batch = pd.read_csv(BytesIO(download_stream.readall()))
        new_data_frames.append(df_batch)
        blobs_to_delete.append(blob_client)

    # Gộp tất cả các file feedback lại
    df_new = pd.concat(new_data_frames, ignore_index=True)
    print(f"Tổng cộng có {len(df_new)} dòng dữ liệu mới.")

    # Xử lý dữ liệu mới để khớp với cấu trúc dữ liệu cũ
    # File gốc có: CustomerID,Age,Gender,Tenure,Usage Frequency,Support Calls,Payment Delay,Subscription Type,Contract Length,Total Spend,Last Interaction,Churn
    if "Predicted_Churn" in df_new.columns:
        df_new.rename(columns={"Predicted_Churn": "Churn"}, inplace=True)
    if "Timestamp" in df_new.columns:
        df_new.drop(columns=["Timestamp"], inplace=True)

    # Chuyển đổi mapping chữ sang giống file gốc (Basic->Basic, Male->Male)
    # Tuy nhiên, trong Gradio code bạn đã map nó thành 0, 1, 2 rồi.
    # Để file gốc đồng nhất, chúng ta cần map ngược lại thành chữ nếu file gốc dùng chữ.
    # Nhìn lại dataset gốc trên Kaggle: Gender là 'Female'/'Male', Subscription Type là 'Basic'/'Standard'/'Premium'
    gender_map = {0: "Female", 1: "Male"}
    sub_map = {0: "Basic", 1: "Standard", 2: "Premium"}
    contract_map = {0: "Monthly", 1: "Quarterly", 2: "Annual"}

    df_new["Gender"] = df_new["Gender"].map(gender_map)
    df_new["Subscription Type"] = df_new["Subscription Type"].map(sub_map)
    df_new["Contract Length"] = df_new["Contract Length"].map(contract_map)

    # Đọc dữ liệu cũ
    print(f"Đang đọc dữ liệu lịch sử từ: {HISTORICAL_DATA_PATH}")
    df_historical = pd.read_csv(HISTORICAL_DATA_PATH)
    
    # Tạo CustomerID mới nối tiếp ID lớn nhất cũ
    max_id = df_historical["CustomerID"].max()
    df_new.insert(0, "CustomerID", range(max_id + 1, max_id + 1 + len(df_new)))

    # Sắp xếp lại cột cho chuẩn
    df_new = df_new[df_historical.columns]

    # Gộp dữ liệu cũ và mới
    df_final = pd.concat([df_historical, df_new], ignore_index=True)

    # Ghi đè lại file cũ
    df_final.to_csv(HISTORICAL_DATA_PATH, index=False)
    print(f"✅ Đã gộp thành công! Tổng số dòng hiện tại: {len(df_final)}")

    # Xóa các file trên Azure để không gộp trùng vào lần sau
    print("Đang dọn dẹp các file feedback trên Azure...")
    for blob_client in blobs_to_delete:
        blob_client.delete_blob()
        print(f" - Đã xóa: {blob_client.blob_name}")

    print("🚀 Hoàn tất Data Merging Pipeline!")

if __name__ == "__main__":
    main()
