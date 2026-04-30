import gradio as gr
import requests
import json
import os
import pandas as pd
from datetime import datetime
from feast import FeatureStore
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env file for local testing
load_dotenv()

# Configure endpoint URLs and credentials
SCORING_URI = os.getenv("AZURE_ML_SCORING_URI", "https://churn-endpoint.southeastasia.inference.ml.azure.com/score")
PRIMARY_KEY = os.getenv("AZURE_ML_PRIMARY_KEY", "")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
CONTAINER_NAME = "churn-feedback"

try:
    store = FeatureStore(repo_path="feature_repo/feature_repo")
except Exception as e:
    store = None
    print(f"Warning: Feast repo is not ready. Error: {e}")

def call_azure_ml(data_dict):
    """Sends inference request to Azure ML endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PRIMARY_KEY}"
    }
    response = requests.post(SCORING_URI, data=json.dumps({"data": [data_dict]}), headers=headers)
    response.raise_for_status()
    result = response.json()
    pred = result["predictions"][0]
    return "CHURN" if pred == 1 else "LOYAL"

def predict_with_feast(customer_id):
    """Predicts churn using features retrieved from Feast online store."""
    if store is None:
        return "Error: Cannot connect to Redis.", None
        
    try:
        if str(customer_id).isdigit():
            search_id = int(customer_id)
        else:
            search_id = customer_id
            
        feature_vector = store.get_online_features(
            features=[
                "churn_features:Age", "churn_features:Gender", "churn_features:Tenure",
                "churn_features:Usage Frequency", "churn_features:Support Calls",
                "churn_features:Payment Delay", "churn_features:Subscription Type",
                "churn_features:Contract Length", "churn_features:Total Spend",
                "churn_features:Last Interaction"
            ],
            entity_rows=[{"customer_id": search_id}]
        ).to_dict()
        
        if "churn_features:Age" not in feature_vector or feature_vector["churn_features:Age"][0] is None:
            return f"Error: ID {customer_id} not found in Redis", None
            
        data = {
            "Age": feature_vector["churn_features:Age"][0],
            "Gender": feature_vector["churn_features:Gender"][0],
            "Tenure": feature_vector["churn_features:Tenure"][0],
            "Usage Frequency": feature_vector["churn_features:Usage Frequency"][0],
            "Support Calls": feature_vector["churn_features:Support Calls"][0],
            "Payment Delay": feature_vector["churn_features:Payment Delay"][0],
            "Subscription Type": feature_vector["churn_features:Subscription Type"][0],
            "Contract Length": feature_vector["churn_features:Contract Length"][0],
            "Total Spend": feature_vector["churn_features:Total Spend"][0],
            "Last Interaction": feature_vector["churn_features:Last Interaction"][0]
        }
        
        label = call_azure_ml(data)
        return label, pd.DataFrame([data])
        
    except Exception as e:
        return f"Error: {str(e)}", None

def predict_and_collect(age, gender, tenure, usage, support, delay, sub_type, contract, spend, last_interaction):
    """Predicts churn and collects feedback data to trigger continuous training."""
    gender_val = 1 if gender == "Male" else 0
    sub_map = {"Basic": 0, "Standard": 1, "Premium": 2}
    contract_map = {"Monthly": 0, "Quarterly": 1, "Annual": 2}
    
    data = {
        "Age": age, "Gender": gender_val, "Tenure": tenure,
        "Usage Frequency": usage, "Support Calls": support,
        "Payment Delay": delay, "Subscription Type": sub_map[sub_type],
        "Contract Length": contract_map[contract], "Total Spend": spend,
        "Last Interaction": last_interaction
    }
    
    label = call_azure_ml(data)
    
    os.makedirs("data/raw", exist_ok=True)
    df = pd.DataFrame([data])
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["Predicted_Churn"] = 1 if "CHURN" in label else 0
    
    file_path = "data/raw/collected_data.csv"
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
        
    total_rows = sum(1 for line in open(file_path)) - 1
    log_msg = f"Saved to CSV (Total: {total_rows}/5 rows to trigger retraining)"
    
    if total_rows >= 5:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"batch_{timestamp}.csv"
            
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
            
            with open(file_path, "rb") as file_data:
                blob_client.upload_blob(file_data)
                
            log_msg += f"\nUploaded batch {blob_name} to Azure Storage to trigger Event Grid."
            
            os.remove(file_path)
        except Exception as e:
            log_msg += f"\nError uploading to Azure Storage: {str(e)}"
            
    return label, log_msg

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MLOps Customer Churn Prediction")
    
    with gr.Tabs():
        with gr.TabItem("Flow 1: Feast Feature Store"):
            gr.Markdown("Retrieve features from Online Store based on Customer ID.")
            with gr.Row():
                customer_id_input = gr.Textbox(label="Customer ID (e.g., 2)", placeholder="2")
                btn_feast = gr.Button("Extract Features & Predict", variant="primary")
            
            out_feast_pred = gr.Textbox(label="Prediction Result")
            out_feast_df = gr.Dataframe(label="Features Retrieved from Redis")
            btn_feast.click(fn=predict_with_feast, inputs=[customer_id_input], outputs=[out_feast_pred, out_feast_df])
            
        with gr.TabItem("Flow 2: Data Collection & Continuous Training"):
            gr.Markdown("Collect manual feedback to trigger the continuous training loop.")
            with gr.Row():
                with gr.Column():
                    age = gr.Slider(18, 100, 35, label="Age")
                    gender = gr.Radio(["Male", "Female"], value="Male", label="Gender")
                    tenure = gr.Slider(0, 120, 12, label="Tenure (Months)")
                    usage = gr.Slider(0, 50, 15, label="Usage Frequency")
                    support = gr.Slider(0, 20, 2, label="Support Calls")
                with gr.Column():
                    delay = gr.Slider(0, 60, 5, label="Payment Delay (Days)")
                    sub_type = gr.Dropdown(["Basic", "Standard", "Premium"], value="Standard", label="Subscription Type")
                    contract = gr.Dropdown(["Monthly", "Quarterly", "Annual"], value="Annual", label="Contract Length")
                    spend = gr.Number(1500, label="Total Spend ($)")
                    last_interaction = gr.Slider(0, 100, 10, label="Last Interaction (Days)")
            
            btn_manual = gr.Button("Predict & Collect Data", variant="primary")
            out_manual_pred = gr.Textbox(label="Prediction Result")
            out_manual_log = gr.Textbox(label="Feedback Loop Log")
            
            btn_manual.click(
                fn=predict_and_collect,
                inputs=[age, gender, tenure, usage, support, delay, sub_type, contract, spend, last_interaction],
                outputs=[out_manual_pred, out_manual_log]
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)
