import os
import json
import joblib
import pandas as pd
import glob

def init():
    """
    Robustly loads the model by searching for model.pkl in the model directory.
    """
    global model
    model_root = os.getenv('AZUREML_MODEL_DIR')
    
    # Azure ML mounts models in a versioned folder. 
    # Let's find the model.pkl file regardless of the subfolder name.
    print(f"Searching for model in: {model_root}")
    
    # Recursive search for model.pkl
    model_files = glob.glob(os.path.join(model_root, "**", "model.pkl"), recursive=True)
    
    if not model_files:
        print(f"CRITICAL: model.pkl not found in {model_root}")
        # List all files for debugging
        for root, dirs, files in os.walk(model_root):
            for file in files:
                print(f"  Found file: {os.path.join(root, file)}")
        return

    model_path = model_files[0]
    print(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model with joblib: {str(e)}")

def run(raw_data):
    """
    Inference function.
    """
    try:
        if 'model' not in globals():
            return {"status": "error", "message": "Model is not initialized. Check container logs."}
            
        input_json = json.loads(raw_data)
        data_list = input_json.get('data')
        
        if not data_list:
            return {"status": "error", "message": "Missing 'data' field."}

        input_df = pd.DataFrame(data_list)
        
        # Ensure column order or names if needed, but for now we predict directly
        predictions = model.predict(input_df)
        
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Inference error: {str(e)}"
        }
