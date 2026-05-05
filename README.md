# MLOps Customer Churn Prediction

## 1. Project Overview

An end-to-end MLOps pipeline for predicting customer churn, built on Microsoft Azure. The system automates the entire machine learning lifecycle: data versioning, feature serving, model training, deployment, and continuous retraining triggered by user feedback. 

The project utilizes a Customer Churn Dataset containing over 440,000 records to train machine learning models that can identify customers at risk of leaving. By predicting churn, businesses can take proactive measures to retain their customer base and improve service quality.

## 2. Features

- Data Versioning: Manages large datasets using Data Version Control (DVC) with Azure Blob Storage as the remote backend, ensuring full reproducibility.
- Feature Store: Implements Feast to maintain a single source of truth for features, utilizing Parquet files for offline training and Azure Cache for Redis for low-latency online serving.
- Experiment Tracking: Integrates MLflow with Azure Machine Learning to log training metrics (Accuracy, F1-Score), compare Random Forest and XGBoost models, and automatically register the best performing model.
- Model Serving: Deploys the registered model to an Azure ML Managed Endpoint for real-time, secure REST API inference.
- Interactive Web Interface: Provides a Gradio application hosted on Azure App Service, offering three main business flows:
  - Feature Store Lookup: Retrieve predictions for existing customers using their Customer ID.
  - Manual Data Collection: Input new customer data, receive predictions, and save the feedback.
  - Batch CSV Upload: Upload multiple customer records for batch processing.
- Continuous Training (CI/CD): Automates the retraining loop. When user feedback accumulates to a specific threshold (5 records), it is uploaded to Azure Blob Storage. Azure Event Grid detects this and triggers a GitHub Actions workflow to merge data, update DVC, materialize features, retrain the model, and deploy the new version without manual intervention.

## 3. Architecture

### System Workflow

```text
User Feedback --> Azure Blob Storage --> Event Grid --> GitHub Actions
                                                              |
                                          [Trigger] <---------+
                                          |
                                    Merge Feedback
                                          |
                                    DVC Push (Version Data)
                                          |
                                    Feast Materialize (CSV to Redis)
                                          |
                                    Azure ML Training Job (RF vs XGBoost)
                                          |
                                    Model Registry (CustomerChurnModel)
                                          |
                                    Update Endpoint (Blue Deployment)
                                          |
                                    Gradio Web App (Live Predictions)
```

### Technology Stack

- Cloud Platform: Microsoft Azure (Machine Learning, App Service, Blob Storage, Redis, Event Grid)
- MLOps Tools: DVC, Feast, MLflow, GitHub Actions
- Machine Learning: scikit-learn (Random Forest), XGBoost, pandas, numpy
- User Interface: Gradio

### Project Structure

```text
.
|-- .github/workflows/
|   |-- mlops_pipeline.yml            # CI/CD: Train model and update endpoint
|   |-- main_churn-gradio-app.yml     # CI/CD: Deploy Gradio app to Azure App Service
|
|-- data/
|   |-- raw/                          # Raw dataset (managed by DVC)
|   |-- processed/                    # Cleaned CSV and Parquet for Feast
|
|-- feature_repo/feature_repo/
|   |-- feature_store.yaml            # Feast config (Redis online store)
|   |-- feature_definitions.py        # Entity and FeatureView definitions
|
|-- src/
|   |-- data_pipeline/
|   |   |-- preprocess.py             # Clean, encode, scale raw data
|   |   |-- merge_feedback.py         # Merge user feedback into historical dataset
|   |
|   |-- training/
|   |   |-- train.py                  # Train RF and XGBoost, register best model
|   |   |-- materialize.py            # Push features from Parquet to Redis
|   |
|   |-- serving/
|       |-- score.py                  # Scoring script for Azure ML Endpoint
|       |-- gradio_app.py             # Web UI with 3 business flows
|       |-- simulate_drift.py         # Demo script to simulate data drift
|       |-- test_endpoint.py          # Manual endpoint testing
|
|-- pipelines/
|   |-- azure_train_job.yml           # Azure ML training job config
|   |-- azure_deployment.yml          # Managed endpoint deployment config
|   |-- azure_endpoint.yml            # Endpoint definition
|
|-- .env.example                      # Environment variables template
|-- requirements.txt                  # Python dependencies
|-- config.json                       # Azure ML workspace config
```

## 4. Installation

### Prerequisites

- Python 3.10 or higher
- Git
- An active Microsoft Azure subscription with the following services provisioned:
  - Azure Machine Learning Workspace and Compute Cluster
  - Azure App Service
  - Azure Blob Storage
  - Azure Cache for Redis
  - Azure Event Grid

### Local Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/adamwhite625/MLOps-Customer-Churn-Prediction.git
cd MLOps-Customer-Churn-Prediction
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
Copy the template file and fill in your Azure credentials.
```bash
cp .env.example .env
```

4. Pull the data from Azure Blob Storage using DVC:
```bash
dvc pull
```

5. Run data preprocessing:
```bash
python src/data_pipeline/preprocess.py
```

6. Start the local Gradio web application:
```bash
python src/serving/gradio_app.py
```

## 5. Environment Variables

To run the project locally and configure the CI/CD pipelines, specific environment variables and secrets are required.

### Local Environment Variables (.env)

These variables are used when running the Gradio app or scripts locally. See `.env.example` for the format.

| Variable | Description |
|----------|-------------|
| AZURE_ML_PRIMARY_KEY | API key for authenticating with the Azure ML Managed Endpoint |
| REDIS_CONNECTION_STRING | Connection string for Azure Cache for Redis (Feast Online Store) |
| AZURE_STORAGE_CONNECTION_STRING | Connection string for Azure Blob Storage (used for DVC and feedback data) |

### GitHub Actions Secrets

These secrets must be configured in your GitHub repository settings to enable the automated CI/CD pipelines.

| Secret Name | Description |
|-------------|-------------|
| AZURE_CREDENTIALS | JSON string containing the Azure service principal credentials |
| AZURE_RESOURCE_GROUP | Name of the Azure Resource Group |
| AZURE_ML_WORKSPACE | Name of the Azure Machine Learning Workspace |
| REDIS_CONNECTION_STRING | Connection string for Azure Cache for Redis |
| AZUREAPPSERVICE_CLIENTID_... | OIDC Client ID for Azure App Service deployment |
| AZUREAPPSERVICE_TENANTID_... | OIDC Tenant ID |
| AZUREAPPSERVICE_SUBSCRIPTIONID_... | OIDC Subscription ID |
