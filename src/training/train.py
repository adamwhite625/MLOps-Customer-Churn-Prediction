import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os
import argparse

def load_data(path):
    """
    Reads the processed CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input data not found at {path}")
    return pd.read_csv(path)

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """
    Generates and saves a Confusion Matrix heatmap as an image.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)

    classes = ['Not Churn (0)', 'Churn (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate each cell with its count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()

    path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(path)
    plt.close(fig)
    return path

def plot_roc_curve(y_true, y_proba, output_dir):
    """
    Generates and saves an ROC Curve plot as an image.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()

    path = os.path.join(output_dir, 'roc_curve.png')
    fig.savefig(path)
    plt.close(fig)
    return path, roc_auc

def train_and_log(data_path):
    """
    Handles data splitting, model training, evaluation, and MLflow logging.
    Includes Confusion Matrix, ROC Curve, and automatic Model Registration.
    """
    df = load_data(data_path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Directory to save chart images before logging to MLflow
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run() as run:
        # Hyperparameters
        n_estimators = 100
        max_depth = 10

        # Log hyperparameters so TV2 can compare across experiments
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)

        # Train
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        print(f"Accuracy:  {acc:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")

        # Log all metrics to Azure ML
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Generate and log Confusion Matrix
        cm_path = plot_confusion_matrix(y_test, predictions, output_dir)
        mlflow.log_artifact(cm_path, "evaluation_charts")

        # Generate and log ROC Curve
        roc_path, roc_auc = plot_roc_curve(y_test, y_proba, output_dir)
        mlflow.log_artifact(roc_path, "evaluation_charts")
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the trained model
        mlflow.sklearn.log_model(model, "customer_churn_model")

        # Register model to Azure ML Model Registry
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/customer_churn_model"
        mlflow.register_model(model_uri, "CustomerChurnModel")

        print(f"ROC AUC:   {roc_auc:.4f}")
        print("All metrics, charts, and model registered successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data", type=str,
        default="data/processed/churn_cleaned.csv"
    )
    args = parser.parse_args()

    train_and_log(args.input_data)
