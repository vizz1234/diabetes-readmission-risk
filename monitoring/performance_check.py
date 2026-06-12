import os
import pandas as pd
from sklearn.metrics import recall_score
import mlflow
from mlflow.tracking import MlflowClient

# Setup local MLflow path if not set
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")

def compute_recent_recall(labelled_outcomes_path: str) -> float:
    """
    Load labelled outcomes (predictions + synthetic actuals). Compute recall
    at the production threshold. Return float, or None if file doesn't exist
    (no labels available yet).
    """
    if not os.path.exists(labelled_outcomes_path):
        print(f"File not found: {labelled_outcomes_path}. Performance recall check skipped.")
        return None
        
    df = pd.read_parquet(labelled_outcomes_path)
    
    if "predicted_probability" not in df.columns or "actual_readmission" not in df.columns:
        raise ValueError("Dataset must contain 'predicted_probability' and 'actual_readmission' columns.")
        
    # Fetch current production threshold
    threshold = 0.35  # default fallback
    try:
        client = MlflowClient()
        prod_versions = client.get_latest_versions("diabetes_readmission_model", stages=["Production"])
        if not prod_versions:
            prod_versions = client.get_latest_versions("diabetes_readmission_model")
        if prod_versions:
            run_id = prod_versions[0].run_id
            run = client.get_run(run_id)
            threshold = run.data.metrics.get("chosen_threshold", 0.35)
            print(f"Fetched production threshold: {threshold:.4f}")
    except Exception as e:
        print(f"Could not fetch threshold from MLflow Model Registry: {e}. Using fallback default: {threshold:.4f}")
        
    y_true = df["actual_readmission"]
    y_pred = (df["predicted_probability"] >= threshold).astype(int)
    
    # Calculate recall
    if sum(y_true == 1) == 0:
        print("No actual positive readmission cases in labelled outcomes.")
        return 1.0
        
    rec = float(recall_score(y_true, y_pred))
    print(f"Computed recent recall: {rec:.4f} (at threshold {threshold:.4f})")
    return rec

if __name__ == "__main__":
    default_outcomes = "data/processed/synthetic/labelled_outcomes_0.3.parquet"
    if os.path.exists(default_outcomes):
        compute_recent_recall(default_outcomes)
    else:
        print(f"Default outcomes file {default_outcomes} not found. Please run generator scripts first.")
