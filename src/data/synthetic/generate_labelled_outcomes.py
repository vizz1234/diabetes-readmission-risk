import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Ensure local imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocessing import apply_preprocessing, load_artifact

# Setup local MLflow path if not set
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")

def generate_labelled_outcomes(decay_rate=0.3, input_path="data/processed/test.parquet", output_dir="data/processed/synthetic"):
    print(f"Loading test data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Reference test file not found at {input_path}. Run metaflow pipeline first.")
        
    df = pd.read_parquet(input_path)
    
    # 1. Load active champion model and prep artifact
    print("Loading champion model from MLflow Model Registry...")
    client = MlflowClient()
    model_name = "diabetes_readmission_model"
    
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            prod_versions = client.get_latest_versions(model_name)
        if not prod_versions:
            raise ValueError("No registered models found.")
            
        ver_info = prod_versions[0]
        run_id = ver_info.run_id
        prep_version = ver_info.tags.get("preprocessing_version", "v1")
        
        # Load model and prep
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        prep = load_artifact(prep_version)
    except Exception as e:
        print(f"Error loading model from MLflow: {e}. Trying to load local resources...")
        # Local fallback
        # Let's find any preprocessed artifact
        import glob
        files = glob.glob("artifacts/preprocessing_*.joblib")
        if not files:
            raise FileNotFoundError("No preprocessing artifacts found in artifacts/")
        files.sort(key=os.path.getmtime)
        prep = load_artifact(os.path.basename(files[-1]).replace("preprocessing_", "").replace(".joblib", ""))
        
        # Load local model
        runs_dirs = glob.glob("mlruns/0/*")
        runs_dirs = [d for d in runs_dirs if os.path.isdir(d) and os.path.basename(d) != "models"]
        if not runs_dirs:
            raise FileNotFoundError("No MLflow runs found locally.")
        runs_dirs.sort(key=os.path.getmtime)
        model = mlflow.sklearn.load_model(os.path.join(runs_dirs[-1], "artifacts", "model"))
        
    # 2. Get predictions
    X, _ = apply_preprocessing(df, prep)
    probs = model.predict_proba(X)[:, 1]
    
    # Set default actual outcomes (mapped from readmitted)
    df["actual_readmission"] = (df["readmitted"] == "<30").astype(int)
    df["predicted_probability"] = probs
    
    # Identify deciles
    df["predicted_decile"] = pd.qcut(df["predicted_probability"], 10, labels=False, duplicates='drop')
    
    # 3. If decay_rate > 0.0, flip labels in the highest-predicted-risk decile
    if decay_rate > 0.0:
        print(f"Applying performance decay rate of {decay_rate}...")
        # Highest risk decile is the max decile value
        max_decile = df["predicted_decile"].max()
        
        # Identify rows in the highest-risk decile where actual_readmission is 0
        target_indices = df[(df["predicted_decile"] == max_decile) & (df["actual_readmission"] == 0)].index
        
        # Determine number of labels to flip
        n_flip = int(len(target_indices) * decay_rate)
        print(f"Flipping {n_flip} outcomes from 0 to 1 in the highest-risk decile (decile {max_decile})...")
        
        if n_flip > 0:
            np.random.seed(42)
            flip_indices = np.random.choice(target_indices, size=n_flip, replace=False)
            df.loc[flip_indices, "actual_readmission"] = 1
            
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"labelled_outcomes_{decay_rate}.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Labelled outcomes file successfully written to {output_path} (records: {len(df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic labeled outcomes.")
    parser.add_argument("--decay-rate", type=float, choices=[0.0, 0.3], default=0.3, help="Decay rate to apply")
    args = parser.parse_args()
    
    generate_labelled_outcomes(decay_rate=args.decay_rate)
