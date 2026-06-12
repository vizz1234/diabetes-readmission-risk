import os
import sys
import glob
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Setup path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocessing import apply_preprocessing, load_artifact
from explain.shap_explainer import ShapExplainer

# Setup local MLflow path if not set
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.prep = None
        self.explainer = None
        self.model_version = "unknown"
        self.prep_version = "unknown"
        self.threshold = 0.35 # Default fallback

    def get_latest_local_prep_version(self) -> str:
        files = glob.glob("artifacts/preprocessing_*.joblib")
        if not files:
            return "unknown"
        # Sort by modification time to find latest
        files.sort(key=os.path.getmtime)
        basename = os.path.basename(files[-1])
        return basename.replace("preprocessing_", "").replace(".joblib", "")

    def load_resources(self):
        """Loads model and preprocessing artifact from MLflow or local files."""
        print("Initializing Inference Engine resources...")
        client = MlflowClient()
        model_name = "diabetes_readmission_model"
        
        try:
            # Try to get model from production stage
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if not prod_versions:
                # Fallback to latest registered
                prod_versions = client.get_latest_versions(model_name)
                
            if prod_versions:
                ver_info = prod_versions[0]
                self.model_version = ver_info.version
                run_id = ver_info.run_id
                
                # Fetch preprocessing version from tags
                self.prep_version = ver_info.tags.get("preprocessing_version", "unknown")
                
                # Load threshold from run metrics if available
                run = client.get_run(run_id)
                self.threshold = run.data.metrics.get("chosen_threshold", 0.35)
                
                # Load model
                model_uri = f"runs:/{run_id}/model"
                try:
                    self.model = mlflow.sklearn.load_model(model_uri)
                    print(f"Loaded champion model version {self.model_version} from run {run_id}")
                except Exception as load_err:
                    print(f"Direct MLflow load failed: {load_err}. Attempting relative path correction...")
                    # Parse version's meta.yaml directly
                    meta_path = f"mlruns/models/{model_name}/version-{self.model_version}/meta.yaml"
                    loaded = False
                    if os.path.exists(meta_path):
                        import yaml
                        with open(meta_path, "r") as f:
                            meta_data = yaml.safe_load(f)
                        model_id = meta_data.get("model_id")
                        storage_loc = meta_data.get("storage_location", "")
                        
                        # Fallback 1: if storage_location contains a path, construct relative path
                        if "mlruns/" in storage_loc:
                            relative_path = storage_loc[storage_loc.find("mlruns/"):]
                            print(f"Extracted relative path from storage_location: {relative_path}")
                            if os.path.exists(relative_path):
                                self.model = mlflow.sklearn.load_model(relative_path)
                                print(f"Successfully loaded model from storage_location path!")
                                loaded = True
                                
                        # Fallback 2: glob search for model_id folder
                        if not loaded and model_id:
                            matched_paths = glob.glob(f"mlruns/*/models/{model_id}/artifacts")
                            if matched_paths and os.path.exists(matched_paths[0]):
                                print(f"Found corrected local path via glob: {matched_paths[0]}")
                                self.model = mlflow.sklearn.load_model(matched_paths[0])
                                print(f"Successfully loaded model from glob path!")
                                loaded = True
                                
                    if not loaded:
                        raise load_err
            else:
                print("No registered model found in MLflow. Searching local directory...")
                raise ValueError("No MLflow registered model available.")
        except Exception as e:
            print(f"MLflow model load failed or not configured: {e}. Loading latest local run...")
            # Fallback: Search mlruns directory for latest model
            try:
                # Get the latest run dir in mlruns/
                runs_path = "mlruns/0" # Default experiment is usually 0, or check other directories
                if not os.path.exists(runs_path):
                    # Check other experiments
                    exp_dirs = [d for d in glob.glob("mlruns/*") if os.path.basename(d).isdigit()]
                    if exp_dirs:
                        runs_path = exp_dirs[0]
                
                run_dirs = glob.glob(os.path.join(runs_path, "*"))
                # Filter out meta.yaml and directories
                run_dirs = [d for d in run_dirs if os.path.isdir(d) and os.path.basename(d) != "models"]
                if run_dirs:
                    run_dirs.sort(key=os.path.getmtime)
                    latest_run_dir = run_dirs[-1]
                    run_id = os.path.basename(latest_run_dir)
                    
                    self.model_version = f"local_run_{run_id[:8]}"
                    # Load model from local artifact folder
                    model_path = os.path.join(latest_run_dir, "artifacts", "model")
                    self.model = mlflow.sklearn.load_model(model_path)
                    print(f"Loaded fallback model from {model_path}")
                else:
                    raise FileNotFoundError("No local MLflow runs found.")
            except Exception as e_inner:
                print(f"Failed to load fallback local model: {e_inner}")
                raise e_inner

        # Load Preprocessing Artifact
        if self.prep_version == "unknown":
            self.prep_version = self.get_latest_local_prep_version()
            
        try:
            self.prep = load_artifact(self.prep_version)
            print(f"Loaded preprocessing artifact version: {self.prep_version}")
        except Exception as e:
            print(f"Failed to load preprocessing artifact '{self.prep_version}': {e}. Loading latest local...")
            self.prep_version = self.get_latest_local_prep_version()
            self.prep = load_artifact(self.prep_version)
            print(f"Loaded backup preprocessing artifact: {self.prep_version}")

        # Initialize SHAP explainer
        # We can extract a background training dataset sample if available
        X_train_sample = None
        if os.path.exists("data/processed/train.parquet"):
            try:
                train_df = pd.read_parquet("data/processed/train.parquet")
                X_train_sample, _ = apply_preprocessing(train_df.head(100), self.prep)
                print("Loaded background training data sample for SHAP explainer.")
            except Exception as ex:
                print(f"Could not load train parquet for SHAP background: {ex}")
                
        self.explainer = ShapExplainer(self.model, self.prep, X_train_sample=X_train_sample)
        print("SHAP Explainer initialized successfully.")

    def get_risk_band(self, prob: float) -> str:
        """Categorizes prediction probability relative to decision threshold."""
        if prob >= self.threshold:
            return "high"
        elif prob >= (self.threshold / 2):
            return "medium"
        else:
            return "low"

    def predict_single(self, features_dict: dict) -> dict:
        """Transforms and scores a single patient encounter record."""
        # Convert dictionary to 1-row DataFrame
        df = pd.DataFrame([features_dict])
        
        # Apply preprocessing
        X, _ = apply_preprocessing(df, self.prep)
        
        # Predict probability
        prob = float(self.model.predict_proba(X)[0, 1])
        
        # Get risk band
        band = self.get_risk_band(prob)
        
        # Get top 3 factors from SHAP explainer
        top_factors = self.explainer.get_top_factors(X, n=3)
        
        return {
            "risk_score": prob,
            "risk_band": band,
            "top_factors": top_factors,
            "model_version": self.model_version,
            "preprocessing_version": self.prep_version
        }

    def predict_batch(self, list_of_dicts: list[dict]) -> list[dict]:
        """Transforms and scores a list of patient encounter records."""
        df = pd.DataFrame(list_of_dicts)
        
        # Apply preprocessing
        X, _ = apply_preprocessing(df, self.prep)
        
        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]
        
        results = []
        for i in range(len(list_of_dicts)):
            prob = float(probs[i])
            band = self.get_risk_band(prob)
            
            # Extract single row DataFrame for explanation
            row_df = X.iloc[[i]]
            top_factors = self.explainer.get_top_factors(row_df, n=3)
            
            results.append({
                "risk_score": prob,
                "risk_band": band,
                "top_factors": top_factors,
                "model_version": self.model_version,
                "preprocessing_version": self.prep_version
            })
            
        return results

# Singleton instance
engine = InferenceEngine()
