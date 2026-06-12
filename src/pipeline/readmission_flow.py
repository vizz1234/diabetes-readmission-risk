import os
import sys
import pandas as pd
import numpy as np
from metaflow import FlowSpec, step

# Setup Python paths for child processes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

class ReadmissionFlow(FlowSpec):

    @step
    def start(self):
        # Setup local MLflow path if not set
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        if "MLFLOW_TRACKING_URI" not in os.environ:
            os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
            
        print("Loading raw CSV from data/raw/...")
        self.raw_df = pd.read_csv("data/raw/diabetic_data.csv")
        print(f"Data shape: {self.raw_df.shape}")
        
        # Replace '?' with NaN
        self.raw_df = self.raw_df.replace("?", np.nan)
        self.next(self.split)

    @step
    def split(self):
        from sklearn.model_selection import train_test_split
        
        print("Splitting dataset by patient_nbr to avoid patient leakage...")
        # Get patient level target (using 1 if patient was readmitted <30 in any encounter, else 0)
        patient_targets = self.raw_df.groupby("patient_nbr")["readmitted"].apply(
            lambda x: int(any(x == "<30"))
        ).to_frame()
        
        # Split patient IDs 70/30 (stratified)
        train_patients, temp_patients = train_test_split(
            patient_targets.index,
            test_size=0.30,
            stratify=patient_targets["readmitted"],
            random_state=42
        )
        
        # Split temp patient IDs 50/50 -> 15% val, 15% test (stratified)
        temp_targets = patient_targets.loc[temp_patients]
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=0.50,
            stratify=temp_targets["readmitted"],
            random_state=42
        )
        
        # Filter raw dataframe to get the splits
        self.train_df = self.raw_df[self.raw_df["patient_nbr"].isin(train_patients)].copy()
        self.val_df = self.raw_df[self.raw_df["patient_nbr"].isin(val_patients)].copy()
        self.test_df = self.raw_df[self.raw_df["patient_nbr"].isin(test_patients)].copy()
        
        print(f"Split sizes: Train={self.train_df.shape[0]}, Val={self.val_df.shape[0]}, Test={self.test_df.shape[0]}")
        
        # Save training data as parquet to support future drift checks (Reference dataset)
        os.makedirs("data/processed", exist_ok=True)
        self.train_df.to_parquet("data/processed/train.parquet", index=False)
        self.val_df.to_parquet("data/processed/val.parquet", index=False)
        self.test_df.to_parquet("data/processed/test.parquet", index=False)
        
        self.next(self.fit_preprocessing)

    @step
    def fit_preprocessing(self):
        from preprocessing import fit_preprocessing, save_artifact
        from metaflow import current
        
        print("Fitting preprocessing artifact on train split...")
        self.prep_version = current.run_id
        self.prep = fit_preprocessing(self.train_df, version=self.prep_version)
        
        # Save artifact using joblib
        save_artifact(self.prep, self.prep_version)
        print(f"Saved PreprocessingArtifact under version {self.prep_version}")
        
        self.next(self.transform)

    @step
    def transform(self):
        from preprocessing import apply_preprocessing
        
        print("Transforming all splits using fitted artifact...")
        self.X_train, self.y_train = apply_preprocessing(self.train_df, self.prep)
        self.X_val, self.y_val = apply_preprocessing(self.val_df, self.prep)
        self.X_test, self.y_test = apply_preprocessing(self.test_df, self.prep)
        
        print(f"X_train shape: {self.X_train.shape}")
        
        self.next(self.train_logreg, self.train_rf, self.train_lgbm)

    @step
    def train_logreg(self):
        from models.train_eval import run_grid_search
        
        print("Training Logistic Regression...")
        self.result = run_grid_search("logreg", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def train_rf(self):
        from models.train_eval import run_grid_search
        
        print("Training Random Forest...")
        self.result = run_grid_search("random_forest", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def train_lgbm(self):
        from models.train_eval import run_grid_search
        
        print("Training LightGBM...")
        self.result = run_grid_search("lightgbm", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def join(self, inputs):
        # Bring prep_version and other vars to join step
        self.prep_version = inputs[0].prep_version
        self.X_test = inputs[0].X_test
        self.y_test = inputs[0].y_test
        self.test_df = inputs[0].test_df
        self.prep = inputs[0].prep
        
        self.results = [i.result for i in inputs]
        
        from models.train_eval import log_all_to_mlflow, select_best
        
        print("Logging all models to MLflow...")
        log_all_to_mlflow(self.results, self.prep_version)
        
        print("Selecting best model...")
        self.best = select_best(self.results)
        
        self.next(self.final_eval)

    @step
    def final_eval(self):
        from models.train_eval import evaluate_on_test
        from fairness.audit import run_fairness_audit
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Setup local MLflow path if not set
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        if "MLFLOW_TRACKING_URI" not in os.environ:
            os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
            
        print("Running final evaluation on Test split...")
        self.test_metrics = evaluate_on_test(self.best, self.X_test, self.y_test)
        
        # Run fairness audit
        print("Running model fairness audit on test set...")
        run_fairness_audit(
            self.test_df,
            self.best["best_estimator"],
            self.prep,
            self.test_metrics["threshold"],
            output_path="governance/fairness_report.md"
        )
        
        # Log test metrics under the best model run
        best_run_id = self.best["mlflow_run_id"]
        print(f"Logging test metrics to run {best_run_id}...")
        with mlflow.start_run(run_id=best_run_id):
            mlflow.log_metric("test_pr_auc", self.test_metrics["pr_auc"])
            mlflow.log_metric("test_recall_at_p30", self.test_metrics["recall_at_p30"])
            mlflow.log_metric("test_brier", self.test_metrics["brier_score"])
            mlflow.log_metric("chosen_threshold", self.test_metrics["threshold"])
            
        # Register best model in MLflow Model Registry
        print("Registering the best model in the MLflow Model Registry...")
        model_uri = f"runs:/{best_run_id}/model"
        
        try:
            model_details = mlflow.register_model(model_uri, "diabetes_readmission_model")
            
            # Tag the model version with prep_version
            client = MlflowClient()
            client.set_model_version_tag(
                name="diabetes_readmission_model",
                version=model_details.version,
                key="preprocessing_version",
                value=self.prep_version
            )
            print(f"Successfully registered model version {model_details.version} with prep_version={self.prep_version}")
        except Exception as e:
            print(f"Warning: Model registration failed (possibly because MLflow tracking server is local/file-based and registry is not fully supported or active): {e}")
            
        self.next(self.end)

    @step
    def end(self):
        print("\n--- PIPELINE EXECUTION SUCCESSFUL ---")
        print(f"Best Model: {self.best['name']}")
        print(f"Preprocessing Version / Run ID: {self.prep_version}")
        print(f"Test PR-AUC: {self.test_metrics['pr_auc']:.4f}")
        print(f"Brier Score: {self.test_metrics['brier_score']:.4f}")
        print(f"Savings-Optimized Threshold: {self.test_metrics['threshold']:.4f}")
        print(f"Confusion Matrix (at threshold): {self.test_metrics['confusion_matrix']}")

if __name__ == "__main__":
    ReadmissionFlow()
