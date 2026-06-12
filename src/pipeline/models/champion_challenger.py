import os
import mlflow
from mlflow.tracking import MlflowClient

# Setup local MLflow path if not set
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")

def compare_and_promote(challenger_run_id: str, promotion_margin: float = 0.01) -> bool:
    """
    Load champion's current val_pr_auc from MLflow registry (Production stage).
    Load challenger's val_pr_auc from the given run.
    If challenger >= champion + promotion_margin, transition challenger to
    Production, archive previous champion. Return True/False (promoted).
    If no champion exists yet, promote unconditionally.
    """
    print(f"Starting champion/challenger comparison for challenger run: {challenger_run_id}")
    client = MlflowClient()
    
    # 1. Fetch challenger version and PR-AUC
    challenger_run = client.get_run(challenger_run_id)
    challenger_pr_auc = challenger_run.data.metrics.get("val_pr_auc", 0.0)
    print(f"Challenger validation PR-AUC: {challenger_pr_auc:.4f}")
    
    # Ensure challenger is registered in registry
    model_name = "diabetes_readmission_model"
    versions = client.search_model_versions(f"run_id = '{challenger_run_id}'")
    if versions:
        challenger_version = versions[0].version
        print(f"Challenger already registered as version {challenger_version}")
    else:
        print(f"Registering challenger model run as a new version...")
        model_uri = f"runs:/{challenger_run_id}/model"
        model_details = mlflow.register_model(model_uri, model_name)
        challenger_version = model_details.version
        # Tag version with the preprocessing version from the run's params
        prep_version = challenger_run.data.params.get("preprocessing_version", "unknown")
        client.set_model_version_tag(
            name=model_name,
            version=challenger_version,
            key="preprocessing_version",
            value=prep_version
        )
        print(f"Registered challenger as version {challenger_version} with prep_version={prep_version}")
        
    # 2. Fetch current champion (Production stage)
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    
    if not prod_versions:
        print("No active Production champion model found. Promoting challenger unconditionally...")
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {challenger_version} promoted to Production.")
        return True
        
    champion = prod_versions[0]
    champion_version = champion.version
    champion_run_id = champion.run_id
    
    champion_run = client.get_run(champion_run_id)
    champion_pr_auc = champion_run.data.metrics.get("val_pr_auc", 0.0)
    print(f"Current Champion Version: {champion_version} (Run: {champion_run_id})")
    print(f"Champion validation PR-AUC: {champion_pr_auc:.4f}")
    
    # 3. Compare PR-AUC
    if challenger_pr_auc >= (champion_pr_auc + promotion_margin):
        print(f"Challenger exceeds champion by at least margin {promotion_margin:.4f} ({challenger_pr_auc:.4f} >= {champion_pr_auc:.4f} + {promotion_margin:.4f}).")
        print("Promoting challenger to Production and archiving champion...")
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model version {challenger_version} promoted to Production.")
        return True
    else:
        print(f"Challenger does not outperform champion by the required margin {promotion_margin:.4f} ({challenger_pr_auc:.4f} < {champion_pr_auc:.4f} + {promotion_margin:.4f}).")
        print("Keeping current champion in Production.")
        return False
