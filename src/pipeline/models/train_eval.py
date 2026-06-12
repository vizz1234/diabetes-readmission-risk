import os
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve
import mlflow
import mlflow.sklearn

# Set up local MLflow tracking path by default
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
mlflow.set_tracking_uri(tracking_uri)

# Cost Matrix Constants
COST_MISSED_READMISSION = 5000
COST_UNNECESSARY_FOLLOWUP = 200

# Import registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from registry import MODEL_GRID

def get_estimator_class(estimator_name):
    if estimator_name == "LogisticRegression":
        return LogisticRegression
    elif estimator_name == "RandomForestClassifier":
        return RandomForestClassifier
    elif estimator_name == "LGBMClassifier":
        from lightgbm import LGBMClassifier
        return LGBMClassifier
    else:
        raise ValueError(f"Unknown estimator name: {estimator_name}")

def compute_recall_at_precision(y_true, y_probs, target_precision=0.3):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_indices = np.where(precisions >= target_precision)[0]
    # Filter out index corresponding to dummy precision=1, recall=0
    valid_indices = [idx for idx in valid_indices if idx < len(thresholds)]
    if not valid_indices:
        return 0.0, 0.5
    best_idx = valid_indices[np.argmax([recalls[i] for i in valid_indices])]
    return float(recalls[best_idx]), float(thresholds[best_idx])

def select_threshold_by_savings(y_val, val_probs):
    best_savings = -float('inf')
    best_thresh = 0.5
    for thresh in np.linspace(0.0, 1.0, 101):
        y_pred = (val_probs >= thresh).astype(int)
        tp = np.sum((y_val == 1) & (y_pred == 1))
        fp = np.sum((y_val == 0) & (y_pred == 1))
        savings = tp * COST_MISSED_READMISSION - fp * COST_UNNECESSARY_FOLLOWUP
        if savings > best_savings:
            best_savings = savings
            best_thresh = thresh
    return float(best_thresh)

def run_grid_search(model_name, X_train, y_train, X_val, y_val):
    """
    GridSearchCV with StratifiedKFold(5), scoring='average_precision'.
    Refit best estimator on full X_train.
    Evaluate on X_val: pr_auc, recall_at_precision(0.3), brier_score.
    Return dict with metrics and validation probabilities.
    """
    print(f"Running Grid Search for model: {model_name}...")
    model_conf = MODEL_GRID[model_name]
    
    EstClass = get_estimator_class(model_conf["estimator"])
    base_est = EstClass(**model_conf["fixed_params"])
    
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_est,
        param_grid=model_conf["grid"],
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # CV results summary
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_mean_pr_auc = float(grid_search.best_score_)
    # Find standard deviation corresponding to best rank
    best_index = grid_search.best_index_
    cv_std_pr_auc = float(grid_search.cv_results_["std_test_score"][best_index])
    
    # Val evaluation
    val_probs = best_estimator.predict_proba(X_val)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_recall_at_p30, _ = compute_recall_at_precision(y_val, val_probs, 0.3)
    val_brier = float(brier_score_loss(y_val, val_probs))
    
    return {
        "name": model_name,
        "best_params": best_params,
        "best_estimator": best_estimator,
        "cv_results": cv_results_df,
        "cv_mean_pr_auc": cv_mean_pr_auc,
        "cv_std_pr_auc": cv_std_pr_auc,
        "val_pr_auc": val_pr_auc,
        "val_recall_at_p30": val_recall_at_p30,
        "val_brier": val_brier,
        "val_probs": val_probs.tolist(),
        "y_val": y_val.tolist()
    }

def log_all_to_mlflow(results, preprocessing_version):
    """
    For each result, start an mlflow run:
    - log_param('algorithm', name)
    - log_params(best_params)
    ...
    """
    # Use/create experiment
    experiment_name = "diabetes_readmission_experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
    for result in results:
        with mlflow.start_run(experiment_id=experiment_id, run_name=result["name"]) as run:
            # Log params
            mlflow.log_param("algorithm", result["name"])
            mlflow.log_param("preprocessing_version", preprocessing_version)
            for param_name, param_val in result["best_params"].items():
                mlflow.log_param(param_name, param_val)
                
            # Log metrics
            mlflow.log_metric("val_pr_auc", result["val_pr_auc"])
            mlflow.log_metric("val_recall_at_p30", result["val_recall_at_p30"])
            mlflow.log_metric("val_brier", result["val_brier"])
            mlflow.log_metric("cv_mean_pr_auc", result["cv_mean_pr_auc"])
            mlflow.log_metric("cv_std_pr_auc", result["cv_std_pr_auc"])
            
            # Save CV results to temporary CSV and log it
            with tempfile.TemporaryDirectory() as tmpdir:
                cv_csv_path = os.path.join(tmpdir, "cv_results.csv")
                result["cv_results"].to_csv(cv_csv_path, index=False)
                mlflow.log_artifact(cv_csv_path)
                
            # Log scikit-learn model
            # We log with model package info
            mlflow.sklearn.log_model(result["best_estimator"], "model")
            
            # Save the active run ID to result dict so we can reference it later
            result["mlflow_run_id"] = run.info.run_id

def select_best(results):
    """Pick max val_pr_auc, tiebreak on lowest val_brier."""
    sorted_results = sorted(
        results,
        key=lambda x: (x["val_pr_auc"], -x["val_brier"]),
        reverse=True
    )
    best_model = sorted_results[0]
    print(f"Selected best model: {best_model['name']} with val_pr_auc: {best_model['val_pr_auc']}")
    return best_model

def evaluate_on_test(best_result, X_test, y_test):
    """
    Touch test set once. Compute pr_auc, recall_at_precision(0.3),
    calibration curve data, confusion matrix at chosen threshold.
    """
    best_estimator = best_result["best_estimator"]
    
    # Calculate test probabilities
    test_probs = best_estimator.predict_proba(X_test)[:, 1]
    
    # Perform threshold selection using validation set probabilities
    y_val = np.array(best_result["y_val"])
    val_probs = np.array(best_result["val_probs"])
    chosen_threshold = select_threshold_by_savings(y_val, val_probs)
    
    # Calculate test metrics
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_recall_at_p30, _ = compute_recall_at_precision(y_test, test_probs, 0.3)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
    
    # Predictions and confusion matrix at the chosen threshold
    y_pred_test = (test_probs >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_test).tolist()
    
    # Calculate brier score on test
    test_brier = float(brier_score_loss(y_test, test_probs))
    
    return {
        "pr_auc": test_pr_auc,
        "recall_at_p30": test_recall_at_p30,
        "brier_score": test_brier,
        "threshold": chosen_threshold,
        "confusion_matrix": cm,
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }
    }
