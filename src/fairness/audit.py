import os
import sys
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.preprocessing import apply_preprocessing

def fnr_metric(y_true, y_pred):
    """Computes False Negative Rate (FNR = FN / (TP + FN) = 1 - Recall)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    positives = (y_true == 1)
    if np.sum(positives) == 0:
        return 0.0
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(fn / np.sum(positives))

def selection_rate_metric(y_true, y_pred):
    """Computes selection rate (proportion of positive predictions)"""
    y_pred = np.array(y_pred)
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred == 1))

def recall_metric(y_true, y_pred):
    """Computes Recall / True Positive Rate"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    positives = (y_true == 1)
    if np.sum(positives) == 0:
        return 0.0
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return float(tp / np.sum(positives))

def run_fairness_audit(raw_df: pd.DataFrame, model, prep_artifact, threshold: float, output_path: str = "governance/fairness_report.md"):
    print("Running fairness audit...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Preprocess the raw dataframe
    X_test, y_test = apply_preprocessing(raw_df, prep_artifact)
    if y_test is None:
        raise ValueError("Raw dataframe must contain the target 'readmitted' column for fairness auditing.")
        
    # 2. Get predictions
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    
    # Define metrics dictionary
    metrics = {
        "selection_rate": selection_rate_metric,
        "recall_tpr": recall_metric,
        "false_negative_rate": fnr_metric
    }
    
    sensitive_cols = ["age", "gender", "race"]
    reports = {}
    flags = []
    
    # Overall metrics for comparison
    overall_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=pd.Series(["overall"] * len(y_test))
    )
    overall_metrics = overall_frame.overall
    overall_fnr = overall_metrics["false_negative_rate"]
    
    # Compute metrics per demographic group
    for col in sensitive_cols:
        # Fill missing values for demographic grouping if any
        grouping_series = raw_df[col].fillna("Unknown").astype(str)
        
        frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=grouping_series
        )
        
        by_group_df = frame.by_group.copy()
        
        # Format percentages/decimals for readability
        by_group_df["selection_rate"] = by_group_df["selection_rate"].apply(lambda x: f"{x*100:.2f}%")
        by_group_df["recall_tpr"] = by_group_df["recall_tpr"].apply(lambda x: f"{x*100:.2f}%")
        by_group_df["false_negative_rate"] = by_group_df["false_negative_rate"].apply(lambda x: f"{x*100:.2f}%")
        
        reports[col] = by_group_df
        
        # Check FNR flags: if group FNR > overall FNR by more than 5% (0.05)
        for group_name, row in frame.by_group.iterrows():
            group_fnr = row["false_negative_rate"]
            if group_fnr >= overall_fnr + 0.05:
                flags.append({
                    "attribute": col,
                    "group": group_name,
                    "group_fnr": group_fnr,
                    "overall_fnr": overall_fnr,
                    "diff": group_fnr - overall_fnr
                })
                
    # 3. Write fairness report markdown
    with open(output_path, "w") as f:
        f.write("# Model Fairness Audit Report\n\n")
        f.write(f"This report evaluates the model's 30-day readmission risk prediction across key demographic groups. The audit is conducted using the savings-optimized decision threshold of **{threshold:.4f}**.\n\n")
        
        f.write("## Overall Test Performance Reference\n\n")
        f.write(f"- **Overall Selection Rate**: {overall_metrics['selection_rate']*100:.2f}%\n")
        f.write(f"- **Overall Recall (TPR)**: {overall_metrics['recall_tpr']*100:.2f}%\n")
        f.write(f"- **Overall False Negative Rate (FNR)**: {overall_fnr*100:.2f}%\n\n")
        
        for col in sensitive_cols:
            f.write(f"## Disparity Audit by {col.capitalize()}\n\n")
            f.write(reports[col].to_markdown())
            f.write("\n\n")
            
        f.write("## Fairness & Safety Assessment\n\n")
        if flags:
            f.write("> [!WARNING]\n")
            f.write("> **Disparities Detected (High False Negative Rates)**\n")
            f.write("> \n")
            f.write("> The clinical target is to avoid missing high-risk patients (which corresponds to False Negatives). A false negative means a patient at high risk of 30-day readmission is discharged without a follow-up plan.\n")
            f.write("> The following demographic groups exhibit a False Negative Rate (FNR) that is notably higher (>5.0% absolute difference) than the overall average:\n>\n")
            for flag in flags:
                f.write(f"> - **{flag['attribute'].capitalize()}**: `{flag['group']}` has a False Negative Rate of **{flag['group_fnr']*100:.2f}%** (overall average is {flag['overall_fnr']*100:.2f}%, absolute difference: **+{flag['diff']*100:.2f}%**).\n")
            f.write(">\n")
            f.write("> **Clinician Action Required**: These groups are at a higher risk of being missed by the system. Consideration should be given to adjusting post-discharge protocols for patients in these categories or incorporating fairness constraints during retraining.\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> **No Severe Disparities Detected**\n")
            f.write("> \n")
            f.write("> All demographic groups audited (age, gender, race) fall within the safe 5.0% FNR threshold margin of the overall average. Disparities in false negative readmissions are minimal.\n")
            
    print(f"Fairness report successfully written to {output_path}")
