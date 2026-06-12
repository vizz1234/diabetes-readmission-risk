import os
import time
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def compute_drift_score(current_data_path: str, reference_data_path: str = "data/processed/train.parquet") -> float:
    """
    Use Evidently's DataDriftPreset comparing current_data_path against
    reference (training distribution). Return the dataset drift share
    (0-1 fraction of columns with detected drift). Also write the full
    HTML report to monitoring/evidently/drift_report_{timestamp}.html.
    """
    print(f"Comparing current dataset: {current_data_path} with reference: {reference_data_path}")
    if not os.path.exists(reference_data_path):
        raise FileNotFoundError(f"Reference training parquet not found at {reference_data_path}")
    if not os.path.exists(current_data_path):
        raise FileNotFoundError(f"Current batch parquet not found at {current_data_path}")
        
    ref_df = pd.read_parquet(reference_data_path)
    cur_df = pd.read_parquet(current_data_path)
    
    # Align columns by removing ID/Metadata/Target columns to focus on model features
    cols_to_drop = ["encounter_id", "patient_nbr", "readmitted", "weight", "payer_code"]
    common_cols = [col for col in ref_df.columns if col not in cols_to_drop and col in cur_df.columns]
    
    ref_df_clean = ref_df[common_cols].copy()
    cur_df_clean = cur_df[common_cols].copy()
    
    # Initialize and run the Evidently Data Drift Report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df_clean, current_data=cur_df_clean)
    
    # Parse the drift score from the Snapshot dict
    report_dict = snapshot.dict()
    drift_share = 0.0
    for metric in report_dict.get("metrics", []):
        if metric.get("config", {}).get("type") == "evidently:metric_v2:DriftedColumnsCount":
            drift_share = metric.get("value", {}).get("share", 0.0)
            break
            
    # Save the HTML report
    timestamp = int(time.time())
    output_dir = "monitoring/evidently"
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    snapshot.save_html(html_path)
    
    print(f"Evidently report saved to: {html_path}")
    print(f"Data drift share computed: {drift_share:.4f}")
    
    return float(drift_share)

if __name__ == "__main__":
    # If run as standalone, compare severe drifted batch by default
    default_batch = "data/processed/synthetic/drifted_batch_severe.parquet"
    if os.path.exists(default_batch):
        compute_drift_score(default_batch)
    else:
        print(f"Default batch {default_batch} not found. Please run generator scripts first.")
