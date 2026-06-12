import os
import json
import pandas as pd
import numpy as np

def run_profiling(raw_path="data/raw/diabetic_data.csv", output_dir="data/processed"):
    print(f"Loading raw data from {raw_path}...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    print("Replacing '?' with NaN...")
    df = df.replace("?", np.nan)
    
    # Missingness
    missing_pct = df.isnull().mean().to_dict()
    missing_pct_percent = {k: f"{v*100:.2f}%" for k, v in missing_pct.items()}
    
    # Data types
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    
    # Target distribution
    target_col = "readmitted"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. Columns: {list(df.columns)}")
    
    # Create target binary variable: <30 is 1, otherwise 0
    # Let's check original values
    orig_target_counts = df[target_col].value_counts().to_dict()
    
    # Convert to binary target
    binary_target = (df[target_col] == "<30").astype(int)
    class_counts = binary_target.value_counts().to_dict()
    class_pct = binary_target.value_counts(normalize=True).to_dict()
    
    profile = {
        "total_records": len(df),
        "num_columns": len(df.columns),
        "missing_percentage": missing_pct_percent,
        "dtypes": dtypes,
        "original_target_distribution": orig_target_counts,
        "binary_target_distribution": {
            "class_0_count": class_counts.get(0, 0),
            "class_0_pct": f"{class_pct.get(0, 0.0)*100:.2f}%",
            "class_1_count": class_counts.get(1, 0),
            "class_1_pct": f"{class_pct.get(1, 0.0)*100:.2f}%",
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "data_profile.json")
    with open(out_file, "w") as f:
        json.dump(profile, f, indent=4)
        
    print("\n--- DATA PROFILE SUMMARY ---")
    print(f"Total encounter records: {profile['total_records']}")
    print(f"Original target classes: {orig_target_counts}")
    print(f"Binary target distribution: Class 1 (<30 readmission): {profile['binary_target_distribution']['class_1_pct']}, Class 0 ({profile['binary_target_distribution']['class_0_pct']})")
    print("\nTop Columns with Missing Data:")
    sorted_missing = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)
    for col, pct in sorted_missing[:10]:
        if pct > 0:
            print(f"  {col}: {pct*100:.2f}% missing")
    print(f"\nProfile saved to {out_file}")

if __name__ == "__main__":
    run_profiling()
