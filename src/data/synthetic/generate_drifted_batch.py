import os
import argparse
import pandas as pd
import numpy as np

def generate_drifted_data(shift_type="severe", input_path="data/processed/test.parquet", output_dir="data/processed/synthetic"):
    print(f"Loading reference test data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Reference test file not found at {input_path}. Run metaflow pipeline first.")
        
    df = pd.read_parquet(input_path)
    drifted_df = df.copy()
    
    # 1. Shift age distribution one bucket older
    age_map = {
        "[0-10)": "[10-20)",
        "[10-20)": "[20-30)",
        "[20-30)": "[30-40)",
        "[30-40)": "[40-50)",
        "[40-50)": "[50-60)",
        "[50-60)": "[60-70)",
        "[60-70)": "[70-80)",
        "[70-80)": "[80-90)",
        "[80-90)": "[90-100)",
        "[90-100)": "[90-100)"
    }
    drifted_df["age"] = drifted_df["age"].map(lambda x: age_map.get(x, x))
    
    # 2. Scale number_inpatient and number_emergency
    factor = 1.3 if shift_type == "severe" else 1.1
    print(f"Applying factor {factor} to inpatient and emergency visits ({shift_type} shift)...")
    
    # Ensure they are numeric
    for col in ["number_inpatient", "number_emergency"]:
        drifted_df[col] = pd.to_numeric(drifted_df[col], errors="coerce").fillna(0)
        drifted_df[col] = (drifted_df[col] * factor).round().astype(int)
        
    # 3. For severe shift, inject new unseen value in medical_specialty
    if shift_type == "severe":
        print("Injecting unseen category 'SuperSpecialtyXYZ' into medical_specialty...")
        # Replace 20% of values with new unseen category
        mask = np.random.rand(len(drifted_df)) < 0.20
        drifted_df.loc[mask, "medical_specialty"] = "SuperSpecialtyXYZ"
        
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"drifted_batch_{shift_type}.parquet")
    drifted_df.to_parquet(output_path, index=False)
    print(f"Drifted dataset successfully written to {output_path} (records: {len(drifted_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate drifted batch parquet files.")
    parser.add_argument("--shift", type=str, choices=["mild", "severe"], default="severe", help="Type of shift (mild or severe)")
    args = parser.parse_args()
    
    generate_drifted_data(shift_type=args.shift)
