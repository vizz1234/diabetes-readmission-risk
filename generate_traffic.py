import pandas as pd
import requests
import json
import random
import time

def main():
    print("Loading test data for traffic simulation...")
    df = pd.read_parquet("data/processed/test.parquet")
    
    # Replace NaN/None values with standard pydantic defaults to prevent errors
    df = df.fillna("?")
    
    # Select columns matching PatientFeatures Pydantic model
    # Convert dataframe rows to list of dictionaries
    records = df.to_dict(orient="records")
    
    # Select a subset of records to send
    num_single = min(100, len(records))
    single_records = records[:num_single]
    
    print(f"Sending {num_single} online prediction requests to /predict...")
    for i, record in enumerate(single_records):
        # Clean record keys to match expected field names (handling hyphens if needed)
        cleaned_record = {}
        for k, v in record.items():
            if k == "glyburide-metformin":
                cleaned_record["glyburide_metformin"] = v
            elif k == "glipizide-metformin":
                cleaned_record["glipizide_metformin"] = v
            elif k == "glimepiride-pioglitazone":
                cleaned_record["glimepiride_pioglitazone"] = v
            elif k == "metformin-rosiglitazone":
                cleaned_record["metformin_rosiglitazone"] = v
            elif k == "metformin-pioglitazone":
                cleaned_record["metformin_pioglitazone"] = v
            else:
                cleaned_record[k] = v
                
        try:
            resp = requests.post("http://localhost:8000/predict", json=cleaned_record)
            if resp.status_code != 200:
                print(f"Failed at {i}: {resp.text}")
            if i % 20 == 0:
                print(f"Sent {i} single predictions...")
        except Exception as e:
            print(f"Connection failed: {e}")
            break
        time.sleep(0.05)
        
    print("Sending batch prediction requests to /predict/batch...")
    batch1 = records[num_single:num_single+50]
    batch2 = records[num_single+50:num_single+100]
    
    for batch_idx, batch_records in enumerate([batch1, batch2]):
        cleaned_batch = []
        for r in batch_records:
            cleaned_record = {}
            for k, v in r.items():
                if k == "glyburide-metformin":
                    cleaned_record["glyburide_metformin"] = v
                elif k == "glipizide-metformin":
                    cleaned_record["glipizide_metformin"] = v
                elif k == "glimepiride-pioglitazone":
                    cleaned_record["glimepiride_pioglitazone"] = v
                elif k == "metformin-rosiglitazone":
                    cleaned_record["metformin_rosiglitazone"] = v
                elif k == "metformin-pioglitazone":
                    cleaned_record["metformin_pioglitazone"] = v
                else:
                    cleaned_record[k] = v
            cleaned_batch.append(cleaned_record)
            
        payload = {"patients": cleaned_batch}
        try:
            resp = requests.post("http://localhost:8000/predict/batch", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                print(f"Batch {batch_idx+1} succeeded. ID: {data['batch_id']}")
            else:
                print(f"Batch {batch_idx+1} failed: {resp.text}")
        except Exception as e:
            print(f"Connection failed for batch: {e}")
            
    print("Traffic generation completed successfully!")

if __name__ == "__main__":
    main()
