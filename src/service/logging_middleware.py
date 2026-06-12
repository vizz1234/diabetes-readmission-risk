import os
import json
import time
import hashlib

def calculate_input_hash(features_dict: dict) -> str:
    """Computes a stable SHA-256 hash for the patient input dictionary."""
    # Ensure keys are sorted for stability
    serialized = json.dumps(features_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def log_prediction_audit(
    endpoint: str,
    model_version: str,
    prep_version: str,
    threshold: float,
    input_data: dict,
    risk_score: float,
    decision: str
):
    """
    Appends a single structured JSON audit record to governance/audit_log.jsonl.
    """
    log_dir = "governance"
    log_file = os.path.join(log_dir, "audit_log.jsonl")
    
    os.makedirs(log_dir, exist_ok=True)
    
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_version": model_version,
        "preprocessing_version": prep_version,
        "endpoint": endpoint,
        "input_hash": calculate_input_hash(input_data),
        "risk_score": float(risk_score),
        "threshold_used": float(threshold),
        "decision": decision
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")
