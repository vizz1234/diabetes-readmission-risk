import os
import sys
import uuid
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

# Set up local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.schemas import PatientFeatures, PredictResponse, BatchPredictRequest, BatchPredictResponse
from service.inference import engine
from service.logging_middleware import log_prediction_audit
from service.monitoring import setup_monitoring, RISK_SCORE_HISTOGRAM

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and preprocessing resources
    try:
        engine.load_resources()
    except Exception as e:
        print(f"Startup warning: Resource loading failed: {e}. API starting without pre-loaded resources.")
    yield
    # Shutdown: clean up resources if needed
    pass

app = FastAPI(
    title="Diabetes 30-Day Readmission Risk System",
    description="FastAPI service for predicting patient 30-day readmission risk at discharge.",
    version="1.0.0",
    lifespan=lifespan
)

# Wire up Prometheus instrumentation
setup_monitoring(app)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service and resource status."""
    is_ready = engine.model is not None and engine.prep is not None
    return {
        "status": "healthy" if is_ready else "uninitialized",
        "model_version": engine.model_version,
        "preprocessing_version": engine.prep_version,
        "savings_threshold": engine.threshold
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(patient: PatientFeatures):
    """
    Online endpoint: single patient, real-time prediction at discharge.
    """
    if engine.model is None or engine.prep is None:
        raise HTTPException(status_code=503, detail="Model resources are not loaded yet.")
        
    try:
        # Pydantic dict by alias handles the hyphens in the medication names correctly
        patient_dict = patient.model_dump(by_alias=True)
        
        # Core inference
        prediction = engine.predict_single(patient_dict)
        
        # Log to custom Prometheus metric
        RISK_SCORE_HISTOGRAM.observe(prediction["risk_score"])
        
        # Audit log record
        decision = "high" if prediction["risk_score"] >= engine.threshold else "low"
        log_prediction_audit(
            endpoint="predict",
            model_version=prediction["model_version"],
            prep_version=prediction["preprocessing_version"],
            threshold=engine.threshold,
            input_data=patient_dict,
            risk_score=prediction["risk_score"],
            decision=decision
        )
        
        return PredictResponse(
            risk_score=prediction["risk_score"],
            risk_band=prediction["risk_band"],
            top_factors=prediction["top_factors"],
            model_version=prediction["model_version"],
            preprocessing_version=prediction["preprocessing_version"]
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Offline batch endpoint: nightly scoring of all current inpatients.
    Writes results to data/processed/batch_results/{batch_id}.jsonl.
    """
    if engine.model is None or engine.prep is None:
        raise HTTPException(status_code=503, detail="Model resources are not loaded yet.")
        
    try:
        batch_id = str(uuid.uuid4())
        patients_list = [p.model_dump(by_alias=True) for p in request.patients]
        
        # Core inference
        predictions = engine.predict_batch(patients_list)
        
        # Save results to processed directory
        batch_dir = "data/processed/batch_results"
        os.makedirs(batch_dir, exist_ok=True)
        batch_file = os.path.join(batch_dir, f"{batch_id}.jsonl")
        
        results_responses = []
        with open(batch_file, "w") as f:
            for patient_dict, pred in zip(patients_list, predictions):
                # Format single response object
                resp = PredictResponse(
                    risk_score=pred["risk_score"],
                    risk_band=pred["risk_band"],
                    top_factors=pred["top_factors"],
                    model_version=pred["model_version"],
                    preprocessing_version=pred["preprocessing_version"]
                )
                results_responses.append(resp)
                
                # Observe in Prometheus
                RISK_SCORE_HISTOGRAM.observe(pred["risk_score"])
                
                # Write to batch output file
                record = {
                    "input": patient_dict,
                    "prediction": resp.model_dump()
                }
                f.write(json.dumps(record) + "\n")
                
                # Audit log record
                decision = "high" if pred["risk_score"] >= engine.threshold else "low"
                log_prediction_audit(
                    endpoint="predict/batch",
                    model_version=pred["model_version"],
                    prep_version=pred["preprocessing_version"],
                    threshold=engine.threshold,
                    input_data=patient_dict,
                    risk_score=pred["risk_score"],
                    decision=decision
                )
                
        print(f"Batch prediction {batch_id} saved to {batch_file} ({len(patients_list)} records)")
        
        return BatchPredictResponse(
            results=results_responses,
            batch_id=batch_id
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
