from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

# Custom Prometheus metric to monitor risk score distribution
RISK_SCORE_HISTOGRAM = Histogram(
    "diabetes_readmission_risk_score",
    "Distribution of predicted 30-day readmission risk scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

def setup_monitoring(app: FastAPI):
    """
    Configures prometheus-fastapi-instrumentator to collect standard HTTP metrics
    (request counts, latency histograms, error rates) and exposes them on /metrics.
    """
    instrumentator = Instrumentator()
    
    # Instrument the app and expose the /metrics endpoint
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    print("Prometheus monitoring configured on /metrics")
