import os
import sys
import subprocess
from metaflow import FlowSpec, step

# Setup import path configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # monitoring/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # src/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "pipeline", "models"))  # models/

DRIFT_THRESHOLD = 0.3
RECALL_FLOOR = 0.60

class RetrainWatchFlow(FlowSpec):

    @step
    def start(self):
        # Setup local MLflow path if not set
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        if "MLFLOW_TRACKING_URI" not in os.environ:
            os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
        self.next(self.check_drift)

    @step
    def check_drift(self):
        from drift_check import compute_drift_score
        
        # Verify drifted file exists, else use reference test
        drift_file = "data/processed/synthetic/drifted_batch_severe.parquet"
        if not os.path.exists(drift_file):
            print(f"Drift file {drift_file} not found. Defaulting to reference val file for demonstration...")
            drift_file = "data/processed/val.parquet"
            
        self.drift_score = compute_drift_score(current_data_path=drift_file)
        self.next(self.check_performance)

    @step
    def check_performance(self):
        from performance_check import compute_recent_recall
        
        # Verify labelled outcomes file exists, else use reference val
        outcome_file = "data/processed/synthetic/labelled_outcomes_0.3.parquet"
        if not os.path.exists(outcome_file):
            print(f"Outcome file {outcome_file} not found. Skipping recent recall...")
            self.recent_recall = None
        else:
            self.recent_recall = compute_recent_recall(outcome_file)
            
        self.next(self.decide)

    @step
    def decide(self):
        # Check triggers
        self.drift_triggered = self.drift_score > DRIFT_THRESHOLD
        self.recall_triggered = (self.recent_recall is not None) and (self.recent_recall < RECALL_FLOOR)
        self.trigger = self.drift_triggered or self.recall_triggered
        
        from alerts import send_alert
        if self.trigger:
            reason = []
            if self.drift_triggered:
                reason.append(f"drift_score ({self.drift_score:.3f}) > threshold ({DRIFT_THRESHOLD})")
            if self.recall_triggered:
                reason.append(f"recent_recall ({self.recent_recall:.3f}) < floor ({RECALL_FLOOR})")
            
            trigger_reason = " & ".join(reason)
            send_alert(f"Retraining triggered! Reason: {trigger_reason}")
        else:
            print(f"No action required. drift_score={self.drift_score:.3f}, recent_recall={self.recent_recall}")
            
        self.next(self.execute_decision)

    @step
    def execute_decision(self):
        # Setup local MLflow path if not set
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        if "MLFLOW_TRACKING_URI" not in os.environ:
            os.environ["MLFLOW_TRACKING_URI"] = "file://" + os.path.abspath("mlruns")
            
        if not self.trigger:
            print("Skipping retraining because no alert trigger was activated.")
        else:
            print("Launching model retraining flow (readmission_flow.py)...")
            # Run main metaflow pipeline as a subprocess
            result = subprocess.run([sys.executable, "src/pipeline/readmission_flow.py", "run"], capture_output=True, text=True)
            print(result.stdout)
            
            if result.returncode != 0:
                print(result.stderr)
                raise RuntimeError(f"Retraining flow failed with code {result.returncode}")
                
            # Get the latest run ID from MLflow
            import mlflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            experiment = client.get_experiment_by_name("diabetes_readmission_experiment")
            if not experiment:
                raise ValueError("MLflow Experiment 'diabetes_readmission_experiment' not found.")
                
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if not runs:
                raise ValueError("No MLflow runs found after retraining execution.")
                
            latest_run_id = runs[0].info.run_id
            print(f"Retrained champion model run ID found: {latest_run_id}")
            
            # Compare and promote
            from champion_challenger import compare_and_promote
            self.promoted = compare_and_promote(latest_run_id, promotion_margin=0.01)
            
            from alerts import send_alert
            send_alert(f"Retrain complete. Model version from run {latest_run_id[:8]} promoted to Production: {self.promoted}")
            
        self.next(self.end)

    @step
    def end(self):
        print("Retraining watch flow execution finished.")

if __name__ == "__main__":
    RetrainWatchFlow()
