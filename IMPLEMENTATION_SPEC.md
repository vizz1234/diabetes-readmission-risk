# Diabetes 30-Day Readmission Risk System — Implementation Spec

## Context

Hospital network data science project. Build a production-style ML system that
predicts 30-day readmission risk for diabetic patients at discharge. Dataset:
UCI "Diabetes 130-US Hospitals" (101,766 encounters, ~11% positive class,
heavily imbalanced, missing values coded as "?", `weight` column ~97% missing).
Target: binary, `<30` readmission = 1, `{>30, NO}` = 0.

This is not a notebook deliverable. It is a repo containing a reproducible
pipeline, a versioned preprocessing artifact, three trained models compared in
MLflow, a FastAPI service (online + batch endpoints), Docker setup, monitoring,
drift-triggered retraining with alerting, and governance artifacts (fairness
audit, SHAP explanations, model card, audit log).

---

## Repo structure

```
diabetes-readmission/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── artifacts/                       # versioned preprocessing artifacts (joblib)
│
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── profile.py
│   │   └── synthetic/
│   │       ├── generate_drifted_batch.py
│   │       └── generate_labelled_outcomes.py
│   │
│   ├── pipeline/
│   │   ├── readmission_flow.py
│   │   ├── preprocessing.py
│   │   ├── icd9_mapping.py
│   │   └── models/
│   │       ├── registry.py
│   │       ├── train_eval.py
│   │       └── champion_challenger.py
│   │
│   ├── explain/
│   │   └── shap_explainer.py
│   │
│   ├── fairness/
│   │   └── audit.py
│   │
│   └── service/
│       ├── main.py
│       ├── schemas.py
│       ├── inference.py
│       ├── logging_middleware.py
│       └── monitoring.py
│
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/dashboards/readmission_dashboard.json
│   ├── drift_check.py
│   ├── performance_check.py
│   ├── alerts.py
│   └── retrain_flow.py
│
├── governance/
│   ├── model_card.md
│   ├── fairness_report.md
│   ├── reflection.md
│   └── audit_log_sample.jsonl
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_icd9_mapping.py
│   └── test_api.py
│
└── infra/
    ├── cloudrun/deploy.sh
    └── README_DEPLOY.md
```

---

## Build order

Work through these in order. Each numbered item should be a separate
commit/checkpoint.

1. **Data profiling** (`src/data/profile.py`): load raw CSV, replace `"?"` with
   `NaN`, output per-column missingness %, dtypes, class balance for the target.
2. **ICD-9 mapping** (`src/pipeline/icd9_mapping.py`):

```python
def icd9_to_category(code) -> str:
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        n = float(code)
    except ValueError:
        return "other"
    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 250 <= n < 251:
        return "diabetes"
    if 800 <= n <= 999:
        return "injury"
    if 710 <= n <= 739:
        return "musculoskeletal"
    if 580 <= n <= 629 or n == 788:
        return "genitourinary"
    if 140 <= n <= 239:
        return "neoplasms"
    return "other"
```

Apply to `diag_1`, `diag_2`, `diag_3` independently. Also add
`has_diabetes_diag = any(diag in ["diag_1","diag_2","diag_3"] maps to "diabetes")`.

3. **Preprocessing module** (`src/pipeline/preprocessing.py`):

```python
from dataclasses import dataclass
import joblib

@dataclass
class PreprocessingArtifact:
    numeric_impute_values: dict
    categorical_encoders: dict
    scaler: object
    feature_columns: list
    version: str

def fit_preprocessing(train_df) -> PreprocessingArtifact:
    """
    Fit on train_df ONLY:
    - numeric_impute_values: median of each numeric column on train
    - categorical_encoders: OrdinalEncoder per categorical column, fit on train
      categories, with handle_unknown='use_encoded_value', unknown_value=-1
    - scaler: StandardScaler on numeric columns, fit on train
    - feature_columns: final ordered list of feature names after transform
    Drop columns: weight, payer_code (impute "Unknown" if kept), encounter_id,
    patient_nbr (used for split, not as a feature).
    Apply icd9_to_category to diag_1/2/3 before encoding.
    Derived features: total_visits = number_outpatient + number_emergency +
    number_inpatient, medication_change_count from the *change columns,
    num_medications kept as-is.
    """
    ...

def apply_preprocessing(df, artifact: PreprocessingArtifact):
    """Transform df using artifact's fitted values. Never refits. Returns X, y
    (y is None if target column absent, for inference-time use)."""
    ...

def save_artifact(artifact, version, path="artifacts/"):
    joblib.dump(artifact, f"{path}/preprocessing_{version}.joblib")

def load_artifact(version, path="artifacts/"):
    return joblib.load(f"{path}/preprocessing_{version}.joblib")
```

Write unit tests in `tests/test_preprocessing.py`: fit on a small synthetic
train_df, confirm val/test transforms use train-derived medians (not their own),
confirm unknown categories at inference map to -1 without erroring.

4. **Model registry** (`src/pipeline/models/registry.py`):

```python
MODEL_GRID = {
    "logreg": {
        "estimator": "LogisticRegression",
        "fixed_params": {"class_weight": "balanced", "max_iter": 2000, "solver": "liblinear"},
        "grid": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
    },
    "random_forest": {
        "estimator": "RandomForestClassifier",
        "fixed_params": {"class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "grid": {"n_estimators": [200, 400], "max_depth": [8, None], "min_samples_leaf": [1, 10]},
    },
    "lightgbm": {
        "estimator": "LGBMClassifier",
        "fixed_params": {"is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1},
        "grid": {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1], "num_leaves": [15, 31]},
    },
}
```

5. **Train/eval helpers** (`src/pipeline/models/train_eval.py`):

```python
def run_grid_search(model_name, X_train, y_train, X_val, y_val):
    """
    GridSearchCV with StratifiedKFold(5), scoring='average_precision'.
    Refit best estimator on full X_train.
    Evaluate on X_val: pr_auc, recall_at_precision(0.3), brier_score.
    Return dict: {name, best_params, best_estimator, cv_results,
    val_pr_auc, val_recall_at_p30, val_brier}.
    """
    ...

def log_all_to_mlflow(results, preprocessing_version):
    """
    For each result, start an mlflow run:
    - log_param('algorithm', name)
    - log_params(best_params)
    - log_param('preprocessing_version', preprocessing_version)
    - log_metrics(val_pr_auc, val_recall_at_p30, val_brier,
      cv_mean_pr_auc, cv_std_pr_auc)
    - log_model(best_estimator, 'model')
    - log_artifact(cv_results as csv)
    """
    ...

def select_best(results):
    """Pick max val_pr_auc, tiebreak on lowest val_brier."""
    ...

def evaluate_on_test(best_result, X_test, y_test):
    """
    Touch test set once. Compute pr_auc, recall_at_precision(0.3),
    calibration curve data, confusion matrix at chosen threshold.
    Also run threshold selection here using a stated cost matrix:
    cost_missed_readmission = 5000, cost_unnecessary_followup = 200
    (configurable constants at top of file). Pick threshold maximizing
    expected savings = TP*cost_missed_readmission - FP*cost_unnecessary_followup
    (using validation set probabilities for threshold selection, applied
    to test only for final reporting).
    Return dict with all metrics + chosen threshold.
    """
    ...
```

6. **Metaflow pipeline** (`src/pipeline/readmission_flow.py`):

```python
from metaflow import FlowSpec, step

class ReadmissionFlow(FlowSpec):

    @step
    def start(self):
        # load raw CSV from data/raw/, replace "?" with NaN
        self.raw_df = ...
        self.next(self.split)

    @step
    def split(self):
        # split by patient_nbr: 70/15/15 train/val/test, stratify on target
        self.train_df, self.val_df, self.test_df = ...
        self.next(self.fit_preprocessing)

    @step
    def fit_preprocessing(self):
        from preprocessing import fit_preprocessing, save_artifact
        self.prep = fit_preprocessing(self.train_df)
        self.prep_version = self.run_id
        save_artifact(self.prep, self.prep_version)
        self.next(self.transform)

    @step
    def transform(self):
        from preprocessing import apply_preprocessing
        self.X_train, self.y_train = apply_preprocessing(self.train_df, self.prep)
        self.X_val, self.y_val = apply_preprocessing(self.val_df, self.prep)
        self.X_test, self.y_test = apply_preprocessing(self.test_df, self.prep)
        self.next(self.train_logreg, self.train_rf, self.train_lgbm)

    @step
    def train_logreg(self):
        from models.train_eval import run_grid_search
        self.result = run_grid_search("logreg", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def train_rf(self):
        from models.train_eval import run_grid_search
        self.result = run_grid_search("random_forest", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def train_lgbm(self):
        from models.train_eval import run_grid_search
        self.result = run_grid_search("lightgbm", self.X_train, self.y_train, self.X_val, self.y_val)
        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [i.result for i in inputs]
        from models.train_eval import log_all_to_mlflow, select_best
        log_all_to_mlflow(self.results, self.prep_version)
        self.best = select_best(self.results)
        self.next(self.final_eval)

    @step
    def final_eval(self):
        from models.train_eval import evaluate_on_test
        self.test_metrics = evaluate_on_test(self.best, self.X_test, self.y_test)
        # register best model in MLflow Model Registry, tag with prep_version
        self.next(self.end)

    @step
    def end(self):
        print(f"Best: {self.best['name']}, test PR-AUC: {self.test_metrics['pr_auc']}, "
              f"threshold: {self.test_metrics['threshold']}, prep_version: {self.prep_version}")

if __name__ == "__main__":
    ReadmissionFlow()
```

Run: `python src/pipeline/readmission_flow.py run`

7. **Champion/challenger** (`src/pipeline/models/champion_challenger.py`):

```python
def compare_and_promote(challenger_run_id, promotion_margin=0.01):
    """
    Load champion's current val_pr_auc from MLflow registry (Production stage).
    Load challenger's val_pr_auc from the given run.
    If challenger >= champion + promotion_margin, transition challenger to
    Production, archive previous champion. Return True/False (promoted).
    If no champion exists yet, promote unconditionally.
    """
    ...
```

8. **SHAP explainer** (`src/explain/shap_explainer.py`): load registered model +
matching preprocessing artifact, `TreeExplainer` for tree models /
`LinearExplainer` for logreg (branch on algorithm type). Provide
`get_global_importance()` (mean |SHAP| bar data) and
`get_top_factors(single_row, n=3)` returning `[{feature, value, shap_value}]`
for per-patient explanations used by the API.

9. **Fairness audit** (`src/fairness/audit.py`): use Fairlearn `MetricFrame`,
   group by `age` (bucketed), `gender`, `race`. Compute `selection_rate`,
   `recall` (TPR), `false_negative_rate` per group at the chosen threshold.
   Output `governance/fairness_report.md` with a table per group and a
   paragraph flagging any group with FNR notably higher than overall (this is
   the clinically dangerous direction — high-risk patients in that group
   getting missed).

10. **FastAPI service** (`src/service/`):

`schemas.py`:
```python
class PatientFeatures(BaseModel):
    # all raw input columns the model needs, pre-preprocessing
    ...

class PredictResponse(BaseModel):
    risk_score: float
    risk_band: str          # "low" / "medium" / "high" based on threshold
    top_factors: list[dict] # from shap_explainer.get_top_factors
    model_version: str
    preprocessing_version: str

class BatchPredictRequest(BaseModel):
    patients: list[PatientFeatures]

class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    batch_id: str
```

`inference.py`: load champion model + matching preprocessing artifact at
startup (FastAPI lifespan event), expose `score_one(features_dict)` and
`score_batch(list_of_dicts)`.

`main.py`:
```python
@app.post("/predict", response_model=PredictResponse)
async def predict(patient: PatientFeatures):
    # online: single patient, real-time, used at discharge

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    # offline: nightly scoring of all current inpatients, writes results
    # to data/processed/batch_results/{batch_id}.jsonl in addition to returning
```

`logging_middleware.py`: on every `/predict` and `/predict/batch` call, append a
JSON line to `governance/audit_log.jsonl`:
```json
{"timestamp": "...", "model_version": "...", "preprocessing_version": "...",
 "endpoint": "predict", "input_hash": "...", "risk_score": 0.42,
 "threshold_used": 0.35, "decision": "high"}
```

`monitoring.py`: wire up `prometheus-fastapi-instrumentator` on the FastAPI
app (request count, latency histogram, error rate). Also track a custom
Histogram for `risk_score` distribution.

11. **Docker**: `Dockerfile` builds the service (multi-stage: builder installs
deps, runtime copies `src/service`, `artifacts/`, model files). `docker-compose.yml`
brings up: `api` (the FastAPI service), `prometheus`, `grafana`, and `mlflow`
(local tracking server pointed at `./mlflow/` volume). One command:
`docker compose up`.

12. **Monitoring config**: `monitoring/prometheus.yml` scrapes the `api`
service's `/metrics` endpoint every 15s. `monitoring/grafana/dashboards/`
contains a dashboard JSON with panels for request rate, latency p50/p95, error
rate, and risk score distribution histogram.

13. **Synthetic data generators**:

`src/data/synthetic/generate_drifted_batch.py`: takes `data/processed/test.parquet`,
applies configurable shifts (`--shift mild|severe`):
- `age`: shift distribution one bucket older
- `number_inpatient`, `number_emergency`: multiply by 1.3 (severe) or 1.1 (mild)
- inject a new unseen value into `medical_specialty` for severe shift

Outputs `data/processed/synthetic/drifted_batch_{shift}.parquet`.

`src/data/synthetic/generate_labelled_outcomes.py`: takes a batch of test rows,
scores with current champion model, generates synthetic "actual outcomes"
column. Flag `--decay-rate 0.0|0.3`: at 0.0, use real `y_test` labels (recall
should look normal); at 0.3, flip 30% of labels from 0→1 in the
highest-predicted-risk decile (simulates the model's top predictions no longer
correlating with reality, recall drops). Outputs
`data/processed/synthetic/labelled_outcomes_{decay_rate}.parquet`.

14. **Drift & performance checks**:

`monitoring/drift_check.py`:
```python
def compute_drift_score(current_data_path, reference_data_path="data/processed/train.parquet"):
    """
    Use Evidently's DataDriftPreset comparing current_data_path against
    reference (training distribution). Return the dataset drift share
    (0-1 fraction of columns with detected drift). Also write the full
    HTML report to monitoring/evidently/drift_report_{timestamp}.html.
    """
```

`monitoring/performance_check.py`:
```python
def compute_recent_recall(labelled_outcomes_path):
    """
    Load labelled outcomes (predictions + synthetic actuals). Compute recall
    at the production threshold. Return float, or None if file doesn't exist
    (no labels available yet).
    """
```

15. **Alerts** (`monitoring/alerts.py`):
```python
import os, requests

def send_alert(message: str):
    webhook = os.environ.get("ALERT_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"text": message})
    else:
        print(f"[ALERT] {message}")
```

16. **Retrain watcher** (`monitoring/retrain_flow.py`):
```python
from metaflow import FlowSpec, step

DRIFT_THRESHOLD = 0.3
RECALL_FLOOR = 0.60

class RetrainWatchFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.check_drift)

    @step
    def check_drift(self):
        from drift_check import compute_drift_score
        self.drift_score = compute_drift_score(
            current_data_path="data/processed/synthetic/drifted_batch_severe.parquet"
        )
        self.next(self.check_performance)

    @step
    def check_performance(self):
        from performance_check import compute_recent_recall
        self.recent_recall = compute_recent_recall(
            "data/processed/synthetic/labelled_outcomes_0.3.parquet"
        )
        self.next(self.decide)

    @step
    def decide(self):
        self.trigger = (self.drift_score > DRIFT_THRESHOLD) or \
                        (self.recent_recall is not None and self.recent_recall < RECALL_FLOOR)
        from alerts import send_alert
        if self.trigger:
            send_alert(f"Retraining triggered. drift_score={self.drift_score:.3f}, "
                       f"recent_recall={self.recent_recall}")
            self.next(self.retrain)
        else:
            self.next(self.no_action)

    @step
    def retrain(self):
        import subprocess
        subprocess.run(["python", "src/pipeline/readmission_flow.py", "run"])
        from champion_challenger import compare_and_promote
        # get latest run_id, compare_and_promote it
        promoted = compare_and_promote(latest_run_id)
        from alerts import send_alert
        send_alert(f"Retrain complete. Promoted: {promoted}")
        self.next(self.end)

    @step
    def no_action(self):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    RetrainWatchFlow()
```

17. **Governance docs**: write `governance/model_card.md` (intended use,
training data summary, the three models' val metrics, chosen model + reason,
performance + fairness findings, limitations), `governance/fairness_report.md`
(from step 9), and `governance/reflection.md` (1-2 pages, written last,
referencing actual numbers from the run: which model won and why, threshold
chosen and the cost matrix used, fairness disparities found, what would change
in real production e.g. real labels instead of synthetic, real alerting
channel, Feast-style feature store at scale).

18. **Deployment**: `infra/cloudrun/deploy.sh`:
```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/diabetes-api
gcloud run deploy diabetes-api \
  --image gcr.io/$PROJECT_ID/diabetes-api \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi
```
`infra/README_DEPLOY.md`: rollback = `gcloud run services update-traffic
diabetes-api --to-revisions=PREVIOUS_REVISION=100`. Note $300/90-day GCP credit,
set budget alert.

19. **README.md**: top-level quickstart reproducing everything:
```bash
git clone <repo> && cd diabetes-readmission
make data       # download + profile -> data/raw/
make train      # run readmission_flow.py, logs to MLflow
docker compose up   # api + prometheus + grafana + mlflow
curl -X POST localhost:8000/predict -d @sample_patient.json
python monitoring/retrain_flow.py run   # demo drift -> alert -> retrain -> promote
```

---

## Notes for whoever (Antigravity) builds this

- Test set is touched exactly once, inside `final_eval`. No code path outside
  that step should ever load `test_df` or `X_test`/`y_test`.
- `fit_preprocessing` must never be called on val/test/production data — only
  `apply_preprocessing` with a loaded artifact.
- Every model+preprocessing pairing is identified by the same Metaflow
  `run_id` / MLflow `preprocessing_version` tag — the service and the retrain
  flow must always load both from the same version, never mix.
- PR-AUC is the headline metric throughout. Accuracy is not reported as a
  primary metric anywhere, including the model card.
- All four cost-matrix constants (cost_missed_readmission,
  cost_unnecessary_followup, DRIFT_THRESHOLD, RECALL_FLOOR, promotion_margin)
  live as named constants at the top of their respective files, not buried
  inline, so they're easy to find and justify in the reflection.
