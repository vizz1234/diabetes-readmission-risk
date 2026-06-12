# System Reflection and Post-Mortem

## Experimental Results and Winning Model
In this project, we evaluated three models to predict 30-day diabetes readmissions: Logistic Regression, Random Forest, and LightGBM.

- **Winning Model**: LightGBM (LGBMClassifier)
- **Reasoning**: The champion model was selected because it achieved the highest validation PR-AUC of **0.2298**. Accuracy was intentionally excluded from model selection metrics as the dataset is highly imbalanced (~11% readmissions). Logistic Regression provides baseline linear interpretability, but the tree-based models (Random Forest and LightGBM) are better able to capture complex interactions between medication changes and clinical visits.

## Decision Threshold Optimization & Cost Matrix
Standard classifiers output a probability score. To translate this into a binary clinical decision (e.g., whether to deploy post-discharge follow-up care), we optimized the threshold to maximize clinical savings.
We defined the cost matrix as:
- **Cost of a missed readmission (False Negative)**: $5,000 (representing the cost of an emergency re-hospitalization).
- **Cost of unnecessary follow-up (False Positive)**: $200 (representing care manager outreach time).

The savings-maximizing threshold chosen was **0.2700** (derived on validation set predictions). At this threshold, the model achieved:
- **Test Recall (TPR)**: 97.09% (meaning we caught 97.09% of patients who actually readmitted).
- **Expected Savings per Patient**: $376.78 compared to a baseline of discharging all patients without follow-up (total of $5,750,800 savings on the 15,263 test encounters).

## Fairness Disparities Found
Our demographic audit using Fairlearn flagged the following disparities in False Negative Rates:
- **Findings**: The model displayed elevated FNRs in younger cohorts, specifically age groups `[10-20)` at 28.57% and `[20-30)` at 25.00%, compared to the overall test FNR average of 2.91%. This is likely due to the low prevalence of readmission events in younger diabetic patients, making the model overly conservative. In contrast, racial and gender categories did not present significant disparities, maintaining FNR variations within 5.0% of the overall average.

Clinically, a high False Negative Rate is hazardous because it means high-risk patients in these groups are missed. We flag these groups for clinicians to ensure they receive human-in-the-loop oversight at discharge, independent of the model's output.

## Transitioning to a Real Production Environment
To scale this proof-of-concept system into a true enterprise hospital environment, we would recommend several modifications:

1. **Real Labeled Outcomes**: Instead of using synthetic outcome outcomes (where we flipped labels to simulate decay), we would ingest actual hospital admission records from the Electronic Health Record (EHR) database 30 days after each discharge.
2. **Feature Store Integration**: At hospital scale, we would deploy a feature store like Feast. This prevents training-serving skew by serving pre-computed, point-in-time features (like `total_visits` or `medication_change_count`) directly to the FastAPI service.
3. **Real Alerting Infrastructure**: In place of simple console logs, we would connect the `monitoring/alerts.py` webhook to an enterprise Slack channel, PagerDuty, or an email service (like SendGrid) to notify the on-call MLOps engineer immediately when drift triggers.
4. **Active Learning and Retraining**: We would deploy the retraining pipeline on a scheduled cron job (e.g., monthly) using Apache Airflow or Google Cloud Composer, rather than relying on manual file checks.
