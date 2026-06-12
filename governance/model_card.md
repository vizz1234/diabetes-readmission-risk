# Model Card — Diabetes 30-Day Readmission Risk Predictor

## Intended Use
This machine learning model predicts whether a diabetic patient discharged from a hospital network is at high risk of 30-day unplanned readmission. It is designed to be integrated into clinical workflows at discharge to help clinicians identify patients who would benefit from targeted transition-of-care interventions (such as phone call follow-ups, medication reconciliations, or home visits).

- **Primary Users**: Care managers, discharge planners, and clinicians.
- **Out of Scope**: Diagnosing diabetic complications or determining clinical treatments.

## Training Data Summary
- **Source**: UCI "Diabetes 130-US Hospitals" dataset (years 1999-2008).
- **Size**: 101,766 encounter records.
- **Split**: 70% Train, 15% Validation, 15% Test.
- **Characteristics**: Heavily imbalanced target class (approx. 11% readmissions). Missing values coded as "?" (e.g. weight is ~97% missing).
- **Target Variable**: Binary, where `1` represents readmission `<30` days, and `0` represents readmission `>30` days or no readmission.

## Model Performance
The system compares three algorithms: Logistic Regression (`logreg`), Random Forest (`random_forest`), and LightGBM (`lightgbm`). Grid Search is performed using 5-fold Stratified Cross-Validation on average precision (PR-AUC). The champion model is selected based on maximum validation PR-AUC, with Brier score as a tiebreaker.

*Values below will be populated from the active training run.*

| Model Name | CV Mean PR-AUC | Validation PR-AUC | Validation Brier Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.1964 | 0.2040 | 0.2298 |
| **Random Forest** | 0.2048 | 0.2145 | 0.2252 |
| **LightGBM** | 0.2116 | 0.2298 | 0.2171 |

### Final Champion Performance (Test Set)
- **Champion Algorithm**: LightGBM (LGBMClassifier)
- **Test PR-AUC**: 0.2146
- **Brier Score (Calibration)**: 0.2191
- **Cost-Optimized Threshold**: 0.2700
  - *Optimized to maximize expected savings where cost of missed readmission = $5000 and unnecessary follow-up = $200.*
- **Test Recall (TPR) at Threshold**: 97.09%
- **Confusion Matrix**:
  - TN: 1432 | FP: 12146
  - FN: 49 | TP: 1636

## Fairness & Safety Assessment
The model was audited for false negative rates (FNR), selection rates, and recall across gender, race, and age buckets using Fairlearn. 
- **Findings**: The audit detected notable FNR disparities in younger age groups. Patients in the `[10-20)` group had a False Negative Rate of **28.57%** (+25.66% higher than overall average) and the `[20-30)` group had **25.00%** (+22.09% higher than overall average). These represent young patients who are much more likely to be missed by the system. Conversely, race and gender categories showed minimal disparities, staying within a safe 5.0% FNR margin of the overall average.
- **Safety Concern**: False Negatives are the clinically dangerous direction, representing high-risk patients who are missed by the system and discharged without follow-up care.

## Limitations
1. **Historical Data**: The dataset represents clinical practices from 1999-2008. Diagnostic coding patterns (ICD-9 vs ICD-10) and clinical guidelines for diabetes have since changed.
2. **Missing Clinical Indicators**: Vital signs, lab values (other than HbA1c/glucose), and social determinants of health (SDOH) are not present in this dataset.
3. **Synthetic Outcome Limitations**: Retraining triggers rely on synthetic outcome decay, which might not reflect actual clinical outcomes.
