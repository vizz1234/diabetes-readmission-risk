# Model Fairness Audit Report

This report evaluates the model's 30-day readmission risk prediction across key demographic groups. The audit is conducted using the savings-optimized decision threshold of **0.2700**.

## Overall Test Performance Reference

- **Overall Selection Rate**: 90.30%
- **Overall Recall (TPR)**: 97.09%
- **Overall False Negative Rate (FNR)**: 2.91%

## Disparity Audit by Age

| age      | selection_rate   | recall_tpr   | false_negative_rate   |
|:---------|:-----------------|:-------------|:----------------------|
| [0-10)   | 21.43%           | 0.00%        | 0.00%                 |
| [10-20)  | 55.45%           | 71.43%       | 28.57%                |
| [20-30)  | 72.10%           | 75.00%       | 25.00%                |
| [30-40)  | 76.47%           | 100.00%      | 0.00%                 |
| [40-50)  | 82.83%           | 92.17%       | 7.83%                 |
| [50-60)  | 85.51%           | 95.04%       | 4.96%                 |
| [60-70)  | 93.73%           | 98.47%       | 1.53%                 |
| [70-80)  | 94.40%           | 98.20%       | 1.80%                 |
| [80-90)  | 94.73%           | 99.68%       | 0.32%                 |
| [90-100) | 92.23%           | 95.83%       | 4.17%                 |

## Disparity Audit by Gender

| gender   | selection_rate   | recall_tpr   | false_negative_rate   |
|:---------|:-----------------|:-------------|:----------------------|
| Female   | 89.80%           | 97.03%       | 2.97%                 |
| Male     | 90.87%           | 97.16%       | 2.84%                 |

## Disparity Audit by Race

| race            | selection_rate   | recall_tpr   | false_negative_rate   |
|:----------------|:-----------------|:-------------|:----------------------|
| AfricanAmerican | 90.26%           | 96.64%       | 3.36%                 |
| Asian           | 85.00%           | 100.00%      | 0.00%                 |
| Caucasian       | 91.12%           | 97.31%       | 2.69%                 |
| Hispanic        | 83.45%           | 100.00%      | 0.00%                 |
| Other           | 86.34%           | 94.12%       | 5.88%                 |
| Unknown         | 73.65%           | 92.11%       | 7.89%                 |

## Fairness & Safety Assessment

> [!WARNING]
> **Disparities Detected (High False Negative Rates)**
> 
> The clinical target is to avoid missing high-risk patients (which corresponds to False Negatives). A false negative means a patient at high risk of 30-day readmission is discharged without a follow-up plan.
> The following demographic groups exhibit a False Negative Rate (FNR) that is notably higher (>5.0% absolute difference) than the overall average:
>
> - **Age**: `[10-20)` has a False Negative Rate of **28.57%** (overall average is 2.91%, absolute difference: **+25.66%**).
> - **Age**: `[20-30)` has a False Negative Rate of **25.00%** (overall average is 2.91%, absolute difference: **+22.09%**).
>
> **Clinician Action Required**: These groups are at a higher risk of being missed by the system. Consideration should be given to adjusting post-discharge protocols for patients in these categories or incorporating fairness constraints during retraining.
