# Student Early-Warning System: Fair and Calibrated Dropout Prediction

An early-warning system that predicts which students will withdraw from a course, built on the Open University Learning Analytics Dataset (OULAD). The project asks more than "which model ranks students best?" It asks whether the predicted probabilities can be trusted and whether the model's errors are spread fairly across student groups.

> **Status: work in progress.** The cohort definition and the static-feature baseline are done. Temporal features, calibration and fairness analysis are in progress.

## Research questions

1. Which baseline and temporal models perform best for early dropout prediction on OULAD?
2. Are the predicted probabilities well calibrated, overall and within subgroups?
3. Can calibration and fairness-aware methods improve decision quality without a large loss in performance?

## Problem definition

- **Outcome:** `dropout = 1` if `final_result == "Withdrawn"`, otherwise `0` (Pass, Fail and Distinction all count as 0).
- **Prediction point:** day 28 (end of week 4). Only students still enrolled at day 28 are included, so the model never predicts students who had already left.
- **Split:** chronological. Train on presentations 2013B, 2013J and 2014B; test on 2014J. This avoids using future cohorts to predict past ones.
- **Protected attributes:** `gender`, `disability` and `age_band` are **not** used as model inputs. They are kept only to audit fairness.



## Roadmap

- [x] Load and inspect `studentInfo.csv`, define the dropout outcome
- [x] Fix the day-28 cohort and chronological split
- [x] Train a static-feature baseline and save test predictions
- [ ] Full evaluation: precision, recall, F1, ROC-AUC, PR-AUC
- [ ] Weekly VLE activity features from `studentVle.csv` (strictly before the cutoff)
- [ ] Boosted-tree model (XGBoost or CatBoost) and discrete-time survival model
- [ ] Calibration: reliability curves, Brier score, ECE, post-hoc calibration (Platt / isotonic)
- [ ] Fairness audit: TPR, FPR and calibration gaps by gender, disability and age band, with group sizes
- [ ] Mitigation: reweighing and group-specific thresholds

## Repository contents

| File | Description |
|---|---|
| `Dataset.ipynb` | Data loading and inspection notebook |
| `baseline_predictions.csv` | Test-set predictions from the baseline (outcome, probability, group labels) |
| `cohort_day28.csv` | Cleaned cohort of students enrolled at day 28 |



## Tech stack

Python, pandas, scikit-learn, Jupyter Notebook (Anaconda), Git/GitHub. Planned: XGBoost or CatBoost, matplotlib.

