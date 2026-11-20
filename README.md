# Fair and Calibrated Early-Warning Models for Student Dropout Prediction

This repository provides a reproducible implementation package for the thesis:

**Fair and Calibrated Early-Warning Models for Student Dropout Prediction Using Temporal Learning Analytics Data**

The code is designed for the **Open University Learning Analytics Dataset (OULAD)** and supports:

- OULAD-style multi-table ingestion
- non-temporal feature engineering
- weekly temporal snapshot generation
- baseline training with Logistic Regression, XGBoost, and CatBoost
- a lightweight temporal transformer baseline in PyTorch
- a discrete-time survival-style hazard baseline
- calibration with Platt scaling and isotonic regression
- fairness auditing by subgroup
- fairness-aware sample reweighing
- publication-style visualizations
- a synthetic OULAD-like sample dataset for smoke testing

## 1. Project structure

```text
fair_dropout_package/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── Dockerfile
├── main.py
├── config/
│   └── config.yaml
├── data/
│   └── sample_oulad/
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_loading.py
│   ├── synthetic_data.py
│   ├── feature_engineering.py
│   ├── fairness.py
│   ├── calibration.py
│   ├── models.py
│   ├── temporal_model.py
│   ├── survival_baseline.py
│   ├── evaluation.py
│   └── visualization.py
└── tests/
    └── test_smoke.py
```

## 2. Supported data modes

### A. Real OULAD
Place the official OULAD CSV files in a folder such as:

```text
data/raw/oulad/
├── courses.csv
├── assessments.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

Then point `data_dir` in `config/config.yaml` to that path.

### B. Smoke test
Use the bundled synthetic OULAD-like data in `data/sample_oulad/`. This is included only to verify that the pipeline runs end to end without requiring internet access.

## 3. Quick start

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the smoke test:

```bash
python main.py smoke
```

Run the full non-temporal pipeline on real OULAD:

```bash
python main.py prepare --config config/config.yaml
python main.py train_non_temporal --config config/config.yaml
python main.py calibrate --config config/config.yaml
python main.py evaluate --config config/config.yaml
python main.py report --config config/config.yaml
```

Run the temporal transformer baseline:

```bash
python main.py train_temporal --config config/config.yaml
```

Run the survival-style baseline:

```bash
python main.py train_survival --config config/config.yaml
```

## 4. Main outputs

After execution, the package writes:

```text
outputs/
├── features/
│   ├── non_temporal_train.csv
│   ├── non_temporal_valid.csv
│   ├── non_temporal_test.csv
│   ├── temporal_train.parquet
│   ├── temporal_valid.parquet
│   └── temporal_test.parquet
├── models/
│   ├── xgboost.joblib
│   ├── catboost.cbm
│   ├── logistic.joblib
│   ├── temporal_transformer.pt
│   └── survival_logit.joblib
├── predictions/
│   ├── *_valid_predictions.csv
│   ├── *_test_predictions.csv
│   └── *_calibrated_predictions.csv
├── tables/
│   ├── performance_summary.csv
│   ├── fairness_summary.csv
│   ├── subgroup_metrics.csv
│   └── calibration_summary.csv
└── figures/
    ├── roc_curve.png
    ├── precision_recall_curve.png
    ├── reliability_diagram.png
    ├── subgroup_fnr_gap.png
    └── model_comparison.png
```

## 5. Real-study execution order

1. Acquire the official OULAD CSV files from the Open University or UCI.
2. Put the files into `data/raw/oulad/`.
3. Update `config/config.yaml`.
4. Run `prepare`.
5. Run `train_non_temporal`, `train_temporal`, and `train_survival`.
6. Run `calibrate`.
7. Run `evaluate`.
8. Run `report`.
9. Insert the generated tables and figures into the thesis.

## 6. Fairness metrics implemented

The package reports, overall and by subgroup:

- AUROC
- PR-AUC
- Brier score
- Expected calibration error (ECE)
- Accuracy
- Precision
- Recall
- F1 score
- False negative rate (FNR)
- False positive rate (FPR)
- selection rate

The package also computes group gaps such as:

- max FNR gap
- max FPR gap
- max selection-rate gap
- demographic parity difference proxy
- equal opportunity difference proxy

## 7. Fairness-aware intervention included

The package includes **Kamiran-Calders reweighing** for training-set sample weighting. This is a lightweight, reproducible fairness-aware intervention that works across model families. It is supported in the logistic, XGBoost, and CatBoost pipelines.

## 8. Calibration interventions included

Two post-hoc calibration approaches are implemented:

- Platt scaling
- isotonic regression

These are fitted on the validation set and applied to the held-out test set.

## 9. Reproducibility controls

- fixed random seeds
- deterministic train/validation/test splitting by presentation
- config-driven experiment setup
- saved model artifacts
- saved predictions
- smoke-test dataset
- test script

## 10. Common issues

### `fairlearn` is not installed
This package does not require Fairlearn to run. Fairness metrics and reweighing are implemented locally. You may optionally install Fairlearn for extended audits.

### `catboost` or `xgboost` missing
Install them from `requirements.txt`. The code will otherwise skip the unavailable model and continue with the remaining baselines.

### No real OULAD data
Use `python main.py smoke` first. This validates the package using the synthetic sample data.

## 11. Suggested thesis integration

Use the outputs from:

- `outputs/tables/performance_summary.csv`
- `outputs/tables/fairness_summary.csv`
- `outputs/tables/calibration_summary.csv`
- `outputs/figures/reliability_diagram.png`
- `outputs/figures/subgroup_fnr_gap.png`

to populate Chapter 4 and the appendices.
