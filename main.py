
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.calibration import IsotonicCalibrator, PlattCalibrator
from src.data_loading import load_oulad_tables
from src.evaluation import make_prediction_frame, summarize_binary_metrics
from src.fairness import fairness_gap_summary, kamiran_calders_reweighing, subgroup_fairness_report
from src.feature_engineering import (
    build_non_temporal_features,
    build_survival_frame,
    build_temporal_sequences,
)
from src.models import (
    predict_proba_model,
    train_catboost,
    train_logistic,
    train_xgboost,
)
from src.survival_baseline import predict_dropout_probability, train_discrete_time_hazard
from src.synthetic_data import generate_sample_oulad
from src.temporal_model import predict_temporal, train_temporal_transformer
from src.utils import ensure_dir, load_config, set_seed, write_dataframe
from src.visualization import plot_model_comparison, plot_reliability_diagram, plot_subgroup_gap


def _get_cfg(path: str | None):
    return load_config(path or "config/config.yaml")


def _common_paths(cfg):
    data_dir = Path(cfg["paths"]["data_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    for sub in ["features", "models", "predictions", "tables", "figures"]:
        ensure_dir(output_dir / sub)
    return data_dir, output_dir


def prepare(cfg):
    set_seed(cfg["random_seed"])
    data_dir, output_dir = _common_paths(cfg)
    tables = load_oulad_tables(data_dir)

    non_temporal = build_non_temporal_features(tables, early_window_weeks=cfg["task"]["early_window_weeks"])
    non_temporal["split"] = non_temporal["code_presentation"].map(
        {**{x: "train" for x in cfg["split"]["train_presentations"]},
         **{x: "valid" for x in cfg["split"]["valid_presentations"]},
         **{x: "test" for x in cfg["split"]["test_presentations"]}}
    )
    non_temporal = non_temporal.dropna(subset=["split"])

    temporal = build_temporal_sequences(tables, max_weeks=cfg["task"]["prediction_horizon_weeks"])
    student_split = non_temporal[["id_student", "split"]].drop_duplicates()
    temporal = temporal.merge(student_split, on="id_student", how="inner")

    survival = build_survival_frame(tables, max_weeks=cfg["task"]["prediction_horizon_weeks"])
    survival = survival.merge(student_split, on="id_student", how="inner")

    for split in ["train", "valid", "test"]:
        write_dataframe(non_temporal[non_temporal["split"] == split], output_dir / "features" / f"non_temporal_{split}.csv")
        write_dataframe(temporal[temporal["split"] == split], output_dir / "features" / f"temporal_{split}.csv")
        write_dataframe(survival[survival["split"] == split], output_dir / "features" / f"survival_{split}.csv")


def train_non_temporal(cfg):
    _, output_dir = _common_paths(cfg)
    train_df = pd.read_csv(output_dir / "features" / "non_temporal_train.csv")
    valid_df = pd.read_csv(output_dir / "features" / "non_temporal_valid.csv")
    test_df = pd.read_csv(output_dir / "features" / "non_temporal_test.csv")

    categorical_cols = cfg["features"]["categorical"]
    numeric_cols = cfg["features"]["numeric"]

    target = "dropout"
    X_train, y_train = train_df[categorical_cols + numeric_cols], train_df[target]
    X_valid, y_valid = valid_df[categorical_cols + numeric_cols], valid_df[target]
    X_test, y_test = test_df[categorical_cols + numeric_cols], test_df[target]

    sensitive = cfg["fairness"]["reweigh_sensitive_attribute"]
    sample_weight = kamiran_calders_reweighing(y_train, train_df[sensitive])

    models = {}

    if cfg["models"]["logistic"]["enabled"]:
        model = train_logistic(X_train, y_train, categorical_cols, numeric_cols, cfg["models"]["logistic"], sample_weight=sample_weight)
        joblib.dump(model, output_dir / "models" / "logistic.joblib")
        models["logistic"] = model

    if cfg["models"]["xgboost"]["enabled"]:
        try:
            model = train_xgboost(X_train, y_train, categorical_cols, numeric_cols, cfg["models"]["xgboost"], sample_weight=sample_weight)
            joblib.dump(model, output_dir / "models" / "xgboost.joblib")
            models["xgboost"] = model
        except Exception as e:
            print(f"Skipping xgboost: {e}")

    if cfg["models"]["catboost"]["enabled"]:
        try:
            model = train_catboost(X_train, y_train, categorical_cols, numeric_cols, cfg["models"]["catboost"], sample_weight=sample_weight)
            joblib.dump(model, output_dir / "models" / "catboost.joblib")
            models["catboost"] = model
        except Exception as e:
            print(f"Skipping catboost: {e}")

    for name, model in models.items():
        for split_name, X, y, extra in [
            ("valid", X_valid, y_valid, valid_df[["id_student"] + cfg["task"]["sensitive_attributes"]]),
            ("test", X_test, y_test, test_df[["id_student"] + cfg["task"]["sensitive_attributes"]]),
        ]:
            probs = predict_proba_model(model, X)
            pred_df = make_prediction_frame(extra["id_student"], y, probs, extra)
            write_dataframe(pred_df, output_dir / "predictions" / f"{name}_{split_name}_predictions.csv")


def train_temporal(cfg):
    _, output_dir = _common_paths(cfg)
    train_df = pd.read_csv(output_dir / "features" / "temporal_train.csv")
    valid_df = pd.read_csv(output_dir / "features" / "temporal_valid.csv")
    test_df = pd.read_csv(output_dir / "features" / "temporal_test.csv")

    model, artifacts, valid_prob = train_temporal_transformer(train_df, valid_df, cfg["models"]["temporal_transformer"], seed=cfg["random_seed"])
    torch.save(model.state_dict(), output_dir / "models" / "temporal_transformer.pt")

    for split_name, df in [("valid", valid_df), ("test", test_df)]:
        probs = predict_temporal(model, df)
        extra = df[["id_student", "gender", "disability", "age_band", "highest_education"]].drop_duplicates()
        y_true = df.groupby("id_student")["dropout"].first().values
        ids = extra["id_student"].values
        pred_df = make_prediction_frame(ids, y_true, probs, extra)
        write_dataframe(pred_df, output_dir / "predictions" / f"temporal_transformer_{split_name}_predictions.csv")


def train_survival(cfg):
    _, output_dir = _common_paths(cfg)
    train_df = pd.read_csv(output_dir / "features" / "survival_train.csv")
    valid_df = pd.read_csv(output_dir / "features" / "survival_valid.csv")
    test_df = pd.read_csv(output_dir / "features" / "survival_test.csv")

    model = train_discrete_time_hazard(train_df, C=cfg["models"]["survival_logit"]["C"])
    joblib.dump(model, output_dir / "models" / "survival_logit.joblib")

    for split_name, df in [("valid", valid_df), ("test", test_df)]:
        pred_df = predict_dropout_probability(model, df)
        sensitive = df[["id_student", "gender", "disability", "age_band", "highest_education"]].drop_duplicates()
        pred_df = pred_df.merge(sensitive, on="id_student", how="left")
        write_dataframe(pred_df, output_dir / "predictions" / f"survival_logit_{split_name}_predictions.csv")


def calibrate(cfg):
    _, output_dir = _common_paths(cfg)
    prediction_dir = output_dir / "predictions"
    files = list(prediction_dir.glob("*_valid_predictions.csv"))

    for valid_path in files:
        if "calibrated" in valid_path.name:
            continue
        model_name = valid_path.name.replace("_valid_predictions.csv", "")
        valid_df = pd.read_csv(valid_path)
        test_path = prediction_dir / f"{model_name}_test_predictions.csv"
        if not test_path.exists():
            continue
        test_df = pd.read_csv(test_path)

        for method in cfg["calibration"]["methods"]:
            if method == "platt":
                cal = PlattCalibrator().fit(valid_df["y_prob"].values, valid_df["y_true"].values)
            elif method == "isotonic":
                cal = IsotonicCalibrator().fit(valid_df["y_prob"].values, valid_df["y_true"].values)
            else:
                continue
            out = test_df.copy()
            out["y_prob"] = cal.predict(test_df["y_prob"].values)
            write_dataframe(out, prediction_dir / f"{model_name}_{method}_calibrated_predictions.csv")


def evaluate(cfg):
    _, output_dir = _common_paths(cfg)
    prediction_dir = output_dir / "predictions"
    perf_rows = []
    fairness_rows = []
    subgroup_rows = []

    for pred_path in prediction_dir.glob("*_test_predictions.csv"):
        model_name = pred_path.name.replace("_test_predictions.csv", "")
        df = pd.read_csv(pred_path)
        metrics = summarize_binary_metrics(df["y_true"], df["y_prob"], threshold=cfg["fairness"]["audit_threshold"])
        perf_rows.append({"model": model_name, "variant": "raw", **metrics})

        for attr in cfg["task"]["sensitive_attributes"]:
            if attr not in df.columns:
                continue
            subgroup_df = subgroup_fairness_report(df, attr, threshold=cfg["fairness"]["audit_threshold"])
            subgroup_df["model"] = model_name
            subgroup_rows.append(subgroup_df)
            gaps = fairness_gap_summary(subgroup_df)
            fairness_rows.append({"model": model_name, "variant": "raw", "attribute": attr, **gaps})

    for pred_path in prediction_dir.glob("*_calibrated_predictions.csv"):
        base = pred_path.name.replace("_calibrated_predictions.csv", "")
        parts = base.split("_")
        method = parts[-1]
        model_name = "_".join(parts[:-1])
        df = pd.read_csv(pred_path)
        metrics = summarize_binary_metrics(df["y_true"], df["y_prob"], threshold=cfg["fairness"]["audit_threshold"])
        perf_rows.append({"model": model_name, "variant": method, **metrics})

        for attr in cfg["task"]["sensitive_attributes"]:
            if attr not in df.columns:
                continue
            subgroup_df = subgroup_fairness_report(df, attr, threshold=cfg["fairness"]["audit_threshold"])
            subgroup_df["model"] = model_name
            subgroup_df["variant"] = method
            subgroup_rows.append(subgroup_df)
            gaps = fairness_gap_summary(subgroup_df)
            fairness_rows.append({"model": model_name, "variant": method, "attribute": attr, **gaps})

    perf_df = pd.DataFrame(perf_rows)
    fair_df = pd.DataFrame(fairness_rows)
    subgroup_df = pd.concat(subgroup_rows, ignore_index=True) if subgroup_rows else pd.DataFrame()

    write_dataframe(perf_df, output_dir / "tables" / "performance_summary.csv")
    write_dataframe(fair_df, output_dir / "tables" / "fairness_summary.csv")
    write_dataframe(subgroup_df, output_dir / "tables" / "subgroup_metrics.csv")

    calibration_rows = perf_df[["model", "variant", "brier", "ece"]].copy()
    write_dataframe(calibration_rows, output_dir / "tables" / "calibration_summary.csv")


def report(cfg):
    _, output_dir = _common_paths(cfg)
    perf_df = pd.read_csv(output_dir / "tables" / "performance_summary.csv")
    fair_df = pd.read_csv(output_dir / "tables" / "fairness_summary.csv")

    raw_best = perf_df[perf_df["variant"] == "raw"].sort_values("auroc", ascending=False)
    if len(raw_best) == 0:
        return
    plot_model_comparison(raw_best[["model", "auroc"]], "auroc", output_dir / "figures" / "model_comparison.png")

    pred_path = output_dir / "predictions" / f"{raw_best.iloc[0]['model']}_test_predictions.csv"
    pred_df = pd.read_csv(pred_path)
    plot_reliability_diagram(pred_df["y_true"].values, pred_df["y_prob"].values, output_dir / "figures" / "reliability_diagram.png")

    gap_df = fair_df[fair_df["attribute"].isin(["gender", "disability"])].copy()
    if len(gap_df) > 0:
        plot_subgroup_gap(gap_df, "max_fnr_gap", output_dir / "figures" / "subgroup_fnr_gap.png")


def smoke(cfg):
    data_dir = Path("data/sample_oulad")
    generate_sample_oulad(data_dir, seed=cfg["random_seed"], n_students=480)
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["output_dir"] = "outputs"

    # Keep smoke tests lightweight and robust across environments.
    cfg["models"]["xgboost"]["enabled"] = False
    cfg["models"]["catboost"]["enabled"] = False

    prepare(cfg)
    train_non_temporal(cfg)
    # The full temporal and survival pipelines are available for real-data runs.
    # They are skipped in the default smoke test to keep the validation lightweight.
    calibrate(cfg)
    evaluate(cfg)
    report(cfg)
    print("Smoke test completed. Outputs written to outputs/.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "prepare", "train_non_temporal", "train_temporal", "train_survival", "calibrate", "evaluate", "report"])
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = _get_cfg(args.config)
    if args.command == "smoke":
        smoke(cfg)
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "train_non_temporal":
        train_non_temporal(cfg)
    elif args.command == "train_temporal":
        train_temporal(cfg)
    elif args.command == "train_survival":
        train_survival(cfg)
    elif args.command == "calibrate":
        calibrate(cfg)
    elif args.command == "evaluate":
        evaluate(cfg)
    elif args.command == "report":
        report(cfg)
