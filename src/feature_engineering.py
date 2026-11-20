
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir


def _presentation_split(df: pd.DataFrame, train: list[str], valid: list[str], test: list[str]) -> pd.Series:
    split = pd.Series(index=df.index, dtype="object")
    split[df["code_presentation"].isin(train)] = "train"
    split[df["code_presentation"].isin(valid)] = "valid"
    split[df["code_presentation"].isin(test)] = "test"
    return split


def build_non_temporal_features(tables: Dict[str, pd.DataFrame], early_window_weeks: int = 4) -> pd.DataFrame:
    info = tables["studentInfo"].copy()
    reg = tables["studentRegistration"].copy()
    sa = tables["studentAssessment"].copy()
    ass = tables["assessments"].copy()
    sv = tables["studentVle"].copy()

    info["dropout"] = (info["final_result"].astype(str).str.lower() == "withdrawn").astype(int)

    key_cols = ["code_module", "code_presentation", "id_student"]

    reg = reg[key_cols + ["date_registration"]].rename(columns={"date_registration": "registration_offset"})

    ass_merged = sa.merge(ass[["id_assessment", "date"]], on="id_assessment", how="left")
    ass_merged = ass_merged[ass_merged["date"] <= early_window_weeks * 7].copy()

    assessment_features = ass_merged.groupby("id_student").agg(
        assessment_score_mean=("score", "mean"),
        assessment_score_std=("score", "std"),
        assessment_submissions=("score", "count"),
        days_to_first_submission=("date_submitted", "min"),
    ).reset_index()

    assessment_features["assessment_score_std"] = assessment_features["assessment_score_std"].fillna(0.0)
    assessment_features["days_to_first_submission"] = assessment_features["days_to_first_submission"].fillna(early_window_weeks * 7 + 1)

    sv_early = sv[sv["date"] <= early_window_weeks * 7].copy()
    click_features = sv_early.groupby("id_student").agg(
        cumulative_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        distinct_resources=("id_site", "nunique"),
    ).reset_index()

    weekly_clicks = (
        sv_early.assign(week=lambda x: (x["date"] // 7) + 1)
        .groupby(["id_student", "week"])["sum_click"]
        .sum()
        .unstack(fill_value=0)
    )
    weekly_clicks.columns = [f"clicks_week_{int(c)}" for c in weekly_clicks.columns]
    weekly_clicks = weekly_clicks.reset_index()

    feats = (
        info.merge(reg, on=key_cols, how="left")
            .merge(assessment_features, on="id_student", how="left")
            .merge(click_features, on="id_student", how="left")
            .merge(weekly_clicks, on="id_student", how="left")
    )

    for col in [
        "assessment_score_mean", "assessment_score_std", "assessment_submissions",
        "days_to_first_submission", "cumulative_clicks", "active_days", "distinct_resources"
    ]:
        feats[col] = feats[col].fillna(0)

    feats["avg_clicks_per_active_day"] = np.where(
        feats["active_days"] > 0,
        feats["cumulative_clicks"] / feats["active_days"],
        0.0,
    )
    for week in range(1, early_window_weeks + 1):
        col = f"clicks_week_{week}"
        if col not in feats.columns:
            feats[col] = 0
        feats[col] = feats[col].fillna(0)

    return feats


def build_temporal_sequences(tables: Dict[str, pd.DataFrame], max_weeks: int = 10) -> pd.DataFrame:
    info = tables["studentInfo"].copy()
    sv = tables["studentVle"].copy()
    sa = tables["studentAssessment"].copy()
    ass = tables["assessments"].copy()

    info["dropout"] = (info["final_result"].astype(str).str.lower() == "withdrawn").astype(int)

    vle_week = (
        sv.assign(week=lambda x: (x["date"] // 7) + 1)
          .query("week <= @max_weeks")
          .groupby(["id_student", "week"])
          .agg(week_clicks=("sum_click", "sum"),
               week_active_days=("date", "nunique"),
               week_resources=("id_site", "nunique"))
          .reset_index()
    )

    sa_merged = sa.merge(ass[["id_assessment", "date"]], on="id_assessment", how="left")
    sa_week = (
        sa_merged.assign(week=lambda x: (x["date"] // 7) + 1)
        .query("week <= @max_weeks")
        .groupby(["id_student", "week"])
        .agg(week_assessment_score=("score", "mean"),
             week_assessment_submissions=("score", "count"))
        .reset_index()
    )

    rows = []
    base = info[["id_student", "dropout", "gender", "disability", "age_band", "highest_education"]].drop_duplicates()

    for _, row in base.iterrows():
        sid = row["id_student"]
        user_v = vle_week[vle_week["id_student"] == sid].set_index("week")
        user_a = sa_week[sa_week["id_student"] == sid].set_index("week")
        cum_clicks = 0.0
        cum_submissions = 0.0
        for week in range(1, max_weeks + 1):
            vc = float(user_v["week_clicks"].get(week, 0.0)) if not user_v.empty else 0.0
            va = float(user_v["week_active_days"].get(week, 0.0)) if not user_v.empty else 0.0
            vr = float(user_v["week_resources"].get(week, 0.0)) if not user_v.empty else 0.0
            ac = float(user_a["week_assessment_score"].get(week, 0.0)) if not user_a.empty else 0.0
            asub = float(user_a["week_assessment_submissions"].get(week, 0.0)) if not user_a.empty else 0.0
            cum_clicks += vc
            cum_submissions += asub
            rows.append({
                "id_student": sid,
                "week": week,
                "week_clicks": vc,
                "week_active_days": va,
                "week_resources": vr,
                "week_assessment_score": ac,
                "week_assessment_submissions": asub,
                "cum_clicks": cum_clicks,
                "cum_assessment_submissions": cum_submissions,
                "dropout": int(row["dropout"]),
                "gender": row["gender"],
                "disability": row["disability"],
                "age_band": row["age_band"],
                "highest_education": row["highest_education"],
            })
    return pd.DataFrame(rows)


def build_survival_frame(tables: Dict[str, pd.DataFrame], max_weeks: int = 10) -> pd.DataFrame:
    info = tables["studentInfo"].copy()
    reg = tables["studentRegistration"].copy()
    sv = tables["studentVle"].copy()

    info["dropout"] = (info["final_result"].astype(str).str.lower() == "withdrawn").astype(int)
    reg = reg[["id_student", "date_unregistration"]].copy()
    reg["week_of_dropout"] = np.ceil(reg["date_unregistration"] / 7.0)
    reg["week_of_dropout"] = reg["week_of_dropout"].fillna(max_weeks + 1)

    base = info.merge(reg, on="id_student", how="left")
    click_week = (
        sv.assign(week=lambda x: (x["date"] // 7) + 1)
        .query("week <= @max_weeks")
        .groupby(["id_student", "week"])["sum_click"]
        .sum()
        .reset_index()
    )

    rows = []
    for _, row in base.iterrows():
        sid = row["id_student"]
        user_clicks = click_week[click_week["id_student"] == sid].set_index("week")["sum_click"].to_dict()
        cumulative = 0.0
        dropout_week = int(min(max_weeks + 1, row["week_of_dropout"])) if pd.notna(row["week_of_dropout"]) else max_weeks + 1
        for week in range(1, max_weeks + 1):
            weekly = float(user_clicks.get(week, 0.0))
            cumulative += weekly
            event = int(row["dropout"] == 1 and week == dropout_week)
            at_risk = int(week <= dropout_week if row["dropout"] == 1 else 1)
            if at_risk == 0:
                continue
            rows.append({
                "id_student": sid,
                "week": week,
                "week_clicks": weekly,
                "cum_clicks": cumulative,
                "gender": row["gender"],
                "disability": row["disability"],
                "age_band": row["age_band"],
                "highest_education": row["highest_education"],
                "event": event,
            })
    return pd.DataFrame(rows)
