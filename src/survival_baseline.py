
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_discrete_time_hazard(df: pd.DataFrame, C: float = 1.0):
    features = ["week", "week_clicks", "cum_clicks", "gender", "disability", "age_band", "highest_education"]
    categorical = ["gender", "disability", "age_band", "highest_education"]
    numeric = ["week", "week_clicks", "cum_clicks"]

    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=False)),
        ]), numeric),
    ])

    clf = LogisticRegression(C=C, max_iter=400)
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(df[features], df["event"])
    return pipe


def predict_dropout_probability(pipe, df: pd.DataFrame) -> pd.DataFrame:
    features = ["week", "week_clicks", "cum_clicks", "gender", "disability", "age_band", "highest_education"]
    temp = df.copy()
    temp["hazard"] = pipe.predict_proba(temp[features])[:, 1]
    out = []
    for sid, sdf in temp.groupby("id_student"):
        sdf = sdf.sort_values("week")
        survival = 1.0
        for _, row in sdf.iterrows():
            survival *= (1.0 - row["hazard"])
        dropout_prob = 1.0 - survival
        out.append({"id_student": sid, "y_prob": float(dropout_prob), "y_true": int(sdf["event"].max())})
    return pd.DataFrame(out)
