
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def make_preprocessor(categorical_cols, numeric_cols) -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipe, categorical_cols),
        ("numeric", numeric_pipe, numeric_cols),
    ])


def train_logistic(X_train, y_train, categorical_cols, numeric_cols, config: Dict[str, Any], sample_weight=None):
    pipe = Pipeline([
        ("prep", make_preprocessor(categorical_cols, numeric_cols)),
        ("clf", LogisticRegression(C=config.get("C", 1.0), max_iter=config.get("max_iter", 400))),
    ])
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
    return pipe


def train_xgboost(X_train, y_train, categorical_cols, numeric_cols, config: Dict[str, Any], sample_weight=None):
    if XGBClassifier is None:
        raise ImportError("xgboost is not available")
    prep = make_preprocessor(categorical_cols, numeric_cols)
    Xtr = prep.fit_transform(X_train)
    clf = XGBClassifier(
        n_estimators=config.get("n_estimators", 120),
        max_depth=config.get("max_depth", 4),
        learning_rate=config.get("learning_rate", 0.08),
        subsample=config.get("subsample", 0.9),
        colsample_bytree=config.get("colsample_bytree", 0.9),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(Xtr, y_train, sample_weight=sample_weight)
    return {"prep": prep, "clf": clf}


def train_catboost(X_train, y_train, categorical_cols, numeric_cols, config: Dict[str, Any], sample_weight=None):
    if CatBoostClassifier is None:
        raise ImportError("catboost is not available")
    X_train = X_train.copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
    for col in numeric_cols:
        X_train[col] = X_train[col].astype(float)
    feature_cols = categorical_cols + numeric_cols
    clf = CatBoostClassifier(
        iterations=config.get("iterations", 150),
        depth=config.get("depth", 5),
        learning_rate=config.get("learning_rate", 0.08),
        loss_function=config.get("loss_function", "Logloss"),
        verbose=False,
        random_seed=42,
    )
    cat_idx = list(range(len(categorical_cols)))
    clf.fit(X_train[feature_cols], y_train, cat_features=cat_idx, sample_weight=sample_weight)
    return {"feature_cols": feature_cols, "cat_cols": categorical_cols, "clf": clf}


def predict_proba_model(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, Pipeline):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, dict) and "prep" in model and "clf" in model:
        Xt = model["prep"].transform(X)
        return model["clf"].predict_proba(Xt)[:, 1]
    if isinstance(model, dict) and "feature_cols" in model and "clf" in model:
        Xc = X.copy()
        for col in model["feature_cols"]:
            if col in model["cat_cols"]:
                Xc[col] = Xc[col].astype(str)
            else:
                Xc[col] = Xc[col].astype(float)
        return model["clf"].predict_proba(Xc[model["feature_cols"]])[:, 1]
    raise ValueError("Unknown model type")
