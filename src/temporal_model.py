
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TemporalArtifacts:
    scaler: StandardScaler
    encoders: Dict[str, LabelEncoder]
    feature_names: list[str]
    sensitive_columns: list[str]


class TinyTemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        logits = self.cls(pooled).squeeze(-1)
        return logits


def prepare_temporal_tensor(df: pd.DataFrame, max_weeks: int = 10) -> Tuple[np.ndarray, np.ndarray, TemporalArtifacts]:
    static_cat = ["gender", "disability", "age_band", "highest_education"]
    dyn_num = ["week_clicks", "week_active_days", "week_resources", "week_assessment_score", "week_assessment_submissions", "cum_clicks", "cum_assessment_submissions"]

    encoders = {}
    temp = df.copy()
    for col in static_cat:
        le = LabelEncoder()
        temp[col] = le.fit_transform(temp[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    temp[dyn_num] = scaler.fit_transform(temp[dyn_num])

    feature_names = dyn_num + static_cat
    students = temp["id_student"].drop_duplicates().tolist()
    X = []
    y = []
    for sid in students:
        sdf = temp[temp["id_student"] == sid].sort_values("week")
        arr = sdf[dyn_num + static_cat].to_numpy(dtype=np.float32)
        if len(arr) < max_weeks:
            pad = np.zeros((max_weeks - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        X.append(arr[:max_weeks])
        y.append(int(sdf["dropout"].iloc[0]))
    artifacts = TemporalArtifacts(scaler=scaler, encoders=encoders, feature_names=feature_names, sensitive_columns=static_cat)
    return np.stack(X), np.array(y, dtype=np.float32), artifacts


def train_temporal_transformer(train_df: pd.DataFrame, valid_df: pd.DataFrame, config: dict, seed: int = 42):
    torch.manual_seed(seed)
    max_weeks = int(train_df["week"].max())
    X_train, y_train, artifacts = prepare_temporal_tensor(train_df, max_weeks=max_weeks)
    X_valid, y_valid, _ = prepare_temporal_tensor(valid_df, max_weeks=max_weeks)

    model = TinyTemporalTransformer(
        input_dim=X_train.shape[-1],
        d_model=config.get("d_model", 32),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-3))
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=config.get("batch_size", 32),
        shuffle=True,
    )

    model.train()
    for _ in range(config.get("epochs", 3)):
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        valid_prob = torch.sigmoid(model(torch.tensor(X_valid))).numpy()
    return model, artifacts, valid_prob


def predict_temporal(model, df: pd.DataFrame) -> np.ndarray:
    max_weeks = int(df["week"].max())
    X, y, _ = prepare_temporal_tensor(df, max_weeks=max_weeks)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X))).numpy()
    return probs
