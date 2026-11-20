
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_FILES = [
    "courses.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
]


def validate_oulad_dir(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required OULAD files in {data_dir}: {missing}"
        )


def load_oulad_tables(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    validate_oulad_dir(data_dir)
    tables = {
        "courses": pd.read_csv(data_dir / "courses.csv"),
        "assessments": pd.read_csv(data_dir / "assessments.csv"),
        "studentAssessment": pd.read_csv(data_dir / "studentAssessment.csv"),
        "studentInfo": pd.read_csv(data_dir / "studentInfo.csv"),
        "studentRegistration": pd.read_csv(data_dir / "studentRegistration.csv"),
        "studentVle": pd.read_csv(data_dir / "studentVle.csv"),
        "vle": pd.read_csv(data_dir / "vle.csv"),
    }
    return tables
