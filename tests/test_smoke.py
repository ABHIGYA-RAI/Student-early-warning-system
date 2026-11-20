
from pathlib import Path
import pandas as pd
import subprocess
import sys


def test_smoke_runs():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "main.py", "smoke"], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    perf = repo_root / "outputs" / "tables" / "performance_summary.csv"
    assert perf.exists()
    df = pd.read_csv(perf)
    assert len(df) > 0
