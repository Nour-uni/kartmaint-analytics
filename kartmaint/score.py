"""Batch scoring: score every kart for the next 14 days using the latest feature snapshot.

Output: data/predictions/latest_scores.parquet (+ .csv), read by the dashboard.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kartmaint.config import FEATURES_DIR, MODEL_DIR, PARTS, PRED_DIR, SERVICE_INTERVAL_HOURS


def run(features_dir: Path = FEATURES_DIR, model_dir: Path = MODEL_DIR, out_dir: Path = PRED_DIR) -> pd.DataFrame:
    bundle = joblib.load(model_dir / "model.joblib")
    latest = pd.read_parquet(features_dir / "latest")
    latest["day"] = pd.to_datetime(latest["day"])

    scores = bundle["pipeline"].predict_proba(latest[bundle["features"]])[:, 1]
    out = latest.copy()
    out["risk_score"] = scores
    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    n = len(out)
    out["risk_band"] = np.where(out["risk_rank"] <= max(int(n * 0.10), 1), "high",
                         np.where(out["risk_rank"] <= int(n * 0.30), "medium", "low"))
    ratios = pd.DataFrame({p: out[f"hours_since_{p}"] / SERVICE_INTERVAL_HOURS[p] for p in PARTS})
    out["most_overdue_part"] = ratios.idxmax(axis=1)
    out["most_overdue_ratio"] = ratios.max(axis=1).round(2)
    out["data_through"] = (out["day"] - pd.Timedelta(days=1)).dt.date.astype(str)
    out["model_name"] = bundle["name"]
    out = out.sort_values("risk_rank").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "latest_scores.parquet", index=False)
    out.to_csv(out_dir / "latest_scores.csv", index=False)
    print(out[["kart_id", "model", "risk_score", "risk_band", "most_overdue_part", "most_overdue_ratio"]].head(10).to_string(index=False))
    print(f"\nscored {len(out)} karts with model '{bundle['name']}', data through {out['data_through'].iloc[0]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=PRED_DIR)
    a = ap.parse_args()
    run(a.features_dir, a.model_dir, a.out_dir)
