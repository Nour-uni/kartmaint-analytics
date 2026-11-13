"""Train and evaluate the breakdown-risk model.

Honest-evaluation choices:
  * TIME-BASED split (train < validation < test in chronological order), with a 14-day gap between
    splits because each label looks 14 days into the future. No random shuffling.
  * Class imbalance is real (~9% positives), so we report PR-AUC (average precision) and
    precision/recall in the top 10% of risk scores, not accuracy. We do NOT re-weight classes, so the
    predicted probabilities stay roughly calibrated (see mean_predicted_probability vs positive_rate).
  * Two comparison points: a naive rule ("days since last preventive maintenance") and logistic
    regression. The gradient-boosting model has to earn its place.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kartmaint.config import FEATURES_DIR, HORIZON_DAYS, MODEL_DIR
from kartmaint.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

LABEL = "breakdown_next_14d"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def time_split(df: pd.DataFrame, train_frac=0.6, val_frac=0.2):
    days = np.sort(df["day"].unique())
    t_val = days[int(len(days) * train_frac)]
    t_test = days[int(len(days) * (train_frac + val_frac))]
    gap = pd.Timedelta(days=HORIZON_DAYS)
    train = df[df["day"] < t_val - gap]
    val = df[(df["day"] >= t_val) & (df["day"] < t_test - gap)]
    test = df[df["day"] >= t_test]
    return train, val, test


def metrics(y, score, top_frac=0.10):
    k = max(int(len(y) * top_frac), 1)
    top = np.argsort(-score)[:k]
    y = np.asarray(y)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "precision_at_top10pct": float(y[top].mean()),
        "recall_at_top10pct": float(y[top].sum() / max(y.sum(), 1)),
        "positive_rate": float(y.mean()),
    }


def make_models():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return {
        "logistic_regression": Pipeline(
            [("pre", pre), ("clf", LogisticRegression(max_iter=2000))]
        ),
        "gradient_boosting": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=42
                    ),
                ),
            ]
        ),
    }


def run(features_dir: Path = FEATURES_DIR, model_dir: Path = MODEL_DIR) -> dict:
    df = pd.read_parquet(features_dir / "train_set")
    df["day"] = pd.to_datetime(df["day"])
    train, val, test = time_split(df)
    print(f"rows  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"dates train<{train['day'].max().date()}  val<{val['day'].max().date()}  test<={test['day'].max().date()}")

    results = {"split_sizes": {"train": len(train), "val": len(val), "test": len(test)}}
    candidates = make_models()

    # naive rule baseline: the longer since the last preventive service, the riskier
    results["baseline_rule_days_since_preventive"] = {
        "validation": metrics(val[LABEL], val["days_since_last_preventive"].to_numpy()),
        "test": metrics(test[LABEL], test["days_since_last_preventive"].to_numpy()),
    }

    # choose the best candidate on VALIDATION, then refit on train+val and report TEST once
    val_pr = {}
    for name, pipe in candidates.items():
        pipe.fit(train[FEATURES], train[LABEL])
        val_pr[name] = metrics(val[LABEL], pipe.predict_proba(val[FEATURES])[:, 1])
        results[name] = {"validation": val_pr[name]}
    best = max(val_pr, key=lambda n: val_pr[n]["pr_auc"])

    trainval = pd.concat([train, val])
    for name, pipe in make_models().items():
        pipe.fit(trainval[FEATURES], trainval[LABEL])
        proba = pipe.predict_proba(test[FEATURES])[:, 1]
        results[name]["test"] = metrics(test[LABEL], proba)
        results[name]["test"]["brier"] = float(brier_score_loss(test[LABEL], proba))
        results[name]["test"]["mean_predicted_probability"] = float(proba.mean())  # calibration sanity check vs positive_rate
        if name == best:
            final = pipe
            final_test_proba = proba

    perm = permutation_importance(
        final, test[FEATURES], test[LABEL], scoring="average_precision", n_repeats=5, random_state=42
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean, "std": perm.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    results["selected_model"] = best
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": FEATURES, "name": best}, model_dir / "model.joblib")
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(
        {"kart_id": test["kart_id"].to_numpy(), "day": test["day"].to_numpy(), "y": test[LABEL].to_numpy(), "p": final_test_proba}
    ).to_parquet(model_dir / "test_predictions.parquet", index=False)

    print(json.dumps({k: v for k, v in results.items() if k != "split_sizes"}, indent=2))
    print("\nTop features (permutation importance on test, PR-AUC drop):")
    print(importance.head(8).to_string(index=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    a = ap.parse_args()
    run(a.features_dir, a.model_dir)
