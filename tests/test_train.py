import numpy as np
import pandas as pd

from kartmaint.train import metrics, time_split


def test_time_split_is_chronological_with_gap():
    days = pd.date_range("2024-01-01", periods=300, freq="D")
    df = pd.DataFrame({"day": np.repeat(days, 2), "kart_id": 1})
    train, val, test = time_split(df)
    assert train["day"].max() + pd.Timedelta(days=14) <= val["day"].min()
    assert val["day"].max() + pd.Timedelta(days=14) <= test["day"].min()


def test_metrics_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    perfect = metrics(y, y.astype(float))
    assert perfect["roc_auc"] == 1.0 and perfect["precision_at_top10pct"] == 1.0
    random = metrics(y, np.random.default_rng(0).random(100))
    assert 0.3 < random["roc_auc"] < 0.8
