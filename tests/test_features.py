"""Independent pandas re-computation of Spark features on random rows: proves no future leakage."""
import numpy as np
import pandas as pd

from kartmaint.config import PARTS
from kartmaint.features import NUMERIC_FEATURES


def _load(pipeline):
    feats = pd.read_parquet(pipeline["features"] / "train_set")
    feats["day"] = pd.to_datetime(feats["day"])
    sessions = pd.read_parquet(pipeline["clean"] / "sessions")
    events = pd.read_parquet(pipeline["clean"] / "events")
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    return feats, sessions, events


def test_features_match_independent_pandas_computation(pipeline):
    feats, sessions, events = _load(pipeline)
    sample = feats.sample(60, random_state=0)
    for _, r in sample.iterrows():
        k, d = r["kart_id"], r["day"]
        s = sessions[sessions.kart_id == k]
        e = events[events.kart_id == k]

        past = lambda n: s[(s.session_date >= d - pd.Timedelta(days=n)) & (s.session_date <= d - pd.Timedelta(days=1))]
        assert np.isclose(past(7).hours_used.sum(), r["hours_7d"])
        assert np.isclose(past(30).hours_used.sum(), r["hours_30d"])

        b = e[e.is_breakdown]
        in_window = lambda n: b[(b.event_date >= d - pd.Timedelta(days=n)) & (b.event_date <= d - pd.Timedelta(days=1))]
        assert len(in_window(90)) == r["breakdowns_90d"]
        assert len(in_window(365)) == r["breakdowns_365d"]

        for p in PARTS:
            svc = e[(e.part == p) & e.event_type.isin(["preventive", "corrective"]) & (e.event_date < d)]
            since = svc.event_date.max() if len(svc) else pd.Timestamp.min
            expected = s[(s.session_date > since) & (s.session_date < d)].hours_used.sum()
            assert np.isclose(expected, r[f"hours_since_{p}"]), (k, d, p)

        # label: any breakdown in days d+1 ... d+14
        future = b[(b.event_date > d) & (b.event_date <= d + pd.Timedelta(days=14))]
        assert int(len(future) > 0) == r["breakdown_next_14d"]


def test_training_rows_are_mondays_with_complete_label(pipeline):
    feats, _, _ = _load(pipeline)
    assert (feats["day"].dt.dayofweek == 0).all()
    assert feats["breakdown_next_14d"].notna().all()
    assert set(NUMERIC_FEATURES) <= set(feats.columns)


def test_latest_snapshot_has_one_row_per_kart(pipeline):
    latest = pd.read_parquet(pipeline["features"] / "latest")
    assert latest["kart_id"].is_unique
    assert latest["breakdown_next_14d"].isna().all()
