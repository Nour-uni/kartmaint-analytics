import numpy as np
import pandas as pd

from kartmaint import generate_data as g
from kartmaint.config import VALID_PARTS


def test_same_seed_same_data():
    a = g.simulate(10, "2024-01-01", "2024-03-01", seed=1)
    b = g.simulate(10, "2024-01-01", "2024-03-01", seed=1)
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)


def test_different_seed_different_data():
    a = g.simulate(10, "2024-01-01", "2024-03-01", seed=1)[1]
    b = g.simulate(10, "2024-01-01", "2024-03-01", seed=2)[1]
    assert not a.equals(b)


def test_clean_simulation_invariants():
    karts, sessions, events = g.simulate(20, "2024-01-01", "2024-06-30", seed=3)
    assert sessions["hours_used"].between(0.2, 0.6).all()
    assert set(events["part"]) <= set(VALID_PARTS)
    assert set(events["event_type"]) <= {"preventive", "corrective", "inspection"}
    merged = events.merge(karts[["kart_id", "purchase_date"]], on="kart_id")
    assert (merged["event_date"] >= merged["purchase_date"]).all()
    # every breakdown is a corrective event
    assert (events.loc[events.is_breakdown == 1, "event_type"] == "corrective").all()
